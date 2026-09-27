"""Stage 4 - pair matcher: two-stage XGBoost on the GPU (CUDA), grouped validation, thresholds.

XGBoost (Apache-2.0) is used instead of LightGBM because the LightGBM pip wheel has no GPU
support on Windows, and training/inference are required to run on the GPU. Hyper-parameters
mirror the LightGBM spec (leaf-wise growth, 127 leaves, 0.8 row/column sampling, L2 = 1).

  v1: pair features -> probability p1.
  v2: pair features + stage-2 cluster features built from p1 (see stage2.py). The fit set's p1
      comes from 3-fold out-of-fold v1 models so v2 never sees in-sample probabilities.

Fit / validation S1 entities are sampled by whole (country, city) partitions, so the S1s that
compete for the same S2/S3 records mostly land in the same set (as they do at test time).

    python train.py [--n-fit 400000] [--n-val 120000] [--lr 0.05]
    python train.py --fit-country US --val-country India      # cross-country check
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

import config as C
import io_utils as io
from blocking import cand_path
from decide import decide, save_thresholds, tune, tune_expected
from features import add_group_context, feature_names, iter_features
from metrics import per_entity_f05
from normalize import load_norm
from stage2 import S2_FEATS, cluster_features

MODEL_PATH = C.MODELS_DIR / "xgb_v1.ubj"
MODEL2_PATH = C.MODELS_DIR / "xgb_v2.ubj"
FEATS_PATH = C.MODELS_DIR / "features_v1.json"
N_OOF = 3
V1_CACHE = C.CACHE_DIR / "train_v1"
CE_BAND = (0.01, 0.995)     # cross-encoder scores only pairs whose v1 probability is uncertain
V2_MIN_P1 = 0.002           # v2 is trained / applied only where v1 is not already certain it is a non-match


def s1_partitions(country=None) -> pl.DataFrame:
    """Train S1 indices with a partition key (country|city) and a random partition order."""
    rec = io.load_records("train", ["idx", "src", "country"]).filter(pl.col("src") == 1)
    if country is not None:
        rec = rec.filter(pl.col("country") == country)
    rec = rec.join(load_norm("train", ["idx", "city"]), on="idx")
    key = pl.when(pl.col("city") != "").then(pl.col("country") + "|" + pl.col("city")) \
            .otherwise(pl.col("country") + "|#" + (pl.col("idx") % 997).cast(pl.Utf8))
    rec = rec.with_columns(key.alias("part"))
    parts = rec.select("part").unique().sort("part")
    rng = np.random.default_rng(C.SEED)
    parts = parts.with_columns(pl.Series("order", rng.permutation(parts.height)))
    return rec.join(parts, on="part").sort("order", "idx")


def split_s1(n_fit: int, n_val: int, fit_country=None, val_country=None):
    """Disjoint fit / validation S1 sets made of whole (country, city) partitions.

    Returns (fit_s1, val_s1, fit_part) where fit_part is the partition key of each fit S1
    (used for grouped out-of-fold folds).
    """
    def take(r: pl.DataFrame, lo: int, hi: int) -> pl.DataFrame:
        """S1s of the partitions whose cumulative size (in random order) falls in (lo, hi]."""
        sizes = r.group_by("order", maintain_order=True).len()
        cum = sizes["len"].cum_sum()
        keep = sizes.filter((cum > lo) & (cum <= hi))["order"]
        return r.filter(pl.col("order").is_in(keep.implode()))

    if fit_country is None:
        r = s1_partitions()
        fit, val = take(r, 0, n_fit), take(r, n_fit, n_fit + n_val)
    else:
        fit = take(s1_partitions(fit_country), 0, n_fit)
        val = take(s1_partitions(val_country), 0, n_val)
    fit, val = fit.sort("idx"), val.sort("idx")
    return fit["idx"].to_numpy(), val["idx"].to_numpy(), fit["order"].to_numpy()


def labelled_matrix(fit_s1: np.ndarray, val_s1: np.ndarray, extra_cols: int):
    """Feature matrix + labels for all candidate pairs of the fit and validation S1s.

    Rows are ordered fit-first, so X[:n_fit] / X[n_fit:] are views (no copies); the matrix is
    preallocated with `extra_cols` spare columns that stage 2 fills in place. Group context
    (ranks, gaps, competition) is computed with every S1 that competes for these S1s'
    candidates, so it matches what test inference sees.
    Returns ids (s1_idx, cand_idx), X (float32), y (int8), n_fit, feature names.
    """
    cands = pl.read_parquet(cand_path("train"))
    sel = pl.Series(np.concatenate([fit_s1, val_s1])).implode()
    their = cands.filter(pl.col("s1_idx").is_in(sel)).select("cand_idx").unique()
    cands = add_group_context(cands.join(their, on="cand_idx", how="semi"))
    cands = (cands.filter(pl.col("s1_idx").is_in(sel))
                  .with_columns(pl.col("s1_idx").is_in(pl.Series(val_s1).implode()).alias("is_val"))
                  .sort("is_val", "s1_idx"))
    n, n_fit = cands.height, int((~cands["is_val"]).sum())
    X, fcols, ids, s = None, None, [], 0
    for fdf in iter_features("train", cands.drop("is_val")):
        if X is None:
            fcols = feature_names(fdf)
            X = np.empty((n, len(fcols) + extra_cols), np.float32)
        X[s:s + fdf.height, :len(fcols)] = fdf.select(fcols).to_numpy()
        ids.append(fdf.select("s1_idx", "cand_idx"))
        s += fdf.height
    del cands
    ids = pl.concat(ids)
    gt = io.load_gt_pairs().with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    y = ids.join(gt, on=["s1_idx", "cand_idx"], how="left", maintain_order="left")["label"]            .fill_null(0).to_numpy().astype(np.int8)
    return ids, X, y, n_fit, fcols


def predict_proba(bst: "xgb.Booster", X: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
    """Probabilities from the best iteration, predicted on the GPU in batches."""
    it = (0, bst.best_iteration + 1) if bst.attr("best_iteration") is not None else (0, 0)
    return np.concatenate([bst.inplace_predict(X[s:s + batch], iteration_range=it)
                           for s in range(0, len(X), batch)]).astype(np.float32)


def fit_xgb(params, X, y, fcols, Xv=None, yv=None, rounds=C.XGB_ROUNDS, label="xgboost", weight=None):
    """Train XGBoost on the GPU; early stopping on (Xv, yv) when given, else fixed rounds."""
    import xgboost as xgb        # imported lazily so spawned feature workers stay light
    with io.stage(f"{label} (cuda)"):
        dtr = xgb.QuantileDMatrix(X, y, weight=weight, feature_names=fcols, max_bin=params["max_bin"])
        if Xv is None:
            return xgb.train(params, dtr, rounds, verbose_eval=False)
        dva = xgb.QuantileDMatrix(Xv, yv, ref=dtr, feature_names=fcols)
        bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")],
                        early_stopping_rounds=C.XGB_EARLY_STOP, verbose_eval=200)
    print(f"  {label}: best iteration {bst.best_iteration} | val logloss {bst.best_score:.5f}")
    return bst


def importance(bst: "xgb.Booster", fcols: list[str]) -> pl.DataFrame:
    """Gain importance of every feature (0 for unused ones)."""
    gain = bst.get_score(importance_type="gain")
    return pl.DataFrame({"feature": fcols, "gain": [gain.get(f, 0.0) for f in fcols]}).sort("gain", descending=True)


def report(pv: pl.DataFrame, val_s1: np.ndarray, truth: pl.DataFrame, owned: pl.Series, tag: str):
    """Tune thresholds (plain + distractor-weighted) and print per-country validation F0.5."""
    with io.stage(f"tune decision {tag}"):
        th1 = tune(pv, val_s1, truth)
        th = tune(pv, val_s1, truth, owned=owned)
        te = tune_expected(pv, val_s1, truth, owned)
    if te["objective"] > th["objective"]:
        th = te
    pred = decide(pv, th)
    rec = io.load_records("train", ["idx", "country"])
    pe = per_entity_f05(val_s1, pred, truth).join(rec.rename({"idx": "s1_idx"}), on="s1_idx")
    print(pe.group_by("country").agg(pl.col("f05").mean(), pl.len()))
    print(f"VALIDATION {tag} macro F0.5 = {pe['f05'].mean():.4f} with {th} "
          f"(plain-optimal thresholds: {th1['f05']:.4f} at t={th1['t']}, t_empty={th1['t_empty']})")
    return th


def main():
    """Train v1 + v2 on the GPU, tune thresholds on validation, save everything."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit", type=int, default=300_000)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=C.XGB_PARAMS["learning_rate"])
    ap.add_argument("--rounds", type=int, default=C.XGB_ROUNDS)
    ap.add_argument("--fit-country", default=None, help="cross-country check: fit only on this label")
    ap.add_argument("--val-country", default=None, help="cross-country check: validate only on this label")
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--stage1-only", action="store_true",
                    help="stop after v1 + OOF probabilities are cached (before stage 2)")
    ap.add_argument("--no-ce", action="store_true", help="do not use the cross-encoder feature")
    ap.add_argument("--no-sib", action="store_true", help="do not use the sibling-business features")
    ap.add_argument("--expand", action="store_true", help="add expand.py rows to the v2 training set")
    ap.add_argument("--pseudo-te", action="store_true",
                    help="add confident test pseudo-labels to the sibling target encodings")
    ap.add_argument("--reuse", action="store_true", help="reuse cached v1 matrices + OOF probabilities")
    a = ap.parse_args()
    C.set_seeds()
    cross = a.fit_country is not None
    fit_s1, val_s1, fit_part = split_s1(a.n_fit, a.n_val, a.fit_country, a.val_country)
    print(f"S1 fit {len(fit_s1):,} | val {len(val_s1):,}")
    from ce import CE_DIR
    use_ce = (not a.no_ce) and (CE_DIR / "config.json").exists()
    extra = [] if a.no_stage2 else S2_FEATS + (["ce_logit"] if use_ce else [])
    gt = io.load_gt_pairs()
    truth = gt.filter(pl.col("s1_idx").is_in(pl.Series(val_s1).implode()))
    owned = gt["cand_idx"]
    params = dict(C.XGB_PARAMS, learning_rate=a.lr)

    if a.reuse and (V1_CACHE / "meta.json").exists():
        # ---- reuse cached v1 features + out-of-fold probabilities (skips ~25 min)
        meta = json.loads((V1_CACHE / "meta.json").read_text())
        n_fit, fcols = meta["n_fit"], meta["fcols"]
        F = len(fcols)
        base = np.load(V1_CACHE / "X.npy", mmap_mode="r")
        X = np.empty((base.shape[0], F + len(extra)), np.float32)
        for s_ in range(0, base.shape[0], 2_000_000):
            X[s_:s_ + 2_000_000, :F] = base[s_:s_ + 2_000_000]
        del base
        ids = pl.read_parquet(V1_CACHE / "ids.parquet")
        y = np.load(V1_CACHE / "y.npy")
        p1_fit, p1_val = np.load(V1_CACHE / "p1_fit.npy"), np.load(V1_CACHE / "p1_val.npy")
        th = json.loads((V1_CACHE / "th_v1.json").read_text())
        y_fit, y_val = y[:n_fit], y[n_fit:]
        fit_ids, val_ids = ids.slice(0, n_fit), ids.slice(n_fit)
        print(f"reused v1 cache: pairs fit {n_fit:,} val {len(y_val):,}")
    else:
        with io.stage("features train"):
            ids, X, y, n_fit, fcols = labelled_matrix(fit_s1, val_s1, len(extra))
        F = len(fcols)
        X_fit, X_val = X[:n_fit, :F], X[n_fit:, :F]            # views, no copies
        y_fit, y_val = y[:n_fit], y[n_fit:]
        fit_ids, val_ids = ids.slice(0, n_fit), ids.slice(n_fit)
        print(f"pairs fit {n_fit:,} val {len(y_val):,} | pos rate {y_fit.mean():.3f} | {F} feats")
        bst1 = fit_xgb(params, X_fit, y_fit, fcols, X_val, y_val, a.rounds, "v1")
        imp1 = importance(bst1, fcols)
        print(imp1.head(15))
        p1_val = predict_proba(bst1, X_val)
        pv1 = val_ids.with_columns(pl.Series("prob", p1_val))
        th = report(pv1, val_s1, truth, owned, "v1")
        if not cross:
            bst1.save_model(str(MODEL_PATH))
            FEATS_PATH.write_text(json.dumps(fcols))
            imp1.write_csv(C.MODELS_DIR / "feature_importance.csv")
            save_thresholds(dict(th, model="v1"))
            pv1.write_parquet(C.CACHE_DIR / "val_probs.parquet")
        if a.no_stage2:
            return
        # out-of-fold p1 on the fit set (held-out fold gets weight 0)
        part_of = dict(zip(fit_s1.tolist(), (fit_part % N_OOF).tolist()))
        fold = np.array([part_of[s] for s in fit_ids["s1_idx"].to_list()], dtype=np.int8)
        p1_fit = np.zeros(n_fit, np.float32)
        n_rounds = bst1.best_iteration + 1
        for k in range(N_OOF):
            b = fit_xgb(params, X_fit, y_fit, fcols, rounds=n_rounds, label=f"v1 oof fold {k}",
                        weight=(fold != k).astype(np.float32))
            p1_fit[fold == k] = predict_proba(b, X_fit)[fold == k]
            del b
        if not cross:
            with io.stage("cache v1 matrices"):
                V1_CACHE.mkdir(exist_ok=True)
                np.save(V1_CACHE / "X.npy", np.ascontiguousarray(X[:, :F]))
                ids.write_parquet(V1_CACHE / "ids.parquet")
                np.save(V1_CACHE / "y.npy", y)
                np.save(V1_CACHE / "p1_fit.npy", p1_fit)
                np.save(V1_CACHE / "p1_val.npy", p1_val)
                (V1_CACHE / "th_v1.json").write_text(json.dumps(th))
                (V1_CACHE / "meta.json").write_text(json.dumps({"n_fit": n_fit, "fcols": fcols}))
        if a.stage1_only:
            print("stage 1 done (v1 model, OOF probabilities and matrices cached)")
            return

    # ---------------- stage 2: cluster / distractor features (+ cross-encoder logit)
    src_of = io.load_records("train", ["src"])["src"].to_numpy()
    with io.stage("stage-2 features"):
        X[:n_fit, F:F + len(S2_FEATS)] = cluster_features(
            "train", fit_ids.with_columns(pl.Series("p1", p1_fit)), src_of).select(S2_FEATS).to_numpy()
        X[n_fit:, F:F + len(S2_FEATS)] = cluster_features(
            "train", val_ids.with_columns(pl.Series("p1", p1_val)), src_of).select(S2_FEATS).to_numpy()
    if use_ce:
        # cross-encoder logits come from `python ce.py --score train` (separate process, GPU freed)
        ce_path = V1_CACHE / "ce_logit.npy"
        if not ce_path.exists():
            raise SystemExit("run `python ce.py --score train` first (needs the v1 cache)")
        X[:, -1] = np.load(ce_path)
    fcols2 = fcols + extra
    # v2 only sees pairs v1 does not already reject (p1 >= V2_MIN_P1); the rest keep p1. This
    # keeps the v2 matrix small enough for 8 GB of VRAM and focuses it on the ambiguous pairs.
    mf, mv = p1_fit >= V2_MIN_P1, p1_val >= V2_MIN_P1
    print(f"v2 rows: fit {mf.sum():,}/{n_fit:,} | val {mv.sum():,}/{len(p1_val):,} | "
          f"positives kept {y_fit[mf].sum() / max(1, y_fit.sum()):.4f}")
    Xf2, Xv2 = X[:n_fit][mf], X[n_fit:][mv]
    use_sib = not a.no_sib
    if use_sib:
        # sibling-business features (house-number gap, target-encoded extra/missing name words,
        # legal-form pair); encodings learned on train S1s outside fit/val
        from sibling import SIB_FEATS, build_target_encodings, sibling_features
        with io.stage("sibling features"):
            build_target_encodings(np.concatenate([fit_s1, val_s1]), pseudo=a.pseudo_te)
            Xf2 = np.hstack([Xf2, sibling_features("train", fit_ids.filter(pl.Series(mf)))])
            Xv2 = np.hstack([Xv2, sibling_features("train", val_ids.filter(pl.Series(mv)))])
        fcols2 = fcols2 + SIB_FEATS
    yf2 = y_fit[mf]
    if a.expand:
        # extra out-of-sample S1s (expand.py): v1 features + stage-2 + CE logit + sibling features
        from expand import EXT_CACHE
        Xe = np.load(EXT_CACHE / "X.npy")
        parts_ = [Xe]
        if use_ce:
            parts_.append(np.load(EXT_CACHE / "ce_logit.npy")[:, None])
        if use_sib:
            parts_.append(sibling_features("train", pl.read_parquet(EXT_CACHE / "ids.parquet")))
        Xe = np.hstack(parts_)
        assert Xe.shape[1] == Xf2.shape[1], (Xe.shape, Xf2.shape)
        Xf2, yf2 = np.vstack([Xf2, Xe]), np.concatenate([yf2, np.load(EXT_CACHE / "y.npy")])
        del Xe, parts_
        print(f"v2 fit rows with expansion: {len(yf2):,}")
    bst2 = fit_xgb(params, Xf2, yf2, fcols2, Xv2, y_val[mv], a.rounds, "v2")
    del Xf2
    imp2 = importance(bst2, fcols2)
    print(imp2.head(20))
    p2_val = p1_val.copy()
    p2_val[mv] = predict_proba(bst2, Xv2)
    pv2 = val_ids.with_columns(pl.Series("prob", p2_val))
    th2 = report(pv2, val_s1, truth, owned, "v2")
    if not cross:
        bst2.save_model(str(MODEL2_PATH))
        (C.MODELS_DIR / "features_v2.json").write_text(json.dumps(fcols2))
        imp2.write_csv(C.MODELS_DIR / "feature_importance_v2.csv")
        if th2["objective"] > th["objective"]:
            save_thresholds(dict(th2, model="v2", ce=use_ce, sib=use_sib))
            pv2.write_parquet(C.CACHE_DIR / "val_probs.parquet")
            print("-> using v2 for inference")
        else:
            print("-> v2 did not improve; keeping v1 for inference")


if __name__ == "__main__":
    main()
