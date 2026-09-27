"""Expand the stage-2 (v2) training set with more train S1 entities.

v1 is trained on the fit S1s only, so its probabilities on any *other* train S1 are genuinely
out-of-sample. For extra S1s (not fit/val, not used to train the cross-encoder, and not in the
half used for the sibling target encodings) we compute v1 features + p1, stage-2 cluster
features, sibling features and cross-encoder logits, and keep only the rows v2 uses
(p1 >= V2_MIN_P1). train.py --reuse --expand then fits v2 on fit + extra rows.

    python expand.py --n-s1 500000        # build cache/train_ext/*
    python ce.py --score ext              # cross-encoder logits for the extra band rows
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

import config as C
import io_utils as io

EXT_CACHE = C.CACHE_DIR / "train_ext"


def extra_s1(n: int) -> np.ndarray:
    """Train S1s outside fit/val, outside the cross-encoder training set, and with odd index
    (even-index S1s feed the sibling target encodings)."""
    from ce import ce_train_s1
    from train import split_s1
    fit_s1, val_s1, _ = split_s1(300_000, 100_000)
    rec = io.load_records("train", ["idx", "src"]).filter(pl.col("src") == 1)
    pool = np.setdiff1d(rec["idx"].to_numpy(), np.concatenate([fit_s1, val_s1, ce_train_s1()]))
    pool = pool[pool % 2 == 1]
    return np.sort(np.random.default_rng(C.SEED + 7).choice(pool, min(n, len(pool)), replace=False))


def build(n_s1: int, batch: int = 200_000) -> None:
    """Compute v2-ready rows for the extra S1s, batch by batch, and cache them."""
    import xgboost as xgb
    from stage2 import S2_FEATS, cluster_features
    from sibling import sibling_features
    from train import FEATS_PATH, MODEL_PATH, V2_MIN_P1, labelled_matrix, predict_proba
    s1 = extra_s1(n_s1)
    print(f"extra S1s: {len(s1):,}")
    bst = xgb.Booster(model_file=str(MODEL_PATH))
    bst.set_param({"device": C.DEVICE})
    fcols = json.loads(FEATS_PATH.read_text())
    src_of = io.load_records("train", ["src"])["src"].to_numpy()
    Xs, ys, ids_l, p1s = [], [], [], []
    for b in range(0, len(s1), batch):
        part = s1[b:b + batch]
        with io.stage(f"extra batch {b // batch}: features"):
            ids, X, y, _, fc = labelled_matrix(part, np.zeros(0, np.int64), 0)
        assert fc == fcols
        p1 = predict_proba(bst, X)
        s2 = cluster_features("train", ids.with_columns(pl.Series("p1", p1)), src_of).select(S2_FEATS).to_numpy()
        m = p1 >= V2_MIN_P1
        Xs.append(np.hstack([X[m], s2[m].astype(np.float32)]))
        ys.append(y[m]); ids_l.append(ids.filter(pl.Series(m))); p1s.append(p1[m])
        print(f"  kept {m.sum():,}/{len(p1):,} rows (positives kept {y[m].sum() / max(1, y.sum()):.4f})")
        del X, s2, ids
    X = np.vstack(Xs); y = np.concatenate(ys); ids = pl.concat(ids_l); p1 = np.concatenate(p1s)
    with io.stage("extra sibling features"):
        sib = sibling_features("train", ids)
    EXT_CACHE.mkdir(exist_ok=True)
    np.save(EXT_CACHE / "X.npy", X)
    np.save(EXT_CACHE / "sib.npy", sib)
    np.save(EXT_CACHE / "y.npy", y)
    np.save(EXT_CACHE / "p1.npy", p1)
    ids.write_parquet(EXT_CACHE / "ids.parquet")
    print(f"saved {len(y):,} extra v2 rows ({y.mean():.3f} positive) to {EXT_CACHE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=500_000)
    a = ap.parse_args()
    with io.stage("expand v2 training set"):
        build(a.n_s1)
