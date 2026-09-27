"""Test inference: features -> v1 -> (stage-2 features -> v2) -> decision -> output/*.tsv
(+ official validator). Pair features are spilled to parquet between the two passes.

    python predict.py
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl

import config as C
import io_utils as io
from blocking import cand_path
from decide import decide, load_thresholds
from features import add_group_context, iter_features
from stage2 import S2_FEATS, cluster_features


def score_split(split: str, bst, fcols: list[str], spill: bool) -> pl.DataFrame:
    """Pass 1: features + v1 probability for every candidate pair (GPU inference).

    With spill=True each chunk's feature matrix is also written to cache/<split>_feats/ so
    the v2 pass can reuse it without recomputing string features.
    """
    from train import predict_proba
    cands = add_group_context(pl.read_parquet(cand_path(split)))
    d = C.CACHE_DIR / f"{split}_feats"
    if spill:
        d.mkdir(exist_ok=True)
        for f in d.glob("*.parquet"):
            f.unlink()
    out, buf, part = [], [], 0
    for fdf in iter_features(split, cands):
        X = fdf.select(fcols).to_numpy().astype(np.float32)
        out.append(fdf.select("s1_idx", "cand_idx").with_columns(
            pl.Series("p1", predict_proba(bst, X))))
        if spill:
            buf.append(fdf.select(fcols))
            if sum(b.height for b in buf) >= 5_000_000:
                pl.concat(buf).write_parquet(d / f"part_{part:04d}.parquet")
                buf, part = [], part + 1
    if spill and buf:
        pl.concat(buf).write_parquet(d / f"part_{part:04d}.parquet")
    return pl.concat(out)


def score_v2(split: str, p1: pl.DataFrame, bst2, fcols: list[str], use_ce: bool = False,
             use_sib: bool = False) -> np.ndarray:
    """Pass 2: stage-2 cluster features (+ cross-encoder logit on the uncertain band) from p1,
    then v2 probabilities (same row order as p1)."""
    from train import V2_MIN_P1, predict_proba
    src_of = io.load_records(split, ["src"])["src"].to_numpy()
    with io.stage("stage-2 features"):
        s2 = cluster_features(split, p1, src_of).select(S2_FEATS).to_numpy().astype(np.float32)
    if use_ce:
        # logits from `python ce.py --score test` (separate process; same row order as test_probs)
        ce = np.load(C.CACHE_DIR / f"{split}_ce_logit.npy")
        assert len(ce) == len(s2), (len(ce), len(s2))
        s2 = np.hstack([s2, ce[:, None]])
    p1v = p1["p1"].to_numpy()
    sib, k = None, 0
    if use_sib:
        from sibling import sibling_features
        with io.stage("sibling features"):
            sib = sibling_features(split, p1.filter(pl.Series(p1v >= V2_MIN_P1)))
    out, s = [], 0
    for f in sorted((C.CACHE_DIR / f"{split}_feats").glob("part_*.parquet")):
        X = pl.read_parquet(f).select(fcols).to_numpy().astype(np.float32)
        pr = p1v[s:s + len(X)].copy()
        m = pr >= V2_MIN_P1            # v2 only where v1 is not already certain (as in training)
        if m.any():
            parts_ = [X[m], s2[s:s + len(X)][m]]
            if sib is not None:
                parts_.append(sib[k:k + m.sum()])
                k += m.sum()
            pr[m] = predict_proba(bst2, np.hstack(parts_))
        out.append(pr)
        s += len(X)
    assert s == len(s2), (s, len(s2))
    return np.concatenate(out)


def id_lists(pairs: pl.DataFrame, rec: pl.DataFrame, col: str) -> pl.DataFrame:
    """One row per S1 (in source order) with a comma-joined, de-duplicated S2/S3 id list."""
    ids = rec.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"))
    grouped = (pairs.select("s1_idx", "cand_idx").unique()
               .join(ids, on="cand_idx")
               .sort("s1_idx", "cand_idx")
               .group_by("s1_idx").agg(pl.col("cid").str.join(",").alias(col)))
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1_idx"),
                                               pl.col("entity_id").alias("source1_entity_id"))
    return (s1.join(grouped, on="s1_idx", how="left", maintain_order="left")
              .with_columns(pl.col(col).fill_null(""))
              .select("source1_entity_id", col))


def write_tsv(df: pl.DataFrame, path) -> None:
    """Write a two-column TSV with no quoting whatsoever."""
    df.write_csv(path, separator="\t", quote_style="never", include_header=True)


def check_outputs(match: pl.DataFrame, cand: pl.DataFrame, rec: pl.DataFrame) -> None:
    """Assert format rules: every S1 once, S2/S3 ids only, no dups, matches within candidates."""
    n_s1 = rec.filter(pl.col("src") == 1).height
    assert match.height == n_s1 == cand.height, (match.height, cand.height, n_s1)
    assert match["source1_entity_id"].n_unique() == n_s1
    for df, col in ((match, "matched_entity_ids"), (cand, "candidate_entity_ids")):
        ex = df.select(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "")
        assert (ex[col].str.starts_with("S2-") | ex[col].str.starts_with("S3-")).all()
        per = df.select(pl.col(col).str.split(",").list.len().alias("n"),
                        pl.col(col).str.split(",").list.n_unique().alias("u"))
        assert (per["n"] == per["u"]).all(), f"duplicate ids in {col}"
    m = match.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m"))
    c = cand.select("source1_entity_id", pl.col("candidate_entity_ids").str.split(",").alias("c"))
    j = m.join(c, on="source1_entity_id").with_columns(
        pl.col("m").list.set_difference("c").list.len().alias("extra"))
    bad = j.filter((pl.col("extra") > 0) & (pl.col("m").list.first() != ""))
    assert bad.height == 0, f"{bad.height} rows have matches outside candidates"
    print("  output assertions OK")


def run_validator(suffix: str = "") -> None:
    """Run the official validator from student_resource/ and print its output."""
    sr = C.DATA_DIR.parent
    cmd = [sys.executable, "utils/validate_submission.py",
           "--matching", str(C.OUTPUT_DIR / f"matching_results{suffix}.tsv"),
           "--candidate", str(C.OUTPUT_DIR / "candidate_pairs.tsv"),
           "--test-dir", "dataset/test"]
    r = subprocess.run(cmd, cwd=sr, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])


def write_candidates(probs: pl.DataFrame, matches: pl.DataFrame, rec: pl.DataFrame) -> None:
    """Write candidate_pairs.tsv = the pairs the final matcher (v2) scores.

    Candidate generation is a cascade: blocking retrieves ~46 candidates per S1, then the
    learned pruning stage (v1) keeps only pairs with p1 >= V2_MIN_P1 (~6 per S1, 99.95% of the
    reachable true matches); only these reach the final matcher, so they are the candidate set.
    Final matches are unioned in as a guarantee that matches are a subset of candidates."""
    from train import V2_MIN_P1
    kept = probs.filter(pl.col("p1") >= V2_MIN_P1).select("s1_idx", "cand_idx")
    cand = pl.concat([kept, matches.select("s1_idx", "cand_idx")]).unique()
    n_s1 = rec.filter(pl.col("src") == 1).height
    print(f"candidate pairs after pruning: {cand.height:,} ({cand.height / n_s1:.2f} per S1; "
          f"retrieved before pruning: {probs.height:,}, {probs.height / n_s1:.2f} per S1)")
    with io.stage("write candidate_pairs.tsv"):
        write_tsv(id_lists(cand, rec, "candidate_entity_ids"), C.OUTPUT_DIR / "candidate_pairs.tsv")


def candidates_only() -> None:
    """Rewrite candidate_pairs.tsv from cached test v1 probabilities, keeping the existing
    matching_results.tsv unchanged (its pairs are unioned into the candidates)."""
    probs = pl.read_parquet(C.CACHE_DIR / "test_probs.parquet", columns=["s1_idx", "cand_idx", "p1"])
    rec = io.load_records("test", ["idx", "entity_id", "src", "country"])
    ids = rec.select(pl.col("idx"), pl.col("entity_id"))
    m = io.read_tsv(C.OUTPUT_DIR / "matching_results.tsv")
    m = (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .filter(pl.col("matched_entity_ids") != "")
          .join(ids.rename({"idx": "s1_idx", "entity_id": "source1_entity_id"}), on="source1_entity_id")
          .join(ids.rename({"idx": "cand_idx", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
          .select("s1_idx", "cand_idx"))
    from train import V2_MIN_P1
    chk = m.join(probs, on=["s1_idx", "cand_idx"], how="left")
    print(f"final matches: {m.height:,} | found in regenerated candidates: {chk['p1'].is_not_null().mean():.5f} "
          f"| with p1 >= {V2_MIN_P1}: {(chk['p1'] >= V2_MIN_P1).mean():.5f}")
    write_candidates(probs, m, rec)
    run_validator("")


def write_only(suffix: str, strict: float | None, suspect_t: float | None = None) -> None:
    """Decide from the cached test probabilities, write matching_results<suffix>.tsv and (for the
    main file) the pruned candidate_pairs.tsv; run the official validator."""
    from decide import apply_rule
    probs = pl.read_parquet(C.CACHE_DIR / "test_probs.parquet", columns=["s1_idx", "cand_idx", "p1", "prob"])
    th = load_thresholds()
    if strict is not None and suspect_t is not None:
        from sibling import region_strict
        pred = region_strict("test", probs, strict, suspect_t)
    elif strict is not None:
        pred = apply_rule(probs, strict, strict)
    else:
        pred = decide(probs, th)
    rec = io.load_records("test", ["idx", "entity_id", "src", "country"])
    match = id_lists(pred, rec, "matched_entity_ids")
    write_tsv(match, C.OUTPUT_DIR / f"matching_results{suffix}.tsv")
    if not suffix:
        write_candidates(probs, pred, rec)
    del probs
    st = (pred.group_by("s1_idx").len()
          .join(rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1_idx"), "country"),
                on="s1_idx", how="right").with_columns(pl.col("len").fill_null(0)))
    print(f"rule: {'t=' + str(strict) + (' suspect_t=' + str(suspect_t) if suspect_t else '') if strict is not None else th}")
    print(st.group_by("country").agg(pl.col("len").mean().alias("avg_matches"),
                                     (pl.col("len") == 0).mean().alias("pct_empty"), pl.len()))
    run_validator(suffix)


def main():
    """Score test candidates, decide, write both output files, validate, print stats."""
    import xgboost as xgb        # heavy CUDA libraries: main process only, never in workers
    from train import FEATS_PATH, MODEL2_PATH, MODEL_PATH
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-v1", action="store_true",
                    help="reuse cached test v1 probabilities + spilled features (skip pass 1)")
    ap.add_argument("--suffix", default="", help="write matching_results<suffix>.tsv (keeps the main file)")
    ap.add_argument("--stage1-only", action="store_true",
                    help="features + v1 only: cache test p1 and spilled features")
    ap.add_argument("--candidates-only", action="store_true",
                    help="rewrite the pruned candidate_pairs.tsv from cached p1; keep matching_results.tsv")
    ap.add_argument("--write-only", action="store_true",
                    help="skip scoring: decide from cached test_probs.parquet and write only the matches")
    ap.add_argument("--strict", type=float, default=None,
                    help="with --write-only: plain threshold t=t_empty=STRICT instead of the tuned rule")
    ap.add_argument("--suspect-t", type=float, default=None,
                    help="with --strict: sibling-suspect pairs need prob >= SUSPECT_T")
    a = ap.parse_args()
    if a.write_only:
        write_only(a.suffix, a.strict, a.suspect_t)
        return
    if a.candidates_only:
        candidates_only()
        return
    C.N_JOBS = C.N_JOBS_INFER            # inference uses more CPU workers (string features, stage 2)
    print(f"inference workers: {C.N_JOBS}")
    bst = xgb.Booster(model_file=str(MODEL_PATH))
    bst.set_param({"device": C.DEVICE})
    fcols = json.loads(FEATS_PATH.read_text())
    th = load_thresholds()
    use_v2 = th.get("model") == "v2"
    print(f"thresholds {th}")
    if a.stage1_only:
        with io.stage("score test (v1)"):
            probs = score_split("test", bst, fcols, spill=True)
        probs.with_columns(pl.col("p1").alias("prob")).write_parquet(C.CACHE_DIR / "test_probs.parquet")
        return
    if a.reuse_v1:
        probs = pl.read_parquet(C.CACHE_DIR / "test_probs.parquet").select("s1_idx", "cand_idx", "p1")
        print(f"reusing cached test v1 probabilities ({probs.height:,} pairs)")
    else:
        with io.stage("score test (v1)"):
            probs = score_split("test", bst, fcols, spill=use_v2)
    if use_v2:
        bst2 = xgb.Booster(model_file=str(MODEL2_PATH))
        bst2.set_param({"device": C.DEVICE})
        with io.stage("score test (v2)"):
            probs = probs.with_columns(pl.Series("prob", score_v2("test", probs, bst2, fcols,
                                                                  use_ce=bool(th.get("ce")),
                                                                  use_sib=bool(th.get("sib")))))
    else:
        probs = probs.with_columns(pl.col("p1").alias("prob"))
    probs.write_parquet(C.CACHE_DIR / "test_probs.parquet")
    pred = decide(probs, th)
    rec = io.load_records("test", ["idx", "entity_id", "src", "country"])
    with io.stage("write outputs"):
        match = id_lists(pred, rec, "matched_entity_ids")
        write_tsv(match, C.OUTPUT_DIR / f"matching_results{a.suffix}.tsv")
        if not a.reuse_v1:      # candidates only change when blocking/v1 scoring is rerun
            cand = id_lists(probs, rec, "candidate_entity_ids")
            write_tsv(cand, C.OUTPUT_DIR / "candidate_pairs.tsv")
            check_outputs(match, cand, rec)
    # per-country stats vs train (mean ~3.46 matches, 5.6% empty)
    st = (pred.group_by("s1_idx").len()
          .join(rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1_idx"), "country"),
                on="s1_idx", how="right").with_columns(pl.col("len").fill_null(0)))
    print(st.group_by("country").agg(pl.col("len").mean().alias("avg_matches"),
                                     (pl.col("len") == 0).mean().alias("pct_empty"), pl.len()))
    run_validator(a.suffix)


if __name__ == "__main__":
    main()
