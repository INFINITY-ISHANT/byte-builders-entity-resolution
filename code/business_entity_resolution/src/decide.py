"""Stage 5 - decision step: one-owner assignment, threshold + empty-rule tuning on macro F0.5."""
from __future__ import annotations

import json

import numpy as np
import polars as pl

import config as C
from metrics import macro_f05

THRESH_PATH = C.MODELS_DIR / "thresholds.json"
# Test has ~5.75 S2/S3 records per S1 vs ~4.68 in train. With the same mean cluster size (3.46)
# that is ~2.29 vs ~1.22 distractors (records owned by no S1) per S1, i.e. ~1.9x more. When
# tuning thresholds, false positives on distractors are weighted by this factor so the chosen
# thresholds target the test distractor density instead of the train one.
DISTRACTOR_WEIGHT = 1.9


def one_owner(p: pl.DataFrame) -> pl.DataFrame:
    """Keep each S2/S3 candidate only for the S1 with the highest probability."""
    return (p.sort(["prob", "s1_idx"], descending=[True, False])
             .unique(subset="cand_idx", keep="first", maintain_order=False))


def apply_rule(p: pl.DataFrame, t: float, t_empty: float) -> pl.DataFrame:
    """Predicted pairs: prob >= t after one-owner, and S1s whose best prob < t_empty emptied.

    p: frame with s1_idx, cand_idx, prob (all candidate pairs).
    """
    q = one_owner(p)
    q = q.filter(pl.col("prob") >= t)
    if t_empty > t:
        q = q.filter(pl.col("prob").max().over("s1_idx") >= t_empty)
    return q.select("s1_idx", "cand_idx")


def expected_f_select(p: pl.DataFrame, prior: float = 0.0, floor: float = 0.05) -> pl.DataFrame:
    """Per-S1 expected-F0.5 selection (after one-owner).

    Candidates sorted by prob; keeping the top k gives E[F] ~ 1.25 * sum(p_1..p_k) /
    (0.25 * (sum(p) + prior) + k), and keeping none gives P(no true match) = prod(1 - p).
    The k with the highest value is kept per S1. Probabilities below `floor` are ignored.
    """
    q = one_owner(p).filter(pl.col("prob") >= floor).sort(["s1_idx", "prob"], descending=[False, True])
    q = q.with_columns(
        pl.col("prob").cum_sum().over("s1_idx").alias("cum"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("k"),
        pl.col("prob").sum().over("s1_idx").alias("esum"),
        (1.0 - pl.col("prob")).clip(1e-6, 1.0).log().sum().over("s1_idx").exp().alias("p_none"),
    ).with_columns((1.25 * pl.col("cum") / (0.25 * (pl.col("esum") + prior) + pl.col("k"))).alias("ef"))
    best = q.group_by("s1_idx").agg(pl.col("ef").max().alias("ef_best"), pl.col("p_none").first())
    q = q.join(best, on="s1_idx").filter(pl.col("ef_best") > pl.col("p_none"))
    kbest = q.filter(pl.col("ef") == pl.col("ef_best")).group_by("s1_idx").agg(pl.col("k").min().alias("kb"))
    return q.join(kbest, on="s1_idx").filter(pl.col("k") <= pl.col("kb")).select("s1_idx", "cand_idx")


def tune_expected(p: pl.DataFrame, s1_ids: np.ndarray, truth: pl.DataFrame, owned: pl.Series,
                  w: float = DISTRACTOR_WEIGHT) -> dict:
    """Grid over the expected-F0.5 rule's prior; objective = distractor-weighted F0.5."""
    best = {"rule": "expected", "prior": 0.0, "objective": -1.0}
    for prior in (0.0, 0.25, 0.5, 1.0):
        pred = expected_f_select(p, prior)
        f = weighted_f05(s1_ids, pred, truth, owned, w)
        if f > best["objective"]:
            best = {"rule": "expected", "prior": prior, "objective": f,
                    "f05": macro_f05(s1_ids, pred, truth)}
    print(f"  expected-F0.5 rule {best}")
    return best


def decide(p: pl.DataFrame, th: dict) -> pl.DataFrame:
    """Apply whichever decision rule was chosen on validation."""
    if th.get("rule") == "expected":
        return expected_f_select(p, th["prior"])
    return apply_rule(p, th["t"], th["t_empty"])


def weighted_f05(s1_ids: np.ndarray, pred: pl.DataFrame, truth: pl.DataFrame,
                 owned: pl.Series, w: float) -> float:
    """Macro F0.5 where a false positive on a distractor (a record owned by no S1) counts w times.

    With w = 1 this equals the official metric. `owned` holds every S2/S3 index that has an
    owner in the ground truth.
    """
    base = pl.DataFrame({"s1_idx": np.asarray(s1_ids, dtype=np.int32)})
    pr = pred.select(pl.col("s1_idx").cast(pl.Int32), pl.col("cand_idx").cast(pl.Int32))
    tr = truth.select(pl.col("s1_idx").cast(pl.Int32), pl.col("cand_idx").cast(pl.Int32))
    pr = pr.join(tr.with_columns(pl.lit(True).alias("hit")), on=["s1_idx", "cand_idx"], how="left")
    pr = pr.with_columns(pl.col("hit").fill_null(False),
                         pl.col("cand_idx").is_in(owned.implode()).alias("owned"))
    agg = pr.group_by("s1_idx").agg(
        pl.col("hit").sum().alias("tp"), pl.len().alias("n_pred"),
        (~pl.col("hit") & ~pl.col("owned")).sum().alias("fp_d"))
    nt = tr.group_by("s1_idx").len().rename({"len": "n_true"})
    d = base.join(agg, on="s1_idx", how="left").join(nt, on="s1_idx", how="left").fill_null(0)
    p = pl.col("tp") / (pl.col("n_pred") + (w - 1.0) * pl.col("fp_d"))
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
           .when((pl.col("n_pred") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    return float(d.select(f.mean()).item())


def tune(p: pl.DataFrame, s1_ids: np.ndarray, truth: pl.DataFrame, owned: pl.Series | None = None,
         w: float = DISTRACTOR_WEIGHT) -> dict:
    """Grid-search (t, t_empty) maximising (distractor-weighted) macro F0.5 on a validation set.

    Returns the chosen thresholds with both the weighted objective and the plain F0.5.
    """
    if owned is None:
        w = 1.0
        owned = pl.Series(np.zeros(0, np.int32))
    q = one_owner(p)            # one-owner does not depend on the thresholds
    best = {"t": 0.5, "t_empty": 0.5, "objective": -1.0}
    for t in np.round(np.arange(0.20, 0.96, 0.05), 3):
        base = q.filter(pl.col("prob") >= t)
        for te in [t] + [x for x in np.round(np.arange(0.30, 0.99, 0.05), 3) if x > t]:
            pred = base if te <= t else base.filter(pl.col("prob").max().over("s1_idx") >= te)
            f = weighted_f05(s1_ids, pred.select("s1_idx", "cand_idx"), truth, owned, w)
            if f > best["objective"]:
                best = {"t": float(t), "t_empty": float(te), "objective": f}
    pred = apply_rule(p, best["t"], best["t_empty"])
    best["f05"] = macro_f05(s1_ids, pred, truth)
    best["distractor_weight"] = w
    print(f"  best threshold {best}")
    return best


def save_thresholds(th: dict) -> None:
    """Persist chosen thresholds to models/thresholds.json."""
    THRESH_PATH.write_text(json.dumps(th, indent=2))


def load_thresholds() -> dict:
    """Load thresholds chosen on validation."""
    return json.loads(THRESH_PATH.read_text())
