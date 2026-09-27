"""Exact macro-averaged F0.5 scorer (matches the official challenge definition).

Per S1 entity:
    truth empty & pred empty      -> 1.0
    exactly one of them empty     -> 0.0
    otherwise P=|pred∩truth|/|pred|, R=|pred∩truth|/|truth|, F=1.25PR/(0.25P+R) (0 if no hit)
Final score = mean over all S1 entities in the evaluation set (singletons included).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def f05_single(pred: set, truth: set) -> float:
    """F0.5 for one S1 entity given predicted and true id sets."""
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def per_entity_f05(s1_ids: np.ndarray, pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F0.5 for many entities at once.

    s1_ids: array of evaluated S1 indices (every one counts, even with no pred/truth).
    pred / truth: frames with columns s1_idx, cand_idx (one row per predicted / true pair).
    Returns a frame with s1_idx, n_pred, n_true, tp, f05.
    """
    base = pl.DataFrame({"s1_idx": np.asarray(s1_ids, dtype=np.int32)})
    pred = pred.select(pl.col("s1_idx").cast(pl.Int32), pl.col("cand_idx").cast(pl.Int32)).unique()
    truth = truth.select(pl.col("s1_idx").cast(pl.Int32), pl.col("cand_idx").cast(pl.Int32)).unique()
    npred = pred.group_by("s1_idx").len().rename({"len": "n_pred"})
    ntrue = truth.group_by("s1_idx").len().rename({"len": "n_true"})
    tp = pred.join(truth, on=["s1_idx", "cand_idx"]).group_by("s1_idx").len().rename({"len": "tp"})
    df = (base.join(npred, on="s1_idx", how="left").join(ntrue, on="s1_idx", how="left")
              .join(tp, on="s1_idx", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
           .when((pl.col("n_pred") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    return df.with_columns(f.alias("f05"))


def macro_f05(s1_ids, pred: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Macro-averaged F0.5 over the given S1 entities."""
    return float(per_entity_f05(s1_ids, pred, truth)["f05"].mean())


def _self_test() -> None:
    """Unit test against the README example (expected 0.714) and edge cases."""
    assert abs(f05_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.7142857) < 1e-6
    assert f05_single(set(), set()) == 1.0
    assert f05_single({"a"}, set()) == 0.0 and f05_single(set(), {"a"}) == 0.0
    pred = pl.DataFrame({"s1_idx": [1, 1, 1, 2], "cand_idx": [47, 193, 812, 5]})
    truth = pl.DataFrame({"s1_idx": [1, 1, 3], "cand_idx": [47, 812, 9]})
    # s1=1 -> 0.714, s1=2 -> 0 (false merge on singleton), s1=3 -> 0 (missed), s1=4 -> 1
    got = macro_f05([1, 2, 3, 4], pred, truth)
    assert abs(got - (0.7142857 + 0 + 0 + 1) / 4) < 1e-6, got
    print("metrics self-test OK")


if __name__ == "__main__":
    _self_test()
