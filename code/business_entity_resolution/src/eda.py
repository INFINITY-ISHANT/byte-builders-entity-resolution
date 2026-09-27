"""Quick EDA report: sizes, label structure and noise statistics (reads the parquet caches).

    python eda.py
"""
from __future__ import annotations

import polars as pl

import io_utils as io
from normalize import load_norm, norm_path


def split_report(split: str) -> None:
    """Print record counts per source/country and simple noise rates for one split."""
    rec = io.load_records(split, ["idx", "src", "country", "name", "address"])
    print(f"\n=== {split}: {rec.height:,} records")
    print(rec.group_by("src", "country").len().sort("src", "country"))
    pool = rec.filter(pl.col("src") != 1)
    print(pool.select(
        (pl.col("address") == "").mean().alias("pool_empty_addr"),
        pl.col("name").str.contains(r"(?i)\.(com|in|co\.in|net|org|co|fr|biz|info)$").mean().alias("pool_domain_name"),
        (~pl.col("name").str.contains(r"^[\x00-\x7F]*$")).mean().alias("pool_non_ascii_name"),
        pl.col("address").str.contains("NULL").mean().alias("pool_NULL_token"),
    ))
    if norm_path(split).exists():
        n = load_norm(split, ["idx", "state", "house_num", "legal_form"]).join(rec.select("idx", "country"), on="idx")
        print(n.group_by("country").agg((pl.col("state") != "").mean().alias("state_found"),
                                        (pl.col("house_num") != "").mean().alias("house_found"),
                                        (pl.col("legal_form") != "").mean().alias("legal_form_found")))


def label_report() -> None:
    """Print the ground-truth structure: singletons, matches per S1, one-owner check."""
    rec = io.load_records("train", ["idx", "src", "country"])
    gt = io.load_gt_pairs()
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1_idx"), "country")
    cnt = (s1.join(gt.group_by("s1_idx").len().rename({"len": "n_match"}), on="s1_idx", how="left")
             .with_columns(pl.col("n_match").fill_null(0)))
    print("\n=== labels")
    print(cnt.group_by("country").agg((pl.col("n_match") == 0).mean().alias("singleton_rate"),
                                     pl.col("n_match").mean().alias("mean_matches"),
                                     pl.col("n_match").max().alias("max")))
    print(cnt.group_by("n_match").len().sort("n_match"))
    print(f"matched ids {gt.height:,}, unique {gt['cand_idx'].n_unique():,} (one-owner holds: "
          f"{gt.height == gt['cand_idx'].n_unique()})")
    n_pool = rec.filter(pl.col("src") != 1).height
    print(f"pool records owned by no S1 (distractors): {1 - gt.height / n_pool:.3f}")


if __name__ == "__main__":
    with io.stage("eda"):
        for sp in ("train", "test"):
            split_report(sp)
        label_report()
