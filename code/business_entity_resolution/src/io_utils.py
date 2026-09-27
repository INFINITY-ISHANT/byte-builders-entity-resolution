"""Fast TSV loading (polars), entity-id <-> int index mapping, cached parquet and writers."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import polars as pl
import psutil

import config as C


@contextmanager
def stage(name: str):
    """Context manager that prints wall time and process RSS for a pipeline stage."""
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    yield
    rss = psutil.Process().memory_info().rss / 2**30
    print(f"[{name}] done in {time.time() - t0:,.1f}s | rss {rss:.2f} GB", flush=True)


def read_tsv(path) -> pl.DataFrame:
    """Read a challenge TSV with every column as a string, no quoting, nulls -> ''."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     has_header=True, encoding="utf8")
    return df.with_columns(pl.all().fill_null(""))


def raw_paths(split: str) -> dict:
    """Return the source file paths for a split."""
    d = C.DATA_DIR / split
    return {s: d / f"{split}_source{s}.tsv" for s in (1, 2, 3)}


def records_path(split: str):
    """Parquet path of the combined raw records table for a split."""
    return C.CACHE_DIR / f"{split}_records.parquet"


def build_records(split: str) -> pl.DataFrame:
    """Concatenate S1, S2, S3 of a split into one table with an int32 `idx`.

    Rows are ordered S1 then S2 then S3, so `idx` doubles as a positional index.
    Columns: idx, entity_id, src (1/2/3), name, address, country.
    """
    parts = []
    for s, p in raw_paths(split).items():
        df = read_tsv(p)
        df = df.rename({"business_name": "name", "business_address": "address"})
        parts.append(df.select("entity_id", pl.lit(s, dtype=pl.Int8).alias("src"),
                               "name", "address", "country"))
    rec = pl.concat(parts)
    rec = rec.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    rec.write_parquet(records_path(split))
    return rec


def load_records(split: str, columns=None) -> pl.DataFrame:
    """Load (building on first use) the combined raw records table for a split."""
    p = records_path(split)
    if not p.exists():
        build_records(split)
    return pl.read_parquet(p, columns=columns)


def gt_pairs_path():
    """Parquet path of the exploded training ground-truth pairs."""
    return C.CACHE_DIR / "train_gt_pairs.parquet"


def load_gt_pairs() -> pl.DataFrame:
    """Ground-truth positive pairs as int indices: columns s1_idx, cand_idx (int32)."""
    p = gt_pairs_path()
    if p.exists():
        return pl.read_parquet(p)
    rec = load_records("train", ["idx", "entity_id"])
    gt = read_tsv(C.DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != ""))
    id2 = rec.rename({"idx": "i", "entity_id": "e"})
    pairs = (gt.join(id2, left_on="source1_entity_id", right_on="e")
               .rename({"i": "s1_idx"})
               .join(id2, left_on="matched_entity_ids", right_on="e")
               .rename({"i": "cand_idx"})
               .select("s1_idx", "cand_idx"))
    pairs.write_parquet(p)
    return pairs


def write_id_lists(path, s1_ids, lists, header_col: str) -> None:
    """Write `source1_entity_id \\t comma-joined ids` rows with no quoting at all.

    s1_ids: sequence of S1 id strings (one row each, in order).
    lists:  sequence of lists of S2/S3 id strings (may be empty).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1, ids in zip(s1_ids, lists):
            f.write(s1 + "\t" + ",".join(ids) + "\n")


def s1_mask(src: np.ndarray) -> np.ndarray:
    """Boolean mask of Source-1 rows."""
    return src == 1
