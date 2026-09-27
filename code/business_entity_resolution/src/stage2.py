"""Stage-2 cluster-context features built on top of the v1 model's probabilities.

For every candidate pair (s1, c) with v1 probability p:
  * v1 context: p, rank of p within the S1, the S1's max / sum / count(p > 0.5), the best
    probability any *other* S1 gives to c and the gap to it.
  * cluster consistency: similarity of c to the S1's other high-probability candidates
    ("anchors" = its top-3 by p, excluding c). True duplicates of one business corroborate each
    other; near-miss distractors and records of another business agree less with the cluster.
    For name, address and house number: max similarity and p-weighted mean over the anchors.
  * source mates: number of the S1's high-probability candidates from the same source as c.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

import config as C
from normalize import load_norm

N_ANCHORS = 3
N_TOP = 8          # distractor-cluster features look at the S1's top-8 candidates by p1
S2_FEATS = ["p1", "p1_rank", "s1_pmax", "s1_psum", "s1_n05", "c_best_other", "c_gap_other",
            "anc_n", "anc_name_max", "anc_name_wmean", "anc_addr_max", "anc_addr_wmean",
            "anc_house_eq_w", "same_src_n05",
            "dm_house_mates", "dm_house_mates_p", "dm_s1_house_support", "dm_closer", "dm_closer_p"]


def _anchor_block(p: pl.DataFrame, names: pl.Series, addrs: pl.Series, houses: pl.Series,
                  chunk: int) -> pl.DataFrame:
    """Anchor-similarity aggregates for a block of pairs covering whole S1 groups (by pid)."""
    anchors = (p.filter(pl.col("p1_rank") <= N_ANCHORS + 1)
                .select("s1_idx", pl.col("cand_idx").alias("anc"), pl.col("p1").alias("anc_p"),
                        pl.col("p1_rank").alias("anc_rank")))
    link = (p.select("pid", "s1_idx", "cand_idx").join(anchors, on="s1_idx")
              .filter(pl.col("anc") != pl.col("cand_idx"))
              .sort("pid", "anc_rank")
              .with_columns(pl.col("anc_rank").rank("ordinal").over("pid").alias("k"))
              .filter(pl.col("k") <= N_ANCHORS).drop("k", "anc_rank"))
    # similarities candidate <-> anchor: rapidfuzz in C++ threads (N_JOBS), house equality in polars
    ci, ai = pl.Series(link["cand_idx"].to_numpy()), pl.Series(link["anc"].to_numpy())
    sn, sa = np.empty(link.height, np.float32), np.empty(link.height, np.float32)
    for s_ in range(0, link.height, chunk):
        c_, a_ = ci.slice(s_, chunk), ai.slice(s_, chunk)
        sn[s_:s_ + chunk] = process.cpdist(names.gather(c_).to_list(), names.gather(a_).to_list(),
                                           scorer=fuzz.token_set_ratio, workers=C.N_JOBS, dtype=np.float32) / 100
        sa[s_:s_ + chunk] = process.cpdist(addrs.gather(c_).to_list(), addrs.gather(a_).to_list(),
                                           scorer=fuzz.token_set_ratio, workers=C.N_JOBS, dtype=np.float32) / 100
    ca, aa, ch, ah = addrs.gather(ci), addrs.gather(ai), houses.gather(ci), houses.gather(ai)
    link = link.with_columns(
        pl.Series("sim_name", sn),
        pl.when((ca == "") | (aa == "")).then(None).otherwise(pl.Series(sa)).alias("sim_addr"),
        pl.when((ch == "") | (ah == "")).then(None).otherwise((ch == ah).cast(pl.Float32)).alias("house_eq"))
    w = pl.col("anc_p")
    agg = link.group_by("pid").agg(
        pl.len().cast(pl.Int8).alias("anc_n"),
        pl.col("sim_name").max().alias("anc_name_max"),
        ((pl.col("sim_name") * w).sum() / w.sum()).alias("anc_name_wmean"),
        pl.col("sim_addr").max().alias("anc_addr_max"),
        ((pl.col("sim_addr") * w).sum() / w.filter(pl.col("sim_addr").is_not_null()).sum()).alias("anc_addr_wmean"),
        ((pl.col("house_eq") * w).sum() / w.filter(pl.col("house_eq").is_not_null()).sum()).alias("anc_house_eq_w"),
    )
    return agg


def _distractor_block(p: pl.DataFrame, names: pl.Series, full: pl.Series, houses: pl.Series,
                      chunk: int) -> pl.DataFrame:
    """Distractor-cluster features for the top-N_TOP candidates of each S1 in a block.

    Near-copy distractors usually come with their own duplicates: records that agree with the
    candidate (same house number, similar name) but not with the S1. For candidate c and every
    other top candidate c' of the same S1:
      dm_house_mates      #c' with house(c') == house(c) != house(S1) and name_sim(c, c') >= 0.8
      dm_house_mates_p    max p1 of those mates
      dm_s1_house_support #c' with house(c') == house(S1)
      dm_closer           #c' whose name+address text is closer to c than to the S1 (by > 0.05)
      dm_closer_p         sum of p1 of those c'
    """
    top = p.filter(pl.col("p1_rank") <= N_TOP).select("pid", "s1_idx", "cand_idx", "p1")
    link = (top.join(top.select("s1_idx", pl.col("cand_idx").alias("oth"), pl.col("p1").alias("oth_p")),
                     on="s1_idx").filter(pl.col("oth") != pl.col("cand_idx")))
    ci, oi, si = (pl.Series(link[c].to_numpy()) for c in ("cand_idx", "oth", "s1_idx"))
    n_cc = np.empty(link.height, np.float32)
    f_cc = np.empty(link.height, np.float32)
    f_sc = np.empty(link.height, np.float32)
    for s_ in range(0, link.height, chunk):
        c_, o_, s1_ = ci.slice(s_, chunk), oi.slice(s_, chunk), si.slice(s_, chunk)
        n_cc[s_:s_ + chunk] = process.cpdist(names.gather(c_).to_list(), names.gather(o_).to_list(),
                                             scorer=fuzz.token_set_ratio, workers=C.N_JOBS, dtype=np.float32) / 100
        fo = full.gather(o_).to_list()
        f_cc[s_:s_ + chunk] = process.cpdist(full.gather(c_).to_list(), fo, scorer=fuzz.ratio,
                                             workers=C.N_JOBS, dtype=np.float32) / 100
        f_sc[s_:s_ + chunk] = process.cpdist(full.gather(s1_).to_list(), fo, scorer=fuzz.ratio,
                                             workers=C.N_JOBS, dtype=np.float32) / 100
    hc, ho, hs = houses.gather(ci), houses.gather(oi), houses.gather(si)
    mate = ((hc != "") & (hc == ho) & (hc != hs) & (hs != "") & pl.Series(n_cc >= 0.8)).to_numpy()
    support = ((ho != "") & (ho == hs)).to_numpy()
    closer = (f_cc - f_sc) > 0.05
    link = link.with_columns(pl.Series("mate", mate), pl.Series("support", support), pl.Series("closer", closer))
    return link.group_by("pid").agg(
        pl.col("mate").sum().cast(pl.Float32).alias("dm_house_mates"),
        pl.col("oth_p").filter(pl.col("mate")).max().fill_null(0.0).alias("dm_house_mates_p"),
        pl.col("support").sum().cast(pl.Float32).alias("dm_s1_house_support"),
        pl.col("closer").sum().cast(pl.Float32).alias("dm_closer"),
        pl.col("oth_p").filter(pl.col("closer")).sum().alias("dm_closer_p"),
    )


def cluster_features(split: str, pairs: pl.DataFrame, src_of: np.ndarray,
                     chunk: int = 2_000_000, s1_chunk_pairs: int = 6_000_000) -> pl.DataFrame:
    """Stage-2 features for pairs (s1_idx, cand_idx, p1). Returns the same rows + S2_FEATS."""
    norm = load_norm(split, ["idx", "name_core", "addr_core", "house_num"])
    names, addrs, houses = norm["name_core"], norm["addr_core"], norm["house_num"]
    full = names + " | " + addrs
    p = pairs.select("s1_idx", "cand_idx", pl.col("p1").cast(pl.Float32))
    p = p.with_columns(pl.Series("src", src_of[p["cand_idx"].to_numpy()]).cast(pl.Int8))
    p = p.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int16).alias("p1_rank"),
        pl.col("p1").max().over("s1_idx").alias("s1_pmax"),
        pl.col("p1").sum().over("s1_idx").alias("s1_psum"),
        (pl.col("p1") > 0.5).sum().over("s1_idx").cast(pl.Int16).alias("s1_n05"),
        ((pl.col("p1") > 0.5).sum().over("s1_idx", "src")
         - (pl.col("p1") > 0.5).cast(pl.Int32)).cast(pl.Int16).alias("same_src_n05"),
    )
    top = p.group_by("cand_idx").agg(pl.col("p1").top_k(2).alias("t2")).with_columns(
        pl.col("t2").list.get(0).alias("cb"), pl.col("t2").list.get(1, null_on_oob=True).fill_null(0.0).alias("cs"))
    p = p.join(top.drop("t2"), on="cand_idx", how="left").with_columns(
        pl.when(pl.col("p1") >= pl.col("cb")).then(pl.col("cs")).otherwise(pl.col("cb")).alias("c_best_other")
    ).with_columns((pl.col("p1") - pl.col("c_best_other")).alias("c_gap_other")).drop("cb", "cs")

    p = p.with_row_index("pid")
    # anchor similarities are per S1: process contiguous S1 ranges to bound memory
    order = p.select("pid", "s1_idx").sort("s1_idx")
    s1_sorted = order["s1_idx"].to_numpy()
    cuts = np.r_[0, np.flatnonzero(np.diff(s1_sorted)) + 1, len(s1_sorted)]
    targets = np.searchsorted(cuts, np.arange(0, len(s1_sorted), s1_chunk_pairs))
    edges = np.unique(np.r_[cuts[targets], len(s1_sorted)])
    aggs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = p.filter(pl.col("pid").is_in(order["pid"].slice(int(lo), int(hi - lo)).implode()))
        aggs.append(_anchor_block(sub, names, addrs, houses, chunk)
                    .join(_distractor_block(sub, names, full, houses, chunk), on="pid", how="full", coalesce=True))
    agg = pl.concat(aggs, how="diagonal_relaxed")
    out = p.join(agg, on="pid", how="left").sort("pid").drop("pid", "src")
    return out.with_columns(pl.col("anc_n").fill_null(0),
                            *[pl.col(c).cast(pl.Float32) for c in S2_FEATS if c not in ("anc_n",)])
