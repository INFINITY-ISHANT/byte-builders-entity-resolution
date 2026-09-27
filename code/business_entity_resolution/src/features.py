"""Stage 3 - pair features (country-agnostic), vectorised with rapidfuzz.cpdist + polars.

Each candidate pair (s1_idx, cand_idx) gets name, address and context features.
String similarities are computed element-wise in multithreaded C++ (rapidfuzz.process.cpdist);
set-overlap features use polars list operations. Pairs are processed in chunks so memory
stays bounded on tens of millions of pairs.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

from normalize import load_norm

FEAT_NORM_COLS = ["idx", "name_clean", "name_core", "name_sorted", "name_nospace",
                  "legal_form", "is_domain", "name_acronym", "name_phon", "name_first",
                  "addr_core", "house_num", "all_nums", "postcode", "street", "city",
                  "state", "addr_empty"]
OPTIONAL_NORM_COLS = ["name_skel"]
N_BITS = 14


_WORKERS = -1   # rapidfuzz threads; set to 1 inside pool workers


def _sim(a: list, b: list, scorer, scale: float = 1.0) -> np.ndarray:
    """Element-wise similarity of two equal-length string lists."""
    return process.cpdist(a, b, scorer=scorer, workers=_WORKERS, dtype=np.float32) / scale


def _nan_if_empty(x: np.ndarray, a: pl.Series, b: pl.Series) -> np.ndarray:
    """Set similarity to NaN where either side is an empty string."""
    m = ((a == "") | (b == "")).to_numpy()
    x = x.astype(np.float32, copy=True)
    x[m] = np.nan
    return x


def _jaccard(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Token-set Jaccard of two space-separated string columns (NaN if either empty)."""
    df = pl.DataFrame({"a": a.str.split(" "), "b": b.str.split(" ")})
    inter = df.select(pl.col("a").list.set_intersection("b").list.len()).to_series()
    union = df.select(pl.col("a").list.set_union("b").list.len()).to_series()
    j = (inter / union).cast(pl.Float32).to_numpy().copy()
    j[((a == "") | (b == "")).to_numpy()] = np.nan
    return j


def pair_features(L: pl.DataFrame, R: pl.DataFrame) -> dict[str, np.ndarray]:
    """String features between left (S1) and right (S2/S3) record frames of equal length."""
    f = {}
    lc, rc = L["name_core"], R["name_core"]
    lcl, rcl = lc.to_list(), rc.to_list()
    f["n_ratio"] = _sim(lcl, rcl, fuzz.ratio, 100)
    f["n_partial"] = _sim(lcl, rcl, fuzz.partial_ratio, 100)
    f["n_tsort"] = _sim(lcl, rcl, fuzz.token_sort_ratio, 100)
    f["n_tset"] = _sim(lcl, rcl, fuzz.token_set_ratio, 100)
    f["n_jw"] = _sim(lcl, rcl, distance.JaroWinkler.normalized_similarity)
    f["n_lev"] = _sim(lcl, rcl, distance.Levenshtein.normalized_similarity)
    f["n_wratio"] = _sim(L["name_clean"].to_list(), R["name_clean"].to_list(), fuzz.WRatio, 100)
    lns, rns = L["name_nospace"].to_list(), R["name_nospace"].to_list()
    f["n_ns_ratio"] = _sim(lns, rns, fuzz.ratio, 100)
    f["n_ns_partial"] = _sim(lns, rns, fuzz.partial_ratio, 100)
    f["n_ns_jw"] = _sim(lns, rns, distance.JaroWinkler.normalized_similarity)
    f["n_tok_jacc"] = _jaccard(lc, rc)
    f["n_exact_core"] = (lc == rc).to_numpy().astype(np.int8)
    f["n_exact_sorted"] = (L["name_sorted"] == R["name_sorted"]).to_numpy().astype(np.int8)
    f["n_exact_ns"] = (L["name_nospace"] == R["name_nospace"]).to_numpy().astype(np.int8)
    f["n_first_eq"] = (L["name_first"] == R["name_first"]).to_numpy().astype(np.int8)
    f["n_phon_eq"] = ((L["name_phon"] == R["name_phon"]) & (L["name_phon"] != "")).to_numpy().astype(np.int8)
    f["n_acr"] = (((L["name_acronym"] == R["name_nospace"]) & (L["name_acronym"] != "")) |
                  ((R["name_acronym"] == L["name_nospace"]) & (R["name_acronym"] != ""))).to_numpy().astype(np.int8)
    ll, rl = L["legal_form"], R["legal_form"]
    f["legal_eq"] = ((ll == rl) & (ll != "")).to_numpy().astype(np.int8)
    f["legal_conflict"] = ((ll != rl) & (ll != "") & (rl != "")).to_numpy().astype(np.int8)
    f["legal_r_missing"] = ((ll != "") & (rl == "")).to_numpy().astype(np.int8)
    f["dom_r"] = R["is_domain"].to_numpy().astype(np.int8)
    ln_, rn_ = lc.str.len_chars().to_numpy(), rc.str.len_chars().to_numpy()
    f["n_len_ratio"] = (np.minimum(ln_, rn_) / np.maximum(np.maximum(ln_, rn_), 1)).astype(np.float32)
    f["n_len_l"] = ln_.astype(np.int16)
    f["n_ntok_r"] = rc.str.count_matches(" ").to_numpy().astype(np.int8) + 1

    # ---------------- address
    la, ra = L["addr_core"], R["addr_core"]
    lal, ral = la.to_list(), ra.to_list()
    f["a_empty_r"] = R["addr_empty"].to_numpy().astype(np.int8)
    f["a_empty_l"] = L["addr_empty"].to_numpy().astype(np.int8)
    f["a_tset"] = _nan_if_empty(_sim(lal, ral, fuzz.token_set_ratio, 100), la, ra)
    f["a_tsort"] = _nan_if_empty(_sim(lal, ral, fuzz.token_sort_ratio, 100), la, ra)
    f["a_partial"] = _nan_if_empty(_sim(lal, ral, fuzz.partial_ratio, 100), la, ra)
    f["a_tok_jacc"] = _jaccard(la, ra)
    lh, rh = L["house_num"], R["house_num"]
    both = ((lh != "") & (rh != "")).to_numpy()
    f["h_eq"] = np.where(both, (lh == rh).to_numpy(), np.nan).astype(np.float32)
    f["h_jw"] = _nan_if_empty(_sim(lh.to_list(), rh.to_list(), distance.JaroWinkler.normalized_similarity), lh, rh)
    f["h_r_missing"] = ((lh != "") & (rh == "")).to_numpy().astype(np.int8)
    f["nums_jacc"] = _jaccard(L["all_nums"], R["all_nums"])
    lp, rp = L["postcode"], R["postcode"]
    f["pc_eq"] = np.where(((lp != "") & (rp != "")).to_numpy(), (lp == rp).to_numpy(), np.nan).astype(np.float32)
    ls, rs = L["street"], R["street"]
    f["st_jw"] = _nan_if_empty(_sim(ls.to_list(), rs.to_list(), distance.JaroWinkler.normalized_similarity), ls, rs)
    f["st_tset"] = _nan_if_empty(_sim(ls.to_list(), rs.to_list(), fuzz.token_set_ratio, 100), ls, rs)
    lci, rci = L["city"], R["city"]
    f["city_jw"] = _nan_if_empty(_sim(lci.to_list(), rci.to_list(), distance.JaroWinkler.normalized_similarity), lci, rci)
    f["city_l_in_r"] = _nan_if_empty(_sim(lci.to_list(), ral, fuzz.partial_ratio, 100), lci, ra)
    f["city_r_in_l"] = _nan_if_empty(_sim(rci.to_list(), lal, fuzz.partial_ratio, 100), rci, la)
    lst, rst = L["state"], R["state"]
    f["state_eq"] = np.where(((lst != "") & (rst != "")).to_numpy(), (lst == rst).to_numpy(), np.nan).astype(np.float32)
    # script-independent consonant skeletons (transliterated vs English names); only when
    # the normalised cache has the column (added after M1)
    if "name_skel" in L.columns:
        lk, rk = L["name_skel"].to_list(), R["name_skel"].to_list()
        f["n_skel_tset"] = _sim(lk, rk, fuzz.token_set_ratio, 100)
        f["n_skel_ratio"] = _sim([x.replace(" ", "") for x in lk], [x.replace(" ", "") for x in rk],
                                 fuzz.ratio, 100)
    # name of one side found inside the other's address / domain vs name tokens
    f["ns_in_ns"] = _sim(lns, rns, fuzz.partial_ratio, 100) * (np.minimum(
        np.array([len(x) for x in lns]), np.array([len(x) for x in rns])) >= 4)
    return f


def context_features(c: pl.DataFrame) -> dict[str, np.ndarray]:
    """Blocking-context features: cosines, ranks, competition, blocker bits, source."""
    f = {k: c[k].to_numpy() for k in ("words_cos", "name4_cos", "addr_cos", "skel4_cos", "cheap_score",
                                       "rank_in_s1", "rank_in_cand", "n_cand_s1", "n_s1_cand",
                                       "name_rank_in_cand", "name_gap_cand", "s1_same_name", "cand_same_name")}
    bits = c["bits"].to_numpy()
    for b in range(N_BITS):
        f[f"blk_{b}"] = ((bits >> b) & 1).astype(np.int8)
    f["cand_src"] = c["cand_src"].to_numpy().astype(np.int8)
    for k in ("gap_s1", "gap_cand", "name4_gap_s1", "addr_gap_s1"):
        f[k] = c[k].to_numpy()
    return f


def add_group_context(c: pl.DataFrame) -> pl.DataFrame:
    """Add score gaps to the best pair of the same S1 and to the best other S1 of the candidate."""
    c = c.with_columns(
        (pl.col("cheap_score") - pl.col("cheap_score").max().over("s1_idx")).alias("gap_s1"),
        (pl.col("name4_cos") - pl.col("name4_cos").max().over("s1_idx")).alias("name4_gap_s1"),
        (pl.col("addr_cos") - pl.col("addr_cos").max().over("s1_idx")).alias("addr_gap_s1"),
    )
    # gap to best *other* S1 for the same candidate (top1 - top2 logic), on the combined score
    # and on name-only similarity (decisive for S2/S3 records with an empty address)
    c = c.with_columns(pl.max_horizontal("name4_cos", "skel4_cos").alias("name_sim"))
    c = c.with_columns(pl.col("name_sim").rank("ordinal", descending=True).over("cand_idx")
                         .cast(pl.Int32).alias("name_rank_in_cand"))
    for col, out in (("cheap_score", "gap_cand"), ("name_sim", "name_gap_cand")):
        top = c.group_by("cand_idx").agg(pl.col(col).top_k(2).alias("t2"))
        top = top.with_columns(pl.col("t2").list.get(0).alias("c_best"),
                               pl.col("t2").list.get(1, null_on_oob=True).fill_null(0.0).alias("c_second")).drop("t2")
        c = c.join(top, on="cand_idx", how="left")
        c = c.with_columns(
            pl.when(pl.col(col) >= pl.col("c_best"))
              .then(pl.col(col) - pl.col("c_second"))
              .otherwise(pl.col(col) - pl.col("c_best")).alias(out)).drop("c_best", "c_second")
    return c.drop("name_sim")


def same_name_counts(split: str) -> np.ndarray:
    """Per record: number of S1 records in the same country with the identical space-less core
    name (S1 names repeat across different businesses, which makes name-only matches ambiguous)."""
    import io_utils as io
    rec = io.load_records(split, ["idx", "src", "country"]).join(
        load_norm(split, ["idx", "name_nospace"]), on="idx")
    cnt = (rec.filter(pl.col("src") == 1).group_by("country", "name_nospace").len()
              .rename({"len": "n"}))
    out = rec.join(cnt, on=["country", "name_nospace"], how="left").sort("idx")
    return out["n"].fill_null(0).to_numpy().astype(np.int32)


def _chunk_worker(args) -> pl.DataFrame:
    """Pool worker: all features of one chunk of pairs (L/R = gathered record fields)."""
    global _WORKERS
    _WORKERS = 1
    L, R, c = args
    f = {"s1_idx": c["s1_idx"].to_numpy(), "cand_idx": c["cand_idx"].to_numpy()}
    f.update(context_features(c))
    f.update(pair_features(L, R))
    return pl.DataFrame(f)


def _norm_cols(split: str) -> list[str]:
    """FEAT_NORM_COLS plus optional columns present in this split's normalised cache."""
    import pyarrow.parquet as pq
    from normalize import norm_path
    have = set(pq.read_schema(norm_path(split)).names)
    return [c for c in FEAT_NORM_COLS + OPTIONAL_NORM_COLS if c in have]


def iter_features(split: str, cands: pl.DataFrame, chunk: int = 100_000):
    """Yield feature frames chunk by chunk, in order (cands must carry group context).

    Chunks are computed in a small process pool (N_JOBS workers, CPU kept light); at most
    N_JOBS + 2 chunks are in flight so memory stays bounded.
    """
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    import config as C
    import io_utils as io
    norm = load_norm(split, _norm_cols(split))
    assert (norm["idx"].to_numpy() == np.arange(norm.height)).all()
    src_of = io.load_records(split, ["src"])["src"].to_numpy()
    same = same_name_counts(split)
    cands = cands.with_columns(pl.Series("cand_src", src_of[cands["cand_idx"].to_numpy()]),
                               pl.Series("s1_same_name", same[cands["s1_idx"].to_numpy()]),
                               pl.Series("cand_same_name", same[cands["cand_idx"].to_numpy()]))
    n = cands.height
    pending = deque()
    done = 0
    import os
    os.environ["POLARS_MAX_THREADS"] = "1"      # inherited by spawned workers only
    with ProcessPoolExecutor(C.N_JOBS) as ex:
        for s in range(0, n, chunk):
            c = cands.slice(s, chunk)
            L = norm.select(pl.all().gather(pl.Series(c["s1_idx"].to_numpy())))
            R = norm.select(pl.all().gather(pl.Series(c["cand_idx"].to_numpy())))
            pending.append(ex.submit(_chunk_worker, (L, R, c)))
            if len(pending) >= C.N_JOBS + 2:
                r = pending.popleft().result()
                done += r.height
                if (done // r.height) % 50 == 0:
                    print(f"  features {done:,}/{n:,}", flush=True)
                yield r
        while pending:
            r = pending.popleft().result()
            done += r.height
            yield r
    print(f"  features {done:,}/{n:,}", flush=True)


def build_features(split: str, cands: pl.DataFrame, chunk: int = 100_000,
                   precomputed_context: bool = False) -> pl.DataFrame:
    """Compute all features for the candidate pairs (s1_idx, cand_idx kept first)."""
    if not precomputed_context:
        cands = add_group_context(cands)
    return pl.concat(list(iter_features(split, cands, chunk)), rechunk=False)


def feature_names(df: pl.DataFrame) -> list[str]:
    """Model input columns (everything except ids and label)."""
    return [c for c in df.columns if c not in ("s1_idx", "cand_idx", "label")]
