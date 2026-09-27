"""Stage 2 - candidate generation (blocking) with recall diagnostics.

Everything is partitioned by the `country` label (an open set: whatever labels appear).
Within a country we union:
  * TF-IDF kNN on a "words" index (name tokens + address tokens + house|street key) and a
    "name4" index (char 4-grams of the space-less core name; catches typos and domain-style
    names), in both directions: S1 -> top-k S2/S3 and S2/S3 -> top-k' S1. Very common
    features (document frequency > cap) are pruned so the sparse top-k product stays
    tractable (char 3-grams over millions of records were measured to be far too common).
    The sparse top-k product runs on the GPU (torch CSR SpGEMM + segmented sort).
  * Exact-key blocks (name_nospace, house+street, first-name-token+house, phonetic+city,
    postcode+first-name-token), dropping oversized blocks.
All candidate pairs then get full-vector cosines (words / name4 / address-only), a cheap
score, and are capped to the best CAND_CAP per S1 (plus each S2/S3 record's best S1).

    python blocking.py --split train|test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

import config as C
import io_utils as io
from normalize import norm_path

BITS = {"words_fwd": 0, "words_rev": 1, "name4_fwd": 2, "name4_rev": 3, "key_nospace": 4,
        "key_house_street": 5, "key_first_house": 6, "key_phon_city": 7, "key_post_first": 8,
        "addr_fwd": 9, "addr_rev": 10, "skel4_fwd": 11, "skel4_rev": 12, "key_sorted": 13}
# kNN indexes: name -> (df cap, top-k S1->pool, top-k pool->S1)
# words: name + address tokens (main index); addr: address only (DBA / trade names);
# name4: name char 4-grams (typos, domains); skel4: consonant-skeleton 4-grams (transliterated names)
KNN = {"words": (50_000, 40, 5), "addr": (20_000, 20, 3), "name4": (5_000, 20, 3),
       "skel4": (5_000, 20, 3)}


# ------------------------------------------------------------------ featurisers
# every document is "name_core<TAB>addr_core<TAB>house_num<TAB>street<TAB>name_skel"
def words_analyzer(doc: str) -> list[str]:
    """Name word tokens + address word tokens + house|first-street-token key."""
    name, addr, house, street, _ = doc.split("\t")
    out = ["n:" + t for t in name.split()]
    out += ["a:" + t for t in addr.split()]
    if house and street:
        out.append("hs:" + house + "|" + street.split()[0])
    return out


def addr_analyzer(doc: str) -> list[str]:
    """Address-only word tokens + house|first-street-token key."""
    name, addr, house, street, _ = doc.split("\t")
    out = ["a:" + t for t in addr.split()]
    if house and street:
        out.append("hs:" + house + "|" + street.split()[0])
    return out


def name4_analyzer(doc: str) -> list[str]:
    """Char 4-grams of the space-less core name, padded with boundary markers."""
    s = " " + doc.split("\t", 1)[0].replace(" ", "") + " "
    return [s[i:i + 4] for i in range(len(s) - 3)]


def skel4_analyzer(doc: str) -> list[str]:
    """Char 4-grams of the space-less consonant skeleton of the name (script independent)."""
    s = " " + doc.rsplit("\t", 1)[1].replace(" ", "") + " "
    return [s[i:i + 4] for i in range(len(s) - 3)]


ANALYZERS = {"words": words_analyzer, "addr": addr_analyzer, "name4": name4_analyzer,
             "skel4": skel4_analyzer}


def _hash_chunk(args):
    """Worker: hash a list of documents into a sparse count matrix."""
    kind, docs = args
    hv = HashingVectorizer(analyzer=ANALYZERS[kind], n_features=2**23, alternate_sign=False,
                           norm=None, dtype=np.float32)
    return hv.transform(docs)


def hash_docs(kind: str, docs: list[str], ex: ProcessPoolExecutor, chunk: int = 200_000):
    """Hash documents in parallel chunks and stack into one CSR count matrix."""
    jobs = [(kind, docs[i:i + chunk]) for i in range(0, len(docs), chunk)]
    return sp.vstack(list(ex.map(_hash_chunk, jobs))).tocsr()


def tfidf(X: sp.csr_matrix):
    """Sublinear-tf IDF weighting + L2 row normalisation in place; returns (X, df)."""
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    X.data = (1.0 + np.log(X.data)) * idf[X.indices]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0
    X = sp.diags(1.0 / norms).dot(X).tocsr().astype(np.float32)
    return X, df


def prune(X: sp.csr_matrix, df: np.ndarray, max_df: int, min_df: int = 2) -> sp.csr_matrix:
    """Drop features whose document frequency is outside [min_df, max_df] (no renorm)."""
    d = df[X.indices]
    Xp = X.copy()
    Xp.data = np.where((d <= max_df) & (d >= min_df), Xp.data, 0).astype(np.float32)
    Xp.eliminate_zeros()
    return Xp


def _torch():
    """Import torch lazily: worker processes (spawned on Windows) must not load the CUDA DLLs."""
    import torch
    return torch, torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")


def to_torch_csr(m: sp.csr_matrix):
    """Copy a scipy CSR matrix to a torch sparse CSR tensor on the GPU."""
    torch, _DEV = _torch()
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)),
                                   torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data.astype(np.float32)),
                                   size=m.shape, device=_DEV)


def topk_pairs(A: sp.csr_matrix, BT, k: int, budget: int = C.KNN_PAIR_BUDGET):
    """Top-k cosine neighbours of each row of A among the columns of BT (= B transposed).

    Runs on the GPU: for each chunk of query rows, sparse(Q) @ sparse(BT) via cuSPARSE, then
    a segmented top-k (sort by score, then stable sort by row). BT may be a scipy CSR matrix or
    an already uploaded torch CSR tensor. Returns numpy (row, col, sim, rank), rank 1 = best.
    """
    torch, _DEV = _torch()
    BTt = BT if isinstance(BT, torch.Tensor) else to_torch_csr(BT)
    # upper bound on the product's non-zeros per query row = sum of posting-list lengths;
    # query batches are cut so each batch's product stays under `budget` entries on the GPU
    post = torch.diff(BTt.crow_indices()).cpu().numpy().astype(np.float64)
    Ab = A.copy()
    Ab.data[:] = 1.0
    cost = np.asarray(Ab @ post).ravel()
    cuts = np.flatnonzero(np.diff((np.cumsum(cost) // budget).astype(np.int64))) + 1
    edges = np.r_[0, cuts, A.shape[0]]
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    out = []
    with torch.no_grad():
        for lo, hi in bounds:
            Q = A[lo:hi]
            if Q.nnz == 0:
                continue
            R = torch.sparse.mm(to_torch_csr(Q), BTt)
            crow, col, val = R.crow_indices(), R.col_indices(), R.values()
            row = torch.repeat_interleave(torch.arange(R.shape[0], device=_DEV), crow.diff())
            m = val >= C.TFIDF_MIN_SIM
            row, col, val = row[m], col[m], val[m]
            o = torch.argsort(-val, stable=True)
            o = o[torch.argsort(row[o], stable=True)]
            row, col, val = row[o], col[o], val[o]
            rank = torch.arange(row.numel(), device=_DEV) - torch.searchsorted(row, row) + 1
            keep = rank <= k
            out.append((row[keep].cpu().numpy() + lo, col[keep].cpu().numpy(),
                        val[keep].cpu().numpy(), rank[keep].cpu().numpy()))
            del R, crow, col, val, row, o, rank, keep
    if not out:
        z = np.zeros(0, np.int32)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.int16)
    r, c, v, rk = (np.concatenate(x) for x in zip(*out))
    return r.astype(np.int32), c.astype(np.int32), v.astype(np.float32), rk.astype(np.int16)


def rowwise_cos(A: sp.csr_matrix, B: sp.csr_matrix, ri: np.ndarray, ci: np.ndarray,
                chunk: int = 1_000_000) -> np.ndarray:
    """Cosine (dot of L2-normalised rows) of A[ri[k]] and B[ci[k]] for every pair k."""
    out = np.empty(len(ri), dtype=np.float32)
    for s in range(0, len(ri), chunk):
        a, b = A[ri[s:s + chunk]], B[ci[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(1)).ravel()
    return out


# ------------------------------------------------------------------ exact keys
def key_pairs(s1: pl.DataFrame, pool: pl.DataFrame, key: pl.Expr) -> pl.DataFrame:
    """Pairs sharing an exact key; blocks larger than MAX_BLOCK on either side dropped.
    s1/pool carry a local row index column `i`; returns columns a (s1 local), b (pool local)."""
    a = s1.select(pl.col("i").alias("a"), key.alias("k")).filter(pl.col("k") != "")
    b = pool.select(pl.col("i").alias("b"), key.alias("k")).filter(pl.col("k") != "")
    ok_a = a.group_by("k").len().filter(pl.col("len") <= C.MAX_BLOCK).select("k")
    ok_b = b.group_by("k").len().filter(pl.col("len") <= C.MAX_BLOCK).select("k")
    ok = ok_a.join(ok_b, on="k")
    return a.join(ok, on="k").join(b, on="k").select("a", "b")


def exact_key_blocks(s1: pl.DataFrame, pool: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """All exact-key blockers for one country partition."""
    street1 = pl.col("street").str.split(" ").list.first().fill_null("")
    first_ok = pl.col("name_first").str.len_chars() >= 3
    keys = {
        "key_nospace": pl.when(pl.col("name_nospace").str.len_chars() >= 4)
                         .then(pl.col("name_nospace")).otherwise(pl.lit("")),
        "key_house_street": pl.when((pl.col("house_num") != "") & (street1 != ""))
                              .then(pl.col("house_num") + "|" + street1).otherwise(pl.lit("")),
        "key_first_house": pl.when(first_ok & (pl.col("house_num") != ""))
                             .then(pl.col("name_first") + "|" + pl.col("house_num")).otherwise(pl.lit("")),
        "key_phon_city": pl.when((pl.col("name_phon") != "") & (pl.col("city") != ""))
                           .then(pl.col("name_phon") + "|" + pl.col("city")).otherwise(pl.lit("")),
        "key_sorted": pl.when(pl.col("name_sorted").str.len_chars() >= 4)
                        .then(pl.col("name_sorted")).otherwise(pl.lit("")),
        "key_post_first": pl.when((pl.col("postcode") != "") & first_ok)
                            .then(pl.col("postcode") + "|" + pl.col("name_first")).otherwise(pl.lit("")),
    }
    return [(name, key_pairs(s1, pool, expr)) for name, expr in keys.items()]


# ------------------------------------------------------------------ driver
NORM_COLS = ["idx", "name_core", "name_nospace", "name_sorted", "name_first", "name_phon",
             "name_skel", "addr_core", "house_num", "street", "city", "postcode"]


def block_country(sub: pl.DataFrame, ex: ProcessPoolExecutor, s1_chunk: int = 60_000) -> pl.DataFrame:
    """Generate capped candidate pairs for one country partition (global idx).

    Reverse kNN (pool -> S1) and exact keys are computed once for the partition and kept as
    slim int arrays; forward kNN, cosines and the per-S1 cap are then done in S1 chunks so
    peak memory stays bounded. A pair is kept if it is in the S1's top CAND_CAP by cheap
    score, or if the S1 is the pool record's rank-1 neighbour in a reverse index.
    """
    is1 = (sub["src"] == 1).to_numpy()
    gidx = sub["idx"].to_numpy()
    s1_rows, pool_rows = np.flatnonzero(is1), np.flatnonzero(~is1)
    print(f"  S1 {len(s1_rows):,} | pool {len(pool_rows):,}", flush=True)
    docs = (sub["name_core"] + "\t" + sub["addr_core"] + "\t" + sub["house_num"] + "\t"
            + sub["street"] + "\t" + sub["name_skel"]).to_list()

    # every candidate source is kept as numpy arrays (a = S1 position, b = pool position,
    # rev1 flag) sorted by a, so the per-S1 merge below only needs binary-search slices
    sources = []   # (a, b, bit, rev1)

    def add_source(a_, b_, bit, rev1=None):
        """Store one candidate source as arrays sorted by S1 position (a_)."""
        o = np.argsort(a_, kind="stable")
        rv = np.zeros(len(a_), bool) if rev1 is None else np.asarray(rev1)[o]
        sources.append((np.asarray(a_, np.int32)[o], np.asarray(b_, np.int32)[o], np.int32(bit), rv))

    # --- exact keys first, so the string frame can be released before the matrices are built
    s1_df = sub[s1_rows].with_columns(pl.int_range(0, len(s1_rows), dtype=pl.Int32).alias("i"))
    pool_df = sub[pool_rows].with_columns(pl.int_range(0, len(pool_rows), dtype=pl.Int32).alias("i"))
    for nm, kp in exact_key_blocks(s1_df, pool_df):
        print(f"  {nm}: {kp.height:,} pairs", flush=True)
        add_source(kp["a"].to_numpy(), kp["b"].to_numpy(), 1 << BITS[nm])
    del s1_df, pool_df, sub

    # --- tf-idf matrices (full ones kept for pair cosines); kNN one index at a time so only
    # that index lives on the GPU (keeping all of them resident overflowed the 8 GB of VRAM)
    mats = {}
    for nm in ("words", "addr", "name4", "skel4"):
        mats[nm], df = tfidf(hash_docs(nm, docs, ex))
        cap, k_fwd, k_rev = KNN[nm]
        Xp = prune(mats[nm], df, cap)
        A, B = Xp[s1_rows], Xp[pool_rows]
        del Xp
        r, c, _, rk = topk_pairs(B, A.T.tocsr(), k_rev)          # pool -> S1
        add_source(c, r, 1 << BITS[nm + "_rev"], rk == 1)
        BT = to_torch_csr(B.T.tocsr())
        del B
        r, c, _, _ = topk_pairs(A, BT, k_fwd)                     # S1 -> pool
        add_source(r, c, 1 << BITS[nm + "_fwd"])
        del A, BT
        _torch()[0].cuda.empty_cache()
        print(f"  {nm} kNN done (both directions)", flush=True)
    del docs

    # --- merge + cosines + cap, per S1 chunk (CPU)
    key_mask = sum(1 << BITS[k] for k in BITS if k.startswith("key_"))
    out, n_raw = [], 0
    for lo in range(0, len(s1_rows), s1_chunk):
        hi = min(lo + s1_chunk, len(s1_rows))
        parts = []
        for a_, b_, bit, rv in sources:
            i0, i1 = np.searchsorted(a_, lo), np.searchsorted(a_, hi)
            parts.append(pl.DataFrame({"a": a_[i0:i1], "b": b_[i0:i1],
                                       "bit": np.full(i1 - i0, bit, np.int32), "rev1": rv[i0:i1]}))
        # bits of distinct blockers are disjoint powers of two; a blocker never emits a pair twice
        pairs = (pl.concat(parts)
                   .group_by("a", "b").agg(pl.col("bit").sum().alias("bits"), pl.col("rev1").any()))
        n_raw += pairs.height
        a, b = s1_rows[pairs["a"].to_numpy()], pool_rows[pairs["b"].to_numpy()]
        pairs = pairs.with_columns([pl.Series(f"{nm}_cos", rowwise_cos(X, X, a, b)) for nm, X in mats.items()])
        key_hit = ((pl.col("bits") & key_mask) > 0).cast(pl.Float32)
        pairs = pairs.with_columns(
            (pl.col("words_cos") + 0.5 * pl.max_horizontal("name4_cos", "skel4_cos")
             + 0.3 * pl.col("addr_cos") + 0.3 * key_hit).alias("cheap_score"))
        pairs = pairs.with_columns(
            pl.col("cheap_score").rank("ordinal", descending=True).over("a").cast(pl.Int32).alias("rank_in_s1"))
        out.append(pairs.filter((pl.col("rank_in_s1") <= C.CAND_CAP) | pl.col("rev1")))
    del mats, sources
    pairs = pl.concat(out)
    pairs = pairs.with_columns(
        pl.col("cheap_score").rank("ordinal", descending=True).over("b").cast(pl.Int32).alias("rank_in_cand"))
    print(f"  pairs before cap {n_raw:,} -> after {pairs.height:,}", flush=True)
    return pairs.with_columns(
        pl.Series("s1_idx", gidx[s1_rows][pairs["a"].to_numpy()]).cast(pl.Int32),
        pl.Series("cand_idx", gidx[pool_rows][pairs["b"].to_numpy()]).cast(pl.Int32),
    ).drop("a", "b", "rev1")


def cand_path(split: str):
    """Parquet path of the candidate pairs of a split."""
    return C.CACHE_DIR / f"{split}_candidates.parquet"


def run(split: str, resume: bool = False) -> pl.DataFrame:
    """Block every country of a split (one at a time, spilled to parquet) and merge the results."""
    rec = pl.scan_parquet(io.records_path(split)).select("idx", "src", "country")
    countries = sorted(rec.select("country").unique().collect()["country"].to_list())
    parts = []
    with ProcessPoolExecutor(C.N_JOBS) as ex:
        for country in countries:
            part = C.CACHE_DIR / f"{split}_candidates_{country}.parquet"
            parts.append(part)
            if resume and part.exists():
                print(f"  {country}: reusing {part.name}")
                continue
            with io.stage(f"block {split}/{country}"):
                sub = (rec.filter(pl.col("country") == country)
                          .join(pl.scan_parquet(norm_path(split)).select(NORM_COLS), on="idx")
                          .sort("idx").collect())
                block_country(sub, ex).write_parquet(part)
                del sub
    cands = pl.concat([pl.read_parquet(p) for p in parts])
    cands = cands.with_columns(
        pl.col("s1_idx").count().over("s1_idx").cast(pl.Int32).alias("n_cand_s1"),
        pl.col("s1_idx").count().over("cand_idx").cast(pl.Int32).alias("n_s1_cand"))
    cands.write_parquet(cand_path(split))
    return cands


def diagnostics(cands: pl.DataFrame) -> None:
    """Print pair recall (overall, per country, per blocker) and the entity-level ceiling."""
    from metrics import macro_f05
    rec = io.load_records("train", ["idx", "src", "country"])
    gt = io.load_gt_pairs()
    s1 = rec.filter(pl.col("src") == 1)
    hit = gt.join(cands.select("s1_idx", "cand_idx", "bits", "rank_in_s1"),
                  on=["s1_idx", "cand_idx"], how="left")
    print(f"total pairs {cands.height:,} | avg cands/S1 {cands.height / s1.height:.1f} | "
          f"p50/p99 {cands['n_cand_s1'].median()}/{cands['n_cand_s1'].quantile(0.99)}")
    found = hit["bits"].is_not_null()
    print(f"PAIR RECALL overall {found.mean():.4f}")
    hc = hit.join(rec.select(pl.col("idx").alias("s1_idx"), "country"), on="s1_idx")
    print(hc.group_by("country").agg(pl.col("bits").is_not_null().mean().alias("recall")))
    for nm, b in BITS.items():
        own = hit.filter(pl.col("bits").is_not_null())
        only = (own["bits"] == (1 << b)).sum()
        print(f"  {nm:18s} recall {((hit['bits'].fill_null(0) & (1 << b)) > 0).mean():.4f}"
              f" | unique-only {only / hit.height:.4f}")
    for cap in (5, 10, 15, 20, 25, 30, 35, 40):
        print(f"  recall @rank<={cap}: {(hit['rank_in_s1'].fill_null(10**9) <= cap).mean():.4f}")
    ceil = macro_f05(s1["idx"].to_numpy(), hit.filter(found).select("s1_idx", "cand_idx"), gt)
    print(f"ENTITY CEILING macro F0.5 (perfect matcher on candidates) {ceil:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=C.SPLITS, nargs="+", default=list(C.SPLITS))
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse per-country parquet parts")
    ar = ap.parse_args()
    for sp_ in ar.split:
        if ar.diag_only:
            c = pl.read_parquet(cand_path(sp_))
        else:
            with io.stage(f"blocking {sp_}"):
                c = run(sp_, resume=ar.resume)
        if sp_ == "train":
            diagnostics(c)
