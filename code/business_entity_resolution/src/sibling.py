"""Sibling-business features for the stage-2 model.

Hard negatives on the test set are mostly "sibling" businesses: the same base name plus an
extra word ("Holding", "Distribution", "Westgate") and/or another legal form, at a nearby but
different house number. True duplicates instead differ by typos, dropped legal forms or
"noise" words, and their house numbers are equal or differ like a typo. Features:

  h_absdiff / h_logdiff   numeric gap between the leading integers of the two house numbers
  h_nearby                gap in 1..20 (sibling signature; in train such pairs match ~4% of the
                          time vs ~12% for other non-zero gaps)
  n_extra / n_missing     name tokens only in the candidate / only in the S1
  te_extra_min/mean/max   target-encoded match rate of the candidate's extra tokens
  te_missing_min          target-encoded match rate of the S1 tokens the candidate lacks
  te_legal                target-encoded match rate of the (S1 legal form, candidate legal form) pair
  sub_word                the candidate gained a word AND lost a word that are not typo variants of
                          each other (best Jaro-Winkler < 0.8): "JD Sportive" -> "JD Culturelle"
  sub_best_jw             best Jaro-Winkler between any gained and any lost word (typo ~ high)
  legal_swap / legal_drop both names carry a legal form and they differ / only one carries one

Target encodings are learned from train S1 entities that are NOT in the XGBoost fit /
validation sets (no leakage) and saved to models/. Tokens are keyed by consonant skeleton of
the token without a trailing "s", so cognates share statistics across languages
("groupe"/"group", "developpement"/"development").
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config as C
import io_utils as io
from normalize import load_norm, skeleton

SIB_FEATS = ["h_absdiff", "h_logdiff", "h_nearby", "n_extra", "n_missing",
             "te_extra_min", "te_extra_mean", "te_extra_max", "te_missing_min", "te_legal",
             "sub_word", "sub_best_jw", "legal_swap", "legal_drop"]
TE_TOK = C.MODELS_DIR / "te_tokens.parquet"
TE_LEGAL = C.MODELS_DIR / "te_legal.parquet"
SMOOTH = 20.0


def _tok_key(tokens: pl.Series) -> dict:
    """Skeleton key for each unique token (trailing 's' removed first)."""
    return {t: skeleton(t[:-1] if len(t) > 3 and t.endswith("s") else t) for t in tokens.unique().to_list() if t}


def _pairs_with_tokens(split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """Attach extra / missing core-name tokens, legal forms and house numbers to pairs."""
    n = load_norm(split, ["idx", "name_core", "legal_form", "house_num"])
    L = n.rename({"idx": "s1_idx", "name_core": "n1", "legal_form": "l1", "house_num": "h1"})
    R = n.rename({"idx": "cand_idx", "name_core": "n2", "legal_form": "l2", "house_num": "h2"})
    d = pairs.with_row_index("rid").join(L, on="s1_idx", how="left").join(R, on="cand_idx", how="left")
    t1, t2 = pl.col("n1").str.split(" "), pl.col("n2").str.split(" ")
    return d.with_columns(t2.list.set_difference(t1).alias("extra"),
                          t1.list.set_difference(t2).alias("missing"))


def pseudo_labels(lo: float = 0.02, hi: float = 0.98) -> pl.DataFrame:
    """Confident test pairs as pseudo-labels (semi-supervised; no test labels exist): pairs the
    current model scores >= hi are positives, pairs in the stage-2 band scored <= lo negatives."""
    from train import V2_MIN_P1
    t = pl.read_parquet(C.CACHE_DIR / "test_probs.parquet", columns=["s1_idx", "cand_idx", "p1", "prob"])
    t = t.filter((pl.col("prob") >= hi) | ((pl.col("prob") <= lo) & (pl.col("p1") >= V2_MIN_P1)))
    return t.select("s1_idx", "cand_idx", (pl.col("prob") >= hi).cast(pl.Int8).alias("y"))


def build_target_encodings(exclude_s1: np.ndarray, max_rank: int = 15, pseudo: bool = False,
                           even_only: bool = True) -> None:
    """Learn token / legal-pair match rates from train S1s outside the fit/val sets."""
    from blocking import cand_path
    c = (pl.read_parquet(cand_path("train"), columns=["s1_idx", "cand_idx", "rank_in_s1"])
           .filter(pl.col("rank_in_s1") <= max_rank)
           .filter(~pl.col("s1_idx").is_in(pl.Series(exclude_s1).implode()))
           .filter((pl.col("s1_idx") % 2 == 0) | (not even_only))
           .select("s1_idx", "cand_idx"))
    gt = io.load_gt_pairs().with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    c = c.join(gt, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    d = _pairs_with_tokens("train", c)
    prior = float(d["y"].mean())
    if pseudo:
        # add confident test pairs so words / legal forms unseen in train (e.g. French) get
        # statistics; train labels of fit/val S1s are still never used
        pt = _pairs_with_tokens("test", pseudo_labels())
        print(f"  + {pt.height:,} pseudo-labelled test pairs ({pt['y'].mean():.3f} positive)")
        d = pl.concat([d.select("y", "extra", "missing", "l1", "l2"),
                       pt.select("y", "extra", "missing", "l1", "l2")])
    ex = d.select("y", pl.col("extra").alias("tok")).explode("tok")
    mi = d.select("y", pl.col("missing").alias("tok")).explode("tok")
    toks = pl.concat([ex, mi]).drop_nulls().filter(pl.col("tok") != "")
    keys = _tok_key(toks["tok"])
    out = []
    for kind, df in (("extra", ex), ("missing", mi)):
        df = df.drop_nulls().filter(pl.col("tok") != "")
        df = df.with_columns(pl.col("tok").replace_strict(keys, default="").alias("key"))
        g = df.group_by("key").agg(pl.len().alias("n"), pl.col("y").sum().alias("pos"))
        out.append(g.with_columns(pl.lit(kind).alias("kind"),
                                  ((pl.col("pos") + SMOOTH * prior) / (pl.col("n") + SMOOTH)).alias("te")))
    pl.concat(out).with_columns(pl.lit(prior).alias("prior")).write_parquet(TE_TOK)
    lg = d.group_by("l1", "l2").agg(pl.len().alias("n"), pl.col("y").sum().alias("pos")).with_columns(
        ((pl.col("pos") + SMOOTH * prior) / (pl.col("n") + SMOOTH)).alias("te"))
    lg.write_parquet(TE_LEGAL)
    print(f"target encodings from {d.height:,} reference pairs (prior {prior:.3f}); "
          f"{len(keys):,} token keys")


def sibling_features(split: str, pairs: pl.DataFrame) -> np.ndarray:
    """SIB_FEATS for pairs (s1_idx, cand_idx) in the given row order -> float32 matrix."""
    te = pl.read_parquet(TE_TOK)
    prior = float(te["prior"][0])
    d = _pairs_with_tokens(split, pairs.select("s1_idx", "cand_idx"))
    num = r"^0*(\d{1,7})"
    d = d.with_columns(
        (pl.col("h1").str.extract(num).cast(pl.Int64) - pl.col("h2").str.extract(num).cast(pl.Int64)).abs().alias("hd"))
    d = d.with_columns(
        pl.col("hd").cast(pl.Float32).alias("h_absdiff"),
        pl.col("hd").cast(pl.Float32).log1p().alias("h_logdiff"),
        ((pl.col("hd") >= 1) & (pl.col("hd") <= 20)).cast(pl.Float32).alias("h_nearby"),
        pl.col("extra").list.len().cast(pl.Float32).alias("n_extra"),
        pl.col("missing").list.len().cast(pl.Float32).alias("n_missing"))
    allt = pl.concat([d["extra"].explode(), d["missing"].explode()]).drop_nulls()
    keys = _tok_key(allt.filter(allt != ""))
    feats = d.select("rid", *[c for c in SIB_FEATS[:5]])
    for kind, col in (("extra", "extra"), ("missing", "missing")):
        tk = te.filter(pl.col("kind") == kind).select("key", "te")
        e = (d.select("rid", pl.col(col).alias("tok")).explode("tok").drop_nulls().filter(pl.col("tok") != "")
               .with_columns(pl.col("tok").replace_strict(keys, default="").alias("key"))
               .join(tk, on="key", how="left").with_columns(pl.col("te").fill_null(prior)))
        if kind == "extra":
            agg = e.group_by("rid").agg(pl.col("te").min().alias("te_extra_min"),
                                        pl.col("te").mean().alias("te_extra_mean"),
                                        pl.col("te").max().alias("te_extra_max"))
        else:
            agg = e.group_by("rid").agg(pl.col("te").min().alias("te_missing_min"))
        feats = feats.join(agg, on="rid", how="left")
    # word substitution vs typo: best Jaro-Winkler between gained and lost tokens
    from rapidfuzz.distance import JaroWinkler
    ex_l, mi_l = d["extra"].to_list(), d["missing"].to_list()
    best = np.full(len(ex_l), np.nan, np.float32)
    for i, (e, m) in enumerate(zip(ex_l, mi_l)):
        if e and m:
            best[i] = max(JaroWinkler.normalized_similarity(a, b) for a in e for b in m)
    l1, l2 = d["l1"].fill_null(""), d["l2"].fill_null("")
    feats = feats.join(pl.DataFrame({"rid": d["rid"], "sub_word": np.where(np.isnan(best), 0.0, (best < 0.8).astype(np.float32)),
                                     "sub_best_jw": best,
                                     "legal_swap": ((l1 != "") & (l2 != "") & (l1 != l2)).cast(pl.Float32),
                                     "legal_drop": ((l1 != "") ^ (l2 != "")).cast(pl.Float32)}), on="rid", how="left")
    lg = pl.read_parquet(TE_LEGAL).select("l1", "l2", pl.col("te").alias("te_legal"))
    feats = feats.join(d.select("rid", "l1", "l2").join(lg, on=["l1", "l2"], how="left").select("rid", "te_legal"),
                       on="rid", how="left")
    return feats.sort("rid").select(SIB_FEATS).to_numpy().astype(np.float32)
