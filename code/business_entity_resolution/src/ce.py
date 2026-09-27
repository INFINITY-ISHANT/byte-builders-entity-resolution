"""Cross-encoder (Ditto-style) pair scorer on the GPU:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params) as a
sequence-classification backbone. (microsoft/mdeberta-v3-base was tried first but diverged /
failed to fit even a tiny batch under transformers 5.16 + torch 2.14, so it was dropped.)

Input is the raw text of both records as a pair: "name | address" [SEP] "name | address".
Trained on candidate pairs of train S1 entities that are NOT in the XGBoost fit/validation
sets (so its score is an honest out-of-sample feature there), with all positives and the
top-ranked (hard) negatives from blocking. The word-embedding matrix (250k-token vocabulary,
most of the parameters) is frozen; bf16 autocast; fused AdamW.

    python ce.py --train              # fine-tune and save to models/ce_minilm/
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import polars as pl

import config as C
import io_utils as io

CE_DIR = C.MODELS_DIR / "ce_minilm"
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 64


def record_texts(split: str) -> pl.Series:
    """'name | address' raw text for every record of a split, as a polars Series (index = idx).
    Kept as Arrow strings (not Python objects) to save memory."""
    rec = io.load_records(split, ["name", "address"])
    return rec["name"] + " | " + rec["address"]


def ce_train_s1(n_s1: int = 90_000, seed: int = C.SEED) -> np.ndarray:
    """The train S1 entities whose candidate pairs train the cross-encoder (outside fit/val)."""
    from train import split_s1
    fit_s1, val_s1, _ = split_s1(300_000, 100_000)
    rec = io.load_records("train", ["idx", "src"]).filter(pl.col("src") == 1)
    free = np.setdiff1d(rec["idx"].to_numpy(), np.concatenate([fit_s1, val_s1]))
    return np.random.default_rng(seed).choice(free, n_s1, replace=False)


def build_train_pairs(n_s1: int = 90_000, n_neg: int = 6, seed: int = C.SEED, pick=None) -> pl.DataFrame:
    """Positives + hard negatives (top blocking ranks) for S1s outside the XGB fit/val sets."""
    from blocking import cand_path
    pick = ce_train_s1(n_s1, seed) if pick is None else pick
    c = (pl.read_parquet(cand_path("train"), columns=["s1_idx", "cand_idx", "rank_in_s1"])
           .filter(pl.col("s1_idx").is_in(pl.Series(pick).implode())))
    gt = io.load_gt_pairs().with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    c = c.join(gt, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("label").fill_null(0))
    neg = (c.filter(pl.col("label") == 0).sort("s1_idx", "rank_in_s1")
             .group_by("s1_idx", maintain_order=True).head(n_neg))
    out = pl.concat([c.filter(pl.col("label") == 1), neg]).select("s1_idx", "cand_idx", "label")
    return out.sample(fraction=1.0, shuffle=True, seed=seed)


def _encode(tok, texts: pl.Series, a_idx: np.ndarray, b_idx: np.ndarray, chunk: int = 20_000):
    """Tokenise record pairs in small chunks -> list of compact int32 id arrays (no padding).

    Tokenising hundreds of thousands of pairs in one call builds full Encoding objects (token
    strings, offsets, masks) and ran the machine out of memory, so chunks are converted to
    numpy immediately and only the input ids are kept."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    out = []
    for s in range(0, len(a_idx), chunk):
        ta = texts.gather(pl.Series(a_idx[s:s + chunk])).to_list()
        tb = texts.gather(pl.Series(b_idx[s:s + chunk])).to_list()
        enc = tok(ta, tb, truncation=True, max_length=MAX_LEN, return_attention_mask=False,
                  return_token_type_ids=False)
        out.extend(np.asarray(x, dtype=np.int32) for x in enc["input_ids"])
    return out


def _batch(ids_list, pad_id, device):
    """Pad a list of id lists to the longest -> (input_ids, attention_mask) on device."""
    import torch
    L = max(len(x) for x in ids_list)
    arr = np.full((len(ids_list), L), pad_id, dtype=np.int64)
    for k, x in enumerate(ids_list):
        arr[k, :len(x)] = x
    t = torch.from_numpy(arr).to(device)
    return t, (t != pad_id).long()


def load_base(num_labels: int = 1):
    """Backbone with a fresh classification head."""
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=num_labels)


def _limit_vram() -> None:
    """Cap PyTorch at 80% of VRAM: on Windows an overflow otherwise spills silently into
    shared system memory and runs ~10x slower instead of failing."""
    import torch
    torch.cuda.set_per_process_memory_fraction(0.8)


def train(epochs: int = 1, bs: int = 64, lr: float = 5e-5, pick=None, out_dir=None) -> None:
    """Fine-tune the cross-encoder and save it (with a held-out AUC report)."""
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer
    torch.manual_seed(C.SEED)
    pairs = build_train_pairs(pick=pick)
    out_dir = CE_DIR if out_dir is None else out_dir
    texts = record_texts("train")
    tok = AutoTokenizer.from_pretrained(BASE)
    with io.stage("tokenise"):
        ids = _encode(tok, texts, pairs["s1_idx"].to_numpy(), pairs["cand_idx"].to_numpy())
    del texts
    y = pairs["label"].to_numpy().astype(np.float32)
    n_hold = 20_000
    tr_ids, tr_y, ho_ids, ho_y = ids[n_hold:], y[n_hold:], ids[:n_hold], y[:n_hold]
    print(f"train pairs {len(tr_ids):,} (pos {tr_y.mean():.3f}) | held-out {n_hold:,}")
    dev = torch.device("cuda")
    _limit_vram()
    model = load_base().to(dev)
    for p in model.base_model.embeddings.word_embeddings.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    # fused AdamW: the foreach / single-tensor CUDA paths produced NaN weights after the
    # first step on this torch build (finite gradients), the fused kernel is correct
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=True)
    steps = epochs * math.ceil(len(tr_ids) / bs)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    pad = tok.pad_token_id
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        order = np.random.default_rng(C.SEED + ep).permutation(len(tr_ids))
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            x, m = _batch([tr_ids[i] for i in b], pad, dev)
            yt = torch.from_numpy(tr_y[b]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 2000 == 0:
                el = time.time() - t0
                print(f"  step {step}/{steps} loss {loss.item():.4f} | {step * bs / el:.0f} pairs/s | "
                      f"eta {(steps - step) * el / step / 60:.1f} min", flush=True)
    model.eval()
    ho = score_ids(model, ho_ids, pad, dev)
    print(f"held-out AUC {roc_auc_score(ho_y, ho):.5f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)


def score_ids(model, ids, pad, dev, bs: int = 256) -> np.ndarray:
    """Logits for tokenised pairs; batches are length-sorted for speed, output in input order."""
    import torch
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.empty(len(ids), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            x, m = _batch([ids[i] for i in b], pad, dev)
            out[b] = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
    return out


def score_pairs(split: str, s1_idx: np.ndarray, cand_idx: np.ndarray, chunk: int = 500_000,
                model_dir=None) -> np.ndarray:
    """Cross-encoder logits for (s1_idx, cand_idx) pairs of a split, on the GPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = CE_DIR if model_dir is None else model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = torch.device("cuda")
    _limit_vram()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    texts = record_texts(split)
    out = np.empty(len(s1_idx), np.float32)
    t0 = time.time()
    for s in range(0, len(s1_idx), chunk):
        ids = _encode(tok, texts, s1_idx[s:s + chunk], cand_idx[s:s + chunk])
        out[s:s + chunk] = score_ids(model, ids, tok.pad_token_id, dev)
        print(f"  ce scored {min(s + chunk, len(s1_idx)):,}/{len(s1_idx):,} "
              f"({(s + chunk) / (time.time() - t0):.0f} pairs/s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def score_band_to_file(split: str) -> None:
    """Score the uncertain-band pairs of a split and save logits to cache (run as its own process
    so the GPU is fully released before XGBoost trains / predicts)."""
    from train import CE_BAND, V1_CACHE
    if split == "ext":
        from expand import EXT_CACHE
        ids = pl.read_parquet(EXT_CACHE / "ids.parquet")
        p1 = np.load(EXT_CACHE / "p1.npy")
        out = EXT_CACHE / "ce_logit.npy"
        split = "train"
    elif split == "train":
        ids = pl.read_parquet(V1_CACHE / "ids.parquet")
        p1 = np.concatenate([np.load(V1_CACHE / "p1_fit.npy"), np.load(V1_CACHE / "p1_val.npy")])
        out = V1_CACHE / "ce_logit.npy"
    else:
        ids = pl.read_parquet(C.CACHE_DIR / "test_probs.parquet", columns=["s1_idx", "cand_idx", "p1"])
        p1 = ids["p1"].to_numpy()
        out = C.CACHE_DIR / "test_ce_logit.npy"
    band = (p1 >= CE_BAND[0]) & (p1 <= CE_BAND[1])
    ce = np.full(len(p1), np.nan, np.float32)
    with io.stage(f"cross-encoder scoring {split} ({band.sum():,} pairs)"):
        ce[band] = score_pairs(split, ids["s1_idx"].to_numpy()[band], ids["cand_idx"].to_numpy()[band])
    np.save(out, ce)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", choices=["train", "test", "ext"])
    a = ap.parse_args()
    if a.train:
        with io.stage("cross-encoder training"):
            train()
    if a.score:
        score_band_to_file(a.score)
