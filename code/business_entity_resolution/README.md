# Business Entity Resolution — reproduction guide

Pipeline: **normalise → block (candidate generation) → pair features → XGBoost matcher (GPU) →
decision step (one-owner + thresholds)**. No external data, APIs or lookups are used; the only
pretrained model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0),
fine-tuned here as a cross-encoder. Every dictionary (legal forms, address abbreviations, state names/codes, old/new
city names) is hand-written in `src/normalize.py`.

## Environment

- Python 3.14 (Anaconda), Windows 11 (any OS works; multiprocessing uses `spawn`-safe entry points)
- `pip install -r requirements.txt`
- Tested on 16 GB RAM, 28 logical cores, NVIDIA RTX 4060 Laptop (8 GB). Heavy numeric work runs on the
  GPU: sparse TF-IDF top-k search (torch CSR SpGEMM) and XGBoost training/inference (`device=cuda`).
  CPU pools (normalisation, hashing, string features, stage 2) use `N_JOBS` = 22 worker processes (override with `BER_N_JOBS`).
- CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu124` (any CUDA build works).

## Data layout

By default the data is read from `<repo>/student_resource/dataset/{train,test}/`.
Override with the environment variable `BER_DATA_DIR=/path/to/dataset`.
Outputs go to `<repo>/output/` (override with `BER_OUTPUT_DIR`).

## Run end-to-end

```bash
cd code/business_entity_resolution/src
python run_pipeline.py --stage all        # or --stage normalize | block | train | predict
```

`run_pipeline.py` runs each step as its own process (GPU memory and RAM are released between
steps), in this order:

| # | Command | What it does | Time* |
|---|---|---|---|
| 1 | `normalize.py --split train test` | clean names/addresses, parse house number, street, city, state/region, postcode, legal form, skeleton key | 5 min |
| 2 | `blocking.py --split train test` | 4 GPU TF-IDF kNN indexes (both directions) + 6 exact keys per country, cap 40/S1; recall diagnostics | 40 min |
| 3 | `ce.py --train` | fine-tune the MiniLM cross-encoder on 825k candidate pairs (GPU) | 11 min |
| 4 | `train.py --stage1-only` | pair features, XGBoost v1 (CUDA), 3-fold out-of-fold v1 probabilities, cached | 30 min |
| 5 | `ce.py --score train` | cross-encoder logits for uncertain pairs | 2 min |
| 6 | `expand.py --n-s1 500000` | extra out-of-sample S1s for stage 2 | 12 min |
| 7 | `ce.py --score ext` | cross-encoder logits for them | 2 min |
| 8 | `train.py --reuse --expand` | stage 2 (cluster + distractor + sibling features + CE) → XGBoost v2 ("v6") | 8 min |
| 9 | `predict.py --stage1-only` | test features + v1 probabilities (cached) | 12 min |
| 10 | `ce.py --score test` | cross-encoder logits for uncertain test pairs | 13 min |
| 11 | `predict.py --reuse-v1 --suffix _v6` | v6 test probabilities (source of confident pseudo-labels) | 20 min |
| 12 | `train.py --reuse --expand --pseudo-te` | v7: target encodings also learn from confident test pseudo-labels | 6 min |
| 13 | `predict.py --reuse-v1 --suffix _v7` | v7 test probabilities | 20 min |
| 14 | `predict.py --write-only --strict 0.9` | final `output/matching_results.tsv` (one owner, threshold 0.9) and `output/candidate_pairs.tsv` (the pruned candidate set, see below) + official validator | 3 min |

*On the machine below. Peak RAM about 10 GB, VRAM < 5 GB.

**Candidate generation is a two-step cascade.** Blocking retrieves about 46 candidates per S1. A learned
pruning stage (the v1 model) then keeps only pairs with v1 probability >= 0.002. Only those reach the
final matcher (v2), so they are what `candidate_pairs.tsv` contains. The final
`predict.py --write-only --strict 0.9` step writes both files from the cached test probabilities.

Other tools: `eda.py` (statistics), `test_normalize.py` and
`metrics.py` (unit tests); `train.py --fit-country US --val-country India --no-stage2` for the
cross-country check.

## Source files

| File | Purpose |
|---|---|
| `config.py` | paths, seeds, caps, hyper-parameters |
| `io_utils.py` | polars TSV loading, int32 id mapping, parquet cache, timing/memory logging |
| `normalize.py` | all text normalisation and field parsing (hand-written dictionaries) |
| `blocking.py` | candidate generation + recall diagnostics |
| `features.py` | pair features (rapidfuzz + polars, multiprocess) |
| `train.py` | XGBoost (GPU) training, validation, threshold tuning |
| `stage2.py` | stage-2 cluster-context and distractor-cluster features from v1 probabilities |
| `sibling.py` | sibling-business features (house-number gap, target-encoded extra/missing words, legal-form change) |
| `ce.py` | MiniLM cross-encoder: training and scoring |
| `expand.py` | extra out-of-sample S1s for stage-2 training |
| `decide.py` | one-owner assignment and thresholds (tuned with distractor-weighted F0.5) |
| `metrics.py` | exact macro F0.5 scorer |
| `predict.py` | test inference and output writing |
| `run_pipeline.py` | end-to-end entry point |
| `eda.py` | quick statistics report |
