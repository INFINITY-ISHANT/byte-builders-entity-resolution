# Business Entity Resolution — Amazon ML Challenge 2026

**Team Byte Builders:** Ishant Kumar (Team Leader), Vedanga Gupta, Amey Gupta, Parth Bansal

Given noisy business records from three independent sources, this project finds, for every entity in the deduplicated reference source (S1), all matching records in sources S2 and S3. The data covers US and India in training; the test set adds **France**, a country never seen in training. The metric is macro-averaged **F0.5**, which weights precision twice as much as recall.

**Result: public leaderboard 0.977** (validation 0.987), up from 0.955 for our first baseline.

## Approach

```
normalise ──► block (GPU kNN + exact keys) ──► XGBoost v1 ──► cluster / sibling features + cross-encoder ──► XGBoost v2 ──► one-owner + threshold
```

1. **Normalisation.** Hand-written, country-agnostic rules clean names and addresses:
   - legal forms, including transliterated Indic and French ones;
   - address abbreviations;
   - US/Indian states, French regions/départements, and native-script state names;
   - transliteration of non-Latin scripts, plus a consonant "skeleton" key so that `राम मार्केटिंग` and `Ram Marketing` compare equal.
2. **Blocking.**
   - Four TF-IDF indexes, searched with sparse top-k on the GPU (torch CSR SpGEMM) in both directions: name+address words, address-only, name 4-grams, and skeleton 4-grams.
   - Six exact keys.
   - 40 candidates kept per S1. This finds **97.9%** of true pairs; a perfect matcher on these candidates would score 0.9925.
3. **Stage 1.** XGBoost (CUDA) on 76 pair features: string similarities, address parts, and competition between S1s for the same record.
4. **Stage 2.** A second XGBoost adds:
   - **cluster features:** does a candidate agree with the S1's other likely duplicates?
   - **sibling-business features:** house-number gap, target-encoded extra/missing name words, and legal-form swaps. Target encodings also learn from confident test pseudo-labels, which covers the unseen country.
   - **a multilingual cross-encoder score:** MiniLM-L12, fine-tuned on candidate pairs.
5. **Decision.** Each S2/S3 record goes to at most one S1, and a match needs probability ≥ 0.9.

**Key insight.** Most test errors were *sibling businesses*: the same base name plus "Holding" / "Distribution", or a swapped legal form, at a nearby house number. Test, and especially France, has far more of them than train. The full write-up, with experiments, blocking recall tables and error analysis, is in [`Documentation_template.md`](Documentation_template.md). Every leaderboard upload is listed in [`submissions_log.md`](submissions_log.md).

## Repository layout

```
code/business_entity_resolution/
  src/               all source code (entry point: run_pipeline.py)
  README.md          exact reproduction steps, runtimes, hardware
  requirements.txt   pinned dependencies
  models/            feature lists, thresholds, feature importances
Documentation_template.md   methodology write-up
submissions_log.md          leaderboard submission history
make_zip.py                 builds the challenge submission package
```

## Running it

The challenge data is not redistributed here. Place it at `student_resource/dataset/{train,test}/`, or set `BER_DATA_DIR`. Then:

```bash
pip install -r code/business_entity_resolution/requirements.txt   # plus a CUDA build of PyTorch
cd code/business_entity_resolution/src
python run_pipeline.py --stage all
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It was developed on a laptop with 16 GB RAM, 28 threads and an RTX 4060 (8 GB); a full run takes about 3–4 hours. See [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for each step.

## Licences

- **Models:** XGBoost (Apache-2.0) and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0).
- **External data:** none. No APIs, geocoders or downloaded gazetteers are used; every dictionary is hand-written in `normalize.py`.
