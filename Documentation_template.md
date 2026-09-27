# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Byte Builders
**Team Members:** Ishant Kumar (Team Leader), Vedanga Gupta, Amey Gupta Gupta, Parth Bansal
**Submission Date:** 27 September 2026

---

## Summary (1 page)

We solve entity resolution with a **blocking → pairwise scoring → cluster-aware re-scoring → decision** pipeline that runs end to end on one laptop (16 GB RAM, 28 cores, RTX 4060 8 GB). The heavy numeric work runs on the GPU.

1. **Normalisation.** Hand-written, country-agnostic dictionaries clean names and addresses: legal forms (incl. transliterated Indic and French), address abbreviations, US/Indian states and French regions/départements, native-script state names, and old/new city names. Non-Latin scripts are transliterated, and a consonant "skeleton" key makes transliterated and English spellings comparable.
2. **Blocking (a two-step cascade).**
   - **Retrieval:** four TF-IDF indexes are searched with sparse top-k on the GPU in both directions (S1→pool, pool→S1), together with 6 exact keys, and capped at 40 candidates per S1 (about 46 with reverse-direction pairs). Pair recall on train is **97.9%**; the entity-level ceiling is **0.9925** macro F0.5.
   - **Learned pruning:** the first-stage model keeps only pairs with probability ≥ 0.002. This leaves **5.8 candidates per S1** (10.05M test pairs, about 8× fewer) while retaining 99.95% of the true matches that survive retrieval. The submitted `candidate_pairs.tsv` is this pruned set: exactly the pairs the final matcher scores.
3. **Matching.**
   - **v1:** XGBoost (CUDA) on 76 pair features.
   - **v2:** a second XGBoost that adds cluster-context features computed from out-of-fold v1 probabilities, "sibling business" features, and the score of a multilingual cross-encoder (MiniLM-L12, Apache-2.0).
4. **Decision.** One owner per S2/S3 record. A per-S1 expected-F0.5 rule is tuned on validation, and the submitted variant uses a stricter probability threshold (0.9), chosen from leaderboard evidence.

**Key insight.** Most test errors were **sibling businesses**: the same base name plus a word such as "Holding" or "Distribution", or a swapped legal form, at a *nearby* house number, often with their own duplicate records. Test, and especially the unseen country France, contains many more of these than train. Features that describe them, a wider v2 training set, and target encodings enriched with confident test pseudo-labels raised the leaderboard from 0.955 to **0.977**.

| Version | Main change | Val macro F0.5 | Public LB |
|---|---|---|---|
| M1 | baseline pipeline | 0.9638 | 0.955 |
| M2 | blocking v2, normalisation v2, stage-2 model | 0.9813 | 0.958 |
| M2 + t=0.9 | stricter decision | 0.9784 | 0.963 |
| v3 + t=0.9 | distractor-cluster features + cross-encoder | 0.9843 | 0.973 |
| v4 + t=0.9 | sibling-business features | 0.9851 | **0.976** |
| v6 + t=0.9 | stage-2 trained on 3× more data | 0.9867 (tuned rule) | 0.976 |
| v7 + t=0.9 | pseudo-label target encodings | 0.9866 (tuned rule) | **0.977** |

---

## 1. Executive Summary
We use a GPU-accelerated blocking-plus-classifier pipeline. There are four TF-IDF kNN indexes and exact keys for recall, followed by a two-stage gradient-boosted matcher. The second stage reasons about each S1's *cluster* of candidates and about sibling businesses, using a small multilingual cross-encoder as an extra signal. Macro F0.5 is optimised directly in the decision step, with one-owner assignment.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA findings (`src/eda.py`):
- **Label structure:** S1 has 2.21M entities (US 1.32M, India 0.88M). **5.6%** of S1s are singletons, the mean number of matches is 3.46 and the maximum is 11. Each S1 has 0–5 S2 matches and 0–6 S3 matches. The one-owner constraint holds exactly. **26%** of S2/S3 records belong to no S1 (distractors).
- **Name noise:** typos, dropped, added or moved legal forms (`Pvt. EFS Print Ventures Ltd.`, `LLC Moncada …`), extra noise words ("Services", "Shri"), repeated tokens, website names (`maurewilliamscolombier.com`, `#empireprogram`), trade names (DBA, `Lumkor trading as …`), and native-script names (Devanagari, Bengali, Kannada, Telugu).
- **Address noise:** reordered segments, `H.NO` / `Door No` / `#` prefixes, state codes versus full names versus native-script state names, `NULL`, and old/new city names. House numbers are perturbed in some true duplicates, and about 4% of pool addresses are empty.
- **Test differs from train:** there are about 23% more pool records per S1. Error analysis showed an excess of *sibling businesses*, especially in France: `Micro Union SAS | 28 Rue …` against `Micro Union Participations SAS | 30 Rue …`. Among ambiguous pairs, test has 3–4× more France pairs than validation.

### 2.2 Solution Strategy
**Approach type:** blocking plus a two-stage classifier (GBDT with a cross-encoder feature), cluster-aware.
**Core innovations:**
- GPU sparse TF-IDF top-k search with torch CSR SpGEMM and a segmented top-k.
- Out-of-fold cluster features.
- Sibling-business features: numeric house-number gap, word-substitution detection, and target-encoded extra/missing words keyed by a script-independent consonant skeleton, so that French and English cognates share statistics.
- Semi-supervised target encodings from confident test pairs, for the unseen country.

---

## 3. Candidate Generation (Blocking)

Everything is partitioned by the `country` label, which is treated as an open set; this is the only way country is used.

- **TF-IDF kNN indexes**, searched on the GPU in both directions:
  - `words`: name tokens, address tokens and a house|street key. df cap 50k, top-40 / reverse top-5.
  - `addr`: address-only (catches trade names). df cap 20k, top-20 / top-3.
  - `name4`: name char 4-grams (typos, website names). df cap 5k, top-20 / top-3.
  - `skel4`: consonant-skeleton 4-grams (transliterated names). df cap 5k, top-20 / top-3.

  Features above the document-frequency cap are pruned so the sparse product stays tractable. Char 3-grams over millions of records were measured to be far too common. We verified that the GPU top-k matches the CPU `sparse_dot_topn` result (99.6% identical pairs, remainder ties).
- **Exact keys**, with blocks larger than 50 per side dropped: `name_nospace`, `name_sorted`, house number + first street token, first name token + house number, phonetic code + city, postcode + first name token.
- **Merge and cap:** union, full-vector cosines for all four indexes, and cheap score = words cos + 0.5 × max(name4, skel4 cos) + 0.3 × addr cos + 0.3 × key hit. Each S1 keeps its top **40**, plus every pair where the S1 is the pool record's rank-1 reverse neighbour.
- **Retrieved candidates:** 99.4M for train (45 per S1) and 80.3M for test (46 per S1).
- **Learned pruning (final candidate set):** the first-stage XGBoost scores the retrieved pairs, and only pairs with probability ≥ 0.002 go on to the final matcher. That is **10.05M test pairs = 5.8 per S1**, the contents of `candidate_pairs.tsv`.
  - On validation this keeps 11.5% of retrieved pairs and 99.95% of the true matches among them, so the pruned set's pair recall is about 97.8%.
  - All 5.74M final matches lie inside it.
  - The cascade exists because the search space must shrink at scale. Cheap vector retrieval narrows millions of records to about 46; a learned filter narrows these to about 6 before the expensive stage-2 features and the cross-encoder are computed.

| Train pair recall | Overall | US | India |
|---|---|---|---|
| M1 blocking (2 indexes, cap 30) | 0.9585 | 0.9723 | 0.9378 |
| final blocking (4 indexes, cap 40) | **0.9789** | **0.9887** | **0.9643** |

Recall versus cap (final): @5 0.876, @10 0.953, @20 0.969, @30 0.974, @40 0.977; 0.979 with reverse rank-1 pairs.
Recall per blocker (and share of pairs found *only* by that blocker): words fwd 0.958 (0.51%), words rev 0.955 (0.30%), addr fwd 0.812, name4 fwd 0.550, skel4 fwd 0.478, key_phon_city 0.452 (0.08%), key_nospace 0.457.
Entity-level ceiling with a perfect matcher on these candidates: **0.9925** macro F0.5.

---

## 4. Matching Model

**Pair features (v1, 76):**
- *Name:* rapidfuzz ratio / partial / token_sort / token_set / WRatio / Jaro-Winkler / Levenshtein on core names, space-less (website) similarity, token Jaccard, exact core / sorted / space-less match, first-token and phonetic equality, acronym match, legal-form equal / conflict / missing, website flag, length ratio, consonant-skeleton similarity.
- *Address:* empty flags, token_set / token_sort / partial, token Jaccard, house number equal / Jaro-Winkler / missing, all-numbers Jaccard, postcode, street Jaro-Winkler and token_set, city Jaro-Winkler, city-in-other-address, state equal.
- *Context:* cosines of the four TF-IDF indexes, cheap score, rank in S1, rank in candidate, candidate count, competing S1 count, score gaps to the S1's best and to the best competing S1, name-only rank and gap among competing S1s, number of S1s with the same name, blocker bits, and source (S2/S3).

**Stage-2 features (v2), computed from out-of-fold v1 probability p1:**
- *v1 context:* p1, rank, S1 max / sum / count(p1 > 0.5), best competing S1 probability and gap.
- *Cluster consistency:* similarity of the candidate to the S1's top-3 other candidates (name, address, house-number agreement, probability-weighted).
- *Distractor clusters:* other top candidates sharing *this candidate's* house number that differs from the S1's, and candidates closer to this one than to the S1.
- *Sibling business:* numeric house-number gap and a "nearby 1–20" flag. On train, a gap of 4–20 matches 3.7% of the time, against 63% for the same number. Also extra and missing name words, target-encoded (e.g. `holding` 0.03%, `dba` 64%), with keys given by the consonant skeleton so that `groupe`/`group` and `développement`/`development` share statistics. Also word substitution versus typo, and legal-form swap/drop with a target-encoded legal pair (SARL→SAS 1%). Encodings are learned on train S1s outside the fit/validation sets, plus confident test pseudo-labels (v7).
- *Cross-encoder:* the logit of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` fine-tuned as a pair classifier on raw "name | address" pairs. The training set is 825k pairs from 90k train S1s outside fit/val, all positives plus the top blocking negatives. Word embeddings are frozen; bf16; fused AdamW; 1 epoch; held-out AUC **0.998**. It scores only pairs with p1 ∈ [0.01, 0.995] (3.5M test pairs). mDeBERTa-v3-base was tried first but did not train under transformers 5.16 / torch 2.14, so it was dropped.

**Model type:** XGBoost 3.4 on CUDA (`hist`, leaf-wise, 127 leaves, lr 0.05, subsample and colsample 0.8, λ = 1, early stopping on validation logloss). LightGBM's pip wheel has no GPU build on Windows. v2 is trained and applied only where p1 ≥ 0.002 (11.5% of pairs, 99.95% of positives). In v6/v7 it uses 4.17M rows: the fit S1s plus 500k extra train S1s on which v1 is out-of-sample.

**Validation:** fit 298k and validation 101k S1s, sampled by whole (country, city) partitions so that competing S1s fall on the same side. Every S1 in the evaluation set counts, singletons included.

**Threshold selection:**
- one owner per record (highest probability);
- grid search of threshold t and empty-S1 threshold, versus a per-S1 expected-F0.5 top-k rule;
- the objective also weights distractor false positives to mimic test's higher distractor density.

Leaderboard probes showed the tuned rule is too lenient on test's siblings, so the submission uses **t = 0.9**. t = 0.95 was worse.

**Cross-country generalisation check** (v1 only; fit about 300k S1s of one country, validate on about 100k S1s of the other):

| Train → validate | Macro F0.5 | Same-country v1 reference |
|---|---|---|
| India → US | 0.958 | US 0.979 |
| US → India | 0.860 | India 0.968 |

Transfer is asymmetric. India's data is noisier (native scripts, long, reordered addresses, `NULL` tokens), so a model trained on it generalises well to the cleaner US data. The reverse drop shows that script and format variety must be seen in training. The response was:
- to train the final model on both countries;
- to make every feature country-agnostic (transliteration, the consonant skeleton, state / region dictionaries applied to all records);
- to let the unseen country (France) contribute word and legal-form statistics through confident test pseudo-labels.

---

## 5. Results & Error Analysis

- **Validation macro F0.5:** 0.9867 overall (US 0.987, India 0.982) with the tuned rule. The submitted t = 0.9 rule scores about 0.985 on validation and 0.977 on the public leaderboard.
- **Common false positives:** sibling businesses: same base name plus "Holding/Distribution/International/Groupe/Westgate", a swapped legal form (SAS↔SARL, Inc↔LLC), or a substituted descriptive word, at an adjacent or nearby house number. Also exact-name businesses in another city, and empty-address records whose name is shared by several S1s.
- **Common false negatives:**
  - native-script names with sparse addresses (`ಸೆವೆನ್ ಇನ್‌ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್ ಲಿಮಿಟೆಡ್ | Ground Floor, Bangalore South`);
  - empty-address records of generic names;
  - trade names with a changed address;
  - blocking misses (about 2.1% of pairs, 3.6% in India).

---

## 6. Conclusion
A GPU-accelerated blocking-plus-two-stage-matcher pipeline reached 0.9867 validation and 0.977 public-leaderboard macro F0.5. The largest gains came from recall-oriented blocking, from cluster-level features, and from diagnosing the train/test shift: sibling businesses and the unseen country. That diagnosis came from targeted leaderboard probes, and the resulting country-agnostic features and semi-supervised encodings address it.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`:
- `run_pipeline.py`: end to end.
- `normalize.py`, `blocking.py`, `features.py`, `stage2.py`, `sibling.py`, `ce.py`, `expand.py`, `train.py`, `decide.py`, `metrics.py`, `predict.py`.
- `eda.py`, `test_normalize.py`.

Exact commands, runtimes and memory are in `code/business_entity_resolution/README.md`.

### B. Licences and data
- **Models and libraries:** XGBoost (Apache-2.0), paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params), PyTorch (BSD), transformers (Apache-2.0), scikit-learn (BSD), polars (MIT), rapidfuzz (MIT), unidecode (GPL-2.0, text transliteration utility; not an ML model), jellyfish (MIT).
- **No external data:** no APIs, geocoders, registries, web data or downloaded gazetteers were used. All dictionaries are hand-written in `normalize.py`. The only downloaded artefact is the pretrained MiniLM checkpoint, which is allowed by the model-licence rule.
- **Test data:** pseudo-labels are derived only from our own model's confident predictions on the provided test data.
