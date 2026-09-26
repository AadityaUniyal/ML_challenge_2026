# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains the complete, reproducible, end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.


## 1. Project Overview & Architecture

* **Source 1:** Deduplicated reference source (ground truth query set).
* **Source 2 & Source 3:** Independent, noisy target sources.
* **Objective:** Map each Source 1 reference record to zero (singleton), one, or many matching records across Source 2 and Source 3.
* **Evaluation Metric:** Macro-averaged $F_{0.5}$ across all Source 1 entities (penalizing false merges twice as heavily as false negatives, with singletons earning a full 1.0 when correctly predicted empty).
* **Candidate Bounding:** Strict candidate generation capping ($\le 25$ candidates per entity) with **94.48% recall ceiling** and **99.9998% search space reduction**.

---

## 2. Repository Structure

```text
amazon-ml-challenge-2026/
│
├── README.md                          # Complete execution and reproduction runbook
├── Documentation_template.md          # 1-2 page official scientific methodology document
├── requirements.txt                   # Pinned dependency versions
├── .gitignore                         # Excludes raw data and bulky temporary binaries
│
├── PLAN/
│   ├── team_contract.md               # Clear division of responsibilities (Role A & B)
│   └── experiment_log.csv             # Iterative experiment tracking log
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Global paths and hyperparameters
│   ├── io_utils.py                    # Robust TSV reader and streamer
│   ├── data_validation.py             # Schema, prefix integrity, and open-set country checks
│   ├── normalization.py               # Canonical multi-view string normalizer
│   ├── blocking.py                    # Multi-channel inverted index retrieval engine
│   ├── features.py                    # 20D C++ RapidFuzz & token interaction feature extractor
│   ├── training.py                    # Model training module (LightGBM / CatBoost)
│   ├── inference.py                   # Batched candidate probability scoring engine
│   ├── thresholding.py                # Macro F0.5 grid-search threshold optimizer
│   ├── evaluation.py                  # Exact official Macro F0.5 evaluation metric
│   ├── output_writer.py               # TSV submission formatter
│   └── pipeline.py                    # Unified end-to-end entity resolution pipeline
│
├── scripts/
│   ├── inspect_data.py                # Dataset profiling runner
│   ├── build_candidates_train.py      # Hard negative pair generator for training
│   ├── train_model.py                 # Model training runner
│   ├── evaluate_validation.py         # Validation evaluation & metric breakdown
│   ├── generate_candidates_test.py    # Test candidate generation (output/candidate_pairs.tsv)
│   ├── predict_test.py                # Test scoring & thresholding (output/matching_results.tsv)
│   ├── validate_submission.py         # Official competition submission validator
│   └── run_all.py                     # One-command full pipeline reproduction script
│
├── configs/
│   ├── baseline.json                  # Baseline blocking and model parameters
│   └── final.json                     # Frozen optimal parameters (tau = 0.65)
│
├── reports/
│   ├── data_profile.md                # 24M record distribution audit
│   ├── blocking_report.md             # 94.48% recall ceiling & 24.79 candidate efficiency report
│   └── validation_report.md           # Model validation metrics and feature importances
│
├── output/
│   ├── matching_results.tsv           # Final matches (uploaded to live leaderboard)
│   └── candidate_pairs.tsv            # Bounded candidate set fed to matching model
│
└── tests/
    ├── test_normalization.py          # Unit tests for text normalization
    ├── test_blocking.py               # Unit tests for multi-channel retrieval
    ├── test_features.py               # Unit tests for feature extraction
    ├── test_evaluation.py             # Unit tests for macro F0.5 calculation
    └── test_output_format.py          # Unit tests for submission formatting
```

---

## 3. Environment Setup

Python 3.8+ is required (tested on Python 3.12).

```bash
# 1. Create and activate a clean virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. How to Reproduce End-to-End

### One-Command Full Pipeline
To execute the automated pipeline from unit tests to final validation:
```bash
python scripts/run_all.py
```

### Modular Pipeline Steps

#### Step 1: Data Profiling & Quality Audit (Role A)
```bash
python scripts/inspect_data.py
```
Generates `reports/data_profile.md`.

#### Step 2: Multi-Channel Candidate Generation (Role A)
```bash
python scripts/generate_candidates_test.py
```
Produces `output/candidate_pairs.tsv`.

#### Step 3: Train Classifier & Tune Threshold (Role B)
```bash
python scripts/train_model.py
```
Produces `artifacts/models/lgbm_er_model.pkl` and freezes $\tau = 0.65$.

#### Step 4: Full Test Set Inference (Role B)
```bash
python scripts/predict_test.py
```
Generates `output/matching_results.tsv`.

#### Step 5: Official Competition Validation
```bash
python scripts/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
```

---

## 5. Compliance & Fair Play Statement

* **Zero External Data:** All representations, blocking keys, and features are computed strictly from the supplied challenge datasets. No commercial APIs, geocoding engines, or external registries are queried.
* **Open-Set Country Generalization:** Country is treated dynamically as an open string field; no hardcoded whitelists exist. (France in test is dynamically supported).
* **Model Constraints:** LightGBM binary classifier is MIT licensed and contains <1 Million parameters (well within the $\le 8\text{B}$ parameter limit).
