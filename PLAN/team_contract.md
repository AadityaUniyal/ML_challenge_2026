# Team Contract: Amazon ML Challenge 2026

## 1. Division of Responsibilities
* **Member 1 / Role A (Owner: You):**
  * Data Understanding & Profiling (`reports/data_profile.md`)
  * Text Normalization & Canonical Representations (`src/normalization.py`)
  * Multi-Channel Blocking Engine (`src/blocking.py`)
  * Candidate Generation for Train & Test (`output/candidate_pairs.tsv`)
  * Unit Testing for Retrieval & Normalization (`tests/test_normalization.py`, `tests/test_blocking.py`)
  * Candidate Metrics: High recall ceiling (>94%), strict candidate capping ($\le 25$ per S1)

* **Member 2 / Role B (Owner: Teammate):**
  * Pairwise Feature Extraction (`src/features.py`)
  * Classifier Training & Hard Negative Optimization (`src/training.py`, `artifacts/models/`)
  * Macro $F_{0.5}$ Evaluation & Decision Gating (`src/thresholding.py`, `src/evaluation.py`)
  * Final Entity Resolution Output (`output/matching_results.tsv`)

## 2. Invariants & Rules
1. **Candidate Subset Invariant:** Every ID in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
2. **Open-Set Country Support:** Open-set string comparison; no hardcoded US/India whitelists (France in test).
3. **No External Lookups:** Zero external APIs or geocoding.
4. **Reproducibility:** All code must run from a single command (`python scripts/run_all.py`).
