# Amazon ML Challenge 2026: Business Entity Resolution

This package contains the complete, self-contained, end-to-end pipeline to reproduce the candidate blocking set (`candidate_pairs.tsv`) and the final entity resolution matches (`matching_results.tsv`).

---

## Environment Setup

Requires Python 3.8+ (tested on Python 3.12).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Text normalization, legal suffix & number extraction
│   ├── blocking.py         # Multi-pass country-partitioned inverted index
│   ├── features.py         # C++ accelerated RapidFuzz pairwise feature extraction
│   ├── model.py            # LightGBM classifier with F0.5 threshold optimization
│   └── metric.py           # Official competition Macro F_0.5 evaluation metric
├── train.py                # Model training script
├── run_inference.py        # End-to-end test inference runner
├── assemble_outputs.py     # Aggregator and validation checker
├── README.md               # Reproduction guide
└── requirements.txt        # Pinned dependencies
```

---

## How to Reproduce End-to-End

### Step 1: Model Training
Train the LightGBM classifier and compute optimal decision threshold:
```bash
python train.py
```
This produces `lgbm_er_model.pkl` and `model_config.json`.

### Step 2: Test Set Inference
Run candidate generation and pairwise scoring across test records:
```bash
python run_inference.py --country France
python run_inference.py --country US
python run_inference.py --country India
```

### Step 3: Assemble & Validate Submission Files
Merge country predictions into final submission files and run verification:
```bash
python assemble_outputs.py
```
This produces:
- `output/matching_results.tsv` (Leaderboard upload)
- `output/candidate_pairs.tsv` (Blocking candidate set)

And validates both against `utils/validate_submission.py`.
