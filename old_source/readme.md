Model Build (Phase 4) — DEPRECATED
==================================

This module contains an older data processing, model architecture, and training pipeline for AI-Sam.
It is kept for reference only. The active implementation is rank-only, per-candidate
in `model_build/scripts/optimized_general_model_v3.py`.

Files
-----
- `data_loader.py`: Load JSONLines logs, normalize two-stage/legacy actions, encode flat features (hand, last_move, etc.). Includes export of normalized artifacts (`X.npy`, `y.npy`, `candidates.jsonl`).
- `model_architecture.py`: Estimator factory (DecisionTree / RandomForest).
- `trainer.py`: Train sklearn model, report accuracy, save model `.pkl`.
- `inference.py`: Programmatic inference helper to map a record → predicted legal move.
- `requirements.txt`, `pyproject.toml`: standalone install metadata.

Quick Start
-----------
```bash
# 1) (Optional) Build/install locally via pyproject
# pip install -e .

# 2) Or just install deps
pip install -r requirements.txt

# 3) Train (example)
python -m model_build.trainer \
  --data ../training_data.jsonl \
  --out runs/phase4_rf.pkl \
  --model random_forest \
  --export runs/phase4_export

# Outputs:
# - runs/phase4_rf.pkl: trained sklearn model
# - runs/phase4_export/{X.npy,y.npy,candidates.jsonl}: normalized dataset artifacts for trace/debug
```

Data Format Assumptions
-----------------------
- Input logs follow the repository-wide schema with two-stage `action` (or legacy flat action which is auto-normalized).
- Required keys: `hand`, `last_move` (nullable), `players_left`, `cards_left`, `meta.legal_moves`.
- Game-agnostic: works for both Sam and TLMN as features are derived from shared fields.

Feature Encoding (baseline, suit-dependent)
------------------------------------------
- Hand one-hot: 52 dims (suit-dependent)
- Last move: combo_type one-hot (6) + rank_value (1)
- Aggregate: players_left_count (1), cards_left_sum (1)
- Label: index of chosen `meta.legal_moves` that matches `action.stage2.cards` (or -1 if unmatched → filtered out at train time)

Inference Usage (programmatic)
------------------------------
```python
from model_build.inference import predict

# `record` must contain fields like in training logs (one state)
move = predict("runs/phase4_rf.pkl", record)
# move is one of legal_moves; fallback to pass if none valid
```

Configuration
-------------
- See `config.example.yaml` for sample paths and hyperparams (informational); CLI flags on `trainer.py` are the source of truth.

Notes
-----
- Deprecated in favor of rank-only per-candidate modeling (suit-agnostic).
- If needed for backward compatibility, keep `data_loader.py` export interfaces stable.


