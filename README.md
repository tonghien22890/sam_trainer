## Layer 2: Style Learner (Overview)

- Purpose: score each legal move (per-candidate) using 51-dim features and pick the best.
- Features (51): 27 original + 9 framework-aware (heavily scaled) + 15 multi-sequence (top-3 sequences × 5).
- Training: supervised, label=chosen move in record. Prediction runs on provided `legal_moves` only.
- Defensive checks: prediction filters out moves whose cards are not subset of `hand`.

### Framework (Layer 1) interaction
- `FrameworkGenerator` supplies `framework` fields (core_combos, strength, alt sequences) that feed the 9 framework and 15 multi-sequence features.

### Hand Variations
- Status: DISABLED by default in `scripts/two_layer/train_style_learner.py` to ensure data integrity.
- Reason: variations must recompute `legal_moves`/`action` for the new hand to avoid label drift.
- Re-enable only after adding recomputation for variation hands.

# 🎯 AI-Sam Model Build - Unbeatable Sequence Model

This module contains:

- Unbeatable Sequence Model for Vietnamese Sam (Báo Sâm)
- Optimized General Gameplay Model V3 (per-candidate)

## 🏗️ Solution Overview

### **Báo Sâm Model (Current)**
- Approach: Rulebase → ML Validation → Pattern Learning → Threshold Learning → Generate Sequence
- Decision: `should_declare_bao_sam = (unbeatable_prob >= user_threshold)`
- Straights: 2 (rank=12) excluded; Ace-high straights allowed

### **General Gameplay Model:**
- **Algorithm**: Per-candidate XGBoost classifier (rank-based)
- **Approach**: Rank all legal moves for the current turn and pick the top-scoring move
- **Features**: 22-dim per-candidate features (includes combo type, rank value, breaks_combo_flag, hand context)
- **Performance**: 67.9% turn@1, 80.2% turn@3 on real user data; realistic, non-overfitting

## 📁 Project Structure

```
model_build/
├── docs/
│   ├── UNBEATABLE_SEQUENCE_MODEL_DESIGN.mdc   # Báo Sâm design (authoritative)
│   ├── OPTIMIZED_GENERAL_MODEL_SOLUTION.md    # General (per-candidate) docs
│   └── stage1.mdc                              # Per-candidate spec
├── data/                                      # Training data (generated)
│   ├── phase1_validation_data.jsonl
│   ├── phase2_pattern_data.jsonl
│   └── phase3_threshold_data.jsonl
├── models/                                    # Saved models
├── logs/                                      # Training/eval logs
├── scripts/
│   ├── general/
│   │   ├── optimized_general_model_v3.py      # General model (per-candidate)
│   │   └── train_optimized_model_v3.py        # General training
│   └── unbeatable/
│       ├── unbeatable_sequence_model.py       # Báo Sâm core implementation
│       ├── train_unbeatable_model.py          # Báo Sâm 3-phase training
│       ├── test_unbeatable_model.py           # Báo Sâm tests
│       └── demo_unbeatable_model.py           # Báo Sâm demo
├── STRUCTURE.md                               # File-to-solution mapping
├── deprecated/                                # Legacy artifacts
└── requirements.txt                           # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Báo Sâm Model (Unbeatable Sequence)

#### Full Training Pipeline
```bash
python scripts/unbeatable/train_unbeatable_model.py
```

#### Generate Synthetic Training Data (all phases)
```bash
python scripts/unbeatable/synthetic_data_generator.py
# Outputs:
# - data/validation_training_data.jsonl
# - data/pattern_training_data.jsonl
# - data/threshold_training_data.jsonl
```

#### Run Tests
```bash
python -m unittest model_build.scripts.unbeatable.test_unbeatable_model
```

#### Demo / Interactive
```bash
python scripts/unbeatable/demo_unbeatable_model.py
```

#### Quick Test
```bash
python scripts/unbeatable/quick_test.py
```

### 3. General Gameplay Model (Per-Candidate)

#### Train Model (using real gameplay logs)
```bash
python scripts/general/train_optimized_model_v3.py
# Reads:   training_data.jsonl (from project root)
# Exports: model_build/formatted_training_data.jsonl (rank-based per-candidate format)
# Creates: model_build/models/optimized_general_model_v3.pkl
```

#### Use in Production
```python
from scripts.optimized_general_model_v3 import OptimizedGeneralModelV3

model = OptimizedGeneralModelV3()
model.load('models/optimized_general_model_v3.pkl')

record = {
    'hand': [...],              # Player's hand
    'last_move': {...},         # Last move
    'cards_left': [...],        # Cards left per player
    'meta': {'legal_moves': [...]}  # Available moves
}
result = model.predict(record)  # Per-candidate ranking over legal_moves
```

## 📊 Performance Metrics

### Báo Sâm Model Results (Indicative)
- See logs under `model_build/logs/` for the latest end-to-end results

### General Gameplay Model Results (Per-Candidate)
- **Per-Candidate Sample Accuracy**: 94.18%
- **Turn Accuracy (Top-1)**: 67.9%
- **Turn Accuracy (Top-3)**: 80.2%
- **Notes**: Trained on real user logs; uses rank-based labels (combo_type + rank_value)

## 🎯 Key Features

### Báo Sâm Model Features
- **Rulebase validation**: Chặn hand yếu, yêu cầu đủ 10 lá hợp lệ
- **ML validation**: Học valid/invalid patterns
- **Pattern learning**: Học cách build combo từ user behavior
- **Threshold learning**: Học ngưỡng ra quyết định của user
- **Straight detection**: Loại 2 khỏi sảnh, consume tránh overlap

### General Gameplay Model Features
- **Per-candidate Ranking**: XGBoost ranks all legal moves
- **Rank-based Labels**: Uses combo_type + rank_value instead of exact cards
- **Combo Breaking Awareness**: `breaks_combo_flag` phạt xé bộ mạnh
- **Contextual Signals**: Hand count, cards_left, last_move alignment

## 📚 Documentation

- `docs/UNBEATABLE_SEQUENCE_MODEL_DESIGN.mdc` (Báo Sâm design)
- `docs/OPTIMIZED_GENERAL_MODEL_SOLUTION.md` (General per-candidate docs)
- `STRUCTURE.md` (File-to-solution mapping)
- `deprecated/` (Legacy Hybrid Conservative artifacts)
 
## 🧰 Utilities

- `synthetic_data_generator.py`: Generate synthetic datasets for all 3 phases used by `train_unbeatable_model.py`.

## 🔧 Model Configuration

### Báo Sâm Model
Refer to `train_unbeatable_model.py` for phase-by-phase model choices and parameters.

### General Gameplay Model (Per-Candidate XGBoost)
```python
import xgboost as xgb

xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)
```

---

*Both models are integrated in production via `GeneralPlayProvider` (general) and `ProductionBaoSamProvider` (Báo Sâm).*

**Last Updated**: 2025-09-18  
**Status**: ACTIVE - Unbeatable Sequence Model  
**Deprecated**: Hybrid Conservative solution (moved to `model_build/deprecated/`)

