# Model Build

Machine learning models for Vietnamese card games (Sam/TLMN).

## Overview

This module provides:
- **Two-Layer Architecture** (Primary): Framework-based move selection using Style Learner
- **Unbeatable Sequence Model**: Báo Sâm (declaration) decision model

## Architecture

### Two-Layer Architecture (Style Learner)

**Purpose**: Score each legal move using 51-dim features and pick the best.

**Layer 1 - Framework Generator**:
- Generates strategic framework from hand using `SequenceEvaluator`
- Outputs: `core_combos`, `framework_strength`, `alternative_sequences`

**Layer 2 - Style Learner**:
- **Features** (51 dims): 
  - 27 original: combo counts, cards_left, hand_count, combo type onehot, hybrid rank, combo length, breaks_combo_flag, individual move strength, enhanced breaks penalty
  - 9 framework-aware (heavily scaled): alignment, priority, breaking severity, strength, position, combo type preference, rank preference, timing preference, sequence compliance
  - 15 multi-sequence: top 3 sequences × 5 features each
- **Training**: Supervised learning-to-rank using XGBRanker
- **Pass Strategy**: Automatic pass option added to `legal_moves` for tactical gameplay

**Note**: Hand variations are DISABLED by default to ensure data integrity (avoids label drift).

### Unbeatable Sequence Model (Báo Sâm)

**Purpose**: Decide whether to declare "Báo Sâm" (unbeatable hand).

**Approach**: 3-phase ML pipeline
1. **Rulebase Validation**: Filters weak hands, requires 10 valid cards
2. **ML Validation**: Learns valid/invalid patterns
3. **Pattern Learning**: Learns combo-building from user behavior
4. **Threshold Learning**: Learns user decision thresholds

**Decision**: `should_declare_bao_sam = (unbeatable_prob >= user_threshold)`

## Project Structure

```
model_build/
├── docs/
│   ├── UNBEATABLE_SEQUENCE_MODEL_DESIGN.mdc   # Báo Sâm design
│   ├── CONSTRAINED_SEQUENCE_PLANNER.md         # Sequence planning docs
│   └── RANKER_MIGRATION.md                     # XGBRanker migration guide
├── data/                                       # Training data
│   ├── phase1_validation_data.jsonl           # Unbeatable phase 1
│   ├── phase2_pattern_data.jsonl              # Unbeatable phase 2
│   └── phase3_threshold_data.jsonl            # Unbeatable phase 3
├── models/                                     # Saved models
│   ├── style_learner_sam.pkl                  # Two-Layer SAM model
│   ├── style_learner_tlmn.pkl                 # Two-Layer TLMN model
│   ├── validation_model.pkl                   # Unbeatable phase 1
│   ├── pattern_model.pkl                      # Unbeatable phase 2
│   └── threshold_model.pkl                    # Unbeatable phase 3
├── scripts/
│   ├── two_layer/                             # Two-Layer Architecture
│   │   ├── framework_generator.py            # Layer 1
│   │   ├── style_learner.py                  # Layer 2
│   │   ├── train_style_learner_core.py       # Core trainer
│   │   ├── train_style_learner_sam.py        # SAM wrapper
│   │   └── train_style_learner_tlmn.py       # TLMN wrapper
│   └── unbeatable/                            # Báo Sâm Model
│       ├── unbeatable_sequence_model.py       # Core implementation
│       ├── train_unbeatable_model.py          # 3-phase training
│       ├── synthetic_data_generator.py        # Data generation
│       ├── test_unbeatable_model.py           # Tests
│       └── demo_unbeatable_model.py           # Demo
├── simple_sam.jsonl                           # Default SAM training data
├── simple_tlmn.jsonl                          # Default TLMN training data
├── train/                                     # Additional training data
└── requirements.txt                           # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Two-Layer Architecture

#### Train SAM Model

```bash
python scripts/two_layer/train_style_learner_sam.py
# Input:  simple_sam.jsonl (default)
# Output: models/style_learner_sam.pkl
```

#### Train TLMN Model

```bash
python scripts/two_layer/train_style_learner_tlmn.py
# Input:  simple_tlmn.jsonl (default)
# Output: models/style_learner_tlmn.pkl
```

#### Custom Data Training

```bash
python scripts/two_layer/train_style_learner_core.py \
  --game_type sam \
  --data_path custom_data.jsonl
```

#### Ensemble Training (Combine Base + New Data)

Combine existing data with newly converted real gameplay logs:

**PowerShell**:
```powershell
python scripts/two_layer/train_style_learner_core.py ^
  --game_type sam ^
  --ensemble ^
  --base_data "d:\Source-Code\AI-Sam\model_build\simple_sam.jsonl" ^
  --new_data "d:\Source-Code\AI-Sam\model_build\simple_sam.jsonl" ^
  --base_weight 1 ^
  --new_weight 5
```

**Bash**:
```bash
python scripts/two_layer/train_style_learner_core.py \
  --game_type sam \
  --ensemble \
  --base_data "/d/Source-Code/AI-Sam/model_build/simple_sam.jsonl" \
  --new_data "/d/Source-Code/AI-Sam/model_build/simple_sam.jsonl" \
  --base_weight 1 \
  --new_weight 5
```

**Parameters**:
- `--base_data`: Existing training data file
- `--new_data`: New data to combine (e.g., from `convert-realdata/`)
- `--base_weight`: How many times to repeat base data (default: 1)
- `--new_weight`: How many times to repeat new data (default: 5)

### 3. Unbeatable Sequence Model (Báo Sâm)

#### Full Training Pipeline

```bash
python scripts/unbeatable/train_unbeatable_model.py
```

#### Generate Synthetic Training Data

```bash
python scripts/unbeatable/synthetic_data_generator.py
# Outputs:
# - data/phase1_validation_data.jsonl
# - data/phase2_pattern_data.jsonl
# - data/phase3_threshold_data.jsonl
```

#### Run Tests

```bash
python -m unittest model_build.scripts.unbeatable.test_unbeatable_model
```

#### Demo / Interactive

```bash
python scripts/unbeatable/demo_unbeatable_model.py
```

## Data Preparation

### Convert Real Gameplay Logs

Use the `convert-realdata/` tool to convert raw gameplay logs into training-ready format:

```bash
cd ../convert-realdata
python convert_log_to_format.py --build_style_data
```

This generates:
- `converted.jsonl`: Basic records matching format
- `../model_build/simple_sam.jsonl`: Enriched records for Style Learner (with `meta.legal_moves`, `action.stage2`, `cards_left`, `framework`)

See `../convert-realdata/README.md` for details.

### Training Data Format

Each line in `*.jsonl` should be a JSON object with:

```json
{
  "game_type": "sam",
  "hand": [8, 15, 17, 45, 49, 1, 41, 42, 43, 37],
  "last_move": {
    "cards": [31, 35, 39]
  },
  "players_count": [1, 0, 8],
  "cards_left": [1, 0, 0, 0],
  "meta": {
    "legal_moves": [
      {
        "combo_type": "pair",
        "rank_value": 11,
        "cards": [45, 49]
      }
    ]
  },
  "action": {
    "stage2": {
      "combo_type": "pair",
      "rank_value": 11,
      "cards": [45, 49]
    }
  },
  "framework": {
    "core_combos": [...],
    "framework_strength": 0.85
  }
}
```

## Model Configuration

### Style Learner (XGBRanker)

```python
xgb.XGBRanker(
    objective='rank:pairwise',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='ndcg@5'
)
```

### Unbeatable Model

Refer to `scripts/unbeatable/train_unbeatable_model.py` for phase-by-phase model configuration.

## Performance

### Style Learner
- Learning-to-rank approach using XGBRanker
- Framework-aware features heavily scaled to override data bias
- Ensemble training supports combining base + new data with weights

### Unbeatable Sequence Model
- 3-phase ML pipeline (validation → pattern → threshold)
- See `logs/` for latest end-to-end results

## Documentation

- `docs/UNBEATABLE_SEQUENCE_MODEL_DESIGN.mdc` - Báo Sâm design
- `docs/CONSTRAINED_SEQUENCE_PLANNER.md` - Sequence planning
- `docs/RANKER_MIGRATION.md` - XGBRanker migration guide
- `STRUCTURE.md` - File-to-solution mapping
- `../convert-realdata/README.md` - Real log conversion tool

## Production Integration

Models are integrated via:
- **Two-Layer**: `TwoLayerAdapter` (in `ai_common/adapters/`)
- **Unbeatable**: `UnbeatableAdapter` (in `ai_common/adapters/`)

Model paths can be configured via `AISAM_MODELS_DIR` environment variable.

---

**Status**: ACTIVE  
**Last Updated**: 2025-01-XX  
**Primary Solution**: Two-Layer Architecture (Style Learner)  
**Features**: Framework-aware ranking, Ensemble training, Pass strategy, Combo preservation
