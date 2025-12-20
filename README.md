# Model Build

Machine learning models for Vietnamese card games (Sam/TLMN).

## Overview

This module provides:
- **Two-Layer Architecture** (Primary): Framework-based move selection using Style Learner
- **RL Agent** (PPO): Reinforcement learning agent with diverse self-play and reward shaping
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
│   ├── rl_agent/                              # RL Agent (PPO)
│   │   ├── trainer.py                        # PPO trainer with opponent pool
│   │   ├── policy.py                         # Policy and value networks
│   │   ├── env.py                            # RL environment with reward shaping
│   │   ├── features.py                       # Feature builder
│   │   └── train_rl_core.py                  # Training script
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

### 1. Install Python, Virtualenv & Dependencies

**Step 1 – Install Python (3.8+)**

- **Windows**: tải từ `https://www.python.org/downloads/`, tick `Add Python to PATH` khi cài.
- **Linux (Ubuntu)**:
  ```bash
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip
  ```

**Step 2 – Tạo virtualenv trong `model_build`**

```bash
cd model_build
python -m venv .venv          # hoặc python3 -m venv .venv

# Windows (PowerShell / CMD)
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

**Step 3 – Cài dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.1. Deploy RL Training to Server/VM

When you want to train RL models (PPO) on a remote server/VM, you **do not** need the whole monorepo, but you **must** keep the relative structure for imports to work.

**Minimum folders/files to copy to the server:**

- `game_engine/` – core game logic (Sam/TLMN)
- `ai_common/` – shared utilities (`SequenceEvaluator`, `ComboAnalyzer`, adapters, etc.)
- `model_build/` – this folder (Two-Layer, RL agent, models, data)
- (Optional) `logs/` – if you want to keep training logs history

**Recommended layout on the server:**

```text
/path/to/AI-Sam/
├── game_engine/
├── ai_common/
└── model_build/
    ├── requirements.txt
    ├── scripts/
    │   ├── two_layer/
    │   └── rl_agent/
    └── models/
```

**Setup steps on server (Linux/macOS shell):**

```bash
cd /path/to/AI-Sam/model_build
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Sanity check: run a tiny RL training
python scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 10 \
  --save_path models/rl_policy_sam_test.pt
```

### 1.2. Chạy training ẩn (background) và xem log

**Linux/macOS (khuyến nghị dùng `nohup` + log file):**

```bash
cd /path/to/AI-Sam
source model_build/.venv/bin/activate   # nếu đã tạo virtualenv

# Chạy training ẩn, ghi log ra file
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/rl_train_sam.log 2>&1 &

# Xem log realtime
tail -f model_build/logs/rl_train_sam.log
```

**Windows (PowerShell / CMD, foreground nhưng log vào file):**

```powershell
cd D:\Source-Code\AI-Sam
.\model_build\.venv\Scripts\activate

# Chạy và ghi log
python model_build\scripts\rl_agent\train_rl_core.py `
  --game_type sam `
  --episodes 50000 `
  --save_path model_build\models\rl_policy_sam.pt `
  > model_build\logs\rl_train_sam.log 2>&1

# Mở log bằng Notepad hoặc tail qua WSL
notepad model_build\logs\rl_train_sam.log
```

**Gợi ý:**
- Luôn ghi log vào `model_build/logs/` để dễ theo dõi lịch sử.
- Với server Linux, có thể dùng `screen`/`tmux` nếu muốn giữ session và xem log trực tiếp.

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

### 3. RL Agent (PPO)

#### Train RL Policy for Sam/TLMN

**Basic Training**:

```bash
python scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 100000 \
  --save_path models/rl_policy_sam.pt
```

**With Custom Configuration**:

```bash
python scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 100000 \
  --seats 2 \
  --lr 3e-4 \
  --hidden_dim 128 \
  --save_path models/rl_policy_sam.pt \
  --self_play  # Enable self-play (default)
```

**Resume Training**:

```bash
python scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --load_path models/rl_policy_sam.pt \
  --save_path models/rl_policy_sam.pt
```

#### RL Agent Features

**PPO (Proximal Policy Optimization)**:
- Stable policy gradient algorithm
- Clipped objective to prevent large policy updates
- Separate value network (critic) for advantage estimation

**Diverse Self-Play with Opponent Pool**:
- **Opponent Pool**: Maintains a pool of previous checkpoints as opponents
- **Variation Parameters**: Each opponent gets unique variation:
  - **Temperature variation**: Random temperature in `[0.7, 1.5]` range
    - Lower (0.7): More deterministic/conservative play
    - Higher (1.5): More random/exploratory play
  - **Weight noise** (optional): Small random noise added to weights for diversity
- **Pool Management**: Automatically saves checkpoints every N episodes, maintains fixed pool size

**Reward Design**:
- **Step Rewards**: Dense reward signal using potential-based reward shaping
  - Encourages agent to reduce cards (`+w1 * d_me`)
  - Potential function: `φ(c_opp, c_me) = k * (min_opp - c_me)`
  - Shaping term: `γ * φ(after) - φ(before)`
- **Final Rewards**: Sparse reward at game end
  - Win: `+sum(opponent_cards_left)`
  - Loss: `-1 × cards_left` (or `-3 × cards_left` if has card 2/four_kind)
  - Special penalty: `-15` if still has all initial cards

**Configuration Parameters**:

```python
TrainerConfig(
    # Basic settings
    game_type="sam",
    seats=2,
    episodes=100000,
    gamma=0.99,
    lr=3e-4,
    hidden_dim=128,
    
    # PPO hyperparameters
    ppo_clip_epsilon=0.2,
    ppo_epochs=6,
    ppo_batch_size=128,
    value_coef=0.5,
    entropy_coef=0.02,
    
    # Opponent pool (diverse self-play)
    opponent_pool_size=4,                    # Number of checkpoints in pool
    opponent_pool_checkpoint_interval=500,   # Save to pool every N episodes
    opponent_temperature_min=0.7,            # Min temperature (conservative)
    opponent_temperature_max=1.5,            # Max temperature (random)
    opponent_weight_noise_std=0.02,          # Weight noise std (0 = disabled)
)
```

#### RL Training Best Practices

1. **Start with shorter episodes**: Test with 1000-10000 episodes first
2. **Monitor training**: Check logs for average reward trends
3. **Use opponent pool**: Enables diverse training, helps agent improve beyond following framework
4. **Step rewards**: Provide dense feedback, especially useful for learning mid-game strategies like "giật cái"
5. **Resume training**: Can continue training from checkpoints to improve further

**See also**: `RL_TRAINING_SETUP.md` for detailed setup guide, `REWARD_DESIGN.md` for reward function details

### 4. Unbeatable Sequence Model (Báo Sâm)

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
- `RL_TRAINING_SETUP.md` - RL training setup and deployment guide
- `REWARD_DESIGN.md` - RL reward function design
- `scripts/rl_agent/GIAT_CAI_PLAN.md` - Strategy improvement plan
- `STRUCTURE.md` - File-to-solution mapping
- `../convert-realdata/README.md` - Real log conversion tool

## Production Integration

Models are integrated via:
- **Two-Layer**: `TwoLayerAdapter` (in `ai_common/adapters/`)
- **RL Agent**: `RLAdapter` (in `ai_common/adapters/`)
- **Unbeatable**: `UnbeatableAdapter` (in `ai_common/adapters/`)

Model paths can be configured via `AISAM_MODELS_DIR` environment variable.

**RL Agent Integration**:
- Set `AISAM_GENERAL_MODE=rl` to use RL policy
- RL policy uses same feature space as Two-Layer for compatibility
- Can switch between Two-Layer and RL at runtime

---

**Status**: ACTIVE  
**Last Updated**: 2025-01-XX  
**Primary Solution**: Two-Layer Architecture (Style Learner)  
**RL Solution**: PPO with diverse self-play, reward shaping, opponent pool  
**Features**: Framework-aware ranking, Ensemble training, Pass strategy, Combo preservation, RL agent with diverse self-play
