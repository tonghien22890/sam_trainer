# RL Agent - Reinforcement Learning for Card Games

Reinforcement Learning (RL) implementation for Vietnamese card games (Sam & TLMN) using PyTorch. This module provides a self-play training pipeline that learns optimal strategies through trial and error.

## Overview

The RL Agent replaces or augments the Two-Layer Architecture's `StyleLearner` (Layer 2) with a policy network trained via REINFORCE algorithm. It uses the same feature space (42 dimensions) as `StyleLearner` to ensure compatibility and leverage existing framework generation.

### Key Features

- **Self-play training**: Agent learns by playing against scripted opponents
- **Framework-aware features**: Uses `FrameworkGenerator` to extract strategic features
- **Reward shaping**: Sophisticated reward function based on game outcomes and card penalties
- **Compatible with existing system**: Integrates seamlessly with `ModelBot` via `RLAdapter`

## Architecture

```
┌─────────────────┐
│  Game Engine    │ (SamGame/TLMNGame)
│  (Game State)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CardGameEnv    │ (RL Environment)
│  - Observation  │
│  - Reward       │
│  - Step/Reset   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RLTrainer      │ (REINFORCE Algorithm)
│  - Collect      │
│  - Update       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PolicyNetwork  │ (PyTorch Neural Network)
│  - Predict      │
│  - Sample       │
└─────────────────┘
```

## File Structure

```
rl_agent/
├── __init__.py           # Package exports
├── env.py                 # CardGameEnv - RL environment wrapper
├── features.py            # FrameworkAwareFeatureBuilder - feature extraction
├── policy.py              # PolicyNetwork & RLLearner - neural network policy
├── trainer.py             # RLTrainer - REINFORCE training loop
├── train_rl_core.py       # Main training script with CLI arguments
├── train_rl_sam.py         # Convenience wrapper for Sam training
├── train_rl_tlmn.py       # Convenience wrapper for TLMN training
└── README.md              # This file
```

## Quick Start

### Training

**Option 1: Use convenience scripts (recommended)**

```cmd
REM Train for Sam (default: 5000 episodes, 2 seats)
python model_build\scripts\rl_agent\train_rl_sam.py

REM Train for TLMN
python model_build\scripts\rl_agent\train_rl_tlmn.py
```

**Option 2: Use core script with custom parameters**

```cmd
python model_build\scripts\rl_agent\train_rl_core.py --game_type sam --episodes 10000 --seats 2 --lr 0.001 --save_path model_build\models\rl_policy_sam.pt
```

#### Checkpoint & mặc định đường dẫn model

- **Đường dẫn model RL mặc định** dùng trong toàn hệ thống cho Sâm:  
  `model_build/models/rl_policy_sam.pt`
- Train **từ đầu (không nối tiếp)**: chỉ truyền `--save_path`, **không truyền `--load_path`**.  
  Ví dụ: model mới hoàn toàn, ghi đè file cũ:
  ```cmd
  python model_build\scripts\rl_agent\train_rl_core.py --game_type sam --episodes 20000 --save_path model_build\models\rl_policy_sam.pt
  ```
- Train **nối tiếp trên model cũ** (làm “thông minh hơn” nếu training ổn định): truyền cả `--load_path` và `--save_path` cùng trỏ tới file model hiện tại:
  ```cmd
  python model_build\scripts\rl_agent\train_rl_core.py --game_type sam --episodes 20000 --load_path model_build\models\rl_policy_sam.pt --save_path model_build\models\rl_policy_sam.pt
  ```
  Khi đó, trainer sẽ load checkpoint ở `load_path`, tiếp tục train thêm `episodes` game, rồi lưu đè lại vào cùng đường dẫn.

### Inference

The trained model is automatically loaded by `ModelBot` when `AISAM_GENERAL_MODE=rl` (default). The `RLAdapter` handles model loading and prediction.

```python
from ai_bots.adapters.rl_adapter import RLAdapter

adapter = RLAdapter("model_build/models/rl_policy_sam.pt", game_type="sam")
move = adapter.predict(game_record, legal_moves)
```

## Configuration Parameters

### Training Parameters (`TrainerConfig`)

| Parameter | Default | Description |
|-----------|---------|--------------|
| `game_type` | `"sam"` | Game type: `"sam"` or `"tlmn"` |
| `seats` | `2` | Number of players (2 = 1v1, 4 = 4 players) |
| `episodes` | `5000` | Number of training episodes (games) |
| `gamma` | `0.99` | Discount factor for future rewards |
| `lr` | `1e-3` | Learning rate for Adam optimizer |
| `hidden_dim` | `128` | Hidden layer size in neural network |
| `max_steps` | `400` | Maximum steps per game (timeout) |
| `log_interval` | `25` | Log progress every N episodes |

### Reward Configuration

The reward function is designed to encourage winning and penalize poor performance:

**Step Reward** (per move):
- `-0.002`: Small penalty to encourage faster games

**Final Reward** (game end):

1. **Agent Wins** (0 cards left):
   - Reward = Sum of remaining cards of all opponents
   - Example: Opponents have [5, 3, 2] cards → Reward = 10

2. **Agent Loses** (>0 cards left):
   - **Normal penalty**: `-1 × remaining_cards`
   - **Special penalties**:
     - Still has all initial cards (10 for Sam, 13 for TLMN): `-15` points
     - Has card 2 (rank 12) or four_kind: `-3 × remaining_cards` (instead of -1)

**Examples:**
- Win with opponents having 10 cards total → Reward = 10
- Lose with 5 cards (normal) → Penalty = -5
- Lose with 5 cards (has card 2) → Penalty = -15
- Lose with all 10 cards → Penalty = -15

## Feature Space

The RL agent uses the same 42-dimensional feature space as `StyleLearner`:

- **Context features (22 dims)**: Legal moves counts, cards left, hand count, combo type onehot, rank value, combo length, hand efficiency, move urgency
- **Framework features (8 dims)**: Framework priority, breaking severity, strength, position, preferences, compliance
- **Multi-sequence features (12 dims)**: Top 3 sequences × 4 features each

This ensures compatibility and allows the RL policy to leverage the same strategic insights as the supervised learning model.

## Training Process

1. **Environment Setup**: `CardGameEnv` wraps the game engine and manages game state
2. **Feature Extraction**: `FrameworkAwareFeatureBuilder` extracts 42-dim features for each legal move
3. **Policy Prediction**: `PolicyNetwork` (MLP) scores each move and selects the best one
4. **Reward Calculation**: Rewards are computed based on game outcome and card penalties
5. **Policy Update**: REINFORCE algorithm updates the policy using discounted returns

### REINFORCE Algorithm

The training uses vanilla REINFORCE (policy gradient):

```python
# Collect episode trajectory
log_probs = []
rewards = []
for step in episode:
    action, log_prob = policy.sample_action(obs)
    reward = env.step(action)
    log_probs.append(log_prob)
    rewards.append(reward)

# Compute discounted returns
returns = discount_rewards(rewards, gamma=0.99)

# Policy gradient update
loss = -sum(log_prob * return for log_prob, return in zip(log_probs, returns))
loss.backward()
optimizer.step()
```

## Model Output

Trained models are saved as PyTorch checkpoints (`.pt` files) containing:
- `model_state_dict`: Policy network weights
- `metadata`: Training configuration and metrics
- `feature_dim`: Feature dimension (42)

**Example checkpoint structure:**
```python
{
    "model_state_dict": {...},
    "metadata": {
        "game_type": "sam",
        "hidden_dim": 128,
        "episodes": 5000,
        "metrics": {"episodes": 5000.0, "avg_reward": -1.876}
    },
    "feature_dim": 42
}
```

## Integration with ModelBot

The RL agent integrates with the existing `ModelBot` system:

1. **Model Loading**: `ModelBot` preloads `RLAdapter` if `AISAM_GENERAL_MODE=rl`
2. **Prediction**: `RLAdapter.predict()` uses the same interface as `TwoLayerAdapter`
3. **Fallback**: If RL model fails, automatically falls back to Two-Layer Architecture

**Environment Variable:**
```cmd
REM Use RL (default)
set AISAM_GENERAL_MODE=rl

REM Use Two-Layer
set AISAM_GENERAL_MODE=two_layer
```

## Performance Tips

1. **Training Time**: 
   - 5000 episodes: ~10-30 minutes (2 seats)
   - 10000 episodes: ~20-60 minutes (2 seats)
   - 4 seats takes ~2x longer

2. **Convergence**:
   - Monitor `avg_reward` in logs
   - Positive rewards indicate winning more often
   - Negative rewards are normal (step penalties + losses)

3. **Hyperparameter Tuning**:
   - Increase `episodes` for better performance
   - Decrease `lr` if training is unstable
   - Adjust `gamma` for long-term vs short-term strategy focus

## Troubleshooting

**Issue**: Model not loading
- **Solution**: Check model path in `RL_POLICY_SAM_MODEL_PATH` or `RL_POLICY_TLMN_MODEL_PATH`

**Issue**: Training too slow
- **Solution**: Reduce `seats` to 2, reduce `max_steps`, or use GPU

**Issue**: Poor performance
- **Solution**: Train for more episodes (10000+), check reward function, verify feature extraction

**Issue**: Import errors
- **Solution**: Ensure project root is in `PYTHONPATH`, check dependencies in `model_build/requirements.txt`

## Dependencies

- `torch >= 2.1.0`: PyTorch for neural network
- `numpy`: Numerical operations
- Game engine modules: `game_engine`, `ai_common`, `model_build.scripts.two_layer`

## References

- **REINFORCE Algorithm**: Policy gradient method for RL
- **Feature Space**: Shared with `StyleLearner` (42 dimensions)
- **Framework Generator**: Reuses `FrameworkGenerator` from Two-Layer Architecture

## License

Part of the AI-Sam project. See main project LICENSE.

