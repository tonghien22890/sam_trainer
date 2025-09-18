# Unbeatable Sequence Model

Complete implementation of the Unbeatable Sequence Model for Vietnamese Sam (Báo Sâm) decision making.

## 🎯 Overview

This model learns to build unbeatable combo sequences from user behavior patterns and makes intelligent Báo Sâm declarations. The system combines rule-based validation with machine learning to create a robust, user-adaptive AI player.

## 🏗️ Architecture

### Core Components

1. **UnbeatableRuleEngine**: Validates hands against Sam-specific rules
2. **SequenceValidationModel**: ML model that learns valid/invalid sequence patterns
3. **PatternLearningModel**: Learns user combo building preferences
4. **ThresholdLearningModel**: Learns user decision thresholds for declarations
5. **UnbeatableSequenceGenerator**: Orchestrates all components for final decisions

### Decision Logic

```
Input: 10-card hand + player_count
↓
Rulebase Validation (reject weak hands)
↓
ML Validation (sequence viability)
↓
Pattern Learning (user preferences)
↓
Threshold Learning (user risk tolerance)
↓
Decision: should_declare_bao_sam = (unbeatable_prob >= user_threshold)
```

## 📊 Features

### Input Features (34 dimensions)
- **Combo-level**: Type, rank, strength, card count (per combo)
- **Sequence-level**: Strength distribution, combo diversity, balance
- **Context**: Player count (minimal context)

### Output
```json
{
  "should_declare_bao_sam": true,
  "unbeatable_probability": 0.87,
  "user_threshold": 0.75,
  "model_confidence": 0.92,
  "reason": "unbeatable_prob_0.87_vs_threshold_0.75",
  "unbeatable_sequence": [...],
  "sequence_stats": {...}
}
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models logs
```

### Basic Usage

```python
from unbeatable_sequence_model import UnbeatableSequenceGenerator

# Initialize generator
generator = UnbeatableSequenceGenerator()

# Analyze a hand
hand = [12, 25, 38, 51, 11, 24, 37, 10, 23, 36]  # Strong hand
result = generator.generate_sequence(hand, player_count=4)

print(f"Should declare: {result['should_declare_bao_sam']}")
print(f"Probability: {result['unbeatable_probability']:.3f}")
```

### Interactive Demo

```bash
python demo_unbeatable_model.py
```

Features:
- Hand scenario testing
- Player count effects
- Interactive hand input
- Model component demonstration
- Full training pipeline

## 🎓 Training

### 3-Phase Training Pipeline

#### Phase 1: Foundation Training
- **Goal**: Learn basic valid/invalid patterns
- **Data**: 1,000 synthetic samples
- **Model**: Random Forest Classifier
- **Success Metric**: 85% accuracy

#### Phase 2: Pattern Learning
- **Goal**: Learn user combo building patterns
- **Data**: 2,000 user behavior samples
- **Model**: Gradient Boosting Regressor
- **Success Metric**: Pattern consistency > 0.8

#### Phase 3: Threshold Optimization
- **Goal**: Learn user decision thresholds
- **Data**: 1,500 decision samples
- **Model**: Random Forest Regressor
- **Success Metric**: Decision accuracy > 0.75

### Run Training

```bash
# Full training pipeline
python train_unbeatable_model.py

# Generate synthetic data only
python synthetic_data_generator.py
```

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all tests
python test_unbeatable_model.py

# Individual component tests
python -m unittest test_unbeatable_model.TestUnbeatableRuleEngine
python -m unittest test_unbeatable_model.TestSequenceValidationModel
```

### Test Coverage
- Rule engine validation
- ML model training and prediction
- Pattern learning
- Threshold learning
- End-to-end integration
- Realistic game scenarios

## 📈 Performance Metrics

### Primary Metrics
- **Unbeatable Rate**: % of sequences actually unbeatable
- **Game Win Rate**: % of games won using generated sequences
- **Sequence Completion Rate**: % of sequences played to completion
- **Expert Approval Rate**: % approved by expert players

### Technical Metrics
- **Inference Time**: < 100ms per decision
- **Memory Usage**: < 512MB
- **Model Size**: < 50MB total

## 🎮 Game Integration

### API Interface

```python
# Simple integration
def should_declare_bao_sam(player_hand, game_context):
    generator = UnbeatableSequenceGenerator()
    result = generator.generate_sequence(
        hand=player_hand, 
        player_count=game_context['player_count']
    )
    return result['should_declare_bao_sam']
```

### Real-time Usage

```python
# Load pre-trained models
generator = UnbeatableSequenceGenerator()
generator.load_models('models/')

# Fast inference
result = generator.generate_sequence(hand, player_count)
# Returns decision in < 100ms
```

## 🔧 Configuration

### Rule Engine Settings

```python
rules = {
    'min_total_cards': 10,           # Minimum cards for valid sequence
    'max_weak_combos': 1,            # Maximum weak combos allowed
    'min_strong_combos': 1,          # Minimum strong combos required
    'min_avg_strength': 0.6,         # Minimum average strength
    'min_unbeatable_combos': 1,      # Minimum unbeatable combos
}
```

### Strength Calculation

Ultra-clear tier separation for Sam-specific combos:

- **Singles**: 2=1.0, A=0.3, others=0.1
- **Pairs**: 2=1.0, A=0.8, others=0.2-0.3
- **Triples**: 2=1.0, A=0.9, faces=0.8, ≥7=0.5, others=0.3-0.35
- **Quads**: 2=1.0, A=0.98, others=0.95+
- **Straights**: 10-card=1.0, Ace-high=1.0, 7+=0.85+, others scaled

## 📁 File Structure

```
model_build/
├── unbeatable_sequence_model.py    # Main model implementation
├── synthetic_data_generator.py     # Training data generation
├── train_unbeatable_model.py       # 3-phase training pipeline
├── test_unbeatable_model.py        # Comprehensive test suite
├── demo_unbeatable_model.py        # Interactive demo
├── README_UNBEATABLE_MODEL.md      # This file
├── requirements.txt                # Dependencies
├── data/                           # Training data
├── models/                         # Saved models
└── logs/                          # Training logs
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not trained**: Run `python train_unbeatable_model.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Data not found**: Run `python synthetic_data_generator.py`
4. **Low accuracy**: Increase training data size or adjust hyperparameters

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enables detailed logging for all components
```

## 🔄 Model Updates

### Retraining

```bash
# Full retraining
python train_unbeatable_model.py

# Individual phases
python -c "from train_unbeatable_model import UnbeatableModelTrainer; trainer = UnbeatableModelTrainer(); trainer.phase1_foundation_training()"
```

### Model Versioning

Models are automatically timestamped and can be versioned:

```python
# Save with version
generator.save_models('models/v1.0/')

# Load specific version
generator.load_models('models/v1.0/')
```

## 📊 Monitoring

### Training Logs

All training runs generate comprehensive logs:
- Phase-by-phase results
- Model performance metrics
- End-to-end evaluation
- Timestamp and duration tracking

### Production Monitoring

```python
# Log all decisions for analysis
result = generator.generate_sequence(hand, player_count)
logger.info(f"Decision: {result['should_declare_bao_sam']}, "
           f"Prob: {result['unbeatable_probability']:.3f}")
```

## 🎯 Future Enhancements

### Planned Features
1. **Online learning**: Update models from real gameplay
2. **Opponent modeling**: Learn opponent patterns
3. **Advanced strategies**: Context-aware decision making
4. **Performance optimization**: Model quantization and acceleration

### Research Directions
1. **Deep learning**: Neural networks for complex pattern recognition
2. **Reinforcement learning**: Self-play optimization
3. **Multi-objective**: Balance multiple success criteria
4. **Explainable AI**: Better decision reasoning

## 📞 Support

For questions, issues, or contributions:

1. Check the test suite for examples
2. Review training logs for debugging
3. Run the interactive demo for hands-on learning
4. Examine the comprehensive documentation

## 📄 License

This implementation is part of the AI-Sam project and follows the same licensing terms.

---

**Built with ❤️ for Vietnamese Sam players**
