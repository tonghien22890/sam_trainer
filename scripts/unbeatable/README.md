# Unbeatable Sequence Models

This directory contains refactored sequence models for different use cases.

## Structure

```
unbeatable/
├── bao_sam_sequence_model.py      # Báo Sâm specific model
├── general_sequence_model.py      # General gameplay model
├── train_bao_sam_model.py         # Train Báo Sâm model
├── test_general_sequence_model.py # Test general model
├── synthetic_data_generator.py    # Original data generator (legacy)
├── unbeatable_sequence_model.py   # Original unified model (legacy)
└── README.md                      # This file
```

## Models

### BaoSamSequenceModel

**Purpose**: Specialized for Báo Sâm declarations with strict validation

**Features**:
- 3-phase training (Validation, Pattern Learning, Threshold Learning)
- Strict rule-based validation
- ML-based validation for Báo Sâm criteria
- User behavior pattern learning
- Decision threshold learning
- Full Báo Sâm decision making

**Usage**:
```python
from bao_sam_sequence_model import BaoSamSequenceGenerator

generator = BaoSamSequenceGenerator()
result = generator.generate_sequence(hand, player_count=4)
print(f"Should declare Báo Sâm: {result['should_declare_bao_sam']}")
```

**Training**:
```bash
python train_bao_sam_model.py --generate-data
```

### GeneralSequenceModel

**Purpose**: Framework generation for general gameplay

**Features**:
- Simplified sequence generation
- No strict validation requirements
- Framework strength calculation
- Combo recommendations
- Hand validation for general play
- Flexible hand sizes

**Usage**:
```python
from general_sequence_model import GeneralSequenceGenerator

generator = GeneralSequenceGenerator()
result = generator.generate_sequence(hand, player_count=4)
print(f"Framework strength: {result['framework_strength']}")
```

**Testing**:
```bash
python test_general_sequence_model.py --test-cases 10
```

## Key Differences

| Feature | BaoSamSequenceModel | GeneralSequenceModel |
|---------|-------------------|-------------------|
| **Purpose** | Báo Sâm declarations | Framework generation |
| **Validation** | Strict (rules + ML) | Basic validation |
| **Hand Size** | Exactly 10 cards | Any size |
| **Training** | 3-phase complex | No training required |
| **Output** | Báo Sâm decision | Framework combos |
| **Use Case** | Báo Sâm feature | General gameplay |

## Integration

### With Data Generation Framework

The models integrate with the data generation framework:

```python
# In sam_general_generator.py
from scripts.unbeatable.general_sequence_model import GeneralSequenceGenerator

generator = GeneralSequenceGenerator()
result = generator.generate_sequence(hand)
```

### With Two-Layer Architecture

The general model can be used as a fallback for framework generation:

```python
# In framework_generator.py
try:
    from scripts.unbeatable.general_sequence_model import GeneralSequenceGenerator
    generator = GeneralSequenceGenerator()
    result = generator.generate_sequence(hand)
except ImportError:
    # Fallback to simple logic
    pass
```

## Migration from Legacy Model

The original `unbeatable_sequence_model.py` has been split into:

1. **BaoSamSequenceModel**: For Báo Sâm specific functionality
2. **GeneralSequenceModel**: For general gameplay framework generation

### Migration Steps

1. **For Báo Sâm functionality**:
   ```python
   # Old
   from unbeatable_sequence_model import UnbeatableSequenceGenerator
   
   # New
   from bao_sam_sequence_model import BaoSamSequenceGenerator
   ```

2. **For general framework generation**:
   ```python
   # Old
   from unbeatable_sequence_model import UnbeatableSequenceGenerator
   result = generator.generate_sequence(hand, context="general")
   
   # New
   from general_sequence_model import GeneralSequenceGenerator
   result = generator.generate_sequence(hand)
   ```

## Training Data

### BaoSamSequenceModel

Requires 3 types of training data:
- **Validation data**: Valid/invalid hands for Báo Sâm
- **Pattern data**: User combo building patterns
- **Threshold data**: User decision thresholds

Generate with:
```bash
python ../data_generation/generate_bao_sam_data.py --all-phases
```

### GeneralSequenceModel

No training required - uses rule-based logic and ComboAnalyzer.

## Performance

- **BaoSamSequenceModel**: High accuracy for Báo Sâm decisions, requires training
- **GeneralSequenceModel**: Fast inference, no training, good for framework generation

## Future Extensions

- Add more game types (TLMN-specific models)
- Implement ensemble methods
- Add real-time learning capabilities
- Optimize for different hand sizes
