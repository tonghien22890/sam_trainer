# Data Generation Framework

This directory contains a modular framework for generating training data for different card game types.

## Structure

```
data_generation/
├── base/                          # Base classes and common functionality
│   ├── __init__.py
│   ├── base_generator.py          # Base data generator class
│   ├── combo_generator.py         # Common combo generation logic
│   └── game_simulator.py          # Common game simulation logic
├── bao_sam/                       # Báo Sâm specific generators
│   ├── __init__.py
│   └── bao_sam_generator.py       # Báo Sâm data generator
├── sam_general/                   # Sam General specific generators
│   ├── __init__.py
│   └── sam_general_generator.py   # Sam General data generator
├── tlmn/                          # TLMN specific generators
│   ├── __init__.py
│   └── tlmn_generator.py          # TLMN data generator
├── generate_bao_sam_data.py       # Entry point for Báo Sâm data generation
├── generate_sam_general_data.py   # Entry point for Sam General data generation
├── generate_tlmn_data.py          # Entry point for TLMN data generation
└── README.md                      # This file
```

## Usage

### Generate Báo Sâm Data

```bash
# Generate all phases of Báo Sâm training data
python generate_bao_sam_data.py --all-phases

# Generate specific phases
python generate_bao_sam_data.py --validation-samples 1000 --pattern-samples 2000 --threshold-samples 1500

# Custom output directory
python generate_bao_sam_data.py --all-phases --output-dir my_data
```

### Generate Sam General Data

```bash
# Generate Sam General training data
python generate_sam_general_data.py --sessions 100

# Custom parameters
python generate_sam_general_data.py --sessions 200 --players 4 --output-file data/my_sam_data.jsonl
```

### Generate TLMN Data

```bash
# Generate TLMN training data
python generate_tlmn_data.py --sessions 100

# Generate with pattern analysis
python generate_tlmn_data.py --sessions 100 --generate-patterns --pattern-samples 1500
```

## Features

### Base Classes

- **BaseDataGenerator**: Common functionality for all generators
- **ComboGenerator**: Handles combo detection and generation for different game types
- **GameSimulator**: Simulates game sessions with realistic player behavior

### Game-Specific Generators

#### BaoSamGenerator
- Generates validation, pattern, and threshold training data
- Simulates user behavior with different profiles (conservative, balanced, aggressive)
- Creates realistic Báo Sâm scenarios

#### SamGeneralGenerator
- Generates turn-based gameplay data
- Includes sequence context and framework information
- Supports variable hand sizes and realistic move selection

#### TLMNGenerator
- Generates TLMN-specific gameplay data
- Includes pattern analysis capabilities
- Handles TLMN-specific rules and strategies

## Data Format

All generators output JSONL format with the following common structure:

```json
{
  "game_id": "game_type_game_1234",
  "game_type": "GameType",
  "players_count": 4,
  "turn_id": 5,
  "player_id": 0,
  "hand": [0, 1, 2, ...],
  "last_move": {...},
  "legal_moves": [...],
  "action": {
    "stage1": {"type": "pass", "value": "pass"},
    "stage2": {...}
  },
  "timestamp": "2024-01-01T00:00:00",
  "synthetic": true,
  "game_progress": 0.25
}
```

## Extending the Framework

To add a new game type:

1. Create a new directory under `data_generation/`
2. Implement a generator class extending `BaseDataGenerator`
3. Implement a game simulator extending `GameSimulator`
4. Create an entry point script
5. Update this README

## Dependencies

- Python 3.7+
- numpy
- ai_common (project internal module)

## Notes

- All generators use the same base classes to ensure consistency
- Game-specific logic is isolated in respective generator classes
- The framework supports both simple and complex data generation scenarios
- Output format is standardized across all game types
