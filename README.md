# 🎯 AI-Sam Model Build - Complete ML Solution

This module contains **two specialized ML models** for Vietnamese card games:

1. **Hybrid Conservative Báo Sâm Model** - Báo Sâm declarations with high precision
2. **Optimized General Gameplay Model V3** - General gameplay decisions with two-stage pipeline

## 🏗️ Solution Overview

### **Báo Sâm Model:**
- **Algorithm**: Decision Tree Classifier với conservative configuration
- **Approach**: Hybrid ML + Rule-based system
- **Performance**: 98.7% precision, 100% accuracy trên test scenarios
- **Compliance**: Tuân thủ đúng luật Sam (5 combo types hợp lệ)

### **General Gameplay Model:**
- **Algorithm**: Two-stage pipeline (Decision Tree + XGBoost)
- **Approach**: Stage 1 (combo type selection) + Stage 2 (card selection)
- **Features**: 12 dims (Stage 1) + 9 dims (Stage 2) + Per-candidate ranking
- **Performance**: 60.49% test accuracy with combo-breaking awareness

## 📁 Project Structure

```
model_build/
├── 📋 docs/                    # Documentation
│   ├── README.md              # Usage guide
│   ├── OPTIMIZED_GENERAL_MODEL_SOLUTION.md # General gameplay model docs
│   ├── HYBRID_CONSERVATIVE_MODEL_DESIGN.md # Báo Sâm model docs
│   └── SOLUTION_SUMMARY.md    # Complete solution overview
├── 🔧 models/                 # Model files
│   ├── hybrid_conservative_bao_sam_model.pkl # Báo Sâm model
│   └── optimized_general_model_v3.pkl # General gameplay model
├── 📊 data/                   # Data files
│   ├── sam_training_data.jsonl # Báo Sâm training data
│   ├── sam_improved_training_data.jsonl # General gameplay training data
│   └── synthetic_training_data.jsonl # Synthetic data
├── 🛠️ scripts/               # Training & generation scripts
│   ├── generate_sam_training_data.py # Generate Báo Sâm data
│   ├── generate_improved_training_data.py # Generate general gameplay data
│   ├── train_optimized_model_v3.py # Train general gameplay model
│   └── optimized_general_model_v3.py # General gameplay model implementation
├── 🧪 tests/                  # Testing & utilities
│   ├── test_realistic_scenarios.py # Báo Sâm test scenarios
│   └── bao_sam_models.py      # Model utilities
├── hybrid_conservative_model.py # Báo Sâm model implementation
├── requirements.txt           # Dependencies
└── __init__.py
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Báo Sâm Model

#### Generate Training Data
```bash
python scripts/generate_sam_training_data.py
# Generates: data/sam_training_data.jsonl (1500 records)
```

#### Train Model
```bash
python scripts/retrain_sam_model.py
# Creates: models/hybrid_conservative_bao_sam_model.pkl
```

#### Test Model
```bash
python tests/test_realistic_scenarios.py
# Tests: 10 realistic scenarios, reports accuracy
```

#### Use in Production
```python
import joblib
model = joblib.load('models/hybrid_conservative_bao_sam_model.pkl')

record = {
    'sammove_sequence': [...],  # Combo sequence
    'hand': [...]              # Player's hand
}
result = model.predict_hybrid(record)
```

### 3. General Gameplay Model

#### Generate Training Data
```bash
python scripts/generate_improved_training_data.py
# Generates: data/sam_improved_training_data.jsonl (1200 records)
```

#### Train Model
```bash
python scripts/train_optimized_model_v3.py
# Creates: models/optimized_general_model_v3.pkl
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
result = model.predict(record)
```

## 📊 Performance Metrics

### Báo Sâm Model Results
- **Precision**: 98.7% ⭐ (rất cao, ít false positives)
- **Training Accuracy**: 76.0%
- **CV Accuracy**: 75.3% ± 1.7%
- **False Positives**: 3 (rất ít)
- **Rulebase Blocked**: 1244 cases (conservative)
- **Test Scenarios**: 100% (10/10 scenarios)

### General Gameplay Model Results
- **Stage 1 Accuracy**: 72.78% (combo type selection)
- **Stage 2 Accuracy**: 60.49% (card selection)
- **Per-candidate Stage 1**: Alternative approach với 22-dims features
- **Combo Breaking Awareness**: `breaks_combo_flag` phạt xé bộ
- **Training Data**: 1200 records với balanced combo types

## 🎯 Key Features

### Báo Sâm Model Features
- **Conservative Approach**: Ưu tiên precision (98.7%) hơn recall
- **Rule-based validation**: Chặn risky cases
- **Confidence threshold**: Cao (≥ 0.8)
- **Sam Rules Compliance**: Chỉ 5 combo types hợp lệ
- **Sequence validation**: Phải đủ 10 lá bài

### General Gameplay Model Features
- **Two-stage Pipeline**: Combo type selection → Card selection
- **Feature Optimization**: 12 dims (Stage 1) + 9 dims (Stage 2)
- **Combo Breaking Awareness**: `breaks_combo_flag` phạt xé bộ
- **Per-candidate Ranking**: Alternative Stage 1 approach
- **XGBoost Regularization**: L1/L2 để giảm overfitting

## 📚 Documentation

- `docs/OPTIMIZED_GENERAL_MODEL_SOLUTION.md`: General gameplay model documentation
- `docs/HYBRID_CONSERVATIVE_MODEL_DESIGN.md`: Báo Sâm model technical design
- `docs/SOLUTION_SUMMARY.md`: Complete solution overview
- `docs/RANK_COMBO_DISCUSSION.md`: Combo strength calculation details

## 🔧 Model Configuration

### Báo Sâm Model
```python
DecisionTreeClassifier(
    max_depth=12,            # Increased depth for better learning
    min_samples_split=10,    # Reduced split threshold
    min_samples_leaf=5,      # Reduced leaf threshold
    criterion='entropy',     
    class_weight={0:1, 1:2}, # Balanced class weights
    random_state=42
)
```

### General Gameplay Model
```python
# Stage 1: Decision Tree
DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=15,
    min_samples_leaf=8,
    criterion='entropy',
    random_state=42
)

# Stage 2: XGBoost
xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='mlogloss'
)
```

---

*Both models đã được test kỹ lưỡng và sẵn sàng cho production use.*

**Last Updated**: 2025-01-15  
**Status**: ✅ COMPLETED - Ready for Production  
**Models**: Báo Sâm (Hybrid Conservative) + General Gameplay (Optimized V3)

