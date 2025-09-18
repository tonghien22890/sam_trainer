# 🎯 Model Build Documentation

This module contains documentation for **two specialized ML models** for Vietnamese card games:

1. **Hybrid Conservative Báo Sâm Model** - Báo Sâm declarations with high precision
2. **Optimized General Gameplay Model V3** - Per-candidate general gameplay decisions

## 📁 Documentation Files

### **Báo Sâm Model:**
- `HYBRID_CONSERVATIVE_MODEL_DESIGN.md` - Technical design and implementation details
- `SOLUTION_SUMMARY.md` - Complete solution overview and usage guide

### **General Gameplay Model:**
- `OPTIMIZED_GENERAL_MODEL_SOLUTION.md` - Two-stage pipeline documentation (legacy)
- `stage1.mdc` - Per-candidate Stage 1 specification and implementation guide

## 🎯 Current Status

### **Báo Sâm Model** ✅ ACTIVE
- **Algorithm**: Decision Tree Classifier với conservative configuration
- **Approach**: Hybrid ML + Rule-based system
- **Performance**: 98.7% precision, 100% accuracy trên test scenarios
- **Compliance**: Tuân thủ đúng luật Sam (5 combo types hợp lệ)

### **General Gameplay Model** ✅ ACTIVE
- **Algorithm**: Per-candidate XGBoost classifier (rank-based)
- **Approach**: Rank all legal moves for the current turn and pick the top-scoring move
- **Performance**: 67.9% turn@1, 80.2% turn@3 on real user data
- **Features**: 22-dim per-candidate features (includes combo type, rank value, breaks_combo_flag)

## 📚 Documentation Guide

### **For Báo Sâm Model:**
1. Read `SOLUTION_SUMMARY.md` for complete overview
2. Read `HYBRID_CONSERVATIVE_MODEL_DESIGN.md` for technical details
3. See main `model_build/README.md` for usage instructions

### **For General Gameplay Model:**
1. Read `stage1.mdc` for current per-candidate implementation
2. Read `OPTIMIZED_GENERAL_MODEL_SOLUTION.md` for legacy two-stage approach
3. See main `model_build/README.md` for current usage

## 🔄 Documentation Status

### **Up-to-date:**
- ✅ `HYBRID_CONSERVATIVE_MODEL_DESIGN.md` - Current Báo Sâm model
- ✅ `SOLUTION_SUMMARY.md` - Current Báo Sâm model
- ✅ `stage1.mdc` - Current per-candidate general gameplay

### **Legacy (for reference):**
- ⚠️ `OPTIMIZED_GENERAL_MODEL_SOLUTION.md` - Two-stage pipeline (replaced by per-candidate)

## 🎯 Key Features

### **Báo Sâm Model:**
- Conservative approach với 98.7% precision
- Rule-based validation chặn risky cases
- Sam rules compliance (5 combo types)

### **General Gameplay Model:**
- Per-candidate ranking với 22-dim features
- Rank-based labels (combo_type + rank_value)
- Combo breaking awareness (breaks_combo_flag)

## 📋 Data Formats

### **Báo Sâm Training Data:**
```json
{
  "game_id": "sam_game_123",
  "player_id": 0,
  "hand": [0, 1, 2, ...],
  "sammove_sequence": [...],
  "result": "success"
}
```

### **General Gameplay Training Data:**
```json
{
  "hand": [...],
  "last_move": {...},
  "meta": {"legal_moves": [...]},
  "action": {"stage2": {"combo_type": "single", "rank_value": 0}}
}
```

## 📚 Documentation Files

- `SOLUTION_SUMMARY.md`: Báo Sâm complete solution overview
- `HYBRID_CONSERVATIVE_MODEL_DESIGN.md`: Báo Sâm technical design details
- `stage1.mdc`: General gameplay per-candidate specification
- `OPTIMIZED_GENERAL_MODEL_SOLUTION.md`: Legacy two-stage approach (reference)
- `README.md`: This documentation index

## 🔧 Model Configurations

### **Báo Sâm Model:**
```python
DecisionTreeClassifier(
    max_depth=8,             # Conservative depth
    min_samples_split=20,    # Large split threshold
    min_samples_leaf=10,     # Large leaf threshold
    criterion='entropy',     
    class_weight={0:1, 1:5}, # Penalize false positives
    random_state=42
)
```

### **General Gameplay Model:**
```python
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

## 🔧 Model Configuration

```python
DecisionTreeClassifier(
    max_depth=8,             # Conservative depth
    min_samples_split=20,    # Large split threshold
    min_samples_leaf=10,     # Large leaf threshold
    criterion='entropy',     
    class_weight={0:1, 1:5}, # Penalize false positives
    random_state=42
)
```

---

*Both models are integrated in production via `GeneralPlayProvider` (general) and `ProductionBaoSamProvider` (Báo Sâm).*

**Last Updated**: 2025-09-17  
**Status**: ✅ ACTIVE - Per-candidate general gameplay + Hybrid Báo Sâm  
**Models**: Báo Sâm (Hybrid Conservative) + General Gameplay (Optimized V3 Per-Candidate)



