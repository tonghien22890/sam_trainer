# Optimized General Model - Per-Candidate Solution

> **✅ CURRENT IMPLEMENTATION**  
> This document describes the **per-candidate approach** implemented in `optimized_general_model_v3.py`.  
> For detailed pipeline specifications, see `stage1.mdc`.

## **📋 OVERVIEW**

Optimized General Model sử dụng **per-candidate ranking approach** để đánh giá từng legal move và chọn move tốt nhất, thay vì two-stage pipeline cũ.

## **🎯 ARCHITECTURE**

### **Per-Candidate Pipeline:**

#### **Single Stage: Move Ranking**
- **Input**: Game state + legal moves
- **Output**: Best move from legal_moves
- **Features**: 25 dims per candidate
- **Approach**: Rank all legal moves, pick highest score

### **Key Advantages:**
- **Direct move evaluation**: Không cần chọn combo type trước
- **Combo breaking awareness**: `breaks_combo_flag` phạt xé bộ mạnh
- **Better generalization**: Học pattern từ từng candidate move

## **🔧 FEATURES DESIGN**

### **Per-Candidate Features (25 dims):**

#### **General Features (11 dims):**
1. **legal_moves_combo_counts** (6 dims):
   ```python
   [single_count, pair_count, triple_count, four_kind_count, straight_count, double_seq_count]
   ```

2. **cards_left** (4 dims):
   ```python
   [player0_cards, player1_cards, player2_cards, player3_cards]
   ```

3. **hand_count** (1 dim):
   ```python
   len(hand)  # Raw count
   ```

4. **REMOVED: combo_strength_relative** - Replaced by individual_move_strength

#### **Combo-Specific Features (14 dims):**
1. **combo_type_onehot** (7 dims):
   ```python
   [single, pair, triple, four_kind, straight, double_seq, pass]
   ```

2. **hybrid_rank_feature** (1 dim):
   ```python
   # HYBRID APPROACH - Auto-selects based on training data size:
   # 
   # Small datasets (<1000 samples): rank_category
   # - single: {2=2, A=1, rest=0}
   # - pair: {2=3, A=2, face=1, rest=0}
   # - triple: {2=3, A=2, >=7=1, rest=0}
   # - four_kind: {2=2, A=1, rest=0}
   # - straight/double_seq: length (0-12)
   #
   # Large datasets (≥1000 samples): rank_value (normalized 0-1)
   # - Uses actual rank_value / 12.0 for fine-grained learning
   # - Allows model to learn subtle rank differences
   ```

3. **combo_length** (1 dim):
   ```python
   len(cards)  # For straight/double_seq
   ```

4. **breaks_combo_flag** (1 dim):
   ```python
   # Severity: 0=no break, 1=normal break, 2=heavy break
   # 2: xé quad hoặc làm mất double_seq
   # 1: xé triple hoặc làm giảm độ dài straight (trước ≥ 5)
   # 0: không xé
   ```

5. **individual_move_strength** (1 dim) - NEW:
   ```python
   # Strength of this specific move (normalized 0-1)
   # Replaces combo_strength_relative for per-candidate evaluation
   # Single(2)=0.32, Pair(2)=0.48, Triple(2)=0.60, Four_kind(2)=1.00
   ```

6. **combo_type_strength_multiplier** (1 dim) - NEW:
   ```python
   # Relative strength between combo types
   # single=1.0, pair=2.0, triple=3.0, four_kind=4.0, straight=2.5, double_seq=3.5
   ```

7. **enhanced_breaks_penalty** (1 dim) - NEW:
   ```python
   # Stronger penalty for breaking combos (0.0/0.3/0.7)
   # Enhanced version of breaks_combo_flag with stronger penalties
   ```

8. **combo_efficiency_score** (1 dim) - NEW:
   ```python
   # Encourages playing stronger combos (0.0-1.0)
   # single=0.2, pair=0.4, triple=0.6, four_kind=0.8, straight=0.5, double_seq=0.7
   ```

## **🎯 PER-CANDIDATE MODEL**

### **Current Implementation:**
Model đánh giá từng move candidate trực tiếp và chọn move tốt nhất:

#### **Training:**
```python
# Per-candidate binary classification
X, y, groups = model.build_stage1_candidate_dataset(records)
model.train_stage1_candidates(records, model_type="xgb")
```

#### **Evaluation:**
- **Turn-level accuracy**: Top-1 accuracy per turn
- **Top-k accuracy**: Top-3 accuracy per turn  
- **Sample-level accuracy**: Binary classification accuracy

#### **Model Options:**
- **XGBoost** (recommended): Best performance với regularization
- **RandomForest**: Good baseline
- **DecisionTree**: Simple but prone to overfitting

## **💪 COMBO STRENGTH CALCULATION**

### **Combo Strength Relative (for general features):**
```python
def calculate_combo_strength_relative(legal_moves):
    """
    Tính sức mạnh tương đối của các combos
    Mỗi combo type có cách tính rank khác nhau
    """
    combo_strengths = []
    
    for move in legal_moves:
        if move.get("type") == "play_cards":
            combo_type = move.get("combo_type")
            rank_value = move.get("rank_value", 0)
            cards = move.get("cards", [])
            
            # Calculate strength based on combo type
            if combo_type == "single":
                # Single: 2, A, Phần còn lại (đánh từ bé đến lớn)
                if rank_value == 1:  # 2
                    strength = 3.0
                elif rank_value == 0:  # A
                    strength = 2.0
                else:  # Phần còn lại
                    strength = 1.0 + (rank_value - 2) / 10.0  # 3-K: 1.0-1.9
                    
            elif combo_type == "pair":
                # Pair: 2, A, Mặt người (J,Q,K), Phần còn lại
                if rank_value == 1:  # 2
                    strength = 4.0
                elif rank_value == 0:  # A
                    strength = 3.0
                elif rank_value >= 10:  # J, Q, K (mặt người)
                    strength = 2.5
                else:  # Phần còn lại
                    strength = 2.0 + (rank_value - 2) / 8.0  # 3-10: 2.0-2.875
                    
            elif combo_type == "triple":
                # Triple: 2, A, >= 7, Phần còn lại
                if rank_value == 1:  # 2
                    strength = 5.0
                elif rank_value == 0:  # A
                    strength = 4.0
                elif rank_value >= 6:  # >= 7 (7,8,9,10,J,Q,K)
                    strength = 3.5
                else:  # Phần còn lại (3,4,5,6)
                    strength = 3.0 + (rank_value - 2) / 4.0  # 3-6: 3.0-3.75
                    
            elif combo_type == "four_kind":
                # Four_kind: A và phần còn lại (2 thì thắng luôn)
                if rank_value == 1:  # 2 - thắng luôn
                    strength = 10.0  # Cực mạnh
                elif rank_value == 0:  # A
                    strength = 9.0
                else:  # Phần còn lại
                    strength = 8.0 + (rank_value - 2) / 11.0  # 3-K: 8.0-8.82
                    
            elif combo_type == "straight":
                # Straight (rank-only): Dây chạm A thì tối đa sức mạnh
                ranks = [c % 13 for c in cards]
                has_ace = any(r == 0 for r in ranks)
                length = len(ranks)
                
                if has_ace:
                    strength = 7.0 + length / 10.0  # A straight: 7.5-8.0
                else:
                    strength = 6.0 + length / 10.0 + (rank_value / 13.0) * 0.5  # Other: 6.5-7.0
                    
            elif combo_type == "double_seq":
                # Double_seq (rank-only): Cực mạnh, vượt trội
                ranks = [c % 13 for c in cards]
                length = len(ranks)
                strength = 9.0 + length / 10.0  # 9.5-10.0
                
            else:
                strength = 0.0
                
            combo_strengths.append(strength)
    
    # Return average strength (0-1 normalized)
    max_possible_strength = 10.0  # 2 four_kind
    normalized_strengths = [s / max_possible_strength for s in combo_strengths]
    return sum(normalized_strengths) / len(normalized_strengths) if normalized_strengths else 0.0
```

### **Breaks Combo Flag Calculation:**
```python
def _breaks_combo_flag(self, hand: List[int], move_cards: List[int]) -> int:
    """
    Return severity if this move breaks stronger structures (0/1/2).
    Heuristics:
      - Quad split → severity 2 (heavy)
      - Double_seq lost → severity 2 (heavy)  
      - Triple split → severity 1 (normal)
      - Straight length reduced (when before >=5) → severity 1 (normal)
    """
    # Implementation details in optimized_general_model_v3.py
    # Returns 0, 1, or 2 based on combo breaking severity
```

## **📊 DATA FORMAT**

### **Input Data Format:**
```json
{
  "hand": [20, 14, 4, 6, 34, 32, 40, 51],
  "cards_left": [9, 8, 3, 3],
  "last_move": {
    "type": "play_cards",
    "cards": [9],
    "combo_type": "triple",
    "rank_value": 9
  },
  "action": {
    "stage2": {
      "type": "play_cards",
      "cards": [20],
      "combo_type": "single",
      "rank_value": 7
    }
  },
  "meta": {
    "legal_moves": [
      {
        "type": "play_cards",
        "cards": [20],
        "combo_type": "single",
        "rank_value": 7
      },
      {
        "type": "play_cards",
        "cards": [14, 40],
        "combo_type": "pair",
        "rank_value": 1
      },
      {
        "type": "pass",
        "cards": [],
        "combo_type": "pass",
        "rank_value": -1
      }
    ]
  }
}
```

### **Output Data Format:**
```json
{
  "type": "play_cards",
  "cards": [20],
  "combo_type": "single",
  "rank_value": 7
}
```

## **🏗️ IMPLEMENTATION FILES**

### **Core Model:**
- `scripts/optimized_general_model_v3.py` - Main per-candidate model implementation
- `scripts/train_optimized_model_v3.py` - Training script
- `docs/stage1.mdc` - Detailed pipeline specifications

### **Data Generation:**
- `scripts/generate_improved_training_data.py` - Generate improved training data
- `data/sam_improved_training_data.jsonl` - Generated training data (1200 records)

### **Model Files:**
- `models/optimized_general_model_v3.pkl` - Trained per-candidate model

## **📈 PERFORMANCE RESULTS**

### **Per-Candidate Model Performance:**
- **Sample-level Accuracy**: ~92% (binary classification)
- **Turn-level Top-1 Accuracy**: ~67% (per-turn move selection)
- **Turn-level Top-3 Accuracy**: ~97% (top-3 moves include correct choice)

### **Model Comparison:**
- **XGBoost**: Best performance với regularization
- **RandomForest**: Good baseline performance
- **DecisionTree**: Prone to overfitting

## **🔧 MODEL PARAMETERS**

### **Per-Candidate Model (Recommended - XGBoost):**
```python
xgb.XGBClassifier(
    max_depth=6,                # Moderate depth
    learning_rate=0.1,          # Standard learning rate
    n_estimators=200,           # Number of trees
    subsample=0.8,              # Subsample ratio
    colsample_bytree=0.8,       # Feature sampling ratio
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    random_state=42,
    eval_metric='logloss'
)
```

### **Alternative Models:**
```python
# RandomForest
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# DecisionTree (prone to overfitting)
DecisionTreeClassifier(
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='entropy',
    random_state=42
)
```

## **🚀 USAGE**

### **Training Per-Candidate Model:**
```python
from scripts.optimized_general_model_v3 import OptimizedGeneralModelV3
import json

# Load training data
with open('data/sam_improved_training_data.jsonl', 'r') as f:
    records = [json.loads(line) for line in f if line.strip()]

# Train per-candidate model
model = OptimizedGeneralModelV3()
sample_acc = model.train_stage1_candidates(records, model_type="xgb")
print(f"Sample accuracy: {sample_acc}")

# Evaluate
eval_res = model.evaluate_stage1_candidates(records)
print(f"Turn accuracy: {eval_res['turn_accuracy']}")

# Save model
model.save('models/optimized_general_model_v3.pkl')
```

### **Generate Training Data:**
```bash
python scripts/generate_improved_training_data.py
```

## **✅ KEY ACHIEVEMENTS**

1. **Per-Candidate Approach**: Thay thế two-stage pipeline bằng single-stage ranking
2. **Feature Engineering**: 25-dims features (11 general + 14 combo-specific)
3. **Individual Move Strength**: Thay thế average strength bằng per-move strength
4. **Combo Type Multiplier**: Khuyến khích combo mạnh hơn (single=1.0, four_kind=4.0)
5. **Enhanced Breaks Penalty**: Penalty mạnh hơn cho việc xé combo (0.0/0.3/0.7)
6. **Combo Efficiency Score**: Khuyến khích sử dụng combo hiệu quả
7. **Hybrid Rank Feature**: Auto-selects rank_category vs rank_value based on training data size
8. **Combo Breaking Awareness**: `breaks_combo_flag` phạt xé bộ mạnh (0/1/2 severity)
9. **Rank-Based Comparison**: So sánh moves theo combo_type + rank_value thay vì exact cards
10. **Multiple Model Support**: XGBoost, RandomForest, DecisionTree
11. **Regularization**: XGBoost parameters tránh overfitting
12. **Turn-Level Evaluation**: Top-1 và Top-k accuracy metrics
13. **Direct Move Selection**: Không cần chọn combo type trước
14. **Adaptive Learning**: Tự động điều chỉnh feature granularity theo dữ liệu

## **📊 PERFORMANCE COMPARISON**

### **Per-Candidate Model Results**

| Model Type | Sample Accuracy | Turn Top-1 | Turn Top-3 | Notes |
|------------|----------------|------------|------------|-------|
| **XGBoost** | ~92% | ~67% | ~97% | **Recommended** - Best performance |
| **RandomForest** | ~90% | ~64% | ~95% | Good baseline |
| **DecisionTree** | ~93% | ~59% | ~92% | Prone to overfitting |

### **Key Findings**

#### **1. Per-Candidate vs Two-Stage**
- **Per-candidate**: Direct move evaluation, better generalization
- **Two-stage**: Sequential decision, potential error propagation
- **Winner**: Per-candidate approach

#### **2. Feature Engineering Impact**
- **25-dims features**: Comprehensive move evaluation with enhanced features
- **individual_move_strength**: Replaces average strength with per-move strength
- **combo_type_strength_multiplier**: Encourages stronger combo types
- **enhanced_breaks_penalty**: Stronger penalties for breaking combos
- **combo_efficiency_score**: Promotes efficient combo usage
- **breaks_combo_flag**: Critical for avoiding bad moves
- **rank_category**: Effective combo type encoding

#### **3. Model Selection**
- **XGBoost**: Best balance of performance and regularization
- **RandomForest**: Good baseline with ensemble benefits
- **DecisionTree**: Overfits but fast training

## **📝 NOTES**

- Model được thiết kế để học phong cách người chơi từ training data
- Không implement complex winning strategies
- Focus vào pattern recognition từ logged data
- Sử dụng legal_moves từ game engine để đảm bảo tính chính xác
- **Per-candidate approach**: Thay thế hoàn toàn two-stage pipeline
- **XGBoost regularization**: Giảm overfitting với L1/L2 regularization
- **Turn-level accuracy**: 67% represents real-world performance
- **Enhanced features**: 4 new features to address single bias and combo preservation

---

**Last Updated**: 2025-09-22
**Status**: ✅ CURRENT - Enhanced Per-Candidate Implementation
**Architecture**: Single-stage per-candidate ranking with 25-dims features + enhanced combo awareness
**Best Model**: XGBoost per-candidate model with individual move strength and combo efficiency
