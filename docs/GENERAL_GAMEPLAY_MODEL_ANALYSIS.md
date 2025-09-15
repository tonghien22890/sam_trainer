# General Gameplay Model - Phân Tích và Thiết Kế

## 📋 Tổng Quan

Model General Gameplay được thiết kế để **học theo cách chơi của user** trong game Sam/TLMN, thay vì sử dụng rule-based AI hiện tại.

## 🎯 Mục Đích

### **Hiện Tại:**
- AI chơi theo rules được lập trình sẵn
- Không học được từ cách chơi của users
- Không thích ứng với meta game

### **Mục Tiêu:**
- **Học từ real gameplay data** của users
- **Predict next move** dựa trên game state
- **Two-stage decision making**: Chọn combo_type trước, sau đó chọn cards cụ thể

## 📊 Data Format Hiện Tại

### **Input Data Structure:**
```json
{
  "game_id": "sam_game_1",
  "player_id": 0,
  "hand": [0, 13, 26, 39, 8, 9, 10, 11, 12, 45],
  "last_move": {
    "type": "play_cards",
    "cards": [1, 14],
    "combo_type": "pair",
    "rank_value": 1
  },
  "players_left": [2, 3, 4],
  "cards_left": [8, 7, 6],
  "action": {
    "stage1": {"type": "combo_type", "value": "four_kind"},
    "stage2": {"type": "play_cards", "cards": [0, 13, 26, 39], "combo_type": "four_kind", "rank_value": 0}
  },
  "meta": {
    "legal_moves": [
      {"type": "play_cards", "cards": [0, 13, 26, 39], "combo_type": "four_kind", "rank_value": 0},
      {"type": "play_cards", "cards": [8, 9, 10, 11, 12], "combo_type": "straight", "rank_value": 8},
      {"type": "play_cards", "cards": [45], "combo_type": "single", "rank_value": 6},
      {"type": "pass", "cards": [], "combo_type": None, "rank_value": None}
    ],
    "game_type": "sam"
  }
}
```

### **Key Components:**
- **hand**: Cards hiện tại của player
- **last_move**: Move của người chơi trước
- **players_left**: Số người chơi còn lại
- **cards_left**: Số lá bài còn lại của mỗi player
- **action**: Move mà player thực sự đã chọn (two-stage)
- **legal_moves**: Tất cả moves có thể chơi

## 🤔 Vấn Đề Hiện Tại

### **1. Single-Stage Approach (Sai):**
```python
# Current: Chọn trực tiếp index từ legal_moves
model.predict(game_state) → move_index → legal_moves[move_index]
```

**Vấn đề:**
- Không học được logic chọn combo_type
- Không phân biệt được stage1 vs stage2
- Model không hiểu được decision flow

### **2. Random Training Data:**
- Data được generate random
- Không có patterns thực tế
- Model không học được gameplay logic

### **3. Feature Engineering Chưa Tối Ưu:**
- 60 features nhưng không capture được game logic
- Không có features cho combo strength
- Không có features cho game context

## 🎯 Yêu Cầu Cụ Thể

### **Two-Stage Decision Making:**

#### **Stage 1: Combo Type Selection**
```
Input: Game State (hand, last_move, context)
Output: Combo Type ("single", "pair", "triple", "four_kind", "straight", "pass")
```

#### **Stage 2: Card Selection**
```
Input: Game State + Chosen Combo Type
Output: Specific Cards từ legal_moves
```

### **Learning Objectives:**
1. **Stage 1**: Học khi nào nên chọn combo_type nào
2. **Stage 2**: Học cách chọn cards cụ thể theo combo_type
3. **Context Awareness**: Hiểu game situation (đầu game, cuối game, etc.)

## 🔧 Các Approach Đề Xuất

### **Approach 1: Two Separate Models**

#### **Stage 1 Model:**
```python
# Features
stage1_features = [
    hand_one_hot,           # 52 dims
    last_move_combo_type,   # 6 dims  
    last_move_rank,         # 1 dim
    players_left,           # 1 dim
    cards_left_sum,         # 1 dim
    game_context            # N dims
]

# Label
stage1_label = record["action"]["stage1"]["value"]  # "four_kind", "pass", etc.

# Training
stage1_model = DecisionTreeClassifier()
stage1_model.fit(stage1_features, stage1_label)
```

#### **Stage 2 Model:**
```python
# Features
stage2_features = [
    stage1_features,        # All stage1 features
    chosen_combo_type,      # 6 dims (one-hot)
    filtered_legal_moves    # Variable dims
]

# Label  
stage2_label = index_of_chosen_move_in_legal_moves

# Training
stage2_model = DecisionTreeClassifier()
stage2_model.fit(stage2_features, stage2_label)
```

#### **Inference:**
```python
def predict(game_record):
    # Stage 1: Choose combo type
    combo_type = stage1_model.predict(extract_stage1_features(game_record))
    
    # Stage 2: Choose specific cards
    filtered_moves = [m for m in legal_moves if m.get("combo_type") == combo_type]
    if filtered_moves:
        move_index = stage2_model.predict(extract_stage2_features(game_record, combo_type))
        return filtered_moves[move_index]
    else:
        return {"type": "pass"}
```

### **Approach 2: Hierarchical Model**

#### **Single Model với 2 Outputs:**
```python
# Features
all_features = extract_all_features(record)

# Labels
labels = [
    stage1_combo_type,      # "four_kind"
    stage2_move_index       # 0, 1, 2, ...
]

# Training
model = DecisionTreeClassifier()
model.fit(all_features, labels)
```

### **Approach 3: Pipeline Model**

#### **Sequential Decision Making:**
```python
def predict(record):
    # Stage 1: Choose combo type
    combo_type = stage1_model.predict(record)
    
    # Filter legal moves by combo type
    filtered_moves = [m for m in legal_moves if m.get("combo_type") == combo_type]
    
    # Stage 2: Choose from filtered moves
    if filtered_moves:
        move_index = stage2_model.predict(record, filtered_moves)
        return filtered_moves[move_index]
    else:
        # Fallback: choose any legal move
        return legal_moves[0] if legal_moves else {"type": "pass"}
```

## 📈 Feature Engineering Chi Tiết

### **Stage 1 Features (Combo Type Selection):**

#### **Basic Features:**
```python
stage1_features = [
    # Hand analysis
    hand_one_hot,                    # 52 dims
    hand_combo_counts,               # 6 dims (count of each combo type)
    
    # Game context
    last_move_combo_type,            # 6 dims
    last_move_rank,                  # 1 dim
    players_left_count,              # 1 dim
    cards_left_sum,                  # 1 dim
    
    # Basic game state
    is_start_of_game,                # 1 dim (no last_move)
    has_last_move,                   # 1 dim (boolean)
]
```

### **Stage 2 Features (Card Selection):**

#### **Combo-Specific Features:**
```python
stage2_features = [
    # All stage1 features
    *stage1_features,
    
    # Chosen combo type
    chosen_combo_type,               # 6 dims (one-hot)
    
    # Available moves for this combo type
    available_moves_count,           # 1 dim
    
    # Basic card features
    card_ranks,                      # 13 dims
    card_suits,                      # 4 dims
]
```

## 🧪 Training Strategy

### **Data Preparation:**
```python
def prepare_training_data(records):
    stage1_data = []
    stage2_data = []
    
    for record in records:
        # Stage 1 data
        stage1_features = extract_stage1_features(record)
        stage1_label = record["action"]["stage1"]["value"]
        stage1_data.append((stage1_features, stage1_label))
        
        # Stage 2 data (only if stage1 is not pass)
        if stage1_label != "pass":
            stage2_features = extract_stage2_features(record, stage1_label)
            stage2_label = get_move_index(record["action"]["stage2"], record["meta"]["legal_moves"])
            stage2_data.append((stage2_features, stage2_label))
    
    return stage1_data, stage2_data
```

### **Model Training:**
```python
# Stage 1 training
stage1_X, stage1_y = zip(*stage1_data)
stage1_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
stage1_model.fit(stage1_X, stage1_y)

# Stage 2 training  
stage2_X, stage2_y = zip(*stage2_data)
stage2_model = DecisionTreeClassifier(max_depth=8, min_samples_split=3)
stage2_model.fit(stage2_X, stage2_y)
```

## 🎯 Evaluation Metrics

### **Stage 1 Metrics:**
- **Accuracy**: Tỷ lệ chọn đúng combo_type
- **Precision/Recall**: Cho từng combo_type
- **Confusion Matrix**: So sánh predicted vs actual

### **Stage 2 Metrics:**
- **Accuracy**: Tỷ lệ chọn đúng cards
- **Fallback Rate**: Tỷ lệ phải dùng fallback logic

### **Overall Metrics:**
- **End-to-End Accuracy**: Tỷ lệ chọn đúng move hoàn chỉnh
- **Pattern Learning**: Mức độ học được patterns từ training data

## ❓ Câu Hỏi Thảo Luận

### **1. Model Architecture:**
- Bạn prefer approach nào? (Two separate models, Hierarchical, Pipeline)
- Có cần thêm complexity không?

### **2. Feature Engineering:**
- Features cơ bản đã đủ chưa?
- Có cần thêm features nào khác không?

### **3. Training Data:**
- Cần bao nhiêu data để train hiệu quả?
- Có cần balance data cho từng combo_type không?

### **4. Evaluation:**
- Accuracy metrics đã đủ chưa?
- Có cần thêm metrics nào khác không?

### **5. Implementation:**
- Pipeline logic có cần optimize không?
- Fallback strategy có phù hợp không?

## 🚀 Next Steps

1. **✅ Chọn approach**: Cách 3 - Pipeline Model
2. **Implement feature engineering** cơ bản
3. **Tạo training data** với format đúng
4. **Train và evaluate** models
5. **Test với random data** trước

---

**File này sẽ được cập nhật dựa trên thảo luận và quyết định của team.**
