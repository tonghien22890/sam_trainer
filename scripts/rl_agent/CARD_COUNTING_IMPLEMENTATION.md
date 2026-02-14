# Card Counting và Rank Power Implementation Plan

## Tổng quan

Document này ghi lại thảo luận và quyết định về việc implement **Card Counting (Seen Vector)** và **Rank Power** features để cải thiện khả năng "giật cái" và đánh giá độ hiếm của lá bài trong RL agent.

## Vấn đề hiện trạng

### 1. Agent chưa học được "giật cái" hiệu quả
- **Nguyên nhân**: Thiếu thông tin về độ hiếm thực tế của lá bài
- **Biểu hiện**: Agent không biết khi nào lá bài trở nên "vô địch" (không còn lá lớn hơn)
- **Hệ quả**: Agent không tự tin giật cái, đặc biệt là bằng single 2 hoặc các combo mạnh

### 2. Agent hội tụ quá sớm
- **Nguyên nhân**: Opponent pool chưa đủ đa dạng, training không đủ thách thức
- **Biểu hiện**: Reward không cải thiện dù đánh với opponent từ 10k, 20k, 50k episodes trước
- **Hệ quả**: Agent không tiếp tục học và cải thiện

### 3. Thiếu Card Counting
- **Vấn đề**: Agent không track các lá đã đánh
- **Hệ quả**: Không thể tính toán "Rank Power" (số lá lớn hơn còn lại)
- **Impact**: Không biết được độ mạnh thực tế của lá bài trong tình huống cụ thể

## Giải pháp đề xuất

### Phase 1: Card Counting Infrastructure (Priority 1 - Critical)

#### 1.1. Seen Vector Tracking

**Cấu trúc:**
```python
# Trong CardGameEnv.__init__:
self.seen_ranks = [0] * 13  # Track 13 ranks (0=3, 1=4, ..., 11=A, 12=2)
                            # Mỗi phần tử = số lá đã thấy (0-4)
```

**Logic Update:**
- Update trong `_apply_move()` sau khi move thành công
- Track cả agent moves và opponent moves
- Reset trong `reset()` khi bắt đầu ván mới

**Quan trọng**: Không track cards trong agent hand vào `seen_ranks` (vì agent biết chúng)

#### 1.2. Rank Power Calculation

**Khái niệm:**
- **Rank Power** = Số lá lớn hơn rank hiện tại CÒN LẠI trong deck (agent KHÔNG biết)
- **Normalized Rank Power** = Rank Power / Max Possible (normalized về [0.0-1.0])

**Công thức:**
```python
def get_rank_power(rank: int, agent_hand: List[int], seen_ranks: List[int]) -> int:
    """
    Tính số lá lớn hơn rank còn lại (KHÔNG tính cards trong agent hand).
    
    Args:
        rank: Rank của lá bài (0-12, 12=2)
        agent_hand: List card_ids trong hand của agent
        seen_ranks: List 13 phần tử, mỗi phần tử = số lá đã thấy
    
    Returns:
        Số lá lớn hơn còn lại (0 nếu không còn, hoặc >= 0)
    """
    if rank >= 12:  # Rank 12 (2) là cao nhất
        return 0
    
    # Đếm số lá mỗi rank trong agent hand
    agent_hand_ranks = [0] * 13
    for card_id in agent_hand:
        r = card_id % 13
        agent_hand_ranks[r] += 1
    
    # Tính số lá lớn hơn còn lại = total - seen - in_agent_hand
    larger_cards_left = 0
    for r in range(rank + 1, 13):
        total_for_rank = 4
        seen_count = seen_ranks[r]
        in_agent_hand = agent_hand_ranks[r]
        remaining = total_for_rank - seen_count - in_agent_hand
        if remaining < 0:
            # Log warning: track lỗi
            import logging
            logging.warning(f"Rank {r} remaining < 0: seen={seen_count}, in_hand={in_agent_hand}")
        larger_cards_left += max(0, remaining)
    
    return larger_cards_left

def get_rank_power_normalized(rank: int, agent_hand: List[int], seen_ranks: List[int]) -> float:
    """
    Normalized rank power [0.0-1.0], trừ đi cards trong agent hand.
    
    Returns:
        0.0 = Không còn lá lớn hơn (unbeatable)
        1.0 = Còn nhiều lá lớn hơn nhất (rất yếu)
    """
    if rank >= 12:
        return 0.0
    
    # Tính max_possible (trừ đi cards trong agent hand)
    agent_hand_ranks = [0] * 13
    for card_id in agent_hand:
        r = card_id % 13
        agent_hand_ranks[r] += 1
    
    max_possible = 0
    for r in range(rank + 1, 13):
        max_possible += (4 - agent_hand_ranks[r])  # Trừ cards trong agent hand
    
    if max_possible == 0:
        return 0.0
    
    raw_power = get_rank_power(rank, agent_hand, seen_ranks)
    return min(1.0, raw_power / max_possible)
```

**Ví dụ:**
```
Bắt đầu ván: Agent cầm đôi A (rank 11)
- seen_ranks = [0, 0, ..., 0, 0]  # Chưa có lá nào ra
- agent_hand có 2 lá rank 11 (A)
- Rank Power của A = số lá rank 12 (2) còn lại = 4 - 0 - 0 = 4
- Normalized = 4 / 4 = 1.0 (yếu nhất, còn 4 lá 2)

Giữa ván: Đã thấy 4 lá 2 ra hết
- seen_ranks[12] = 4
- agent_hand vẫn có 2 lá A
- Rank Power của A = 4 - 4 - 0 = 0 (không còn lá lớn hơn!)
- Normalized = 0.0 (unbeatable, đôi A là lớn nhất)
```

#### 1.3. Top-3 Ranks trong Hand

**Mục đích**: Giúp agent biết còn "tay to" nào để quyết định có nên giật cái không

**Logic:**
```python
def get_top_3_ranks_in_hand(hand: List[int]) -> List[int]:
    """
    Lấy Top-3 rank cao nhất trong hand.
    
    Sorting priority:
    1. Rank 12 (2) luôn đứng đầu
    2. Số lượng cards (ưu tiên bộ: đôi, ba, tứ)
    3. Rank value (cao hơn tốt hơn)
    
    Returns:
        List 3 ranks (có thể < 3 nếu hand ít hơn 3 ranks khác nhau)
    """
    rank_counts = {}
    for card_id in hand:
        rank = card_id % 13
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Sort: (1) Rank 12 trước, (2) Nhiều cards hơn, (3) Rank cao hơn
    sorted_ranks = sorted(
        rank_counts.keys(),
        key=lambda r: (r == 12, rank_counts[r], r),  # True > False, nên rank 12 đứng đầu
        reverse=True
    )
    
    return sorted_ranks[:3]
```

**Ví dụ:**
```
Hand: [A, A, K, K, 5, 5, 3, 3, 2]
rank_counts: {11: 2, 10: 2, 2: 2, 0: 2, 12: 1}

Sorting:
- Rank 12 (2): (True, 1, 12) → Đứng đầu
- Rank 11 (A): (False, 2, 11)
- Rank 10 (K): (False, 2, 10)
- Rank 2 (5): (False, 2, 2)
- Rank 0 (3): (False, 2, 0)

Top-3: [12, 11, 10]  # Rank 2, A, K
```

---

### Phase 2: Features mới (8 dims - "Elite Eight")

#### 2.1. Danh sách Features

1. **`is_lead`** (1 dim)
   - **Value**: 1.0 nếu agent đang có quyền chủ động (không cần block), 0.0 nếu đang blocking
   - **Logic**: `is_lead = 1.0 if last_move is None else 0.0`
   - **Scale**: 10.0 (tương tự blocking_scale)
   - **Mục đích**: Phân biệt rõ khi nào agent đang lead vs blocking

2. **`curr_move_rank_power`** (1 dim)
   - **Value**: Normalized rank power của move hiện tại [0.0-1.0]
   - **Logic**: `get_rank_power_normalized(move_rank, agent_hand, seen_ranks)`
   - **Scale**: 12.0 (tương tự can_beat_scale)
   - **Mục đích**: Đánh giá độ mạnh thực tế của move (dynamic, không chỉ static strength)

3. **`is_unbeatable`** (1 dim)
   - **Value**: 1.0 nếu `rank_power == 0` (không còn lá lớn hơn), 0.0 nếu không
   - **Logic**: `is_unbeatable = 1.0 if curr_move_rank_power == 0.0 else 0.0`
   - **Scale**: 15.0 (quan trọng, scale cao)
   - **Mục đích**: Tín hiệu rõ ràng cho network biết "vé thông hành" (unbeatable move)

4. **`top1_rank_power`** (1 dim)
   - **Value**: Normalized rank power của rank cao nhất trong hand
   - **Scale**: 10.0
   - **Mục đích**: Context về "tay to" nhất còn lại

5. **`top2_rank_power`** (1 dim)
   - **Value**: Normalized rank power của rank thứ 2 trong hand
   - **Scale**: 8.0
   - **Mục đích**: Context về "tay to" thứ 2

6. **`top3_rank_power`** (1 dim)
   - **Value**: Normalized rank power của rank thứ 3 trong hand
   - **Scale**: 6.0
   - **Mục đích**: Context về "tay to" thứ 3

7. **`lose_lead_prob`** (1 dim)
   - **Value**: Xác suất mất quyền chủ động khi đánh move này [0.0-1.0]
   - **Logic**: `remaining_larger_cards / total_remaining_cards`
   - **Scale**: 12.0 (quan trọng cho risk assessment)
   - **Mục đích**: Đánh giá rủi ro khi giật cái

8. **`next_player_danger`** (1 dim)
   - **Value**: HIGH khi next player có ÍT lá (sắp thắng) [0.0-1.0]
   - **Logic**: INVERTED - low cards = high danger
     - ≤2 cards → 1.0 (maximum danger!)
     - ≤4 cards → 0.8 (high danger)
     - ≤6 cards → 0.5 (medium danger)
     - >6 cards → 1.0 - (cards/13)
   - **Scale**: 10.0
   - **Mục đích**: Giúp agent biết khi nào cần chặn/block opponent sắp thắng

#### 2.2. Edge Cases

**Hand rỗng:**
```python
if not hand:
    # Return default values
    return [0.0] * 8
```

**Không đủ 3 ranks:**
```python
# Padding với 1.0 (max rank power = yếu nhất)
while len(top_3_powers) < 3:
    top_3_powers.append(1.0)
```

**Pass move:**
```python
if combo_type == "pass":
    # rank_power = 0.0, is_unbeatable = 0.0
    curr_move_rank_power = 0.0
    is_unbeatable = 0.0
```

**Seen ranks chưa init:**
```python
# Check và init nếu chưa có
seen_ranks = game_record.get("seen_ranks", [0] * 13)
```

---

### Phase 3: Integration

#### 3.1. Truyền Seen Ranks vào Features

**Cách 1 (Recommended)**: Truyền qua `game_record`
```python
# Trong env.py _build_observation():
record = self.game.get_game_record()
record["seen_ranks"] = self.seen_ranks  # Thêm vào record
```

**Cách 2**: Inject vào FeatureBuilder
```python
# Trong features.py:
def build_feature_matrix(self, ..., seen_ranks: Optional[List[int]] = None):
    # ...
```

**Quyết định**: Dùng Cách 1 (truyền qua game_record) vì đơn giản, không cần sửa constructor.

#### 3.2. Vị trí thêm Features

**Đề xuất**: Tạo method mới `_extract_card_counting_features()` và gọi trong `build_feature_matrix()`:
```python
def build_feature_matrix(self, game_record, legal_moves, framework):
    features_list = []
    for move in legal_moves:
        original = self._extract_original_features(move, game_record)
        framework_feat = self._extract_framework_features(move, framework, game_record)
        multi_seq = self._extract_multi_sequence_features(move, framework)
        card_counting = self._extract_card_counting_features(move, game_record)  # ← Mới
        features_list.append(original + framework_feat + multi_seq + card_counting)
    return features_list
```

#### 3.3. Update Feature Dimension

```python
# Trong __init__:
# Base dim: 45
# + ENABLE_BLOCK_LEAD_GATING: +2
# + ENABLE_LEAD_QUALITY_FEATURES: +2
# + CARD_COUNTING_FEATURES: +8  # ← Mới
base_dim = 45
extra_dims = 0
if ENABLE_BLOCK_LEAD_GATING:
    extra_dims += 2
if ENABLE_LEAD_QUALITY_FEATURES:
    extra_dims += 2
if ENABLE_CARD_COUNTING_FEATURES:  # ← Flag mới
    extra_dims += 8
self.feature_dim = base_dim + extra_dims
```

---

## Chi tiết Implementation

### File: `env.py`

#### 1. Thêm Seen Vector vào `__init__`
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Card counting: track seen ranks
    self.seen_ranks = [0] * 13  # 13 ranks, each 0-4 cards seen
```

#### 2. Helper function `_update_seen_cards()`
```python
def _update_seen_cards(self, move: Dict[str, Any]) -> None:
    """
    Update seen_ranks từ move được chơi.
    Chỉ track cards đã được đánh ra, KHÔNG track cards trong agent hand.
    """
    if move.get("type") != "play_cards":
        return
    
    cards = move.get("cards", []) or []
    for card_id in cards:
        rank = card_id % 13
        self.seen_ranks[rank] += 1
        self.seen_ranks[rank] = min(4, self.seen_ranks[rank])  # Cap at 4
```

#### 3. Gọi update trong `_apply_move()`
```python
def _apply_move(self, player_id: int, move: Dict[str, Any]) -> bool:
    if move.get("type") == "pass":
        return self.game.play_move(player_id, PlayerAction.PASS, [])
    
    cards_ids = move.get("cards", []) or []
    cards = CardEncoder.decode_hand(cards_ids)
    success = self.game.play_move(player_id, PlayerAction.PLAY_CARDS, cards)
    
    if success:
        self._update_seen_cards(move)  # ← Update seen cards
    
    return success
```

#### 4. Reset trong `reset()`
```python
def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
    # ... existing code ...
    
    self.seen_ranks = [0] * 13  # Reset seen cards
    # ... rest of reset logic ...
```

#### 5. Truyền vào `game_record` trong `_build_observation()`
```python
def _build_observation(self) -> Dict[str, Any]:
    record = self.game.get_game_record()
    
    # Add seen_ranks to record for feature builder
    record["seen_ranks"] = self.seen_ranks.copy()  # Copy để tránh mutation
    
    # ... rest of observation building ...
```

### File: `features.py`

#### 1. Thêm flag và update feature_dim
```python
ENABLE_CARD_COUNTING_FEATURES = True  # Card counting and rank power features

# Trong __init__:
if ENABLE_CARD_COUNTING_FEATURES:
    extra_dims += 8
```

#### 2. Helper functions cho Rank Power
```python
def _get_rank_power(self, rank: int, agent_hand: List[int], seen_ranks: List[int]) -> int:
    """Tính số lá lớn hơn rank còn lại (raw number)"""
    # Implementation như đã nêu ở trên
    pass

def _get_rank_power_normalized(self, rank: int, agent_hand: List[int], seen_ranks: List[int]) -> float:
    """Normalized rank power [0.0-1.0]"""
    # Implementation như đã nêu ở trên
    pass

def _get_top_3_ranks_in_hand(self, hand: List[int]) -> List[int]:
    """Lấy Top-3 ranks trong hand"""
    # Implementation như đã nêu ở trên
    pass

def _calculate_lose_lead_probability(
    self,
    move_rank: int,
    agent_hand: List[int],
    seen_ranks: List[int],
    cards_left_per_player: List[int]
) -> float:
    """
    Tính xác suất mất quyền chủ động khi đánh move này.
    
    Công thức: remaining_larger_cards / total_remaining_cards
    """
    # Tính số lá lớn hơn còn lại
    remaining_larger = self._get_rank_power(move_rank, agent_hand, seen_ranks)
    
    if remaining_larger == 0:
        return 0.0  # Không còn lá lớn hơn → an toàn
    
    # Tổng số lá còn lại trên bàn
    total_remaining = sum(cards_left_per_player)
    
    if total_remaining == 0:
        return 0.0
    
    # Xác suất đơn giản
    prob = min(1.0, remaining_larger / total_remaining)
    return prob
```

#### 3. Method `_extract_card_counting_features()`
```python
def _extract_card_counting_features(
    self, move: Dict[str, Any], game_record: Dict[str, Any]
) -> List[float]:
    """
    Extract card counting features (8 dims):
    1. is_lead
    2. curr_move_rank_power
    3. is_unbeatable
    4. top1_rank_power
    5. top2_rank_power
    6. top3_rank_power
    7. lose_lead_prob
    8. next_player_danger
    """
    features = []
    
    if not ENABLE_CARD_COUNTING_FEATURES:
        return [0.0] * 8
    
    # Get seen_ranks and agent hand from game_record
    seen_ranks = game_record.get("seen_ranks", [0] * 13)
    agent_hand = game_record.get("hand", []) or []
    cards_left = game_record.get("cards_left", []) or []
    current_player_id = game_record.get("current_player_id", 0)
    
    # 1. is_lead: 1.0 nếu agent có quyền chủ động
    is_lead = 1.0 if game_record.get("last_move") is None else 0.0
    features.append(is_lead)
    
    # 2. curr_move_rank_power
    combo_type = move.get("combo_type", "pass")
    move_rank = move.get("rank_value", 0) or 0
    if combo_type == "pass":
        curr_move_rank_power = 0.0
    else:
        curr_move_rank_power = self._get_rank_power_normalized(
            move_rank, agent_hand, seen_ranks
        )
    features.append(curr_move_rank_power)
    
    # 3. is_unbeatable
    is_unbeatable = 1.0 if curr_move_rank_power == 0.0 and combo_type != "pass" else 0.0
    features.append(is_unbeatable)
    
    # 4-6. Top-3 rank powers
    if agent_hand:
        top_3_ranks = self._get_top_3_ranks_in_hand(agent_hand)
        top_3_powers = [
            self._get_rank_power_normalized(rank, agent_hand, seen_ranks)
            for rank in top_3_ranks
        ]
        # Padding nếu không đủ 3
        while len(top_3_powers) < 3:
            top_3_powers.append(1.0)  # Max rank power = yếu nhất
        features.extend(top_3_powers[:3])
    else:
        features.extend([1.0, 1.0, 1.0])  # Default: yếu nhất
    
    # 7. lose_lead_prob
    if combo_type == "pass":
        lose_lead_prob = 0.0
    else:
        lose_lead_prob = self._calculate_lose_lead_probability(
            move_rank, agent_hand, seen_ranks, cards_left
        )
    features.append(lose_lead_prob)
    
    # 8. next_player_danger: HIGH when opponent has FEW cards (about to win)
    num_players = len(cards_left) if cards_left else 4
    next_player_id = (current_player_id + 1) % num_players
    next_player_cards = cards_left[next_player_id] if next_player_id < len(cards_left) else 13
    
    # INVERTED: low cards = high danger!
    if next_player_cards <= 2:
        next_player_danger = 1.0  # Maximum danger - opponent about to win!
    elif next_player_cards <= 4:
        next_player_danger = 0.8  # High danger
    elif next_player_cards <= 6:
        next_player_danger = 0.5  # Medium danger
    else:
        next_player_danger = max(0.0, 1.0 - (next_player_cards / 13.0))
    
    features.append(next_player_danger)
    
    return features
```

#### 4. Apply scales trong `build_feature_matrix()`
```python
def build_feature_matrix(self, game_record, legal_moves, framework):
    features_list = []
    for move in legal_moves:
        # ... existing features ...
        card_counting = self._extract_card_counting_features(move, game_record)
        
        # Apply scales
        card_counting_scaled = [
            card_counting[0] * 10.0,  # is_lead
            card_counting[1] * 12.0,  # curr_move_rank_power
            card_counting[2] * 15.0,  # is_unbeatable
            card_counting[3] * 10.0,  # top1_rank_power
            card_counting[4] * 8.0,   # top2_rank_power
            card_counting[5] * 6.0,   # top3_rank_power
            card_counting[6] * 12.0,  # lose_lead_prob
            card_counting[7] * 10.0,  # next_player_danger
        ]
        
        features_list.append(original + framework_feat + multi_seq + card_counting_scaled)
    return features_list
```

---

## Lưu ý quan trọng

### 1. Không track cards trong agent hand
- **Lý do**: Agent biết chính xác cards trong hand của mình (perfect information)
- **Cách xử lý**: Trừ `agent_hand_ranks` khi tính Rank Power
- **Impact**: Tránh "tự hù dọa chính mình" (ví dụ: cầm đôi 2 nhưng nghĩ đối thủ có thể có 2)

### 2. Chặt chồng (Tứ quý / Đôi thông)
- **Vấn đề**: Rank Power tính theo rank đơn, nhưng Tứ quý có thể chặt được đôi 2
- **Giải pháp**: KHÔNG đưa vào Rank Power (quá phức tạp)
- **Lý do**: PPO sẽ tự học thông qua `lose_lead_prob` và kinh nghiệm khi bị chặt trong self-play
- **Đủ dữ liệu**: `seen_ranks` track đúng các lá đã ra là đủ cho network học

### 3. Edge Cases
- **Remaining < 0**: Log warning (indicates tracking error)
- **Hand rỗng**: Return default values [0.0] * 8
- **Không đủ 3 ranks**: Padding với 1.0 (max rank power)
- **Pass move**: rank_power = 0.0, is_unbeatable = 0.0

---

## Feature Scales Summary

| Feature | Raw Range | Scale | Scaled Range | Purpose |
|---------|-----------|-------|--------------|---------|
| `is_lead` | [0.0, 1.0] | 10.0 | [0.0, 10.0] | Lead vs blocking |
| `curr_move_rank_power` | [0.0, 1.0] | 12.0 | [0.0, 12.0] | Dynamic move strength |
| `is_unbeatable` | [0.0, 1.0] | 15.0 | [0.0, 15.0] | Unbeatable signal |
| `top1_rank_power` | [0.0, 1.0] | 10.0 | [0.0, 10.0] | Top card context |
| `top2_rank_power` | [0.0, 1.0] | 8.0 | [0.0, 8.0] | Second card context |
| `top3_rank_power` | [0.0, 1.0] | 6.0 | [0.0, 6.0] | Third card context |
| `lose_lead_prob` | [0.0, 1.0] | 12.0 | [0.0, 12.0] | Risk assessment |
| `next_player_danger` | [0.0, 1.0] | 10.0 | [0.0, 10.0] | HIGH when opponent about to win |

**Tổng impact**: 8 features × scale trung bình ~10 = ~80 scaled units
- So với framework features (priority=15, breaking=15, compliance=16, seq_order=20): Vừa phải, không lấn át
- Đủ mạnh để network học được sự khác biệt

---

## Testing Checklist

### Unit Tests
- [ ] `_get_rank_power()` với các edge cases:
  - Rank 12 (2) → return 0
  - Hand rỗng
  - All cards seen
  - Agent hand có nhiều cards cùng rank
- [ ] `_get_rank_power_normalized()` với edge cases:
  - Max possible = 0
  - Raw power > max_possible
- [ ] `_get_top_3_ranks_in_hand()` với:
  - Hand có < 3 ranks khác nhau
  - Hand có rank 12
  - Hand rỗng
- [ ] `_update_seen_cards()` với:
  - Pass move (không update)
  - Play cards move
  - Cards vượt quá 4 (cap at 4)

### Integration Tests
- [ ] Seen ranks được update đúng sau mỗi move
- [ ] Seen ranks được reset trong `reset()`
- [ ] Seen ranks được truyền vào `game_record`
- [ ] Features được tính đúng với seen_ranks

### Training Tests
- [ ] Feature dimension đúng (base + 8)
- [ ] Network có thể load checkpoint cũ (feature_dim mismatch handling)
- [ ] Training không crash với features mới

---

## Kế hoạch triển khai

### Step 1: Infrastructure (CardGameEnv)
1. Thêm `self.seen_ranks` trong `__init__`
2. Implement `_update_seen_cards()`
3. Gọi update trong `_apply_move()`
4. Reset trong `reset()`
5. Truyền vào `game_record` trong `_build_observation()`

### Step 2: Helper Functions (features.py)
1. Implement `_get_rank_power()`
2. Implement `_get_rank_power_normalized()`
3. Implement `_get_top_3_ranks_in_hand()`
4. Implement `_calculate_lose_lead_probability()`

### Step 3: Features Extraction
1. Implement `_extract_card_counting_features()`
2. Handle edge cases (hand rỗng, pass, không đủ 3 ranks)
3. Apply feature scales
4. Integrate vào `build_feature_matrix()`

### Step 4: Update Configuration
1. Thêm flag `ENABLE_CARD_COUNTING_FEATURES`
2. Update `feature_dim` calculation
3. Update documentation

### Step 5: Testing
1. Unit tests cho helper functions
2. Integration tests
3. Training test với feature mới

---

## Kỳ vọng

### Immediate (Sau khi implement)
- Agent có thông tin về độ hiếm thực tế của lá bài
- Features cung cấp context tốt hơn cho network

### Short-term (100k-200k episodes)
- Agent học được khi nào lá bài trở nên "unbeatable"
- Agent tự tin hơn khi giật cái bằng single 2 hoặc combo mạnh
- Cải thiện winrate trong các tình huống cần giật cái

### Long-term (500k+ episodes)
- Agent master việc đánh giá độ hiếm
- Agent biết khi nào nên giật cái vs khi nào nên giữ bài
- Agent học được patterns về "chặt chồng" thông qua experience

---

## Notes

- **Không thay thế Combo Strength**: Rank Power là bổ sung, không thay thế. Combo Strength (static) vẫn quan trọng.
- **Không ép lối chơi**: Features chỉ cung cấp thông tin, network tự học cách sử dụng.
- **Có thể tune scales**: Nếu training cho thấy features quá yếu/mạnh, có thể điều chỉnh scales.

