# Reward Design Documentation

## Tổng quan

Hệ thống reward hiện tại sử dụng **Potential-based Reward Shaping** để cung cấp feedback liên tục ở mỗi step, kết hợp với final reward khi game kết thúc.

---

## 1. Step Reward (Reward ở mỗi step)

### 1.1. Công thức

```python
_step_reward(cards_before, cards_after, w1=0.5, w2=0.7, k=0.5, gamma=0.99, time_penalty=0.05)
```

**Các thành phần:**

1. **Agent giảm bài** (`d_me`):
   - `d_me = c_me_before - c_me_after`
   - `>= 0` nếu agent đánh bài
   - Reward: `+w1 * d_me` (mặc định `w1 = 0.5`)

2. **Đối thủ giảm bài** (`d_opp_avg`):
   - `d_opp_total = sum(c_opp_list_before) - sum(c_opp_list_after)`
   - `d_opp_avg = d_opp_total / max(1, len(c_opp_list_before))`
   - Penalty: `-w2 * d_opp_avg` (mặc định `w2 = 0.7`)

3. **Potential-based Shaping**:
   - Potential function: `φ(c_opp_list, c_me) = k * (min_opp - c_me)`
     - `min_opp`: số bài ít nhất của đối thủ
     - `c_me`: số bài của agent
     - `k = 0.5`: scaling factor
   - Shaping term: `γ * φ(after) - φ(before)`
     - `γ = 0.99`: discount factor

4. **Time penalty**:
   - `-time_penalty` (mặc định `-0.05`) cho mỗi step

### 1.2. Công thức tổng hợp

```
r_step = w1 * d_me - w2 * d_opp_avg + shaping - time_penalty
r_step = clip(r_step, -1.0, 1.0)  # Giới hạn trong [-1, 1]
```

### 1.3. Ví dụ

**Tình huống:**
- Trước: Agent có 8 lá, Opponent 1 có 5 lá, Opponent 2 có 7 lá
- Agent đánh 2 lá → còn 6 lá
- Opponent 1 đánh 1 lá → còn 4 lá
- Opponent 2 pass → còn 7 lá

**Tính toán:**
- `d_me = 8 - 6 = 2` → `+0.5 * 2 = +1.0`
- `d_opp_avg = (5+7) - (4+7) / 2 = 1/2 = 0.5` → `-0.7 * 0.5 = -0.35`
- `φ(before) = 0.5 * (5 - 8) = -1.5`
- `φ(after) = 0.5 * (4 - 6) = -1.0`
- `shaping = 0.99 * (-1.0) - (-1.5) = 0.51`
- `r_step = 1.0 - 0.35 + 0.51 - 0.05 = 1.11` → clip → `1.0`

---

## 2. Final Reward (Reward khi game kết thúc)

### 2.1. Agent thắng (0 lá còn lại)

```
reward = sum(số lá còn lại của tất cả đối thủ)
```

**Ví dụ:**
- Opponent 1 còn 3 lá, Opponent 2 còn 5 lá
- `reward = 3 + 5 = 8`

### 2.2. Agent thua (>0 lá còn lại)

#### 2.2.1. Trường hợp đặc biệt: Còn đủ 10 lá (Sam) hoặc 13 lá (TLMN)

```
reward = -15.0
```

#### 2.2.2. Trường hợp có card 2 (rank 12) hoặc four_of_a_kind

```
reward = -3.0 * số_lá_còn_lại
```

**Ví dụ:**
- Agent còn 5 lá, trong đó có card 2
- `reward = -3.0 * 5 = -15.0`

#### 2.2.3. Trường hợp bình thường

```
reward = -1.0 * số_lá_còn_lại
```

**Ví dụ:**
- Agent còn 3 lá
- `reward = -3.0`

---

## 3. Kết hợp Step Reward và Final Reward

### 3.1. Các step bình thường (game chưa kết thúc)

```
reward = _step_reward(cards_before, cards_after)
```

### 3.2. Step cuối cùng (game kết thúc)

```
reward = _final_reward() + _step_reward(cards_before, cards_after)
```

**Lưu ý:** Step reward cho move cuối được cộng vào final reward để agent nhận được feedback cho cả việc đánh bài và kết quả cuối cùng.

---

## 4. Các tham số hiện tại

| Tham số | Giá trị | Mô tả |
|---------|--------|-------|
| `w1` | 0.5 | Weight cho agent giảm bài |
| `w2` | 0.7 | Weight cho đối thủ giảm bài |
| `k` | 0.5 | Scaling factor cho potential function |
| `gamma` | 0.99 | Discount factor (từ TrainerConfig) |
| `time_penalty` | 0.05 | Penalty cho mỗi step |
| `win_multiplier` | 1.0 | Multiplier cho final reward khi thắng (sum of opponent cards) |
| `loss_base` | -1.0 | Base penalty cho mỗi lá còn lại khi thua |
| `loss_special` | -3.0 | Penalty cho mỗi lá khi có card 2 hoặc four_kind |
| `loss_all_cards` | -15.0 | Penalty khi còn đủ số lá ban đầu |

---

## 5. Logic trong code

### 5.1. Flow trong `step()`

```python
# 1. Track cards trước khi agent đánh
cards_before = self._get_cards_state()

# 2. Agent đánh
success = self._apply_move(self.agent_id, selected_move)

# 3. Opponents đánh (trong _advance_until_agent_turn)
self._advance_until_agent_turn()

# 4. Track cards sau khi tất cả đánh xong
cards_after = self._get_cards_state()

# 5. Tính reward
if game.is_finished:
    reward = _final_reward_with_shaping(cards_before)  # = final + step
else:
    reward = _step_reward(cards_before, cards_after)
```

### 5.2. Potential Function (`_phi`)

```python
def _phi(self, c_opp_list: List[int], c_me: int, k: float = 0.5) -> float:
    if not c_opp_list:
        return 0.0
    min_opp = min(c_opp_list)  # Focus vào đối thủ nguy hiểm nhất
    return k * (min_opp - c_me)
```

**Ý nghĩa:**
- `φ > 0` khi agent ít bài hơn đối thủ nguy hiểm nhất → khuyến khích
- `φ < 0` khi agent nhiều bài hơn → penalty
- Shaping term `γ * φ(after) - φ(before)` đảm bảo không thay đổi optimal policy (theoretical guarantee)

---

## 6. Điểm mạnh và điểm yếu

### 6.1. Điểm mạnh

1. **Continuous feedback**: Agent nhận reward ở mỗi step, không chỉ cuối game
2. **Potential-based shaping**: Đảm bảo không thay đổi optimal policy (theoretical guarantee)
3. **Khuyến khích giảm bài**: `w1 * d_me` khuyến khích agent đánh bài
4. **Penalty đối thủ giảm bài**: `-w2 * d_opp_avg` khuyến khích agent chặn đối thủ
5. **Final reward rõ ràng**: Thắng được reward cao, thua bị penalty rõ ràng

### 6.2. Điểm yếu / Cần xem xét

1. **w1 vs w2**: Hiện tại `w2 = 0.7 > w1 = 0.5`, có nghĩa là penalty khi đối thủ giảm bài lớn hơn reward khi agent giảm bài. Có thể cần điều chỉnh.

2. **Time penalty**: `-0.05` mỗi step có thể quá nhỏ hoặc quá lớn tùy vào độ dài game.

3. **Potential function**: Dùng `min_opp` có thể không phản ánh đúng tình huống khi có nhiều đối thủ.

4. **Final reward scale**: Final reward có thể rất lớn (ví dụ: thắng với 3 đối thủ còn nhiều bài → reward = 10+), trong khi step reward bị clip ở [-1, 1]. Có thể cần normalize.

---

## 7. Gợi ý điều chỉnh

### 7.1. Cân bằng w1 và w2

- **Hiện tại**: `w1 = 0.5`, `w2 = 0.7`
- **Gợi ý**: Có thể thử `w1 = w2 = 0.6` để cân bằng hơn

### 7.2. Normalize final reward

- Có thể chia final reward cho số lá ban đầu để scale về cùng range với step reward
- Ví dụ: `final_reward = final_reward / initial_hand_size`

### 7.3. Điều chỉnh time penalty

- Nếu game thường dài (>100 steps), có thể tăng `time_penalty` lên `0.1`
- Nếu game ngắn (<50 steps), có thể giảm xuống `0.02`

---

## 8. File liên quan

- **`model_build/scripts/rl_agent/env.py`**:
  - `_step_reward()`: Dòng 421-458
  - `_final_reward()`: Dòng 331-375
  - `_final_reward_with_shaping()`: Dòng 460-472
  - `_phi()`: Dòng 408-419
  - `step()`: Dòng 90-147

- **`model_build/scripts/rl_agent/trainer.py`**:
  - `TrainerConfig.gamma`: Dòng 24 (mặc định 0.99)












