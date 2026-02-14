# RL Agent — Status Report

**Ngày**: 2026-02-12  
**Mục tiêu**: Model có khả năng đánh bài (Sam / TLMN) với tỉ lệ thắng người cao nhất có thể  
**Trạng thái**: Đang phát triển — Agent chưa đạt mục tiêu, cần cải thiện đáng kể

---

## 1. Tổng quan hệ thống

Hệ thống RL Agent sử dụng **PPO (Proximal Policy Optimization)** với self-play để học chiến lược đánh bài. Kiến trúc gồm 4 thành phần chính:

- **CardGameEnv** (`env.py`): Bọc game engine (Sam/TLMN) thành RL environment, xử lý reward và card counting
- **FrameworkAwareFeatureBuilder** (`features.py`): Trích xuất 57 features cho mỗi nước đi hợp lệ
- **PolicyNetwork / ValueNetwork** (`policy.py`): Mạng neural 3 lớp (MLP) cho Actor-Critic
- **RLTrainer** (`trainer.py`): Vòng lặp training PPO với opponent pool

Đã train thử với 150k episodes x 3 lần (TLMN, 2 seats) nhưng agent vẫn **chưa thể hiện khả năng chơi thông minh** — cụ thể là không biết chặn đối thủ sắp thắng bằng lá mạnh.

---

## 2. Các vấn đề nghiêm trọng đang gặp

### 2.1. BUG: `_assign_opponents_to_seats()` không được gọi

Hàm `_assign_opponents_to_seats()` trong `env.py` được thiết kế để gán checkpoint cố định cho mỗi opponent seat mỗi ván. Tuy nhiên, hàm này **không bao giờ được gọi** trong `reset()` hay bất kỳ đâu. Hệ quả: `_seat_to_opponent` dict luôn rỗng, `_get_opponent_network_for_seat()` luôn fallback về random selection, opponent bị random lại **mỗi lượt** thay vì cố định mỗi ván.

**Impact**: Training thiếu consistency — agent đối mặt với opponent khác nhau mỗi lượt trong cùng một ván, làm tín hiệu học bị nhiễu.

### 2.2. Feature Scale mất cân bằng nghiêm trọng

Các feature scales hiện tại rất chênh lệch:

| Feature | Scale | Max Value | Nhận xét |
|---------|-------|-----------|----------|
| `lead_candidate_score` | **60.0** | **~150.0** | Cao nhất, có thể áp đảo mọi feature khác |
| `seq_order_penalty` | 20.0 | ~40.0+ | Penalty framework rất mạnh |
| `compliance` | 16.0 | 16.0 | Ép follow framework |
| `breaking` | 15.0 | ~30.0 | Phạt phá framework |
| `is_unbeatable` | 15.0 | 15.0 | Binary signal |
| `can_beat` | 15.0 | 15.0 | Binary signal |
| `blocking` | 12.0 | 12.0 | Binary signal |
| `next_player_danger` | 10.0 | 10.0 | Danger context |
| `combo_type_pref` | 3.0 | 3.0 | Rất yếu |
| `timing_pref` | 3.0 | **1.5** (luôn 0.5) | **Placeholder, không có thông tin** |

`lead_candidate_score` (max ~150.0) có thể áp đảo hoàn toàn các feature framework (max ~40.0) khi urgency cao. Nhưng khi urgency thấp, feature này = 0.0. Sự chênh lệch cực lớn tạo tín hiệu cực đoan khiến network khó hội tụ ổn định.

### 2.3. Agent không học được hành vi "chặn đối thủ sắp thắng"

**Biểu hiện**: Khi đối thủ còn 2 lá, agent vẫn chặn bằng lá yếu (6) thay vì lá mạnh (A, 2).

**Phân tích nguyên nhân**:
- Feature `next_player_danger` (scale 10.0) quá yếu so với framework features (scale 15-20)
- `lead_candidate_score` chỉ có giá trị khi `is_lead_situation = True`, nhưng `can_beat` là binary (1.0 cho mọi blocking move, kể cả lá 6 và lá A) nên không phân biệt được sức mạnh giữa các blocking moves
- Framework penalty (breaking + compliance + seq_order) có thể phạt nặng lá mạnh hơn lá yếu (vì lá mạnh thường nằm cuối sequence)

### 2.4. Reward Structure có vấn đề

| Vấn đề | Chi tiết |
|--------|----------|
| Chat reward quá lớn | +15.0 cho tứ quý/đôi thông, so với step reward clip [-1, 1] và final reward thường -1 đến +13 |
| `w2 = 0.0` | Hoàn toàn bỏ qua việc đối thủ giảm bài trong step reward. Comment ghi "reduced to 0.1" nhưng code là 0.0 |
| Dead code reward | `_calculate_situation_bonus()` (+0.5 khi block opponent < 3 cards) đã implement nhưng **không được gọi** |
| Dead code reward | `_calculate_winning_bonus()` và `_final_reward_with_shaping()` cũng không được sử dụng |

### 2.5. Value Network state representation kém

Value network dùng `move_features.mean(dim=0)` làm state representation — lấy **trung bình** tất cả feature vectors của mọi legal move. Đây là approximation rất thô: state embedding không phân biệt được tình huống có 3 legal moves vs 15 legal moves, mất thông tin về distribution.

### 2.6. Các bug logic nhỏ trong feature extraction

- **Double-counting game phase**: `_infer_game_phase()` tính `total_on_table = total_cards_left + hand_count`, nhưng `cards_left` đã bao gồm agent's cards. Agent hand bị đếm 2 lần, game phase luôn skew về "early" lâu hơn thực tế.
- **`_timing_preference()` là placeholder**: Luôn return 0.5, nhân scale 3.0 = luôn ra 1.5. Feature không chứa thông tin gì.

---

## 3. Phân tích kiến trúc

### 3.1. Network Architecture

PolicyNetwork và ValueNetwork đều là MLP: `57 → 128 → 128 → 1`.

| Đặc điểm | Hiện tại | Đánh giá |
|-----------|----------|----------|
| Loại network | MLP 3 lớp | Không có memory, mỗi turn quyết định độc lập |
| Cross-move comparison | Không có | Score từng move riêng lẻ, không so sánh với nhau |
| Hidden dim | 128 | Có thể thiếu capacity cho 57 features |
| Regularization | Không có | Không dropout, không layer norm |

### 3.2. Feature Space (57 dims)

| Nhóm | Dims | Flag | Vai trò |
|------|------|------|---------|
| Context (original) | 24 | Luôn bật | Cards_left, hand_size, combo_type, rank, blocking, can_beat, game_phase, ... |
| Framework-aware | 11 | `ENABLE_LEAD_QUALITY_FEATURES` | Priority, breaking, compliance, seq_order_penalty, lead_candidate, lead_waste |
| Multi-sequence | 12 | Luôn bật | So sánh 3 sequences (priority, breaking, compliance, seq_order x 3) |
| Card counting | 8 | `ENABLE_CARD_COUNTING_FEATURES` | Rank power, is_unbeatable, top ranks in hand, lose_lead_prob, next_player_danger |
| **Tổng** | **57** | | |

### 3.3. Self-play & Opponent Pool

| Config | Value | Đánh giá |
|--------|-------|----------|
| Pool size | 10 | Hợp lý |
| Checkpoint interval | 25,000 episodes | Khá dài, pool đa dạng chậm |
| Temperature range | 0.5 - 2.0 | Rộng, tốt cho diversity |
| Weight noise | std = 0.05 | Hợp lý |
| **Bug**: Seat assignment | Không hoạt động | Opponent random mỗi lượt |

### 3.4. PPO Hyperparameters

| Parameter | Value | Nguồn | Đánh giá |
|-----------|-------|-------|----------|
| Learning rate | **3e-4** (CLI) vs **5e-4** (config) | **Mâu thuẫn** — CLI ghi đè config | Nên thống nhất |
| PPO clip epsilon | 0.2 | Config | Standard |
| PPO epochs | 3 | Config | Hợp lý (giảm từ 6) |
| Batch size | 128 steps | Config | Nhỏ, tính theo steps không episodes |
| Entropy coef | 0.1 | Config | Cao — khuyến khích exploration |
| Gamma | 0.99 | Config | Standard |
| Normalize returns | False | Config | Giữ magnitude |
| Max grad norm | 1.0 | Config | Standard |

---

## 4. Reward System chi tiết

### Step Reward

`r_step = w1 * d_me - w2 * d_opp_avg + k * (phi_new - phi_old)`, clip [-1, 1]

- `w1 = 0.5`: Thưởng agent giảm bài
- `w2 = 0.0`: **Không thưởng khi opponent giảm bài** (bị vô hiệu hóa)
- `k = 0.1`: Potential shaping scale

### Chat Reward (per step, khi đánh chặt)

- Tứ quý: +15.0
- Ba đôi thông: +15.0 (TLMN only)
- Bốn đôi thông: +15.0 (TLMN only)

### Final Reward

- Thắng: `+sum(opponent remaining cards)` (thường 5-13 per opponent)
- Thua (full hand): -15.0
- Thua (giữ 2 hoặc tứ quý): `-3.0 * cards_left`
- Thua (bình thường): `-1.0 * cards_left`

### Reward chưa kích hoạt (dead code)

- `_calculate_situation_bonus()`: +0.5 khi block opponent < 3 cards
- `_calculate_winning_bonus()`: Bonus cho nước đi cuối khi thắng

---

## 6. Root Cause Analysis

### Tại sao agent chưa đạt mục tiêu sau 450k episodes?

**Nguyên nhân cốt lõi #1: Scale War**

Mỗi lần thấy agent chơi sai, ta tăng scale feature tương ứng. Lịch sử: `lead_candidate_scale` đi từ 15 → 22 → 30 → 60. Kết quả là một "cuộc chiến scale" giữa framework features (ép follow sequence) và lead features (ép chặn mạnh). Network không thể cân bằng tín hiệu chênh lệch cực lớn (1.5 vs 150.0).

Bản chất: Ta đang dùng feature scale để làm công việc mà lẽ ra **network weights** phải tự học. Scale cực lớn = ta đang "hardcode" hành vi vào features thay vì để network khám phá.

**Nguyên nhân cốt lõi #2: Thiếu reward signal trực tiếp**

Agent không nhận được feedback khi chặn đúng/sai. `_calculate_situation_bonus()` đã implement nhưng không được gọi. Agent chỉ biết kết quả cuối ván (thắng/thua), quá xa với hành vi cụ thể mỗi lượt.

**Nguyên nhân cốt lõi #3: Bug opponent pool**

`_assign_opponents_to_seats()` không được gọi. Opponents random mỗi lượt thay vì mỗi ván. Training signal bị nhiễu — agent không thể xây dựng model nhất quán về opponent behavior trong một ván.

**Nguyên nhân cốt lõi #4: Feature design tạo tín hiệu cực đoan**

Framework features (max 40-60) luôn phạt việc phá sequence, kể cả khi phá sequence là nước đi đúng. `lead_candidate_score` chỉ counter-act khi urgency cao, tạo ra tín hiệu binary (bình thường = 0, khẩn cấp = 150) thay vì gradient mượt. Network nhận "0 hoặc 150" thay vì dải giá trị liên tục, rất khó generalize.

---

## 7. Đề xuất hành động

### Phase 1: Fix Bugs & Stabilize (ngay lập tức)

| # | Hành động | Effort | Expected Impact |
|---|-----------|--------|-----------------|
| 1 | Gọi `_assign_opponents_to_seats()` trong `reset()` | 1 dòng | Training consistent |
| 2 | Kích hoạt `_calculate_situation_bonus()` trong `step()` | 2-3 dòng | Reward signal cho blocking |
| 3 | Giảm chat reward 15.0 → 5.0 | 1 dòng | Reward cân bằng |
| 4 | Fix `_infer_game_phase()` double-counting | 1 dòng | Game phase chính xác |
| 5 | Xóa hoặc implement `_timing_preference()` | 3-5 dòng | Bỏ feature rác |
| 6 | Thống nhất LR giữa CLI và config | 1 dòng | Config nhất quán |

**Sau Phase 1**: Train lại 100k episodes, đánh giá. Nếu cải thiện → tiếp Phase 2. Nếu không → vẫn tiếp Phase 2.

### Phase 2: Normalize Features (ưu tiên cao)

| # | Hành động | Effort | Expected Impact |
|---|-----------|--------|-----------------|
| 1 | Normalize toàn bộ features về [0, 1] | Trung bình | Network học hiệu quả |
| 2 | Bỏ scale escalation — đưa tất cả scale về 1.0-3.0 | Trung bình | Không còn scale war |
| 3 | Retrain từ đầu | — | Baseline mới, sạch |

### Phase 3: Architecture Improvement (nếu Phase 1+2 chưa đủ)

| # | Hành động | Effort | Expected Impact |
|---|-----------|--------|-----------------|
| 1 | Cải thiện Value Network (tách state vs move features) | Trung bình | PPO update chính xác |
| 2 | Tăng hidden dim (256 hoặc 512) | Thấp | Thêm capacity |
| 3 | Thử attention mechanism cho cross-move comparison | Cao | Đánh giá moves tương đối |
| 4 | Thử RNN/LSTM cho memory | Cao | Agent nhớ context ván bài |

---

## 8. Kết luận

Hệ thống có foundation tốt: PPO implementation đúng, game engine wrapper hoạt động, feature set phong phú, self-play mechanism có sẵn.

**3 vấn đề chính đang ngăn cản agent học hiệu quả**:

1. **Bug `_assign_opponents_to_seats()`** — opponents không cố định mỗi ván
2. **Scale war giữa features** — chênh 100x giữa feature mạnh nhất (150) và yếu nhất (1.5)
3. **Dead code reward** — bonus cho blocking đã implement nhưng chưa kích hoạt

Vấn đề **không phải thiếu features hay thiếu episodes**. Vấn đề là cách features được scale và reward signal bị thiếu/mất cân bằng. Fix Phase 1 (bugs + dead code) có thể tạo ra cải thiện đáng kể với effort rất thấp. Phase 2 (normalize features) là thay đổi quan trọng nhất để agent thực sự "tự học" thay vì bị ép bởi scale.
