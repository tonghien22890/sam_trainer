# Reward Design Documentation (Synced with Phase 4)

## Tổng quan

Hệ thống reward hiện tại sử dụng **Potential-based Reward Shaping** kết hợp với **Normalization** (chuẩn hóa theo số lá ban đầu) để đảm bảo độ ổn định khi training RL.

---

## 1. Step Reward (Reward ở mỗi bước)

### 1.1. Công thức
```python
r_step = (w1 * d_me - w2 * d_opp_avg + shaping)
```

**Các thành phần (Giá chuẩn trong code):**
- `w1 = 0.5`: Thưởng khi Agent giảm được số bài trên tay.
- `w2 = 0.1`: Phạt dựa trên mức trung bình đối thủ giảm bài (đang để thấp để agent tập trung vào bài mình).
- `k = 0.1`: Hệ số Potential function.
- `gamma = 0.99`: Discount factor.
- **Time Penalty**: Đã loại bỏ để tránh việc AI đánh bừa cho xong ván.

**Clip (ổn định training):** Step reward được **clip trong khoảng [-3.0, 3.0]** để situation bonus và chặt reward vẫn có trọng lượng tương đối so với final reward.

---

## 2. Final Reward (Reward khi ván kết thúc)

> [!IMPORTANT]
> Toàn bộ Final Reward đều được **Chia cho số lá ban đầu** (10 với Sâm, 13 với TLMN) để scale về khoảng [-1.5, 3.0].

### 2.1. Agent Thắng (ACL = 0)
`Reward = (Tổng số lá còn lại của tất cả đối thủ) / initial_hand_size`

### 2.2. Agent Thua (ACL > 0)

1. **Móm (Treo bài - Còn đủ 10/13 lá)**:
   `Reward = -15.0 / initial_hand_size` (Tương đương -1.5 hoặc -1.15)
2. **Thối Heo (Rank 12) hoặc Tứ Quý**:
   `Reward = (-3.0 * ACL) / initial_hand_size`
3. **Thua bình thường**:
   `Reward = (-1.0 * ACL) / initial_hand_size`

---

## 3. Bonus (Situation & Chặt)

Các bonus này được cộng vào reward tại bước tương ứng (không chuẩn hóa), giúp agent học block và chặt đúng lúc.

### 3.1. Situation Bonus (Block đối thủ sắp về)
- **Điều kiện:** Có ít nhất một đối thủ còn **&lt; 3 lá** và agent **chặn** (can_beat nước trước).
- **Giá trị:** `+0.5` mỗi lần thỏa điều kiện.

### 3.2. Chặt Reward
- **Điều kiện:** Agent đánh combo **chặt** (tứ quý, 3 đôi thông, 4 đôi thông) và **đè đúng nước 2 (Heo)** của đối thủ.
- **Giá trị:** `+15.0 / initial_hand_size` (Chuẩn hóa = +1.5 với Sâm, tương đương ăn 1 người Móm).
- **Lưu ý:** Chặn lá thường (đè combo không phải 2) không mang lại giá trị thưởng.

---

## 4. Phase 4: Strategic Failure Penalty (Mới)

Đây là hình phạt bổ sung (không chuẩn hóa) dùng để ép AI không được găm hàng mạnh khi đối thủ sắp về:

1. **Găm Heo (2)**: `-1.0 Phạt` (Nếu thua mà vẫn còn Heo trên tay).
2. **Găm Dây dài (>= 5 quân)**: `-0.5 Phạt`.

---

## 5. Tổng Reward mỗi ván

Reward **tổng** của một ván (dùng trong evaluation: *Avg Reward* = trung bình tổng reward các ván) là:

```
reward_van = final_reward + sum(step_r mỗi bước) + sum(situation_bonus) + sum(chat_reward) + strat_penalty
```

- `final_reward`: chỉ tính **một lần** khi ván kết thúc (đã chuẩn hóa).
- `step_r`: mỗi bước agent chơi (đã clip [-3, 3]).
- `situation_bonus`, `chat_reward`: cộng tại từng bước thỏa điều kiện.
- `strat_penalty`: chỉ khi agent **thua** (găm Heo / dây dài).

---

## 6. Bảng Tra cứu Tham số Thực tế

| Tham số | Giá trị | Trạng thái |
|---------|--------|-------|
| `w1` | 0.5 | Active |
| `w2` | 0.1 | Active (Giảm từ 0.2) |
| `k` | 0.1 | Active (Giảm từ 0.5) |
| `time_penalty` | 0.0 | **Disabled** |
| Step reward clip | [-3.0, 3.0] | Active |
| `Normalization` | Yes | `Value / HandSize` |
| Situation bonus | +0.5 | Khi block đối thủ &lt; 3 lá |
| Chặt reward | +15/HandSize | Chỉ khi chặt được 2 (Heo) |
| `Strat Penalty` | Yes | New in Phase 4 |




