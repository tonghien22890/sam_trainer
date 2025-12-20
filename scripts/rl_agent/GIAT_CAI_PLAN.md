## Kế hoạch cải tiến chiến thuật “giật cái” cho PPO

### 1. Mục tiêu tổng quan

- **Giữ lại ưu điểm hiện tại**: Agent vẫn ưu tiên chơi theo framework sequence (khung combo) vốn đang hoạt động tốt khi bài có khung đẹp.
- **Mở thêm khả năng tuỳ biến** trong 3 lớp quyết định:
  1. **Khi nào nên giật cái** (chọn thời điểm cướp lượt).
  2. **Giật cái bằng lá/combo nào** (chọn “lá giật cái” hợp lý: single 2 đẹp, bộ to phá khung, hạn chế đốt đôi 2/three 2 vô tội vạ).
  3. **Chơi gì sau khi đã giật cái** (tận dụng lượt để xả rác hiệu quả, nhưng vẫn tôn trọng khung khi cần).
- **Không phá vỡ cấu trúc sequence hiện tại**: các tín hiệu mới chỉ là “lớp mềm” bổ sung, không lấn át framework.

---

### 2. Nguyên tắc thiết kế

- **Framework là gốc**, PPO chỉ học *tinh chỉnh*:
  - Khi không có lý do đặc biệt → ưu tiên tuyệt đối framework.
  - Khi có tín hiệu đặc biệt (tay rác nhiều, đối thủ sắp về, v.v.) → cho phép lệch một chút.
- **Gating theo trạng thái**:
  - Các logic “giật cái” chỉ nên tác động mạnh khi `is_blocking == 0` (mình mở vòng / nắm lượt).
  - Khi đang block (`is_blocking == 1`) → framework + `can_beat` vẫn là tiếng nói chính.
- **Scale cẩn thận**:
  - Các feature/framework hiện tại (`priority`, `position`, `compliance`, `seq_order_penalty`) giữ scale cao hơn.
  - Các tín hiệu “giật cái / rác / lead quality” giữ scale nhỏ hơn để chỉ làm tie-break trong một số trạng thái.
- **Reward cuối không đổi**:
  - Final reward vẫn đo thắng/thua + số lá còn lại + phạt 2/four_kind như hiện tại.
  - Shaping chỉ thêm tín hiệu nhỏ, nếu thiết kế sai → PPO sẽ tự “bóp” weight các feature đó (do winrate giảm).

---

### 3. Phân lớp bài toán

#### 3.1. Lớp 1 – Thời điểm giật cái (WHEN)

Ý tưởng: chỉ nên giật cái khi:
- Tay có nhiều rác/combo yếu cần xả.
- Đối thủ chưa quá nguy hiểm (không ai sắp về).
- Việc giật cái mở ra khả năng xả được nhiều lá trong 1–2 lượt tiếp theo.

Thông tin sử dụng (không nhất thiết phải thêm feature mới ngay):
- `cards_left` của đối thủ (đã có).
- `hand_size`, `efficiency`, `framework_strength`, `breaking_severity` (đã có).
- Có thể dùng thêm các heuristics trong framework/`ComboAnalyzer` để ước lượng:
  - **Trash load**: tỉ lệ single/pair yếu trong tay.
  - **Coverability**: nếu đánh combo này để giật cái, trong 1–2 lượt kế có thể xả thêm bao nhiêu lá yếu.

Giai đoạn đầu (Phase 1) có thể **chưa implement lớp này**, tập trung vào “lá giật cái” trước cho rủi ro thấp.

#### 3.2. Lớp 2 – Lá/combo dùng để giật cái (WHAT)

Ưu tiên:
- **Đẹp nhất**: giật bằng **single 2**.
- Tiếp theo: những combo strength = 1 **nhưng không phải** đôi 2 / three 2 / four 2.
- Với combo: ưu tiên những bộ **to thực sự** phá thứ tự sequence có chủ đích (dọn đường cho khung), hơn là những bộ chỉ tốt để chặn.

Ý tưởng feature (Phase 1 – rủi ro thấp):
- **`lead_candidate_score`** (0–1):
  - Cao khi:
    - combo rất mạnh (strength ≈ 1) *và*
    - không phải đôi 2/three 2/four 2.
  - Trung bình cho single 2 (vì là option đẹp, nhưng cần tuỳ tình huống).
  - Thấp cho combo yếu/chỉ đủ block.
- **`lead_waste_penalty`** (0–1):
  - Cao khi combo chứa nhiều lá 2 hoặc là bộ nên để dành để block.
  - Thấp khi combo phù hợp để “đốt” trước nhằm mở bài.

Sử dụng:
- Khi `is_blocking == 0` (mở vòng/giật cái), policy nhìn thêm 2 scalar này.
- Scales nhỏ (ví dụ 3–4) để không lấn át framework.
- Không cần đổi reward: chính feature sẽ hướng PPO chọn lá giật cái hợp lý hơn.

#### 3.3. Lớp 3 – Sequence sau khi giật cái (AFTER)

Sau khi đã nắm lượt (vừa giật cái xong, bước tiếp theo):
- `is_blocking == 0`, `last_move` là chính mình.

Mong muốn:
- Khi **bàn an toàn** (không ai sắp về):
  - Ưu tiên xả rác: single/pair yếu, combo ít ảnh hưởng khung.
  - Tận dụng lượt để giảm tối đa `hand_size`.
- Khi **bàn nguy hiểm** (min_opponent_cards rất thấp):
  - Hạn chế xả rác vô tội vạ.
  - Ưu tiên các nước đi giữ tài nguyên block / giảm nguy cơ bị về sau đó.

Thông tin dùng (đa phần đã có):
- `is_blocking`, `cards_left`, `rank_value`, `efficiency`,
- `framework_strength`, `breaking_severity`, `can_beat`.

Shaping mềm (tuỳ chọn):
- Khi `is_blocking == 0` **và** không có threat cao:
  - Thưởng nhẹ cho:
    - `efficiency` cao (xả được nhiều lá),
    - `breaking_severity` thấp, `rank_value` thấp (xả rác).
- Khi threat cao:
  - Không thưởng xả rác, chỉ để final reward + bonus blocking dẫn hướng.

---

### 4. Lộ trình triển khai (phased plan)

#### Phase 1 – Tối ưu “lá giật cái” (rủi ro thấp)

Mục tiêu:
- Không đụng thêm feature dimension lớn, không đổi reward nhiều.
- Tập trung dạy PPO chọn **lá/combo giật cái hợp lý hơn**, trong khi framework vẫn giữ vai trò chính.

Việc làm:
1. Thiết kế `lead_candidate_score` và `lead_waste_penalty` trong framework/tầng feature:
   - Dựa trên combo type (single/pair/triple/four_kind),
   - Có/không chứa 2,
   - Strength của combo và vị trí trong sequence.
2. Thêm 1–2 dim feature tương ứng, với scale nhỏ (≈3–4).
3. Không thay đổi reward (giữ step reward + final reward như hiện tại).
4. Train thử (ví dụ 100k–200k episodes) và:
   - So sánh các lượt mở vòng: tỉ lệ single 2 / bộ to phá khung / đốt đôi 2.
   - Đảm bảo behavior trên bài có khung vẫn gần giống hiện tại.

Kỳ vọng:
- Khi có khung tốt, agent vẫn đánh rất giống bản hiện tại (vì framework vẫn chiếm ưu thế).
- Khi cần giật cái, chọn được lá/combo “đẹp” hơn (ít phá khung vô ích, ít đốt 2 vô duyên).

#### Phase 2 – Thời điểm giật cái + xử lý trash (rủi ro vừa)

Mục tiêu:
- Bổ sung hiểu biết về:
  - Khi nào cần giật cái để xả rác,
  - Khi nào không nên giật vì đối thủ đang sắp về.

Việc làm:
1. (Tuỳ chọn) Thêm 1–2 scalar nội bộ (không nhất thiết là feature public):
   - Trash load (tỉ lệ single/pair yếu),
   - Opponent threat level (dựa trên `cards_left`).
2. Dùng các scalar đó **trong reward shaping nhẹ**:
   - Bonus nhỏ khi:
     - Tay rác nhiều,
     - Threat thấp,
     - Agent giật cái và sau đó xả được nhiều lá yếu.
   - Không bonus (hoặc ngược lại) khi threat cao nhưng vẫn giật cái “liều”.
3. Theo dõi:
   - Winrate,
   - Số lá rác còn lại khi vào late game,
   - Behavior khi min_opponent_cards < 3.

Kỳ vọng:
- Agent biết “nhìn thời điểm” để giật cái, không chỉ nhìn strength combo.
- Vẫn không phá khung quá đà vì framework features vẫn mạnh hơn.

#### Phase 3 – Tune tinh & đánh giá dài hạn (rủi ro cao hơn, tuỳ nhu cầu)

Nếu Phase 1–2 cho kết quả tích cực:
- Xem xét:
  - Điều chỉnh nhẹ scale framework vs. lead/trash features,
  - Thử curriculum training (train trước 2 seats, sau đó 3–4 seats),
  - Thêm evaluation script ngoài training để test một số kịch bản “giật cái” điển hình.

Nếu Phase 1–2 không cải thiện:
- Cân nhắc:
  - Giữ nguyên framework-based (rule-based) cho phần này,
  - Khoanh vùng RL chỉ làm một số nhiệm vụ nhỏ hơn (vd: chọn giữa vài sequence candidate).

---

### 5. Tiêu chí đánh giá thành công

- **Không tệ hơn baseline**:
  - Khi không cần giật cái, agent đánh gần giống bản sequence hiện tại.
- **Cải thiện ở các tình huống đặc biệt**:
  - Trên tay nhiều rác + đối thủ còn nhiều lá:
    - Agent biết chọn lúc và lá giật cái hợp lý hơn → xả rác hiệu quả hơn.
  - Khi đối thủ sắp về:
    - Agent ít “giật cái ngu” bằng đôi 2 / bộ lớn vô nghĩa,
    - Tập trung block / chơi an toàn hơn.
- **Không cần quá nhiều episodes để thấy khác biệt**:
  - Trong khoảng 100k–200k episodes đã bắt đầu thấy pattern mới rõ ràng hơn baseline.


