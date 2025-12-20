# Feature Scales Analysis

## Tổng kết Feature Scales hiện tại

### 1. Context Features (Original) - 22-24 dims

| Feature | Scale | Range/Notes |
|---------|-------|-------------|
| Combo counts (6 dims) | Normalized | [0, 1] - normalized by total legal moves |
| Cards left (4 dims) | Normalized | [0, 1] - normalized by max 13 cards |
| Hand size (1 dim) | Normalized | [0, 1] - normalized by max 13 cards |
| Combo type onehot (7 dims) | 1.0 | Binary (0 or 1) |
| Rank value (1 dim) | Normalized | [0, 1] - normalized by max 12 |
| Combo length (1 dim) | Normalized | [0, 1] - normalized by max 10 |
| Efficiency (1 dim) | 1.0 | [0, 1] - cards played / hand size |
| Urgency/Phase (1 dim) | 1.0 | [0.1, 0.5, 1.0] |
| **Blocking** | **12.0** | [0, 12] - binary * 12 |
| **Can beat** | **15.0** | [0, 15] - binary * 15 |
| Efficiency block (if enabled) | 1.0 | [0, 1] - efficiency * is_blocking |
| Efficiency lead (if enabled) | 1.0 | [0, 1] - efficiency * (1 - is_blocking) |

**Đánh giá**: Context features có scale hợp lý, chủ yếu normalized [0, 1]. Chỉ có `blocking` và `can_beat` có scale cao (12-15) để nhấn mạnh tầm quan trọng.

---

### 2. Framework-Aware Features - 9 dims

| Feature | Scale | Range/Notes | Impact |
|---------|-------|-------------|--------|
| Priority score | **5.0** | [0, 5] - reduced from 15.0 | 🟡 Medium |
| Breaking severity | **15.0** | [0, -30] - negative, penalty up to -30 | 🔴 **HIGH** (penalty) |
| Framework strength | **8.0** | [0, 8] | 🟡 Medium-High |
| Position | **12.0** | [0, 12] | 🟠 High |
| Combo type preference | **3.0** | [0, 3] | 🟢 Low |
| Rank preference | **4.0** | [0, 4] | 🟢 Low |
| Timing preference | **3.0** | [0, 3] | 🟢 Low |
| Sequence compliance | **16.0** | [0, 16] | 🔴 **HIGH** |
| Sequence order penalty | **20.0** | [0, -40+] - negative, major penalty | 🔴 **VERY HIGH** (penalty) |

**Đánh giá**: 
- ⚠️ **Vấn đề nghiêm trọng**: `seq_order_penalty_scale = 20.0` là scale cao nhất, phạt rất nặng khi đánh sai thứ tự
- `compliance_scale = 16.0` cũng rất mạnh, khuyến khích follow framework
- `breaking_scale = 15.0` phạt khi phá framework
- `position_scale = 12.0` khuyến khích đánh đúng vị trí

**Tổng impact framework**: Rất mạnh, ép agent follow framework nghiêm ngặt.

---

### 3. Lead Quality Features (Optional) - 2 dims

| Feature | Scale | Range/Notes | Impact |
|---------|-------|-------------|--------|
| Lead candidate score | **15.0** | [0, 15] - max when single 2 + urgency | 🟠 High |
| Lead waste penalty | **12.0** | [0, -12] - negative, penalty for pair/triple/four 2 | 🟠 High (penalty) |

**Range thực tế**:
- `lead_candidate_score`: 
  - Single 2: 0.6-1.0 → scaled: **9.0-15.0**
  - Strong combo: 0.5-0.8 → scaled: **7.5-12.0**
  - Weak combo: 0.0-0.6 → scaled: **0.0-9.0**
- `lead_waste_penalty`: 
  - Pair/triple/four 2: 1.0 → scaled: **-12.0** (giảm xuống -6.0 to -7.8 khi urgency cao)

**Đánh giá**: 
- Scale **15.0** cho lead candidate là khá mạnh, nhưng vẫn thua `seq_order_penalty` và `compliance`
- Khi agent muốn giật cái bằng single 2, có thể bị:
  - `+15.0` (lead candidate, single 2, urgency=1.0)
  - `-20.0 * penalty` (seq_order_penalty, có thể -8.8 sau khi giảm 56%)
  - `-15.0 * breaking` (nếu phá framework)
  - `-16.0 * (1 - compliance)` (nếu không comply)
  
  → **Tổng có thể vẫn âm**, agent sẽ không giật cái!

---

### 4. Multi-Sequence Features - 12 dims (3 sequences × 4 dims)

| Feature | Scale | Range/Notes | Impact |
|---------|-------|-------------|--------|
| Priority (per sequence) | **2.0** | [0, 2] × 3 | 🟢 Low |
| Breaking (per sequence) | **2.0** | [0, -4] × 3 - negative | 🟢 Low |
| Position (per sequence) | **2.0** | [0, 2] × 3 | 🟢 Low |
| Compliance (per sequence) | **2.0** | [0, 2] × 3 | 🟢 Low |

**Đánh giá**: Multi-sequence features có scale thấp (2.0), ảnh hưởng nhẹ, chủ yếu để cung cấp thông tin bổ sung.

---

## Phân tích tổng hợp

### Scale Ranking (từ cao đến thấp):

1. **🔴 Sequence order penalty: 20.0** (penalty, có thể lên đến -40+)
2. **🔴 Compliance: 16.0** (reward, up to +16)
3. **🟠 Breaking severity: 15.0** (penalty, up to -30)
4. **🟠 Lead candidate: 15.0** (reward, up to +15)
5. **🟠 Can beat: 15.0** (reward, binary +15)
6. **🟠 Lead waste: 12.0** (penalty, up to -12)
7. **🟠 Position: 12.0** (reward, up to +12)
8. **🟠 Blocking: 12.0** (reward, binary +12)
9. **🟡 Framework strength: 8.0** (reward, up to +8)
10. **🟢 Priority: 5.0** (reward, up to +5, đã giảm từ 15.0)
11. **🟢 Rank preference: 4.0** (reward, up to +4)
12. **🟢 Combo type preference: 3.0** (reward, up to +3)
13. **🟢 Timing preference: 3.0** (reward, up to +3)
14. **🟢 Multi-sequence: 2.0** (reward/penalty, up to ±2)

### Vấn đề chính:

**Khi agent muốn "giật cái" bằng single 2:**

**Trường hợp tốt nhất (urgency cao, penalty giảm 80%):**
- Lead candidate: **+15.0** (single 2, urgency=1.0)
- Seq order penalty: **-4.0** (penalty gốc -1.0, giảm 80% → -0.2, scaled -4.0)
- Breaking (giả sử nhẹ): **-3.0** (breaking=0.2 × 15)
- Compliance (giả sử 50%): **-8.0** (compliance=0.5 × 16)
- **Tổng: +15.0 - 4.0 - 3.0 - 8.0 = 0.0** ⚠️ Cân bằng!

**Trường hợp xấu hơn (urgency thấp, không giảm penalty):**
- Lead candidate: **+9.0** (single 2, urgency=0.1)
- Seq order penalty: **-20.0** (penalty gốc -1.0, không giảm)
- Breaking: **-7.5** (breaking=0.5 × 15)
- Compliance: **-8.0** (compliance=0.5 × 16)
- **Tổng: +9.0 - 20.0 - 7.5 - 8.0 = -26.5** ❌ Rất âm!

### Kết luận:

**Framework features quá mạnh**, đặc biệt là:
- `seq_order_penalty_scale = 20.0` → phạt rất nặng khi đánh sai thứ tự
- `compliance_scale = 16.0` → khuyến khích follow framework quá mạnh
- `breaking_scale = 15.0` → phạt khi phá framework

**Lead quality features (`lead_candidate_scale = 15.0`) không đủ mạnh** để vượt qua các penalty trên, đặc biệt khi urgency thấp hoặc khi phá framework nhiều.

**Vấn đề với giảm scales:**
- ❌ Đã thử giảm scales xuống 10-12 nhưng agent đánh rất tệ
- ❌ Agent không giữ được combo đúng, thường xuyên phá combo sequence
- ✅ Kết luận: Framework scales phải giữ nguyên cao để đảm bảo agent follow framework tốt

**Giải pháp thay thế (thay vì giảm framework scales):**

### Option 1: Tăng `lead_candidate_scale` mạnh hơn
- Hiện tại: `lead_candidate_scale = 15.0`
- Đề xuất: **20.0-25.0** để vượt qua các penalty framework khi urgency cao
- Single 2 + urgency=1.0: 1.0 × 25.0 = **+25.0** (thay vì +15.0)

### Option 2: Cải thiện `lead_candidate_score` calculation
- Hiện tại: Single 2 = 0.6 + (urgency × 0.4) = **0.6-1.0**
- Đề xuất: 
  - Base cao hơn: `0.8 + (urgency × 0.2)` = **0.8-1.0** (tăng base)
  - Hoặc: `0.7 + (urgency × 0.5)` = **0.7-1.2** (cho phép > 1.0 khi urgency rất cao)
- Kết hợp với scale 20.0 → **+16.0-+24.0**

### Option 3: Giảm penalty mạnh hơn cho single 2 khi urgency cao
- Hiện tại: Giảm 56-80% seq_order_penalty khi single 2 + urgency >= 0.7
- Đề xuất: 
  - Giảm 90-100% (gần như xóa penalty) khi single 2 + urgency >= 0.7
  - Hoặc: Giảm cả `breaking_penalty` và `compliance_penalty` khi single 2 + urgency cao

### Option 4: Kết hợp (Recommended)
- Tăng `lead_candidate_scale`: 15.0 → **22.0-25.0**
- Cải thiện `lead_candidate_score`: Cho phép > 1.0 khi urgency rất cao (min_opp <= 2)
- Giảm penalty mạnh hơn: 90-100% thay vì 56-80%

**Tính toán mới (Option 4 - urgency rất cao, min_opp <= 2):**
- Lead candidate: **+22.0** (single 2, urgency=1.0, score=1.0, scale=22.0)
- Seq order penalty: **-2.0** (penalty gốc -1.0, giảm 90% → -0.1, scaled -2.0)
- Breaking (giả sử nhẹ, giảm 50% khi urgency cao): **-1.5** (breaking=0.2 × 15 × 0.5)
- Compliance (giảm 50% khi urgency cao): **-4.0** (compliance=0.5 × 16 × 0.5)
- **Tổng: +22.0 - 2.0 - 1.5 - 4.0 = +14.5** ✅ Dương rõ ràng!

**Khuyến nghị cuối cùng:**
- ✅ **Giữ nguyên** framework scales (20.0, 16.0, 15.0)
- ✅ **Tăng** `lead_candidate_scale`: 15.0 → **22.0-25.0**
- ✅ **Cải thiện** `lead_candidate_score` calculation để đạt > 1.0 khi urgency rất cao
- ✅ **Tăng** penalty reduction: 80% → **90-95%** cho single 2 khi urgency >= 0.7

