## RL Training Setup & Deployment Guide

### 1. Cài Python, virtualenv, dependencies

**Bước 1 – Cài Python (3.8+)**

- **Windows**: tải từ `https://www.python.org/downloads/`, tick `Add Python to PATH` khi cài.
- **Linux (Ubuntu)**:
  ```bash
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip
  ```

**Bước 2 – Tạo virtualenv trong `model_build`**

```bash
cd model_build
python -m venv .venv          # hoặc python3 -m venv .venv

# Windows (PowerShell / CMD)
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

**Bước 3 – Cài dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Deploy RL training lên server/VM

Khi cần train RL (PPO) trên server/VM, **không bắt buộc** phải copy toàn bộ monorepo, nhưng **phải** giữ nguyên quan hệ thư mục để import đúng.

**Tối thiểu cần copy:** (giữ chung 1 thư mục gốc, ví dụ `AI-Sam/`)

- `game_engine/` – core game logic (Sam/TLMN)
- `ai_common/` – shared utilities (`SequenceEvaluator`, `ComboAnalyzer`, adapters, v.v.)
- `model_build/` – toàn bộ module model build (Two-Layer, RL agent, models, data)
- (Tuỳ chọn) `logs/` – nếu muốn giữ lịch sử log training

**Layout khuyến nghị trên server:**

```text
/path/to/AI-Sam/
├── game_engine/
├── ai_common/
└── model_build/
    ├── requirements.txt
    ├── scripts/
    │   ├── two_layer/
    │   └── rl_agent/
    └── models/
```

**Cài đặt nhanh trên server (Linux/macOS):**

```bash
cd /path/to/AI-Sam/model_build
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Sanity check: chạy thử 10 episodes
python scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 10 \
  --save_path models/rl_policy_sam_test.pt
```

---

### 3. Chạy training ẩn (background) và xem log

#### 3.1. Linux/macOS – `nohup` + log file

```bash
cd /path/to/AI-Sam
source model_build/.venv/bin/activate   # nếu đã tạo virtualenv

# Chạy training ẩn, ghi log ra file
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/rl_train_sam.log 2>&1 &

# Xem log realtime
tail -f model_build/logs/rl_train_sam.log
```

Gợi ý:
- Có thể dùng `screen` hoặc `tmux` nếu muốn giữ session lâu, xem log trực tiếp.
- Nên tạo thư mục `model_build/logs/` trước: `mkdir -p model_build/logs`.

#### 3.2. Windows – PowerShell / CMD (foreground nhưng ghi log)

```powershell
cd D:\Source-Code\AI-Sam
.\model_build\.venv\Scripts\activate

# Chạy và ghi log
python model_build\scripts\rl_agent\train_rl_core.py `
  --game_type sam `
  --episodes 50000 `
  --save_path model_build\models\rl_policy_sam.pt `
  > model_build\logs\rl_train_sam.log 2>&1

# Xem log
notepad model_build\logs\rl_train_sam.log
```

---

### 4. Quản lý process khi chạy ẩn (Linux/macOS)

Khi chạy training với `nohup ... &`, process sẽ chạy trong background. Dưới đây là các lệnh quản lý:

#### 4.1. Lưu Process ID (PID) khi khởi động

```bash
# Cách 1: Lưu PID vào biến
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/rl_train_sam.log 2>&1 &
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# Cách 2: Lưu PID vào file
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/rl_train_sam.log 2>&1 &
echo $! > model_build/logs/train_sam.pid
```

#### 4.2. Kiểm tra process có đang chạy không

```bash
# Tìm process theo tên
ps aux | grep train_rl_core.py

# Hoặc nếu đã lưu PID
PID=$(cat model_build/logs/train_sam.pid)
ps -p $PID

# Xem chi tiết (CPU, RAM, thời gian chạy)
ps -p $PID -o pid,pcpu,pmem,etime,cmd
```

#### 4.3. Dừng process

```bash
# Cách 1: Dùng PID đã lưu
PID=$(cat model_build/logs/train_sam.pid)
kill $PID

# Cách 2: Tìm và kill theo tên
pkill -f "train_rl_core.py"

# Cách 3: Kill mạnh (nếu process không dừng)
kill -9 $PID
# hoặc
pkill -9 -f "train_rl_core.py"
```

#### 4.4. Xem log realtime và theo dõi

```bash
# Xem log realtime (theo dõi liên tục)
tail -f model_build/logs/rl_train_sam.log

# Xem 100 dòng cuối
tail -n 100 model_build/logs/rl_train_sam.log

# Xem log và tìm kiếm
grep "Episode" model_build/logs/rl_train_sam.log | tail -20
grep "avg_reward" model_build/logs/rl_train_sam.log | tail -10
```

#### 4.5. Kiểm tra tiến độ training

```bash
# Xem dòng cuối cùng (thường có thông tin episode hiện tại)
tail -1 model_build/logs/rl_train_sam.log

# Đếm số dòng log (ước tính tiến độ)
wc -l model_build/logs/rl_train_sam.log

# Xem checkpoint đã được lưu chưa
ls -lh model_build/models/rl_policy_sam.pt
```

#### 4.6. Quản lý nhiều training jobs

```bash
# Chạy nhiều training với log riêng
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam --episodes 50000 \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/train_sam.log 2>&1 &
echo $! > model_build/logs/train_sam.pid

nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type tlmn --episodes 50000 \
  --save_path model_build/models/rl_policy_tlmn.pt \
  > model_build/logs/train_tlmn.log 2>&1 &
echo $! > model_build/logs/train_tlmn.pid

# Xem tất cả training đang chạy
ps aux | grep train_rl_core.py

# Dừng tất cả
pkill -f "train_rl_core.py"
```

#### 4.7. Kiểm tra resource usage (CPU, RAM)

```bash
# Xem CPU và RAM của process
PID=$(cat model_build/logs/train_sam.pid)
top -p $PID

# Hoặc dùng htop (nếu có cài)
htop -p $PID

# Xem tổng resource của tất cả training
ps aux | grep train_rl_core.py | awk '{sum+=$3; sum2+=$4} END {print "CPU: " sum "%, RAM: " sum2 "%"}'
```

#### 4.8. Resume training sau khi dừng

```bash
# Nếu training bị dừng giữa chừng, có thể resume từ checkpoint
nohup python model_build/scripts/rl_agent/train_rl_core.py \
  --game_type sam \
  --episodes 50000 \
  --load_path model_build/models/rl_policy_sam.pt \
  --save_path model_build/models/rl_policy_sam.pt \
  > model_build/logs/rl_train_sam.log 2>&1 &
echo $! > model_build/logs/train_sam.pid
```

---

### 5. Lưu ý

- Luôn chạy lệnh từ thư mục gốc project (`AI-Sam/`) hoặc `model_build/` như trong guide để import không lỗi.
- Với server có GPU, PyTorch sẽ tự dùng nếu `torch.cuda.is_available()` là `True`. Không cần đổi code.
- Khi train nối tiếp (resume), nhớ copy cả file `.pt` cũ và dùng `--load_path` trong `train_rl_core.py`.


Cài tmux
Cài Htop