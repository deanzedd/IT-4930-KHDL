# Agent Điền Dữ Liệu Bất Động Sản Hà Nội

Pipeline 2 phase sử dụng Qwen2.5-3B-Instruct (vLLM) + regression để tự động điền và dự đoán các trường dữ liệu còn thiếu trong file CSV bất động sản.

```
Input CSV
    │
    ▼
[Phase 1]  agent.py          → LLM fill: area, bedrooms, bathrooms,
    │                           floors, width_m, depth_m, full_address
    │                         → Post-process: unfill nếu area ≈ width_m
    │                         → output_phase1.csv
    ▼
[Phase 2]  predict_attribute.py  → Lọc ra input_phase2.csv
                                 → Dự đoán floors / width_m / depth_m
                                   bằng math + regression (không dùng LLM)
                                 → output_phase2.csv
```

---

## Phần 1: Host LLM Local với vLLM (Docker)

### 1.1 Yêu cầu hệ thống

- GPU NVIDIA (RTX 3080 10GB trở lên)
- Docker đã cài đặt
- NVIDIA Container Toolkit đã cài đặt
- ~6GB dung lượng tải model từ HuggingFace

### 1.2 Kiểm tra GPU trước khi chạy

```bash
# Xem danh sách GPU và trạng thái
nvidia-smi

# Kiểm tra GPU theo index cụ thể (0, 1, 2...)
nvidia-smi -L
```

> ⚠️ **Lưu ý quan trọng**: Nếu server có GPU bị lỗi phần cứng (hiển thị `Unknown Error` trong nvidia-smi),
> bạn PHẢI chỉ định rõ GPU theo index để tránh lỗi khi khởi động container.
> Ví dụ server lab có GPU 3 bị lỗi → chỉ dùng GPU 0 và 1.

### 1.3 Kiểm tra NVIDIA Container Toolkit

```bash
# Kiểm tra nvidia runtime đã được cài chưa
docker info | grep -i runtime
# Kết quả phải có: Runtimes: ... nvidia ...
```

Nếu chưa cài, xem hướng dẫn tại: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### 1.4 Chạy vLLM Server

#### Cách 1: Dùng 1 GPU (tensor-parallel-size 1)

```bash
docker run --runtime nvidia \
    --gpus '"device=0"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

#### Cách 2: Dùng 2 GPU (tensor-parallel-size 2) — KHUYẾN NGHỊ

```bash
docker run --runtime nvidia \
    --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2
```

> **Giải thích `--gpus '"device=0,1"'`**: Chỉ định rõ GPU index 0 và 1 (bỏ qua GPU lỗi).
> Dấu nháy kép bên trong là bắt buộc để Docker hiểu đúng cú pháp.

#### Chạy trong TMUX (khuyến nghị cho server lab)

```bash
# Tạo session mới
tmux new -s vllm

# Chạy lệnh Docker bên trong tmux
docker run --runtime nvidia \
    --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2

# Detach khỏi tmux (giữ server chạy ngầm): Ctrl+B, sau đó nhấn D
# Attach lại session: tmux attach -t vllm
```

### 1.5 Kiểm tra server đang chạy

```bash
# Kiểm tra danh sách model đang serve
curl http://localhost:8000/v1/models

# Test thử một request (Vietnamese)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Nhà 3 phòng ngủ 2 tắm thì bedrooms_detail là bao nhiêu? Trả lời chỉ 1 số."}],
    "max_tokens": 10
  }'
```

Server sẵn sàng khi bạn thấy output như:
```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Phần 2: Chạy Pipeline 2 Phase

### 2.1 Cài đặt dependencies

```bash
cd /mnt/disk1/theanh/LLM_uncer/src/preprocess/agents
pip install -r requirements.txt
pip install scikit-learn  # Cần thêm cho Phase 2 regression
```

---

### Phase 1 — LLM Fill (agent.py)

#### 2.2 Chạy agent (trong TMUX session riêng)

```bash
# Tạo session tmux mới cho agent
tmux new -s agent

# Chạy agent — output là output_phase1.csv
python agent.py \
    --input /mnt/disk1/theanh/LLM_uncer/data/HN_finalDATA.csv \
    --output /mnt/disk1/theanh/LLM_uncer/data/output_phase1.csv \
    --api-url http://localhost:8001 \
    --workers 8 \
    --batch-size 50

# Detach: Ctrl+B, D
# Reattach: tmux attach -t agent
```

> **Lưu ý port**: Nếu port 8000 đã bị chiếm bởi service khác,
> dùng `-p 8001:8000` trong lệnh docker và `--api-url http://localhost:8001` ở đây.

#### 2.3 Theo dõi tiến độ

```bash
# Xem log real-time
tail -f /mnt/disk1/theanh/LLM_uncer/data/agent_run.log

# Kiểm tra số hàng đã xử lý
wc -l /mnt/disk1/theanh/LLM_uncer/data/output_phase1.csv
```

#### 2.4 Resume nếu bị gián đoạn

Agent tự động checkpoint. Nếu bị dừng giữa chừng, chỉ cần chạy lại **cùng lệnh** — agent sẽ skip các hàng đã xử lý.

---

### Phase 2 — Predict Attributes (predict_attribute.py)

Sau khi `agent.py` chạy xong và tạo ra `output_phase1.csv`, chạy:

```bash
# Chạy trong tmux session riêng hoặc trực tiếp (không cần GPU)
python predict_attribute.py \
    --phase1-output /mnt/disk1/theanh/LLM_uncer/data/output_phase1.csv \
    --data-dir      /mnt/disk1/theanh/LLM_uncer/data
```

Script sẽ tự động:
1. **Post-process Phase 1**: Nếu `area ≈ width_m` → unfill `width_m` về NaN (tránh lỗi fill sai)
2. **Lọc** ra `input_phase2.csv` (chỉ rows đủ các trường bắt buộc, còn thiếu floors/width_m/depth_m)
3. **Dự đoán** theo logic:
   - `floors` còn null → regression từ giá, diện tích, số phòng
   - `width_m` null, `depth_m` có → `width_m = ceil(area / depth_m)`
   - `width_m` có, `depth_m` null → `depth_m = ceil(area / width_m)`
   - Cả 2 null → regression
4. **Xuất** `output_phase2.csv`

#### Theo dõi log Phase 2

```bash
tail -f /mnt/disk1/theanh/LLM_uncer/data/predict_attribute.log
```

---

## Cấu trúc file

```
agents/
├── README.md                ← File này
├── agent.py                 ← Phase 1: LLM fill dữ liệu (không sửa)
├── predict_attribute.py     ← Phase 2: Post-process + Predict thuộc tính
├── requirements.txt         ← Dependencies Python
└── prompt.md                ← Ghi chú prompt
```

## Output

| File | Phase | Mô tả |
|---|---|---|
| `output_phase1.csv` | 1 | Kết quả sau khi LLM fill xong + đã fix area≈width_m |
| `input_phase2.csv` | 2 | Subset rows đủ dữ liệu bắt buộc, cần predict floors/width/depth |
| `output_phase2.csv` | 2 | File hoàn chỉnh cuối cùng sau dự đoán |
| `agent_run.log` | 1 | Log chi tiết agent.py (thành công / thất bại / skip) |
| `predict_attribute.log` | 2 | Log chi tiết predict_attribute.py |
| `checkpoint.json` | 1 | Checkpoint để resume agent.py nếu bị gián đoạn |
