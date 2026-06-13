# DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN HÀ NỘI 5/2026

## 📂 Cấu trúc dự án

```
KHDL/
├── data/                          # Dữ liệu thô và đã xử lý
├── src/
│   ├── preprocess/
│   │   ├── agents/                # Pipeline LLM điền dữ liệu thiếu
│   │   └── ...
│   └── model/                     # Model dự đoán giá + SHAP
├── req.txt                        # Dependencies tổng hợp
└── README.md                      # File này
```

---

## ⚙️ Cài đặt môi trường

### Bước 1 — Tạo môi trường ảo Python

#### 🪟 Windows (PowerShell / CMD)

```powershell
# Tạo môi trường ảo tên "venv"
python -m venv venv 
# Kích hoạt môi trường ảo
venv\Scripts\activate

# Kiểm tra phiên bản Python bên trong venv
python --version
```


#### 🍎 macOS / 🐧 Linux

```bash
# Tạo môi trường ảo tên "venv"
python3.10 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate

# Kiểm tra phiên bản Python bên trong venv
python --version
```

> Nếu chưa có Python 3.10, cài qua:
> - **macOS**: `brew install python@3.10`
> - **Ubuntu/Debian**: `sudo apt install python3.10 python3.10-venv`

---

### Bước 2 — Tải dependencies

Sau khi **đã kích hoạt môi trường ảo**, chạy lệnh sau từ thư mục gốc dự án:

```bash
pip install -r req.txt
```

#### Nâng cấp pip trước nếu cần

```bash
pip install --upgrade pip
pip install -r req.txt
```

---

### Bước 3 — Tắt môi trường ảo (khi xong việc)

```bash
deactivate
```

---

## 🚀 Chạy scripts

### Model dự đoán giá (XGBoost / CatBoost + SHAP)

```bash
# Kích hoạt venv trước
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Chạy toàn bộ pipeline
cd src/model
python run_pipeline.py

# So sánh log-transform vs raw price
python run_pipeline.py --compare-log-transform

# Dùng raw price (không log-transform)
python run_pipeline.py --no-log
```

> Chi tiết xem thêm tại [src/model/README.md](src/model/README.md)

---

### Agent điền dữ liệu thiếu (LLM Pipeline)

```bash
cd src/preprocess/agents

# Phase 1 — LLM fill
python agent.py \
    --input ../../data/HN_finalDATA.csv \
    --output ../../data/output_phase1.csv \
    --api-url http://localhost:8000 \
    --workers 8 --batch-size 50

# Phase 2 — Predict attributes (không cần GPU)
python predict_attribute.py \
    --phase1-output ../../data/output_phase1.csv \
    --data-dir ../../data
```

> Chi tiết cài vLLM + Docker xem tại [src/preprocess/agents/README.md](src/preprocess/agents/README.md)

---

## 📦 Danh sách dependencies (`req.txt`)

| Package | Phiên bản | Dùng cho |
|---------|-----------|----------|
| `pandas` | ≥ 2.0.0 | Xử lý dữ liệu |
| `numpy` | ≥ 1.24.0 | Tính toán số học |
| `scikit-learn` | ≥ 1.4.0 | Preprocessing, metrics, regression |
| `xgboost` | ≥ 2.0.0 | Model XGBoost |
| `catboost` | ≥ 1.2.0 | Model CatBoost |
| `shap` | ≥ 0.45.0 | Phân tích SHAP Intrinsic/Extrinsic |
| `matplotlib` | ≥ 3.8.0 | Biểu đồ |
| `seaborn` | ≥ 0.13.0 | Biểu đồ nâng cao |
| `joblib` | ≥ 1.3.0 | Lưu/load model |
| `aiohttp` | ≥ 3.9.0 | Agent HTTP async (vLLM API) |
| `tqdm` | ≥ 4.66.0 | Progress bar cho agent |
