# 🏠 House Price Prediction — Web UI

Demo web app dự đoán giá bất động sản Hà Nội sử dụng mô hình **XGBoost** đã được huấn luyện.

---

## 📁 Cấu trúc thư mục

```
src/Web/
├── app.py                  # Flask backend (entry point)
├── templates/
│   └── index.html          # Giao diện người dùng
├── static/                 # (tuỳ chọn) CSS/JS tĩnh
└── README.md               # File này
```

---

## ⚙️ Yêu cầu hệ thống

- Python 3.9+
- Model đã được train và lưu tại:
  ```
  src/model/saved_models/xgboost_pipeline.pkl
  ```

---

## 🚀 Cách chạy local (Deploy demo)

### Bước 1 — Cài dependencies

```bash
pip install flask joblib scikit-learn xgboost numpy pandas
```

### Bước 2 — Đảm bảo model đã được train

Nếu chưa có file `.pkl`, chạy pipeline trước:

```bash
cd /mnt/disk1/theanh/LLM_uncer
python src/model/run_pipeline.py
```

File model sẽ được lưu tại:
```
src/model/saved_models/xgboost_pipeline.pkl
```

### Bước 3 — Chạy web server

```bash
cd /mnt/disk1/theanh/LLM_uncer/src/Web
python app.py
```

Hoặc từ thư mục gốc:

```bash
cd /mnt/disk1/theanh/LLM_uncer
python src/Web/app.py
```

### Bước 4 — Mở trình duyệt

Truy cập: **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/`       | Giao diện chính |
| `POST` | `/predict` | Dự đoán giá nhà |
| `GET`  | `/health`  | Kiểm tra trạng thái server |

### Ví dụ gọi `/predict` bằng curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 60,
    "property_type": "nha_pho",
    "bedrooms_detail": 3,
    "bathrooms_detail": 2,
    "floors": 4,
    "width_m": 4.5,
    "depth_m": 14,
    "district": "Quận Đống Đa",
    "is_pho_co": 0,
    "dist_hoan_kiem_km": 3.5,
    "dist_hospital_nearest_km": 0.8,
    "hospital_count_2km": 3,
    "dist_university_nearest_km": 1.2,
    "university_count_2km": 2,
    "dist_mall_nearest_km": 1.5,
    "mall_count_2km": 1,
    "dist_lake_nearest_km": 0.5,
    "lake_count_2km": 2
  }'
```

**Response:**
```json
{
  "success": true,
  "price": 12.85,
  "price_low": 10.92,
  "price_high": 14.78,
  "unit": "tỷ đồng"
}
```

---

## 📊 Thông tin mô hình

| Metric | Giá trị |
|--------|---------|
| Algorithm | XGBoost |
| R² (test set) | 0.7235 |
| MAPE | 24.47% |
| Dữ liệu train | 3,827 BĐS Hà Nội |
| Features | 18 thuộc tính |

---

## 🔧 Tuỳ chỉnh

- **Đổi port**: Sửa dòng cuối trong `app.py`:
  ```python
  app.run(host="0.0.0.0", port=8080)  # đổi 5000 → 8080
  ```

- **Dùng model khác (sau khi ensemble tune)**:
  Cập nhật `MODEL_PATH` trong `app.py`:
  ```python
  MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "saved_models", "ensemble_model.pkl")
  ```

---

## 🛑 Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `ModuleNotFoundError: No module named 'flask'` | `pip install flask` |
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` |
| `FileNotFoundError: xgboost_pipeline.pkl` | Chạy `python src/model/run_pipeline.py` trước |
| `Port 5000 in use` | Đổi port hoặc `kill $(lsof -t -i:5000)` |
