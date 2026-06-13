# 🏠 Unified Tree-based Model + SHAP Analytics
### Dự đoán & Phân rã Giá Bất Động Sản Hà Nội

---

## 📌 Tổng quan

Module này xây dựng một mô hình **Gradient Boosting** (XGBoost / CatBoost) để dự đoán giá bất động sản (đơn vị: **tỷ đồng**) tại Hà Nội, sau đó áp dụng **SHAP** để phân rã giá trị dự đoán thành 2 nhóm đóng góp rõ ràng:

$$\text{Price} = \text{Baseline} + \underbrace{\sum \text{SHAP}_{\text{intrinsic}}}_{\text{đặc điểm vật lý}} + \underbrace{\sum \text{SHAP}_{\text{extrinsic}}}_{\text{vị trí \& tiện ích}}$$

---

## 📂 Cấu trúc thư mục

```
src/model/
├── config.py           # Cấu hình: nhóm feature, đường dẫn, hyperparameters
├── preprocessor.py     # Pipeline tiền xử lý (imputation, encoding, scaling)
├── trainer.py          # Huấn luyện & cross-validation XGBoost / CatBoost
├── shap_analyzer.py    # Tính SHAP values, phân rã theo nhóm, vẽ biểu đồ
├── evaluator.py        # Đánh giá mô hình (metrics + plots)
├── run_pipeline.py     # ⚡ Entry point — chạy toàn bộ pipeline
├── README.md           # Tài liệu này
├── saved_models/       # Model .pkl được lưu sau khi train
│   ├── xgboost_pipeline.pkl
│   ├── catboost_pipeline.pkl
│   └── preprocessor.pkl
└── outputs/            # Kết quả: plots, reports, CSV
    ├── evaluation_report.txt
    ├── shap_group_importance_summary.png
    ├── shap_group_bar.png
    ├── shap_beeswarm_intrinsic.png
    ├── shap_beeswarm_extrinsic.png
    ├── shap_group_scatter.png
    ├── shap_waterfall_sample*.png
    ├── prediction_vs_actual_*.png
    ├── residuals_*.png
    └── shap_decomposition_test.csv
```

---

## 🗂️ Nhóm Features

Toàn bộ 18 features được chia thành **2 nhóm** phục vụ phân tích SHAP:

### Nhóm 1 — Intrinsic (Đặc điểm vật lý)
| Feature | Mô tả |
|---------|-------|
| `area` | Diện tích (m²) |
| `property_type` | Loại bất động sản (`nha_mat_pho`, `nha_ngo`, ...) |
| `bedrooms_detail` | Số phòng ngủ |
| `bathrooms_detail` | Số phòng tắm |
| `floors` | Số tầng |
| `width_m` | Mặt tiền (m) |
| `depth_m` | Chiều sâu (m) |

### Nhóm 2 — Extrinsic (Vị trí & Tiện ích)
| Feature | Mô tả |
|---------|-------|
| `district` | Quận/huyện |
| `is_pho_co` | Thuộc khu phố cổ (0/1) |
| `dist_hoan_kiem_km` | Khoảng cách đến Hoàn Kiếm (km) |
| `dist_hospital_nearest_km` | Khoảng cách đến bệnh viện gần nhất (km) |
| `hospital_count_2km` | Số bệnh viện trong bán kính 2km |
| `dist_university_nearest_km` | Khoảng cách đến trường đại học gần nhất (km) |
| `university_count_2km` | Số trường đại học trong bán kính 2km |
| `dist_mall_nearest_km` | Khoảng cách đến trung tâm thương mại gần nhất (km) |
| `mall_count_2km` | Số TTTM trong bán kính 2km |
| `dist_lake_nearest_km` | Khoảng cách đến hồ gần nhất (km) |
| `lake_count_2km` | Số hồ trong bán kính 2km |

---

## ⚙️ Pipeline Xử lý

```
CSV Data
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Preprocessor (ColumnTransformer)               │
│  ┌──────────────────────────────────────────┐   │
│  │ Numeric features                         │   │
│  │   → SimpleImputer(median)                │   │
│  │   → StandardScaler                       │   │
│  ├──────────────────────────────────────────┤   │
│  │ Categorical features (property_type,     │   │
│  │                        district)         │   │
│  │   → SimpleImputer(most_frequent)         │   │
│  │   → OrdinalEncoder                       │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Model (sklearn Pipeline)                       │
│  ├─ XGBRegressor  (n_est=500, lr=0.05, d=6)    │
│  └─ CatBoostRegressor (iter=500, lr=0.05, d=6) │
│  → 5-Fold CV → chọn model tốt hơn (RMSE thấp) │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│  SHAP (TreeExplainer)                           │
│  → SHAP values per feature                      │
│  → Sum by group: Intrinsic | Extrinsic          │
│  → Plots: waterfall, beeswarm, scatter, bar     │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Cách chạy

> **Yêu cầu:** Chạy từ thư mục `src/model/`

```bash
cd src/model
```

### 1. Cài dependencies

```bash
pip install xgboost catboost shap scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Chạy toàn bộ pipeline (mặc định: log-transform target)

```bash
python run_pipeline.py
```

### 3. Chạy với raw price (không log-transform)

```bash
python run_pipeline.py --no-log
```

### 4. So sánh log-transform vs raw price trước khi train

```bash
python run_pipeline.py --compare-log-transform
```

### 5. Chỉ train (không SHAP)

```python
# Trong Python script hoặc notebook
from trainer import train
results = train(log_transform=True)
```

### 6. Chỉ chạy SHAP trên model đã train

```python
import joblib, pandas as pd
from shap_analyzer import SHAPAnalyzer
from preprocessor import get_feature_names_after_transform

pipeline = joblib.load("saved_models/xgboost_pipeline.pkl")
preprocessor = pipeline.named_steps["preprocessor"]
feat_names = get_feature_names_after_transform(preprocessor)

analyzer = SHAPAnalyzer(pipeline, feat_names, log_transform=True)

# Load dữ liệu
X_test = pd.read_csv("../../data/final_features_clean.csv")[feat_names]
shap_vals = analyzer.compute_shap_values(X_test.head(100))

# Xem phân rã cho 1 căn nhà
analyzer.print_sample_decomposition(shap_vals, X_test.head(100), sample_idx=0)

# Vẽ waterfall chart
analyzer.plot_waterfall(shap_vals, X_test, sample_idx=0)
```

---

## 📊 Outputs

Sau khi chạy xong, toàn bộ kết quả được lưu tại `outputs/`:

| File | Mô tả |
|------|-------|
| `evaluation_report.txt` | Metrics chi tiết (CV + test set) |
| `prediction_vs_actual_*.png` | Scatter: giá thực vs giá dự đoán |
| `residuals_*.png` | Phân tích phần dư |
| `feature_importance_*.png` | Feature importance (XGBoost gain) |
| `shap_group_importance_summary.png` | Tổng mean\|SHAP\| theo nhóm |
| `shap_group_bar.png` | Bar chart từng feature trong mỗi nhóm |
| `shap_beeswarm_intrinsic.png` | Beeswarm plot – Intrinsic features |
| `shap_beeswarm_extrinsic.png` | Beeswarm plot – Extrinsic features |
| `shap_group_scatter.png` | Scatter: Σ SHAP_intrinsic vs Σ SHAP_extrinsic |
| `shap_waterfall_sample*.png` | Waterfall cho 3 mẫu đại diện |
| `shap_decomposition_test.csv` | Bảng phân rã cho toàn bộ test set |

### Ví dụ output console

```
═══════════════════════════════════════════════════════
  Price Decomposition — Sample #0
═══════════════════════════════════════════════════════
  District      : Quận Hoàn Kiếm
  Property type : nha_mat_pho
  Area          : 35.0 m²
───────────────────────────────────────────────────────
  Baseline price          :   15.234 tỷ
  + Intrinsic contribution:  +0.8421  (log-scale)
  + Extrinsic contribution:  +1.2310  (log-scale)
───────────────────────────────────────────────────────
  Predicted price         :   42.500 tỷ đồng
═══════════════════════════════════════════════════════
```

---

## 🔧 Cấu hình tùy chỉnh

Chỉnh sửa `config.py` để thay đổi:

```python
# Thay đổi hyperparameters XGBoost
XGB_PARAMS = {
    "n_estimators" : 800,    # tăng số cây
    "max_depth"    : 5,
    "learning_rate": 0.03,
    ...
}

# Thay đổi target transform
LOG_TRANSFORM = False        # dùng raw price

# Thay đổi tỷ lệ train/test
TEST_SIZE = 0.15             # 85/15
```

---

## 📦 Dependencies

```
xgboost>=2.0
catboost>=1.2
shap>=0.45
scikit-learn>=1.4
pandas>=2.0
numpy>=1.24
matplotlib>=3.8
seaborn>=0.13
joblib>=1.3
```
