"""
app.py
------
Flask backend for House Price Prediction UI.

Run:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# ── Resolve model path ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "saved_models", "xgboost_pipeline.pkl")

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────
print(f"[App] Loading model from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)
print("[App] ✅ Model loaded successfully!")

# ── Feature definitions (must match training config) ─────────────
INTRINSIC_FEATURES = [
    "area",
    "property_type",
    "bedrooms_detail",
    "bathrooms_detail",
    "floors",
    "width_m",
    "depth_m",
]

EXTRINSIC_FEATURES = [
    "district",
    "is_pho_co",
    "dist_hoan_kiem_km",
    "dist_hospital_nearest_km",
    "hospital_count_2km",
    "dist_university_nearest_km",
    "university_count_2km",
    "dist_mall_nearest_km",
    "mall_count_2km",
    "dist_lake_nearest_km",
    "lake_count_2km",
]

ALL_FEATURES = INTRINSIC_FEATURES + EXTRINSIC_FEATURES

# ── Categorical options ───────────────────────────────────────────
PROPERTY_TYPES = [
    "nha_mat_pho",
    "nha_pho",
    "nha",
    "can_ho",
    "biet_thu",
    "dat",
    "chung_cu",
    "khac",
]

DISTRICTS = [
    "Quận Hoàn Kiếm",
    "Quận Ba Đình",
    "Quận Đống Đa",
    "Quận Hai Bà Trưng",
    "Quận Hoàng Mai",
    "Quận Cầu Giấy",
    "Quận Long Biên",
    "Quận Thanh Xuân",
    "Quận Bắc Từ Liêm",
    "Quận Nam Từ Liêm",
    "Quận Hà Đông",
    "Quận Tây Hồ",
    "Huyện Gia Lâm",
    "Huyện Hoài Đức",
    "Huyện Thanh Trì",
    "Huyện Đông Anh",
    "Huyện Đan Phượng",
    "Huyện Thạch Thất",
    "Huyện Mỹ Đức",
    "Thị Xã Sơn Tây",
]


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        property_types=PROPERTY_TYPES,
        districts=DISTRICTS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build DataFrame with correct column order
        row = {
            # Intrinsic
            "area"              : float(data.get("area", 0)),
            "property_type"     : str(data.get("property_type", "nha_pho")),
            "bedrooms_detail"   : float(data.get("bedrooms_detail", 2)),
            "bathrooms_detail"  : float(data.get("bathrooms_detail", 1)),
            "floors"            : float(data.get("floors", 1)),
            "width_m"           : float(data.get("width_m", 4)),
            "depth_m"           : float(data.get("depth_m", 10)),
            # Extrinsic
            "district"                  : str(data.get("district", "Quận Đống Đa")),
            "is_pho_co"                 : int(data.get("is_pho_co", 0)),
            "dist_hoan_kiem_km"         : float(data.get("dist_hoan_kiem_km", 5)),
            "dist_hospital_nearest_km"  : float(data.get("dist_hospital_nearest_km", 1)),
            "hospital_count_2km"        : int(data.get("hospital_count_2km", 2)),
            "dist_university_nearest_km": float(data.get("dist_university_nearest_km", 2)),
            "university_count_2km"      : int(data.get("university_count_2km", 1)),
            "dist_mall_nearest_km"      : float(data.get("dist_mall_nearest_km", 2)),
            "mall_count_2km"            : int(data.get("mall_count_2km", 1)),
            "dist_lake_nearest_km"      : float(data.get("dist_lake_nearest_km", 1)),
            "lake_count_2km"            : int(data.get("lake_count_2km", 1)),
        }

        X = pd.DataFrame([row], columns=ALL_FEATURES)
        y_log = pipeline.predict(X)[0]

        # Inverse log1p transform → original price in tỷ đồng
        price = float(np.expm1(y_log))

        # Confidence range ±15%
        price_low  = round(price * 0.85, 2)
        price_high = round(price * 1.15, 2)
        price      = round(price, 2)

        return jsonify({
            "success"    : True,
            "price"      : price,
            "price_low"  : price_low,
            "price_high" : price_high,
            "unit"       : "tỷ đồng",
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "XGBoost"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
