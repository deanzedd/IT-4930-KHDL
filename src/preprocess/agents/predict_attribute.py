#!/usr/bin/env python3
"""
predict_attribute.py
====================
Script xử lý 2 bước sau khi agent.py đã điền dữ liệu:

POST-PROCESSING Phase 1:
  - Đọc output_phase1.csv (kết quả của agent.py)
  - Nếu diện tích (area) == mặt tiền (width_m) → unfill width_m về NaN và ghi lại

Phase 2:
  - Lọc từ output_phase1.csv ra input_phase2.csv
    (chỉ giữ rows đủ TẤT CẢ các trường bắt buộc)
  - Dự đoán floors, width_m, depth_m theo logic:
      if floors == null:
          dự đoán floors dựa trên giá tiền, area, bedrooms_detail, bathrooms_detail
      else:
          if width_m == null và depth_m != null:  width_m = area / depth_m  (làm tròn lên)
          if width_m != null và depth_m == null:  depth_m = area / width_m  (làm tròn lên)
          if width_m == null và depth_m == null:  dùng regression để dự đoán cả 2
  - Xuất output_phase2.csv

Cách chạy:
  python predict_attribute.py \
      --phase1-output /mnt/disk1/theanh/LLM_uncer/data/output_phase1.csv \
      --data-dir     /mnt/disk1/theanh/LLM_uncer/data
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("predict_attribute")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─── Hằng số ──────────────────────────────────────────────────────────────────

# Các trường bắt buộc để một row được giữ lại cho Phase 2
REQUIRED_FIELDS = [
    "title", "description", "area", "district", "property_type",
    "bedrooms_detail", "bathrooms_detail", "floors",
    "price_billion", "width_m", "depth_m", "full_address",
]

# Ngưỡng sai số để coi area ≈ width_m (tính bằng %)
AREA_EQUALS_WIDTH_TOLERANCE = 0.01  # 1%


# ─── Post-processing Phase 1 ──────────────────────────────────────────────────

def postprocess_phase1(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Sau khi agent.py fill xong: nếu area ≈ width_m thì unfill width_m về NaN.
    Lý do: diện tích không thể bằng mặt tiền → đây là lỗi fill sai.
    """
    fixed_count = 0
    for idx, row in df.iterrows():
        area = row.get("area")
        width = row.get("width_m")

        if pd.isna(area) or pd.isna(width):
            continue

        try:
            area_val = float(area)
            width_val = float(width)
        except (ValueError, TypeError):
            continue

        if area_val == 0:
            continue

        # Nếu width_m ≈ area (sai số trong ngưỡng) → lỗi fill
        if abs(area_val - width_val) / area_val <= AREA_EQUALS_WIDTH_TOLERANCE:
            logger.warning(
                f"Row {idx}: area={area_val} ≈ width_m={width_val} → "
                f"unfill width_m về NaN"
            )
            df.at[idx, "width_m"] = float("nan")
            fixed_count += 1

    logger.info(f"Post-processing Phase 1: đã unfill {fixed_count} row có area ≈ width_m")
    return df


# ─── Tạo input_phase2.csv ─────────────────────────────────────────────────────

def create_input_phase2(
    df: pd.DataFrame, output_path: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    Lọc các row còn thiếu ít nhất 1 trong REQUIRED_FIELDS (trừ floors/width_m/depth_m
    vì chúng sẽ được predict ở Phase 2).
    
    Phase 2 predict: floors, width_m, depth_m
    → Các trường còn lại PHẢI đủ để row được giữ lại.
    """
    # Các trường phải có sẵn (không predict ở Phase 2)
    must_have = [f for f in REQUIRED_FIELDS if f not in ("floors", "width_m", "depth_m")]

    # Giữ lại rows có đủ must_have fields
    mask_complete = df[must_have].notna().all(axis=1)

    # Trong những row đó, chỉ giữ những row còn thiếu floors / width_m / depth_m
    # (những row đã đủ hết rồi không cần Phase 2)
    predict_fields = ["floors", "width_m", "depth_m"]
    mask_needs_predict = df[predict_fields].isna().any(axis=1)

    df_phase2_input = df[mask_complete & mask_needs_predict].copy()

    logger.info(
        f"input_phase2: {len(df_phase2_input)} rows cần dự đoán "
        f"(floors/width_m/depth_m) trong tổng {len(df)} rows"
    )

    df_phase2_input.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu: {output_path}")
    return df_phase2_input


# ─── Regression để dự đoán width_m, depth_m ──────────────────────────────────

def train_dimension_regression(df_train: pd.DataFrame, logger: logging.Logger):
    """
    Huấn luyện 2 mô hình linear regression từ dữ liệu đã có đủ width_m và depth_m.
    Features: area, price_billion, bedrooms_detail, bathrooms_detail
    Targets:  width_m, depth_m
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("Cần cài scikit-learn: pip install scikit-learn")
        return None, None, None

    features = ["area", "price_billion", "bedrooms_detail", "bathrooms_detail"]
    df_complete = df_train.dropna(subset=features + ["width_m", "depth_m"]).copy()

    if len(df_complete) < 10:
        logger.warning(
            f"Chỉ có {len(df_complete)} rows để train regression — kết quả có thể không chính xác"
        )

    if len(df_complete) == 0:
        logger.error("Không có dữ liệu đủ để train regression width/depth.")
        return None, None, None

    X = df_complete[features].values
    y_width = df_complete["width_m"].values
    y_depth = df_complete["depth_m"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_width = LinearRegression().fit(X_scaled, y_width)
    reg_depth = LinearRegression().fit(X_scaled, y_depth)

    score_w = reg_width.score(X_scaled, y_width)
    score_d = reg_depth.score(X_scaled, y_depth)
    logger.info(f"Regression R² — width_m: {score_w:.3f}, depth_m: {score_d:.3f}")

    return scaler, reg_width, reg_depth


def train_floors_regression(df_train: pd.DataFrame, logger: logging.Logger):
    """
    Huấn luyện mô hình dự đoán floors.
    Features: area, price_billion, bedrooms_detail, bathrooms_detail
    Target:   floors
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("Cần cài scikit-learn: pip install scikit-learn")
        return None, None

    features = ["area", "price_billion", "bedrooms_detail", "bathrooms_detail"]
    df_complete = df_train.dropna(subset=features + ["floors"]).copy()

    if len(df_complete) == 0:
        logger.error("Không có dữ liệu đủ để train regression floors.")
        return None, None

    X = df_complete[features].values
    y = df_complete["floors"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = LinearRegression().fit(X_scaled, y)
    logger.info(f"Regression R² — floors: {reg.score(X_scaled, y):.3f}")
    return scaler, reg


# ─── Phase 2: Predict ─────────────────────────────────────────────────────────

def predict_phase2(
    df_full: pd.DataFrame,
    df_needs: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Thực hiện dự đoán floors, width_m, depth_m theo logic trong note.txt.
    df_full : toàn bộ DataFrame (dùng để train regression)
    df_needs: subset các rows cần dự đoán
    """
    features = ["area", "price_billion", "bedrooms_detail", "bathrooms_detail"]

    # Train models từ toàn bộ dữ liệu đã có
    scaler_dim, reg_width, reg_depth = train_dimension_regression(df_full, logger)
    scaler_floors, reg_floors = train_floors_regression(df_full, logger)

    stats = {"floors": 0, "width_simple": 0, "depth_simple": 0, "dim_regression": 0}

    for idx, row in df_needs.iterrows():
        floors = row.get("floors")
        width  = row.get("width_m")
        depth  = row.get("depth_m")
        area   = row.get("area")

        # ── Dự đoán floors ──────────────────────────────────────────────────
        if pd.isna(floors):
            if scaler_floors is not None and reg_floors is not None:
                feat_vals = [row.get(f) for f in features]
                if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in feat_vals):
                    import numpy as np
                    x = np.array([feat_vals])
                    x_scaled = scaler_floors.transform(x)
                    pred_floors = reg_floors.predict(x_scaled)[0]
                    # Làm tròn lên số nguyên gần nhất, tối thiểu 1
                    pred_floors = max(1, math.ceil(pred_floors))
                    df_needs.at[idx, "floors"] = float(pred_floors)
                    stats["floors"] += 1
                    logger.info(f"Row {idx}: floors dự đoán = {pred_floors}")
                else:
                    logger.warning(f"Row {idx}: thiếu features để dự đoán floors, bỏ qua")
            # Cập nhật giá trị floors để dùng cho logic bên dưới
            floors = df_needs.at[idx, "floors"]

        # ── Dự đoán width_m / depth_m ───────────────────────────────────────
        if not pd.isna(floors):
            area_val = None
            if area is not None and not (isinstance(area, float) and math.isnan(area)):
                try:
                    area_val = float(area)
                except (ValueError, TypeError):
                    area_val = None

            width_missing = pd.isna(width)
            depth_missing = pd.isna(depth)

            if width_missing and not depth_missing:
                # width_m = area / depth_m, làm tròn lên
                try:
                    depth_val = float(depth)
                    if depth_val > 0 and area_val and area_val > 0:
                        pred_width = math.ceil(area_val / depth_val)
                        df_needs.at[idx, "width_m"] = float(pred_width)
                        stats["width_simple"] += 1
                        logger.info(f"Row {idx}: width_m = ceil({area_val}/{depth_val}) = {pred_width}")
                except (ValueError, TypeError):
                    pass

            elif not width_missing and depth_missing:
                # depth_m = area / width_m, làm tròn lên
                try:
                    width_val = float(width)
                    if width_val > 0 and area_val and area_val > 0:
                        pred_depth = math.ceil(area_val / width_val)
                        df_needs.at[idx, "depth_m"] = float(pred_depth)
                        stats["depth_simple"] += 1
                        logger.info(f"Row {idx}: depth_m = ceil({area_val}/{width_val}) = {pred_depth}")
                except (ValueError, TypeError):
                    pass

            elif width_missing and depth_missing:
                # Dùng regression
                if scaler_dim is not None and reg_width is not None:
                    feat_vals = [row.get(f) for f in features]
                    if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in feat_vals):
                        import numpy as np
                        x = np.array([feat_vals])
                        x_scaled = scaler_dim.transform(x)
                        pred_w = max(1.0, reg_width.predict(x_scaled)[0])
                        pred_d = max(1.0, reg_depth.predict(x_scaled)[0])
                        # Làm tròn lên 1 chữ số thập phân
                        pred_w = round(math.ceil(pred_w * 10) / 10, 1)
                        pred_d = round(math.ceil(pred_d * 10) / 10, 1)
                        df_needs.at[idx, "width_m"] = pred_w
                        df_needs.at[idx, "depth_m"] = pred_d
                        stats["dim_regression"] += 1
                        logger.info(f"Row {idx}: width_m={pred_w}, depth_m={pred_d} (regression)")
                    else:
                        logger.warning(f"Row {idx}: thiếu features để dự đoán width/depth, bỏ qua")

    logger.info(
        f"Phase 2 hoàn tất — "
        f"floors: {stats['floors']}, "
        f"width_simple: {stats['width_simple']}, "
        f"depth_simple: {stats['depth_simple']}, "
        f"dim_regression: {stats['dim_regression']}"
    )
    return df_needs


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-process Phase 1 và chạy Phase 2 dự đoán thuộc tính bất động sản"
    )
    parser.add_argument(
        "--phase1-output",
        default="/mnt/disk1/theanh/LLM_uncer/data/output_phase1.csv",
        help="File CSV output từ agent.py (Phase 1)",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/disk1/theanh/LLM_uncer/data",
        help="Thư mục chứa các file trung gian và output",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Bỏ qua bước post-process Phase 1 (area ≈ width_m check)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log_path = str(data_dir / "predict_attribute.log")
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("predict_attribute.py khởi động")
    logger.info(f"  Phase 1 output : {args.phase1_output}")
    logger.info(f"  Data dir       : {args.data_dir}")

    # ── Đọc output Phase 1 ────────────────────────────────────────────────────
    logger.info(f"Đọc file: {args.phase1_output}")
    df = pd.read_csv(args.phase1_output)
    logger.info(f"Loaded {len(df)} rows")

    # ── Post-process Phase 1: unfill area ≈ width_m ───────────────────────────
    if not args.skip_postprocess:
        logger.info("─── Post-processing Phase 1 ───")
        df = postprocess_phase1(df, logger)
        # Lưu lại Phase 1 sau khi đã fix
        fixed_phase1_path = str(data_dir / "output_phase1.csv")
        df.to_csv(fixed_phase1_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu output_phase1.csv (đã fix): {fixed_phase1_path}")
    else:
        logger.info("Bỏ qua post-processing Phase 1.")

    # ── Tạo input_phase2.csv ──────────────────────────────────────────────────
    logger.info("─── Tạo input_phase2.csv ───")
    input_phase2_path = str(data_dir / "input_phase2.csv")
    df_needs = create_input_phase2(df, input_phase2_path, logger)

    if len(df_needs) == 0:
        logger.info("Không có row nào cần dự đoán ở Phase 2. Kết thúc.")
        # Vẫn lưu output_phase2.csv từ phase 1 đã đủ dữ liệu
        output_phase2_path = str(data_dir / "output_phase2.csv")
        df.to_csv(output_phase2_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu output_phase2.csv: {output_phase2_path}")
        return

    # ── Phase 2: Dự đoán ─────────────────────────────────────────────────────
    logger.info("─── Phase 2: Dự đoán floors, width_m, depth_m ───")
    df_predicted = predict_phase2(df, df_needs, logger)

    # Merge kết quả dự đoán trở lại df gốc
    for idx in df_predicted.index:
        for field in ["floors", "width_m", "depth_m"]:
            val = df_predicted.at[idx, field]
            if not pd.isna(val):
                df.at[idx, field] = val

    # ── Lưu output_phase2.csv ─────────────────────────────────────────────────
    output_phase2_path = str(data_dir / "output_phase2.csv")
    df.to_csv(output_phase2_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu output_phase2.csv: {output_phase2_path}")

    # ── Thống kê ──────────────────────────────────────────────────────────────
    df_out = pd.read_csv(output_phase2_path)
    logger.info("─── Null counts sau Phase 2 ───")
    for f in ["floors", "width_m", "depth_m"]:
        if f in df_out.columns:
            null_count = df_out[f].isna().sum()
            logger.info(f"  {f}: {null_count} null còn lại")

    logger.info("predict_attribute.py hoàn thành!")


if __name__ == "__main__":
    main()
