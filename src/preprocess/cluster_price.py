"""
cluster_price.py
----------------
Lọc các hàng dữ liệu từ file CSV đầu vào chỉ giữ lại những hàng có
`price_billion` nằm trong khoảng [1.0, 150.0] (tỷ đồng, inclusive).

Cách dùng:
    python cluster_price.py --input <input_csv> [--output <output_csv>]
                            [--min_price <float>] [--max_price <float>]

Ví dụ:
    python cluster_price.py \
        --input /mnt/disk1/theanh/LLM_uncer/data/output_phase2.csv \
        --output /mnt/disk1/theanh/LLM_uncer/data/output_phase2_price_filtered.csv

Nếu không truyền --output, tên file output sẽ được tự động tạo ra bằng cách
thêm hậu tố `_price_filtered` vào tên file input, và lưu vào folder
/mnt/disk1/theanh/LLM_uncer/data/.
"""

import argparse
import os
import sys

import pandas as pd


# --------------------------------------------------------------------------- #
# Hằng số mặc định                                                            #
# --------------------------------------------------------------------------- #
DEFAULT_OUTPUT_DIR = "/mnt/disk1/theanh/LLM_uncer/data"
DEFAULT_MIN_PRICE = 1.0
DEFAULT_MAX_PRICE = 150.0
PRICE_COLUMN = "price_billion"


# --------------------------------------------------------------------------- #
# Hàm chính                                                                    #
# --------------------------------------------------------------------------- #
def filter_by_price(
    input_path: str,
    output_path: str,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
) -> None:
    """
    Đọc file CSV đầu vào, lọc các hàng có `price_billion` nằm trong
    khoảng [min_price, max_price] rồi ghi ra file CSV đầu ra.

    Parameters
    ----------
    input_path : str
        Đường dẫn tuyệt đối / tương đối đến file CSV đầu vào.
    output_path : str
        Đường dẫn tuyệt đối / tương đối đến file CSV đầu ra.
    min_price : float
        Giá trị nhỏ nhất (tỷ đồng), mặc định 1.0.
    max_price : float
        Giá trị lớn nhất (tỷ đồng), mặc định 150.0.
    """
    # --- Kiểm tra file đầu vào ------------------------------------------------
    if not os.path.isfile(input_path):
        print(f"[ERROR] Không tìm thấy file đầu vào: {input_path}")
        sys.exit(1)

    print(f"[INFO] Đọc file: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    total_rows = len(df)
    print(f"[INFO] Tổng số hàng ban đầu : {total_rows:,}")

    # --- Kiểm tra cột price_billion -------------------------------------------
    if PRICE_COLUMN not in df.columns:
        print(
            f"[ERROR] Không tìm thấy cột '{PRICE_COLUMN}' trong file CSV.\n"
            f"        Các cột hiện có: {list(df.columns)}"
        )
        sys.exit(1)

    # --- Chuyển sang kiểu số (coerce các giá trị không hợp lệ thành NaN) -----
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")

    # --- Lọc dữ liệu ----------------------------------------------------------
    mask = df[PRICE_COLUMN].between(min_price, max_price, inclusive="both")
    df_filtered = df[mask].copy()

    filtered_rows = len(df_filtered)
    removed_rows = total_rows - filtered_rows
    print(f"[INFO] Số hàng sau khi lọc [{min_price} – {max_price} tỷ] : {filtered_rows:,}")
    print(f"[INFO] Số hàng bị loại bỏ                                 : {removed_rows:,}")

    # --- Tạo thư mục đầu ra nếu chưa tồn tại ---------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Đã tạo thư mục đầu ra: {output_dir}")

    # --- Ghi file đầu ra ------------------------------------------------------
    df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã lưu file đầu ra  : {output_path}")

    # --- Thống kê nhanh -------------------------------------------------------
    if filtered_rows > 0:
        print("\n[STATS] Thống kê cột price_billion sau khi lọc:")
        stats = df_filtered[PRICE_COLUMN].describe()
        for stat_name, stat_val in stats.items():
            print(f"        {stat_name:8s}: {stat_val:.4f}")


# --------------------------------------------------------------------------- #
# Xây dựng output path mặc định                                               #
# --------------------------------------------------------------------------- #
def build_default_output_path(input_path: str) -> str:
    """
    Tạo tên file đầu ra dựa trên tên file đầu vào.
    Ví dụ: output_phase2.csv  →  output_phase2_price_filtered.csv
    File được lưu vào DEFAULT_OUTPUT_DIR.
    """
    basename = os.path.basename(input_path)          # output_phase2.csv
    name, ext = os.path.splitext(basename)           # output_phase2 | .csv
    out_name = f"{name}_price_filtered{ext}"         # output_phase2_price_filtered.csv
    return os.path.join(DEFAULT_OUTPUT_DIR, out_name)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lọc các hàng có price_billion nằm trong khoảng [min_price, max_price] "
            "từ file CSV đầu vào và lưu kết quả ra file CSV đầu ra."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="INPUT_CSV",
        help="Đường dẫn đến file CSV đầu vào. "
             "Ví dụ: /mnt/disk1/theanh/LLM_uncer/data/output_phase2.csv",
    )
    parser.add_argument(
        "--output", "-o",
        required=False,
        default=None,
        metavar="OUTPUT_CSV",
        help=(
            "Đường dẫn đến file CSV đầu ra. "
            f"Mặc định: <tên_file_input>_price_filtered.csv lưu trong {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--min_price",
        type=float,
        default=DEFAULT_MIN_PRICE,
        metavar="FLOAT",
        help=f"Giá trị nhỏ nhất của price_billion (mặc định: {DEFAULT_MIN_PRICE}).",
    )
    parser.add_argument(
        "--max_price",
        type=float,
        default=DEFAULT_MAX_PRICE,
        metavar="FLOAT",
        help=f"Giá trị lớn nhất của price_billion (mặc định: {DEFAULT_MAX_PRICE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Xác định output path
    output_path = args.output if args.output else build_default_output_path(args.input)

    # Kiểm tra khoảng giá hợp lệ
    if args.min_price >= args.max_price:
        print(
            f"[ERROR] min_price ({args.min_price}) phải nhỏ hơn max_price ({args.max_price})."
        )
        sys.exit(1)

    filter_by_price(
        input_path=args.input,
        output_path=output_path,
        min_price=args.min_price,
        max_price=args.max_price,
    )


if __name__ == "__main__":
    main()
