"""
alex.py - Công cụ tương tác với dữ liệu bất động sản Hà Nội
File: data/hanoi_final_filtered_features.csv

Cách dùng:
    python alex.py                  # Chạy chế độ tương tác (menu)
    python alex.py --info 0         # Xem thông tin hàng đầu tiên (index 0)
    python alex.py --desc 0         # Xem description hàng đầu tiên
    python alex.py --search Hoàn Kiếm  # Tìm kiếm theo quận
    python alex.py --top 5          # Xem 5 nhà đắt nhất
    python alex.py --cheap 5        # Xem 5 nhà rẻ nhất
    python alex.py --stats          # Thống kê tổng quan
    # Dùng file CSV mặc định, tự động tạo file *_complete.csv
    python alex.py --filter
    # Chỉ định file CSV đầu vào và đường dẫn output tùy chỉnh
    python alex.py --file data/input.csv --filter --output data/output_clean.csv
"""

import sys
import io
import argparse
import os

# Fix encoding cho Windows terminal (tranh loi UnicodeEncodeError voi tieng Viet/emoji)
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# ─── Cấu hình đường dẫn ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "hanoi_final_filtered_features.csv")


# ─── Load dữ liệu ───────────────────────────────────────────────────────────
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Đọc file CSV và trả về DataFrame."""
    if not os.path.exists(path):
        print(f"[LỖI] Không tìm thấy file: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    return df


# ─── Các hàm truy xuất dữ liệu ──────────────────────────────────────────────
def get_row(df: pd.DataFrame, index: int) -> pd.Series:
    """Lấy một hàng theo chỉ số (index)."""
    if index < 0 or index >= len(df):
        print(f"[LỖI] Index {index} nằm ngoài phạm vi. Dữ liệu có {len(df)} hàng (0 đến {len(df)-1}).")
        sys.exit(1)
    return df.iloc[index]


def print_row_info(row: pd.Series, index: int):
    """In toàn bộ thông tin của một hàng."""
    print(f"\n{'='*60}")
    print(f"  [NHADAT] THONG TIN BAT DONG SAN - Hang so {index}")
    print(f"{'='*60}")
    fields = {
        "title":           "Tieu de",
        "district":        "Quan/Huyen",
        "property_type":   "Loai BDS",
        "area":            "Dien tich (m2)",
        "price_billion":   "Gia (ty dong)",
        "width_m":         "Mat tien (m)",
        "depth_m":         "Chieu sau (m)",
        "floors":          "So tang",
        "bedrooms_detail": "Phong ngu",
        "bathrooms_detail":"Phong tam",
        "full_address":    "Dia chi day du",
    }
    for key, label in fields.items():
        value = row.get(key, "N/A")
        if pd.isna(value):
            value = "N/A"
        print(f"  {label:<22}: {value}")
    print(f"{'='*60}")


def print_description(row: pd.Series, index: int):
    """In description của một hàng."""
    desc = row.get("description", "Khong co mo ta.")
    if pd.isna(desc):
        desc = "Khong co mo ta."
    title = row.get('title', '')
    print(f"\n{'='*60}")
    print(f"  [MO TA] Hang so {index}: {title}")
    print(f"{'='*60}")
    print(desc)
    print(f"{'='*60}\n")


def search_by_district(df: pd.DataFrame, keyword: str):
    """Tìm kiếm theo tên quận/địa chỉ."""
    mask = (
        df["district"].str.contains(keyword, case=False, na=False) |
        df["full_address"].str.contains(keyword, case=False, na=False) |
        df["title"].str.contains(keyword, case=False, na=False)
    )
    results = df[mask]
    if results.empty:
        print(f"\n[!] Khong tim thay ket qua nao cho: '{keyword}'")
        return
    print(f"\n{'='*60}")
    print(f"  [TIM KIEM] '{keyword}' --- {len(results)} ket qua")
    print(f"{'='*60}")
    for i, (orig_idx, row) in enumerate(results.iterrows()):
        price = row.get("price_billion", "N/A")
        area  = row.get("area", "N/A")
        dist  = row.get("district", "N/A")
        title = str(row.get("title", ""))[:60]
        print(f"  [{orig_idx:>5}] {title}")
        print(f"         >> {dist} | {area}m2 | {price} ty")
        print()


def show_top(df: pd.DataFrame, n: int = 5, cheapest: bool = False):
    """Hiển thị n bất động sản đắt hoặc rẻ nhất."""
    df_valid = df.dropna(subset=["price_billion"])
    sorted_df = df_valid.sort_values("price_billion", ascending=cheapest).head(n)
    label = "RE NHAT" if cheapest else "DAT NHAT"
    print(f"\n{'='*60}")
    print(f"  [TOP {n}] BAT DONG SAN {label}")
    print(f"{'='*60}")
    for rank, (orig_idx, row) in enumerate(sorted_df.iterrows(), 1):
        title = str(row.get("title", ""))[:55]
        price = row.get("price_billion", "N/A")
        dist  = row.get("district", "N/A")
        area  = row.get("area", "N/A")
        print(f"  #{rank} [{orig_idx}] {title}")
        print(f"       >> {dist} | {area}m2 | {price} ty")
        print()


def show_stats(df: pd.DataFrame):
    """Hiển thị thống kê tổng quan."""
    print(f"\n{'='*60}")
    print(f"  [THONG KE] TONG QUAN DU LIEU")
    print(f"{'='*60}")
    print(f"  Tong so ban ghi     : {len(df):,}")
    print(f"  So cot              : {len(df.columns)}")
    print(f"  Cac cot             : {', '.join(df.columns.tolist())}")

    if "price_billion" in df.columns:
        p = df["price_billion"].dropna()
        print(f"\n  --- Gia (ty dong) ---")
        print(f"  Nho nhat  : {p.min():.2f} ty")
        print(f"  Lon nhat  : {p.max():.2f} ty")
        print(f"  Trung binh: {p.mean():.2f} ty")
        print(f"  Trung vi  : {p.median():.2f} ty")

    if "area" in df.columns:
        a = df["area"].dropna()
        print(f"\n  --- Dien tich (m2) ---")
        print(f"  Nho nhat  : {a.min():.0f} m2")
        print(f"  Lon nhat  : {a.max():.0f} m2")
        print(f"  Trung binh: {a.mean():.1f} m2")

    if "district" in df.columns:
        top_dist = df["district"].value_counts().head(5)
        print(f"\n  --- Top 5 quan/huyen nhieu tin nhat ---")
        for dist, cnt in top_dist.items():
            print(f"  {dist:<30}: {cnt:>5} tin")

    if "property_type" in df.columns:
        print(f"\n  --- Phan loai bat dong san ---")
        for ptype, cnt in df["property_type"].value_counts().items():
            print(f"  {ptype:<30}: {cnt:>5} tin")
    print(f"{'='*60}\n")


# ─── Lọc dữ liệu đầy đủ & Xuất CSV + Báo cáo ──────────────────────────────
# Danh sách các cột được coi là "quan trọng" (phải đầy đủ)
REQUIRED_FIELDS = [
    "title", "description", "area", "district", "property_type",
    "bedrooms_detail", "bathrooms_detail", "floors",
    "price_billion", "width_m", "depth_m", "full_address",
]


def filter_and_export(df: pd.DataFrame, input_path: str, output_csv: str | None = None):
    """
    Lọc chỉ giữ lại những hàng có đầy đủ tất cả các trường trong REQUIRED_FIELDS.
    Xuất ra file CSV mới, thống kê ra terminal và ghi file .txt báo cáo.
    """
    import datetime

    # ── Xác định cột nào thực sự tồn tại trong df ──
    existing_required = [c for c in REQUIRED_FIELDS if c in df.columns]
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]

    total_before = len(df)

    # ── Lọc: chỉ giữ hàng không có NaN ở các cột required ──
    df_clean = df.dropna(subset=existing_required).copy()
    # Loại thêm các hàng có chuỗi rỗng (sau khi strip) ở các cột chuỗi
    str_cols = [c for c in existing_required if df_clean[c].dtype == object]
    for col in str_cols:
        df_clean = df_clean[df_clean[col].str.strip() != ""]

    total_after  = len(df_clean)
    dropped      = total_before - total_after
    keep_pct     = total_after / total_before * 100 if total_before else 0

    # ── Đường dẫn output ──
    if output_csv is None:
        base  = os.path.splitext(input_path)[0]
        output_csv = base + "_complete.csv"
    output_txt = os.path.splitext(output_csv)[0] + "_report.txt"

    # ── Lưu CSV ──
    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ── Xây dựng nội dung báo cáo ──
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    sep  = "=" * 62

    def add(text=""):
        lines.append(text)

    add(sep)
    add("  BAO CAO LOC DU LIEU HOAN CHINH")
    add(f"  Thoi gian tao : {now}")
    add(sep)
    add()
    add("[NGUON DU LIEU]")
    add(f"  File goc      : {input_path}")
    add(f"  File ket qua  : {output_csv}")
    add(f"  File bao cao  : {output_txt}")
    add()
    add("[TIEU CHI LOC]")
    add(f"  Cac truong bat buoc ({len(existing_required)}):")
    for c in existing_required:
        add(f"    - {c}")
    if missing_cols:
        add(f"  [!] Cac cot khong ton tai trong file (da bo qua):")
        for c in missing_cols:
            add(f"    - {c}")
    add()
    add("[THONG KE SO LUONG]")
    add(f"  Truoc khi loc : {total_before:>10,} ban ghi")
    add(f"  Sau khi loc   : {total_after:>10,} ban ghi")
    add(f"  Da loai bo    : {dropped:>10,} ban ghi ({100-keep_pct:.1f}% tong so)")
    add(f"  Ti le giu lai : {keep_pct:.1f}%")
    add()

    if total_after > 0:
        # ── Thống kê giá ──
        if "price_billion" in df_clean.columns:
            p = df_clean["price_billion"]
            add("[THONG KE GIA (ty dong)]")
            add(f"  Nho nhat  : {p.min():.2f}")
            add(f"  Lon nhat  : {p.max():.2f}")
            add(f"  Trung binh: {p.mean():.2f}")
            add(f"  Trung vi  : {p.median():.2f}")
            add(f"  Do lech TC: {p.std():.2f}")
            add()

        # ── Thống kê diện tích ──
        if "area" in df_clean.columns:
            a = df_clean["area"]
            add("[THONG KE DIEN TICH (m2)]")
            add(f"  Nho nhat  : {a.min():.0f}")
            add(f"  Lon nhat  : {a.max():.0f}")
            add(f"  Trung binh: {a.mean():.1f}")
            add(f"  Trung vi  : {a.median():.1f}")
            add()

        # ── Thống kê mặt tiền ──
        if "width_m" in df_clean.columns:
            w = df_clean["width_m"]
            add("[THONG KE MAT TIEN (m)]")
            add(f"  Nho nhat  : {w.min():.1f}")
            add(f"  Lon nhat  : {w.max():.1f}")
            add(f"  Trung binh: {w.mean():.2f}")
            add()

        # ── Phân bố quận ──
        if "district" in df_clean.columns:
            dist_counts = df_clean["district"].value_counts()
            add("[PHAN BO THEO QUAN/HUYEN (top 10)]")
            for dist, cnt in dist_counts.head(10).items():
                pct = cnt / total_after * 100
                add(f"  {dist:<32}: {cnt:>5,}  ({pct:.1f}%)")
            add()

        # ── Phân loại BĐS ──
        if "property_type" in df_clean.columns:
            ptype_counts = df_clean["property_type"].value_counts()
            add("[PHAN LOAI BAT DONG SAN]")
            for ptype, cnt in ptype_counts.items():
                pct = cnt / total_after * 100
                add(f"  {ptype:<32}: {cnt:>5,}  ({pct:.1f}%)")
            add()

        # ── Thống kê số tầng ──
        if "floors" in df_clean.columns:
            f_col = df_clean["floors"]
            add("[THONG KE SO TANG]")
            add(f"  Nho nhat  : {f_col.min():.0f}")
            add(f"  Lon nhat  : {f_col.max():.0f}")
            add(f"  Pho bien  : {f_col.mode().iloc[0]:.0f}")
            add()

        # ── Missing value còn lại (cột ngoài required) ──
        extra_cols = [c for c in df_clean.columns if c not in existing_required]
        if extra_cols:
            missing_summary = df_clean[extra_cols].isnull().sum()
            missing_summary = missing_summary[missing_summary > 0]
            if not missing_summary.empty:
                add("[GIA TRI THIEU (cot khong bat buoc, trong file ket qua)]")
                for col, cnt in missing_summary.items():
                    add(f"  {col:<32}: {cnt:>5,} hang thieu")
                add()

    add(sep)
    report_text = "\n".join(lines)

    # ── In ra terminal ──
    print("\n" + report_text)

    # ── Ghi file .txt ──
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\n[OK] Da luu CSV : {output_csv}")
    print(f"[OK] Da luu bao cao: {output_txt}\n")


# ─── Chế độ tương tác (Interactive Menu) ────────────────────────────────────
def interactive_mode(df: pd.DataFrame):
    """Chạy menu tương tác."""
    print(f"\n{'='*60}")
    print("  [BDS] CONG CU KHAM PHA DU LIEU BDS HA NOI")
    print(f"  File: data/hanoi_final_filtered_features.csv")
    print(f"  So ban ghi: {len(df):,}")
    print(f"{'='*60}")

    menu = """
  [1] Xem thong tin hang theo index
  [2] Xem description cua hang theo index
  [3] Tim kiem theo quan / tu khoa
  [4] Top nha dat nhat
  [5] Top nha re nhat
  [6] Thong ke tong quan
  [0] Thoat
"""
    while True:
        print(menu)
        choice = input("  Chon chuc nang: ").strip()

        if choice == "1":
            idx = input("  Nhap index (0 = hang dau tien): ").strip()
            if idx.isdigit():
                row = get_row(df, int(idx))
                print_row_info(row, int(idx))
            else:
                print("  [!] Vui long nhap so nguyen.")

        elif choice == "2":
            idx = input("  Nhap index (0 = hang dau tien): ").strip()
            if idx.isdigit():
                row = get_row(df, int(idx))
                print_description(row, int(idx))
            else:
                print("  [!] Vui long nhap so nguyen.")

        elif choice == "3":
            kw = input("  Nhap tu khoa (vd: Hoan Kiem, Cau Giay): ").strip()
            if kw:
                search_by_district(df, kw)

        elif choice == "4":
            n = input("  So luong ket qua (mac dinh 5): ").strip()
            n = int(n) if n.isdigit() else 5
            show_top(df, n, cheapest=False)

        elif choice == "5":
            n = input("  So luong ket qua (mac dinh 5): ").strip()
            n = int(n) if n.isdigit() else 5
            show_top(df, n, cheapest=True)

        elif choice == "6":
            show_stats(df)

        elif choice == "0":
            print("\n  Tam biet!\n")
            break
        else:
            print("  [!] Lua chon khong hop le, vui long thu lai.")


# ─── CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Công cụ tương tác với dữ liệu BĐS Hà Nội",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--info",   type=int,   metavar="INDEX", help="Xem toàn bộ thông tin hàng theo index")
    parser.add_argument("--desc",   type=int,   metavar="INDEX", help="Xem description hàng theo index")
    parser.add_argument("--search", type=str,   metavar="KEYWORD", help="Tìm kiếm theo quận / từ khóa")
    parser.add_argument("--top",    type=int,   metavar="N",     help="Top N bất động sản đắt nhất")
    parser.add_argument("--cheap",  type=int,   metavar="N",     help="Top N bất động sản rẻ nhất")
    parser.add_argument("--stats",  action="store_true",         help="Hien thi thong ke tong quan")
    parser.add_argument("--filter", action="store_true",         help="Loc hang du lieu day du, xuat CSV moi va bao cao .txt")
    parser.add_argument("--output", type=str,   default=None,    help="Duong dan CSV ket qua (dung voi --filter, mac dinh: *_complete.csv)")
    parser.add_argument("--file",   type=str,   default=CSV_PATH, help=f"Duong dan toi file CSV (mac dinh: {CSV_PATH})")
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    df = load_data(args.file)

    # Nếu không có argument nào, chạy chế độ tương tác
    if not any([args.info is not None, args.desc is not None,
                args.search, args.top, args.cheap, args.stats, args.filter]):
        interactive_mode(df)
        return

    if args.info is not None:
        row = get_row(df, args.info)
        print_row_info(row, args.info)

    if args.desc is not None:
        row = get_row(df, args.desc)
        print_description(row, args.desc)

    if args.search:
        search_by_district(df, args.search)

    if args.top:
        show_top(df, args.top, cheapest=False)

    if args.cheap:
        show_top(df, args.cheap, cheapest=True)

    if args.stats:
        show_stats(df)

    if args.filter:
        filter_and_export(df, input_path=args.file, output_csv=args.output)


if __name__ == "__main__":
    main()
