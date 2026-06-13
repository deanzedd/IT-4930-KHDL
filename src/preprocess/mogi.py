"""
mogi.py - Preprocessing script for Mogi housing data.

Filters selected columns from the raw CSV and saves the result to:
    <project_root>/data/2026/mogi/hanoi_houses_mogi.csv

Usage:
    python src/preprocess/mogi.py
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to this file so no hard-coded absolute paths are used)
# ---------------------------------------------------------------------------

# Directory of this script  ->  src/preprocess/
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root  ->  KHDL/
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Input file
INPUT_FILE = PROJECT_ROOT / "data" / "hanoi_houses_final.csv"

# Output directory & file
OUTPUT_DIR = PROJECT_ROOT / "data" / "2026" / "mogi"
OUTPUT_FILE = OUTPUT_DIR / "hanoi_houses_mogi.csv"

# ---------------------------------------------------------------------------
# Columns to keep
# ---------------------------------------------------------------------------
SELECTED_COLUMNS = [
    "property_type",
    "district",
    "area_text",
    "bedrooms",
    "bathrooms",
    "price_text",
    "posted_at",
]


# Pattern: optional "X tỷ" part  +  optional "Y triệu" part
# Supports integers and decimals in each part, e.g. "16 tỷ 800 triệu", "6,5 tỷ"
_PRICE_RE = re.compile(
    r"(?:([\d,\.]+)\s*t[ỷy])?"
    r"\s*(?:([\d,\.]+)\s*tri[eệ]u)?",
    re.IGNORECASE | re.UNICODE,
)


def parse_price(text: str) -> float | None:
    """
    Convert a Vietnamese price string to an integer value in VNĐ.

    Examples
    --------
    >>> parse_price("25 tỷ")             # -> 25_000_000_000
    >>> parse_price("67 tỷ 500 triệu")  # -> 67_500_000_000
    >>> parse_price("6,5 tỷ")           # -> 6_500_000_000
    >>> parse_price("800 triệu")        # -> 800_000_000
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    match = _PRICE_RE.search(text)
    if not match:
        return None

    ty_str, trieu_str = match.group(1), match.group(2)

    # Must capture at least one unit
    if not ty_str and not trieu_str:
        return None

    def to_float(s: str | None) -> float:
        if not s:
            return 0.0
        return float(s.replace(",", "."))

    ty = to_float(ty_str)
    trieu = to_float(trieu_str)

    total = ty * 1_000_000_000 + trieu * 1_000_000
    return int(total) if total > 0 else None


def preprocess(input_path: Path, output_path: Path, columns: list[str]) -> None:
    """
    Load the raw CSV, select the desired columns, derive a numeric ``price``
    column from ``price_text``, and save the result.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    output_path : Path
        Path where the filtered CSV will be written.
    columns : list[str]
        Column names to retain (must include ``price_text``).
    """
    print(f"[INFO] Reading data from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")

    # Validate that every requested column actually exists
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in source file: {missing}")

    # Filter columns
    df_filtered = df[columns].copy()

    # Derive numeric price column from price_text
    df_filtered["price"] = df_filtered["price_text"].apply(parse_price)

    parsed_count = df_filtered["price"].notna().sum()
    print(f"[INFO] Rows kept      : {len(df_filtered)}")
    print(f"[INFO] Price parsed   : {parsed_count} / {len(df_filtered)}")
    print(f"[INFO] Columns        : {df_filtered.columns.tolist()}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved to       : {output_path}")


if __name__ == "__main__":
    preprocess(INPUT_FILE, OUTPUT_FILE, SELECTED_COLUMNS)
