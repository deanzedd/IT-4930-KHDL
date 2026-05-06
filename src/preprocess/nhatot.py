
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
INPUT_FILE = PROJECT_ROOT / "data" / "house_price_db.listings.csv"

# Output directory & file
OUTPUT_DIR = PROJECT_ROOT / "data" / "2026" / "nhatot"
OUTPUT_FILE = OUTPUT_DIR / "hanoi_houses_nhatot.csv"

# ---------------------------------------------------------------------------
# Columns to keep
# ---------------------------------------------------------------------------
SELECTED_COLUMNS = [
    "area",
    "district",
    "property_type",
    "price",
    "listing_type",
    "price_text"
]
## area, district, property_type, price, listing_type, price_text

# Only keep rows with this listing type
LISTING_TYPE_FILTER = "ban"


def preprocess(input_path: Path, output_path: Path, columns: list[str]) -> None:
    """
    Load the raw CSV, select the desired columns, filter by listing_type,
    and save the result.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    output_path : Path
        Path where the filtered CSV will be written.
    columns : list[str]
        Column names to retain (must include ``listing_type``).
    """
    print(f"[INFO] Reading data from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"[INFO] Total rows loaded : {len(df)}")

    # Validate that every requested column actually exists
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in source file: {missing}")

    # Filter rows: keep only listing_type == LISTING_TYPE_FILTER
    if "listing_type" not in df.columns:
        raise ValueError("[ERROR] Column 'listing_type' not found in source file.")
    df = df[df["listing_type"] == LISTING_TYPE_FILTER]
    print(f"[INFO] Rows after filter  : {len(df)}  (listing_type == '{LISTING_TYPE_FILTER}')")

    # Select columns
    df_filtered = df[columns].copy()

    print(f"[INFO] Columns            : {df_filtered.columns.tolist()}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved to           : {output_path}")


if __name__ == "__main__":
    preprocess(INPUT_FILE, OUTPUT_FILE, SELECTED_COLUMNS)
