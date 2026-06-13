"""
main.py - Crawl dữ liệu nhà đất Hà Nội từ nhiều nguồn
"""
import sys
import argparse
sys.stdout.reconfigure(encoding="utf-8")

from db.mongo import save_many, count_listings, close


def _crawl_nhatot(pages):
    from scrapers.nhatot import crawl
    return crawl(max_pages=pages)

def _crawl_batdongsan(pages):
    from scrapers.batdongsan import crawl
    return crawl(max_pages=pages)

def _crawl_mogi(pages):
    from scrapers.mogi import crawl
    return crawl(max_pages=pages)

def _crawl_alonhadat(pages):
    from scrapers.alonhadat import crawl
    return crawl(max_pages=pages)

def _crawl_chotot(pages):
    from scrapers.chotot import crawl
    return crawl(max_pages=pages)


SOURCES = {
    "nhatot":     _crawl_nhatot,
    "batdongsan": _crawl_batdongsan,
    "mogi":       _crawl_mogi,
    "alonhadat":  _crawl_alonhadat,
    "chotot":     _crawl_chotot,
}


def run(source: str, max_pages: int):
    targets = SOURCES if source == "all" else {source: SOURCES[source]}

    for name, crawl_fn in targets.items():
        print(f"\n{'='*55}")
        print(f"  Crawl: {name.upper()}  (max {max_pages} trang/category)")
        print(f"{'='*55}")

        listings = crawl_fn(max_pages)
        if listings:
            stats = save_many(listings)
            print(f"\n[DB] +{stats['inserted']} mới | ~{stats['updated']} update | x{stats['failed']} lỗi")
        else:
            print("Không có dữ liệu.")

    print(f"\n{'='*55}")
    print(f"  Tổng DB: {count_listings()} listings")
    for name in targets:
        print(f"  - {name}: {count_listings(name)} listings")
    print(f"{'='*55}")
    close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl nhà đất Hà Nội từ nhiều nguồn",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["nhatot", "batdongsan", "mogi", "alonhadat", "chotot", "all"],
        default="nhatot",
        help=(
            "Nguồn crawl (default: nhatot)\n"
            "  nhatot     - nhatot.com qua API (nhanh nhất)\n"
            "  batdongsan - batdongsan.com.vn (bị Cloudflare, ít dùng)\n"
            "  mogi       - mogi.vn qua Selenium\n"
            "  alonhadat  - alonhadat.com.vn qua requests (không cần Selenium)\n"
            "  all        - Tất cả nguồn"
        ),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Số trang mỗi category (default=5)",
    )
    args = parser.parse_args()
    run(args.source, args.pages)
