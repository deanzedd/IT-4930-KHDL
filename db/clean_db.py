"""
db/clean_db.py - Xóa các listings không phải BĐS khỏi MongoDB
Chạy: python db/clean_db.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from db.mongo import get_db, close

# Từ khóa xe cộ và hàng hóa không phải BĐS trong title
VEHICLE_KEYWORDS = [
    "toyota", "honda", "ford", "hyundai", "kia", "mazda", "mitsubishi",
    "vinfast", "suzuki", "chevrolet", "lexus", "peugeot", "nissan",
    "iphone", "samsung", "laptop", "xe đạp", "xe máy", "cbr", "sh ",
    "triton", "ranger", "fortuner", "camry", "vios", "civic",
    "air blade", "wave ", "vision ", "mg mgs", "mg zs",
]

def clean(dry_run: bool = True):
    db = get_db()
    col = db["listings"]

    total_before = col.count_documents({})
    print(f"Tong listings truoc khi clean: {total_before}")

    # Tìm bằng regex OR trên title
    regex_parts = [{"title": {"$regex": kw, "$options": "i"}} for kw in VEHICLE_KEYWORDS]
    vehicle_filter = {"$or": regex_parts}

    vehicle_count = col.count_documents(vehicle_filter)
    print(f"Listings xe cộ/hang hoa can xoa: {vehicle_count}")

    if vehicle_count == 0:
        print("Khong co gi can xoa.")
        close()
        return

    # In vài ví dụ trước khi xóa
    print("\nVi du (5 listings se bi xoa):")
    for doc in col.find(vehicle_filter, {"title": 1, "price": 1}).limit(5):
        print(f"  - {doc.get('title','')[:60]}")

    if dry_run:
        print("\n[DRY RUN] Them --delete de xoa that su.")
    else:
        result = col.delete_many(vehicle_filter)
        total_after = col.count_documents({})
        print(f"\nDa xoa: {result.deleted_count} listings")
        print(f"Con lai: {total_after} listings")

    close()


if __name__ == "__main__":
    import sys
    do_delete = "--delete" in sys.argv
    clean(dry_run=not do_delete)
