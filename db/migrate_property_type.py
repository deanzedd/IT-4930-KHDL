"""
db/migrate_property_type.py
Fix property_type cho các listings bi gán 'khac' do detection cu qua hep.
Chay: python -m db.migrate_property_type
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from db.mongo import get_db, close
from models.listing import detect_property_type
from collections import Counter


def migrate(dry_run: bool = True):
    db = get_db()
    col = db["listings"]

    khac_docs = list(col.find({"property_type": "khac"}, {"_id": 1, "title": 1, "url": 1, "bedrooms": 1, "area": 1}))
    print(f"Listings can migrate: {len(khac_docs)}")

    updates = []
    new_type_counts = Counter()

    for doc in khac_docs:
        title = doc.get("title", "")
        url   = doc.get("url", "")
        new_type = detect_property_type(title, url)

        # Neu van la 'khac' thi dung them signal
        if new_type == "khac":
            # Co bedrooms/bathrooms -> gần chắc là nhà ở
            if doc.get("bedrooms") is not None:
                new_type = "nha_pho"
            # Area rất lớn, không có bedrooms -> đất
            elif doc.get("area") and doc.get("area") > 200 and doc.get("bedrooms") is None:
                new_type = "dat"

        new_type_counts[new_type] += 1
        if new_type != "khac":
            updates.append((doc["_id"], new_type))

    print(f"\nKet qua phan loai:")
    for k, v in new_type_counts.most_common():
        print(f"  {k}: {v}")

    print(f"\nSe update: {len(updates)} listings")
    print(f"Van giu 'khac': {new_type_counts['khac']}")

    if dry_run:
        print("\n[DRY RUN] Them --apply de chay that.")
        close()
        return

    updated = 0
    for doc_id, new_type in updates:
        col.update_one({"_id": doc_id}, {"$set": {"property_type": new_type}})
        updated += 1
        if updated % 500 == 0:
            print(f"  Da update {updated}/{len(updates)}...")

    print(f"\nHoan thanh: update {updated} listings.")

    # In ket qua cuoi
    final = Counter(d.get("property_type") for d in col.find({}, {"property_type": 1}))
    print("\nPhan bo property_type sau migrate:")
    for k, v in final.most_common():
        print(f"  {k}: {v}")

    close()


if __name__ == "__main__":
    dry_run = "--apply" not in sys.argv
    migrate(dry_run=dry_run)
