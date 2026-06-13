"""
db/mongo.py - Kết nối MongoDB và các hàm lưu dữ liệu
"""
import os
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

load_dotenv()

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(uri)
        _db = _client[os.getenv("MONGO_DB", "house_price_db")]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db):
    """Tạo index để tránh duplicate và tăng tốc query"""
    col = db["listings"]
    col.create_index([("url", ASCENDING)], unique=True)
    col.create_index([("source", ASCENDING), ("crawled_at", ASCENDING)])
    col.create_index([("district", ASCENDING), ("price", ASCENDING)])
    col.create_index([("district", ASCENDING), ("ward", ASCENDING), ("street", ASCENDING)])
    col.create_index([("lat", ASCENDING), ("lng", ASCENDING)])
    print("[MongoDB] Indexes ready.")


def save_listing(listing: dict) -> bool:
    """
    Lưu 1 listing vào collection 'listings'.
    Trả về True nếu insert thành công, False nếu đã tồn tại.
    """
    db = get_db()
    # Dùng bản copy để insert_one không mutate dict gốc (PyMongo tự thêm _id vào dict)
    doc = {**listing, "crawled_at": datetime.utcnow()}
    try:
        db["listings"].insert_one(doc)
        return True
    except DuplicateKeyError:
        # URL đã có → chỉ update các field thay đổi, bỏ qua _id
        update_fields = {k: v for k, v in doc.items() if k != "_id"}
        update_fields["updated_at"] = datetime.utcnow()
        db["listings"].update_one(
            {"url": listing["url"]},
            {"$set": update_fields}
        )
        return False


def save_many(listings: list[dict]) -> dict:
    """Lưu nhiều listings, trả về thống kê inserted/updated"""
    stats = {"inserted": 0, "updated": 0, "failed": 0}
    for listing in listings:
        try:
            ok = save_listing(listing)
            if ok:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1
        except Exception as e:
            print(f"[DB ERROR] {e}")
            stats["failed"] += 1
    return stats


def count_listings(source: str = None) -> int:
    db = get_db()
    query = {"source": source} if source else {}
    return db["listings"].count_documents(query)


def close():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
