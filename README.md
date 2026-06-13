# House Price Crawler 🏠

Crawler BĐS Hà Nội từ 3 nguồn: **batdongsan.com.vn**, **nhatot.com**, **mogi.vn**  
Lưu dữ liệu vào **MongoDB**.

## Cài đặt

```bash
# 1. Cài Python packages
pip install -r requirements.txt

# 2. Cài MongoDB (nếu chưa có)
# Windows: https://www.mongodb.com/try/download/community
# Ubuntu:  sudo apt install mongodb

# 3. Config .env (sửa nếu cần)
# MONGO_URI=mongodb://localhost:27017
# MONGO_DB=house_price_db
# HEADLESS=true        ← false nếu muốn xem browser chạy
# DELAY_MIN=2          ← delay tối thiểu giữa các request (giây)
# DELAY_MAX=5          ← delay tối đa
```

## Chạy

```bash
# Crawl tất cả nguồn, mỗi nguồn 5 trang
python main.py

# Chỉ crawl 1 nguồn
python main.py --sources batdongsan

# Nhiều nguồn + tăng số trang
python main.py --sources batdongsan nhatot --pages 10
```

## Cấu trúc project

```
house-price-crawler/
├── scrapers/
│   ├── base.py           # Abstract base class
│   ├── batdongsan.py     # Scraper batdongsan.com.vn
│   ├── nhatot.py         # Scraper nhatot.com
│   └── mogi.py           # Scraper mogi.vn
├── db/
│   └── mongo.py          # Kết nối & lưu MongoDB
├── models/
│   └── listing.py        # Schema + hàm parse giá/diện tích
├── main.py               # Entry point
├── .env                  # Config
└── requirements.txt
```

## Schema MongoDB

Collection: `listings`

| Field | Type | Mô tả |
|---|---|---|
| source | str | batdongsan / nhatot / mogi |
| url | str | URL gốc (unique index) |
| title | str | Tiêu đề tin đăng |
| listing_type | str | "ban" hoặc "thue" |
| property_type | str | nha_pho / chung_cu / dat / biet_thu |
| price | float | Giá (VNĐ) |
| price_per_m2 | float | Giá/m² (VNĐ) |
| area | float | Diện tích (m²) |
| district | str | Quận/Huyện |
| city | str | Hà Nội |
| bedrooms | int | Số phòng ngủ |
| bathrooms | int | Số WC |
| crawled_at | datetime | Thời gian crawl |

## Query MongoDB mẫu

```js
// Xem trong MongoDB Compass hoặc mongosh

// Tất cả nhà bán quận Hoàn Kiếm dưới 5 tỷ
db.listings.find({
  listing_type: "ban",
  district: { $regex: "Hoàn Kiếm" },
  price: { $lt: 5000000000 }
})

// Thống kê theo nguồn
db.listings.aggregate([
  { $group: { _id: "$source", count: { $sum: 1 } } }
])

// Giá trung bình theo quận
db.listings.aggregate([
  { $match: { listing_type: "ban", price: { $ne: null } } },
  { $group: { _id: "$district", avg_price: { $avg: "$price" } } },
  { $sort: { avg_price: -1 } }
])
```

## Lưu ý

- Các trang BĐS **thay đổi HTML thường xuyên** → selector có thể cần cập nhật
- Đặt `HEADLESS=false` để debug xem browser đang load gì
- Tăng `DELAY_MIN/MAX` nếu bị block IP
- Nên chạy vào ban đêm hoặc chia nhỏ số trang để tránh bị chặn
