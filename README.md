# House Price Crawler 🏠

Crawler BĐS Hà Nội từ 5 nguồn: **batdongsan.com.vn**, **nhatot.com**, **mogi.vn**, **alonhadat.com.vn**, **chotot.com**  
Lưu dữ liệu vào **MongoDB**.

## Cài đặt

```bash
# 1. Cài Python packages
pip install -r requirements.txt

# 2. Cài MongoDB (nếu chưa có)
# Windows: https://www.mongodb.com/try/download/community
# Ubuntu:  sudo apt install mongodb

# 3. Tạo file .env (copy từ ví dụ dưới)
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
python main.py --sources nhatot chotot --pages 10
```

## Cấu trúc project

```
house-price-crawler/
├── scrapers/
│   ├── base.py           # Abstract base class (Selenium)
│   ├── batdongsan.py     # Scraper batdongsan.com.vn (undetected-chromedriver)
│   ├── mogi.py           # Scraper mogi.vn (Selenium)
│   ├── alonhadat.py      # Scraper alonhadat.com.vn (requests + BS4)
│   ├── nhatot.py         # Scraper nhatot.com (JSON API)
│   └── chotot.py         # Scraper chotot.com (JSON API)
├── db/
│   ├── mongo.py          # Kết nối & lưu MongoDB
│   ├── clean_db.py       # Xóa duplicate / dọn dữ liệu
│   └── migrate_property_type.py
├── models/
│   └── listing.py        # Schema + hàm parse giá/diện tích/địa chỉ
├── main.py               # Entry point
├── .env                  # Config (không commit)
└── requirements.txt
```

## Schema MongoDB

Collection: `listings`

| Field | Type | Mô tả |
|---|---|---|
| source | str | batdongsan / nhatot / mogi / alonhadat / chotot |
| url | str | URL gốc (unique index) |
| title | str | Tiêu đề tin đăng |
| listing_type | str | "ban" hoặc "thue" |
| property_type | str | nha_pho / chung_cu / dat / biet_thu |
| price | float | Giá (VNĐ) |
| price_per_m2 | float | Giá/m² (VNĐ) |
| area | float | Diện tích (m²) |
| district | str | Quận/Huyện |
| ward | str | Phường/Xã |
| street | str | Tên đường/phố |
| lat / lng | float | Tọa độ địa lý |
| bedrooms | int | Số phòng ngủ |
| bathrooms | int | Số WC |
| floors | int | Số tầng |
| facade | float | Mặt tiền (m) |
| legal_status | str | Pháp lý (sổ đỏ, sổ hồng...) |
| posted_at | str | Ngày đăng |

## Query MongoDB mẫu

```js
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
- **nhatot** và **chotot** dùng JSON API (nhanh, ổn định hơn)
- **batdongsan** cần `undetected-chromedriver` để bypass Cloudflare
