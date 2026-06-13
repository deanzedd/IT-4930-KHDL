"""
scrapers/chotot.py - Crawl chotot.com.vn qua public JSON API
Không cần Selenium hay HTML parsing — API trả về đủ dữ liệu.
"""
import sys
import time
import random
import requests
sys.stdout.reconfigure(encoding="utf-8")

from models.listing import Listing, detect_property_type, parse_address_components

SOURCE = "chotot"
BASE_URL = "https://www.chotot.com"
API_URL = "https://gateway.chotot.com/v1/public/ad-listing"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

# Hà Nội region
REGION_V2 = "12000"

# Tất cả quận/huyện Hà Nội (area_v2 → tên)
HN_DISTRICTS: dict[int, str] = {
    12073: "Quận Hoàn Kiếm",
    12074: "Quận Ba Đình",
    12075: "Quận Đống Đa",
    12076: "Quận Hai Bà Trưng",
    12077: "Quận Thanh Xuân",
    12078: "Quận Tây Hồ",
    12079: "Quận Cầu Giấy",
    12080: "Quận Hoàng Mai",
    12081: "Quận Long Biên",
    12082: "Huyện Đông Anh",
    12083: "Huyện Sóc Sơn",
    12084: "Huyện Thanh Trì",
    12086: "Quận Hà Đông",
    12087: "Thị xã Sơn Tây",
    12088: "Huyện Đan Phượng",
    12089: "Huyện Hoài Đức",
    12090: "Huyện Quốc Oai",
    12091: "Huyện Thạch Thất",
    12092: "Huyện Chương Mỹ",
    12093: "Huyện Thường Tín",
    12094: "Huyện Phú Xuyên",
    12121: "Quận Nam Từ Liêm",
    12122: "Huyện Ba Vì",
    12123: "Huyện Gia Lâm",
    12124: "Huyện Mê Linh",
    12129: "Quận Bắc Từ Liêm",
}

# (listing_type, cg, category_label)
CATEGORIES = [
    ("ban",  1010, "can-ho-chung-cu"),
    ("ban",  1020, "nha-o"),
    ("ban",  1040, "dat"),
    ("thue", 1050, "phong-tro"),
]

LEGAL_MAP = {
    1: "Sổ đỏ/Sổ hồng",
    2: "Đang chờ sổ",
    3: "Giấy tờ hợp lệ khác",
    4: "Chưa có sổ",
}

FURNITURE_MAP = {
    1: "Không có nội thất",
    2: "Nội thất cơ bản",
    3: "Đầy đủ nội thất",
}

DIRECTION_MAP = {
    1: "Đông", 2: "Tây", 3: "Nam", 4: "Bắc",
    5: "Đông Nam", 6: "Đông Bắc", 7: "Tây Nam", 8: "Tây Bắc",
}


def _fetch_district(cg: int, area_v2: int, offset: int = 0, limit: int = 20) -> list[dict]:
    """Lấy listings của 1 quận với offset pagination."""
    params = {
        "cg":                 cg,
        "area_v2":            area_v2,
        "limit":              limit,
        "o":                  offset,
        "key_param_included": "true",
    }
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("ads", [])
    except Exception as e:
        print(f"[{SOURCE}] API error area {area_v2} offset={offset}: {e}")
        return []


def _parse_ad(ad: dict, listing_type: str, cat_label: str) -> dict | None:
    try:
        ad_id = ad.get("ad_id") or ad.get("list_id")
        title = ad.get("subject", "").strip()
        if not title or not ad_id:
            return None

        url = f"{BASE_URL}/mua-ban-bat-dong-san/{cat_label}/{ad_id}.htm"

        price = float(ad["price"]) if ad.get("price") else None
        area = float(ad["size"]) if ad.get("size") else None
        price_per_m2 = (
            float(ad["price_million_per_m2"]) * 1_000_000
            if ad.get("price_million_per_m2") else
            (round(price / area, 0) if price and area else None)
        )

        district = ad.get("area_name", "")
        ward = ad.get("ward_name", "")
        street = ad.get("street_name", "")
        address_parts = [p for p in [street, ward, district, "Hà Nội"] if p]
        address = ", ".join(address_parts)

        legal_code = ad.get("property_legal_document")
        legal_status = LEGAL_MAP.get(legal_code, "")

        furniture_code = ad.get("furnishing_sell")
        furniture = FURNITURE_MAP.get(furniture_code, "")

        dir_code = ad.get("direction")
        direction = DIRECTION_MAP.get(dir_code, "")

        property_type = detect_property_type(title, url)

        images = [img for img in (ad.get("images") or []) if img][:10]

        listing = Listing(
            source=SOURCE,
            url=url,
            title=title,
            listing_type=listing_type,
            property_type=property_type,
            price=price,
            price_text=ad.get("price_string", ""),
            price_per_m2=price_per_m2,
            area=area,
            area_text=f"{area} m²" if area else "",
            address=address,
            district=district,
            ward=ward,
            city="Hà Nội",
            street=street,
            lat=float(ad["latitude"]) if ad.get("latitude") else None,
            lng=float(ad["longitude"]) if ad.get("longitude") else None,
            bedrooms=int(ad["rooms"]) if ad.get("rooms") else None,
            bathrooms=int(ad["toilets"]) if ad.get("toilets") else None,
            floors=int(ad["floors"]) if ad.get("floors") else None,
            facade=float(ad["width"]) if ad.get("width") else None,
            direction=direction,
            legal_status=legal_status,
            furniture=furniture,
            description=(ad.get("body") or "")[:500],
            posted_at=ad.get("date", ""),
            images=images,
        )
        return listing.to_dict() if listing.is_valid() else None

    except Exception as e:
        print(f"[{SOURCE}] Parse error ad {ad.get('ad_id')}: {e}")
        return None


def crawl(max_pages: int = 5) -> list[dict]:
    """Query từng quận × từng category × nhiều trang để tối đa số listing unique."""
    all_listings = []
    seen_urls: set[str] = set()

    for listing_type, cg, cat_label in CATEGORIES:
        print(f"[{SOURCE}] === {listing_type.upper()} | cg={cg} ({cat_label}) ===")
        cat_count = 0

        for area_v2, district_name in HN_DISTRICTS.items():
            district_new = 0
            for page in range(max_pages):
                offset = page * 20
                ads = _fetch_district(cg, area_v2, offset=offset)
                if not ads:
                    break

                new_count = 0
                for ad in ads:
                    d = _parse_ad(ad, listing_type, cat_label)
                    if d and d["url"] not in seen_urls:
                        seen_urls.add(d["url"])
                        all_listings.append(d)
                        new_count += 1

                district_new += new_count
                time.sleep(random.uniform(0.2, 0.5))

                # Nếu trang này không có listing mới → dừng sớm
                if new_count == 0:
                    break

            if district_new:
                print(f"[{SOURCE}]   {district_name}: {district_new} mới")
            cat_count += district_new

        print(f"[{SOURCE}]   → {cat_count} listings mới từ category này")

    print(f"[{SOURCE}] Xong. Tổng: {len(all_listings)} listings unique.")
    return all_listings


if __name__ == "__main__":
    listings = crawl(max_pages=1)
    if listings:
        print("\n=== Sample listing ===")
        for k, v in listings[0].items():
            if k != "images":
                print(f"  {k}: {v}")
