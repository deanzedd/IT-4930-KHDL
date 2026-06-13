"""
scrapers/nhatot.py - Crawl nhatot.com qua API gateway.chotot.com
Đọc features trực tiếp từ list API (không cần detail API riêng).
"""
import sys
import time
import random
import re
import requests
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
from models.listing import Listing, detect_property_type, parse_address_components

SOURCE = "nhatot"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nhatot.com/",
}

CATEGORIES   = [1010, 1020, 1040, 1050]  # 1010=Chung cu, 1020=Nha o, 1040=Dat, 1050=Phong tro
REGION_HN    = 12000          # Ha Noi
LIST_URL     = "https://gateway.chotot.com/v1/public/ad-listing"

# Keyword loại bỏ listing không phải BĐS
_NON_RE_KEYWORDS = [
    # Ô tô
    "toyota", "honda cr", "ford ", "hyundai", "kia ", "mazda", "mitsubishi",
    "vinfast", "suzuki ", "chevrolet", "lexus", "peugeot", "nissan",
    "mercedes", "bmw ", "audi ", "porsche", "bentley", "ferrari",
    "range rover", "land rover", "xpander", "fortuner", "camry ", "vios ",
    "innova", "inova ", "everest ", "daewoo", "lacetti",
    "glk ", "glc ", "c300 ", "c200 ", "cls ",
    # Xe máy / scooter
    "yamaha ", "piaggio", "vespa ", "sym ", "gpx ", "exciter ", "exciter 1",
    "airblade", "winner v", "winer v", "sh350", "sh125", "sh150",
    "pcx 125", "scoopy", "vision 1", "lead 125", "vario",
    "jupiter fi", "sirius 11", "nouvo", "janus 20",
    "kimco", "kymco", "zoomer",
    # cc pattern (xe máy)
    "50cc", "100cc", "110cc", "125cc", "135cc", "150cc", "200cc", "250cc",
    # Đồ dùng / điện tử
    "iphone", "samsung", "laptop", "máy tính", "xe đạp",
]

LEGAL_MAP = {
    1: "So do/So hong",
    2: "Hop dong mua ban",
    3: "Dang cho so",
    4: "Giay to khac",
}

CATEGORY_TYPE_MAP = {
    "nhà ở":    "nha_pho",
    "đất":       "dat",
    "căn hộ":   "chung_cu",
    "biệt thự": "biet_thu",
    "chung cư":      "chung_cu",
}


def fetch_page(category_id: int, offset: int) -> list[dict]:
    params = {
        "cg":                 category_id,
        "region_v2":          REGION_HN,
        "limit":              20,
        "o":                  offset,
        "key_param_included": "true",
    }
    try:
        resp = requests.get(LIST_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("ads", [])
    except Exception as e:
        print(f"[{SOURCE}] Loi API cg={category_id} offset={offset}: {e}")
        return []


def _is_non_real_estate(title: str, cg: int, cg_name: str) -> bool:
    """Trả về True nếu listing không phải BĐS."""
    if not (1000 <= cg < 2000):
        return True
    title_lower = title.lower()
    if any(kw in title_lower for kw in _NON_RE_KEYWORDS):
        return True
    bad_cg = ["xe", "dien thoai", "may tinh", "thu cung", "vat dung"]
    cg_norm = cg_name.lower()
    if any(kw in cg_norm for kw in bad_cg):
        return True
    return False


def _extract_floors(text: str) -> int | None:
    """
    Tìm số tầng từ title/description.
    '4T', '4 tang', '7 TANG', '5 tầng', '5T thang may'
    """
    if not text:
        return None
    t = text.lower()
    # Pattern: số + "tầng" hoặc số + "t" (viết tắt phổ biến)
    for pat in [
        r'\b(\d+)\s*t[aầ]ng\b',
        r'\b(\d+)\s*t(?:\s+thang|\s+elevator|\b)',
    ]:
        m = re.search(pat, t)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 50:  # sanity check
                return val
    return None


def _extract_facade(text: str) -> float | None:
    """
    Tìm mặt tiền từ title/description.
    'MT:3.9m', 'MT 4m', 'mat tien 4m', 'ngang 4m', 'MN:4'
    """
    if not text:
        return None
    t = text.lower()
    for pat in [
        r'\bmt[:\s]?\s*([\d.]+)\s*m',
        r'\bm[aặ]t\s*ti[eề]n[:\s]?\s*([\d.]+)\s*m',
        r'\bngang[:\s]?\s*([\d.]+)\s*m',
        r'\bmn[:\s]?\s*([\d.]+)',
    ]:
        m = re.search(pat, t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100:
                return val
    return None


def _parse_posted_at(ad: dict) -> str:
    """Chuyển list_time (Unix ms) sang ISO datetime string."""
    ts = ad.get("list_time")
    if ts:
        try:
            dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return ad.get("date", "")


def _detect_listing_type(ad: dict) -> str:
    """
    Dùng field 'type' của API để phân biệt bán/thuê.
    's' = ban (sale), 'u' = cho thue (used/rental)
    """
    api_type = str(ad.get("type", "s") or "s").lower()
    title_lower = (ad.get("subject", "") or "").lower()
    if api_type == "u" or "cho thue" in title_lower or "can thue" in title_lower:
        return "thue"
    return "ban"


def parse_ad(ad: dict) -> dict | None:
    try:
        ad_id   = ad.get("ad_id", "")
        title   = ad.get("subject") or (ad.get("body", "")[:80])
        url     = f"https://www.nhatot.com/mua-ban-nha-dat/{ad_id}.htm"
        cg      = ad.get("category", 0)
        cg_name = ad.get("category_name", "")

        if _is_non_real_estate(title, cg, cg_name):
            return None

        listing_type = _detect_listing_type(ad)

        # Listing "thue" mà không có area → xe cộ hoặc hàng hóa, bỏ qua
        area_check = float(ad.get("size", 0) or 0)
        if listing_type == "thue" and area_check == 0:
            return None

        # ── Giá & Diện tích ────────────────────────────────
        price      = float(ad.get("price", 0) or 0) or None
        price_text = ad.get("price_string", "")
        area       = float(ad.get("size", 0) or 0) or None

        # ── Tọa độ ─────────────────────────────────────────
        lat = _safe_float(str(ad.get("latitude", "") or ""))
        lng = _safe_float(str(ad.get("longitude", "") or ""))
        if not lat or not lng:
            loc_str = str(ad.get("location", "") or "")
            if "," in loc_str:
                try:
                    lat, lng = (float(x) for x in loc_str.split(",", 1))
                except ValueError:
                    pass

        # ── Features từ top-level API fields ───────────────
        bedrooms  = _safe_int(str(ad.get("rooms", "") or ""))
        bathrooms = _safe_int(str(ad.get("toilets", "") or ""))
        facade    = _safe_float(str(ad.get("width", "") or ""))
        floors    = None
        direction = ""
        furniture = ""
        year_built = None

        legal_code   = ad.get("property_legal_document")
        legal_status = LEGAL_MAP.get(legal_code, "")

        # ── Bổ sung từ params[] ─────────────────────────────
        for p in (ad.get("params", []) or []):
            pid   = str(p.get("id", "") or "").lower()
            label = str(p.get("label", "") or "").lower()
            value = str(p.get("value", "") or "")

            if pid in ("direction", "house_direction") or "huong" in label or "hướng" in label:
                direction = value.strip()
            elif pid in ("interior", "furniture") or "nội thất" in label:
                furniture = value.strip()
            elif pid in ("year_built", "construction_year") or "năm xây" in label:
                year_built = _safe_int(value)
            elif pid in ("rooms",) and bedrooms is None:
                bedrooms = _safe_int(value)
            elif pid in ("toilets", "bathrooms") and bathrooms is None:
                bathrooms = _safe_int(value)
            elif pid == "floors" and floors is None:
                floors = _safe_int(value)
            elif pid == "paper" and not legal_status:
                legal_status = value.strip()

        # ── Fallback: extract floors, facade, direction từ text
        description = ad.get("body", "")
        search_text = f"{title} {description}"

        if floors is None:
            floors = _extract_floors(search_text)
        if facade is None:
            facade = _extract_facade(search_text)

        # ── Địa chỉ ────────────────────────────────────────
        area_name   = ad.get("area_name", "")
        ward_name   = ad.get("ward_name", "")
        region_name = ad.get("region_name", "")
        street_name = ad.get("street_name", "")

        address_parts = [p for p in [street_name, ward_name, area_name, region_name] if p]
        address = ", ".join(address_parts)

        addr_comps   = parse_address_components(address)
        house_number = addr_comps["house_number"]
        alley        = addr_comps["alley"]
        street       = street_name or addr_comps["street"]
        ward         = ward_name or addr_comps["ward"]

        # ── Meta ────────────────────────────────────────────
        posted_at = _parse_posted_at(ad)
        contact   = ad.get("account_name", "")
        images    = [img for img in (ad.get("images", []) or []) if img][:10]

        # Property type từ category_name trước, fallback detect từ title
        cg_norm = cg_name.lower()
        prop_type = next(
            (v for k, v in CATEGORY_TYPE_MAP.items() if k in cg_norm),
            detect_property_type(title, url)
        )

        listing = Listing(
            source        = SOURCE,
            url           = url,
            title         = title,
            listing_type  = listing_type,
            property_type = prop_type,
            price         = price if price and price > 0 else None,
            price_text    = price_text,
            price_per_m2  = round(price / area, 0) if price and area else None,
            area          = area,
            area_text     = f"{area} m2" if area else "",
            address       = address,
            district      = area_name,
            ward          = ward,
            city          = region_name or "Ha Noi",
            street        = street,
            house_number  = house_number,
            alley         = alley,
            lat           = lat,
            lng           = lng,
            bedrooms      = bedrooms,
            bathrooms     = bathrooms,
            floors        = floors,
            facade        = facade,
            direction     = direction,
            legal_status  = legal_status,
            furniture     = furniture,
            year_built    = year_built,
            posted_at     = posted_at,
            contact       = contact,
            description   = description[:500] if description else "",
            images        = images,
        )
        return listing.to_dict() if listing.is_valid() else None

    except Exception as e:
        print(f"[{SOURCE}] Parse error ad_id={ad.get('ad_id','?')}: {e}")
        return None


def _safe_int(text: str):
    m = re.search(r"\d+", str(text))
    return int(m.group()) if m else None

def _safe_float(text: str):
    m = re.search(r"[\d.]+", str(text))
    return float(m.group()) if m else None


def crawl(max_pages: int = 5) -> list[dict]:
    all_listings = []
    for cat_id in CATEGORIES:
        print(f"[{SOURCE}] category={cat_id}")
        for page in range(max_pages):
            offset = page * 20
            ads = fetch_page(cat_id, offset)
            if not ads:
                print(f"  offset={offset}: het data")
                break
            parsed = [p for p in (parse_ad(a) for a in ads) if p]
            all_listings.extend(parsed)
            print(f"  offset={offset}: {len(ads)} ads -> {len(parsed)} valid")
            time.sleep(random.uniform(1, 2))
    print(f"[{SOURCE}] Tong: {len(all_listings)} listings")
    return all_listings


if __name__ == "__main__":
    import json
    listings = crawl(max_pages=1)
    if listings:
        print("\n=== Sample listing ===")
        sample = listings[0]
        for k, v in sample.items():
            if k != "images":
                print(f"  {k}: {v}")
