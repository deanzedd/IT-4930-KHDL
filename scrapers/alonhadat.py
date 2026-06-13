"""
scrapers/alonhadat.py - Crawl alonhadat.com.vn bằng requests + BeautifulSoup
Site dùng server-side render (PHP), không cần Selenium.
"""
import re
import sys
import time
import random
import requests
sys.stdout.reconfigure(encoding="utf-8")

from bs4 import BeautifulSoup
from models.listing import (
    Listing, parse_price, parse_area,
    detect_property_type, parse_address_components,
)

SOURCE = "alonhadat"
BASE_URL = "https://alonhadat.com.vn"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

# (listing_type, base_path)  — pagination dùng ?page=N
CATEGORIES = [
    ("ban",  "/can-ban-nha-dat/ha-noi"),
    ("thue", "/can-thue-nha-dat/ha-noi"),
]

# Session dùng chung để giữ cookies qua các request
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
        # Ghé thăm homepage để lấy cookies ban đầu
        try:
            _session.get(BASE_URL, timeout=10)
            time.sleep(random.uniform(1.0, 2.0))
        except Exception:
            pass
    return _session


def _get(url: str, referer: str = BASE_URL) -> BeautifulSoup | None:
    sess = _get_session()
    sess.headers.update({"Referer": referer})
    try:
        resp = sess.get(url, timeout=15, allow_redirects=True)
        # Nếu bị redirect sang trang block → trả về None
        if "vui-long-thu-lai" in resp.url or "thu-lai" in resp.url:
            print(f"[{SOURCE}] Bị rate-limit, chờ 30s rồi thử lại...")
            time.sleep(30)
            resp = sess.get(url, timeout=15)
            if "vui-long-thu-lai" in resp.url:
                print(f"[{SOURCE}] Vẫn bị block: {url[:60]}")
                return None
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[{SOURCE}] GET error {url}: {e}")
        return None


def _get_listing_urls(list_url: str) -> list[str]:
    soup = _get(list_url)
    if not soup:
        return []

    links = []
    # Mỗi listing nằm trong article.property-item
    for article in soup.select("article.property-item"):
        a = article.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        full = BASE_URL + href if href.startswith("/") else href
        if full not in links:
            links.append(full)

    return links


def _parse_listing(url: str, listing_type: str, referer: str = BASE_URL) -> dict | None:
    soup = _get(url, referer=referer)
    if not soup:
        return None
    try:
        # ── Tiêu đề ──────────────────────────────────
        title_el = soup.select_one("h1.title, h1")
        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            return None

        property_type = detect_property_type(title, url)

        # ── Giá  (selector chính: data.value, fallback span.price) ──
        price_text = ""
        price_el = soup.select_one("section.more-info data.value")
        if price_el:
            price_text = price_el.get_text(strip=True)
        else:
            el = soup.select_one("section.more-info span.price")
            if el:
                price_text = el.get_text(strip=True).replace("Giá:", "").strip()
        price = parse_price(price_text)

        # ── Diện tích ─────────────────────────────────
        area_text = ""
        area_el = soup.select_one("section.more-info span.area")
        if area_el:
            area_text = area_el.get_text(strip=True).replace("Diện tích:", "").strip()
        area = parse_area(area_text)

        # ── Địa chỉ ──────────────────────────────────
        address = ""
        addr_el = soup.select_one("p.old-address")
        if addr_el:
            address = addr_el.get_text(strip=True)
        if not address:
            for sel in ["div.address", "span.address", "div.location"]:
                el = soup.select_one(sel)
                if el:
                    address = el.get_text(strip=True)
                    break

        # ── Thông số kỹ thuật ─────────────────────────
        main_info = soup.select_one("section.more-info")
        bedrooms = bathrooms = floors = facade = None
        direction = legal_status = furniture = ""

        if main_info:
            fl_el = main_info.select_one("span.floors")
            if fl_el:
                m = re.search(r"\d+", fl_el.get_text())
                floors = int(m.group()) if m else None

            bed_el = main_info.select_one("span.bedroom")
            if bed_el:
                m = re.search(r"\d+", bed_el.get_text())
                bedrooms = int(m.group()) if m else None

            # Mặt tiền: dạng "X,Xm" trước span.floors
            for txt_node in main_info.find_all(string=True):
                m = re.search(r"([\d.,]+)\s*m\s*$", txt_node.strip())
                if m and facade is None:
                    try:
                        facade = float(m.group(1).replace(",", "."))
                    except ValueError:
                        pass

        # Bảng thông số bổ sung
        for row in soup.select("table.moreinfor tr, div.property-table tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            lbl = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True)
            if not val or val == "---":
                continue

            if any(k in lbl for k in ("toilet", "vệ sinh", "tắm")):
                m = re.search(r"\d+", val)
                bathrooms = int(m.group()) if m else None
            elif "số phòng ngủ" in lbl or "phòng ngủ" in lbl:
                if bedrooms is None:
                    m = re.search(r"\d+", val)
                    bedrooms = int(m.group()) if m else None
            elif "số tầng" in lbl or "tầng" in lbl:
                if floors is None:
                    m = re.search(r"\d+", val)
                    floors = int(m.group()) if m else None
            elif "mặt tiền" in lbl or "ngang" in lbl:
                m = re.search(r"[\d.]+", val)
                facade = float(m.group()) if m else None
            elif "hướng" in lbl:
                direction = val
            elif "pháp lý" in lbl or "giấy tờ" in lbl:
                legal_status = val
            elif "nội thất" in lbl:
                furniture = val

        district = _extract_district(soup, address)
        addr = parse_address_components(address)

        # ── Mô tả ─────────────────────────────────────
        desc_el = soup.select_one("div.detail-content, div#content-detail, div.description")
        description = desc_el.get_text(" ", strip=True)[:500] if desc_el else ""

        # ── Ngày đăng ─────────────────────────────────
        posted_at = ""
        for sel in ["span.date", "div.date", "time", "span[class*='time']"]:
            el = soup.select_one(sel)
            if el:
                posted_at = el.get_text(strip=True)
                break

        # ── Ảnh ───────────────────────────────────────
        images = []
        for img in soup.select("div.photo-list img, div.gallery img, div[class*='slide'] img"):
            src = img.get("src") or img.get("data-src", "")
            if src and src not in images:
                images.append(src)
        images = images[:10]

        listing = Listing(
            source=SOURCE,
            url=url,
            title=title,
            listing_type=listing_type,
            property_type=property_type,
            price=price,
            price_text=price_text,
            price_per_m2=round(price / area, 0) if price and area else None,
            area=area,
            area_text=area_text,
            address=address,
            district=district,
            ward=addr.get("ward", ""),
            city="Hà Nội",
            street=addr.get("street", ""),
            house_number=addr.get("house_number", ""),
            alley=addr.get("alley", ""),
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floors=floors,
            facade=facade,
            direction=direction,
            legal_status=legal_status,
            furniture=furniture,
            description=description,
            posted_at=posted_at,
            images=images,
        )
        return listing.to_dict() if listing.is_valid() else None

    except Exception as e:
        print(f"[{SOURCE}] Parse error {url}: {e}")
        return None


def _extract_district(soup: BeautifulSoup, address: str) -> str:
    for a in soup.select("div.breadcrumb a, ol.breadcrumb li a"):
        t = a.get_text(strip=True)
        if re.search(r"(quận|huyện)\s", t, re.I):
            return t
    m = re.search(r"(Quận|Huyện|quận|huyện)\s+[\w\s]+", address)
    return m.group(0).strip() if m else ""


def crawl(max_pages: int = 5) -> list[dict]:
    global _session
    _session = None  # fresh session mỗi lần crawl

    all_listings = []
    total_requests = 0

    for listing_type, base_path in CATEGORIES:
        cat_name = base_path.split("/")[1]
        print(f"[{SOURCE}] === {listing_type.upper()} | {cat_name} ===")
        for page in range(1, max_pages + 1):
            list_url = BASE_URL + base_path + (f"?page={page}" if page > 1 else "")
            print(f"[{SOURCE}] Page {page}: {list_url}")

            urls = _get_listing_urls(list_url)
            total_requests += 1
            print(f"[{SOURCE}]   -> {len(urls)} links")
            if not urls:
                print(f"[{SOURCE}]   Hết data, dừng category này.")
                break

            for u in urls:
                d = _parse_listing(u, listing_type, referer=list_url)
                total_requests += 1
                if d:
                    all_listings.append(d)
                    print(f"[{SOURCE}]   ✓ {d.get('title', '')[:55]}")
                time.sleep(random.uniform(1.0, 2.5))

                # Mỗi 50 requests nghỉ dài để tránh rate-limit
                if total_requests % 50 == 0:
                    print(f"[{SOURCE}]   Cooldown 90s sau {total_requests} requests...")
                    time.sleep(90)

            time.sleep(random.uniform(3.0, 6.0))

    print(f"[{SOURCE}] Xong. Tổng: {len(all_listings)} listings.")
    return all_listings


if __name__ == "__main__":
    listings = crawl(max_pages=1)
    if listings:
        print("\n=== Sample listing ===")
        for k, v in listings[0].items():
            if k != "images":
                print(f"  {k}: {v}")
