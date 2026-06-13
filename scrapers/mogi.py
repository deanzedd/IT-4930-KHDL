"""
scrapers/mogi.py - Crawl mogi.vn qua Selenium
Hỗ trợ: bán và cho thuê nhà đất tại Hà Nội
"""
import re
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from .base import BaseScraper
from models.listing import (
    Listing, parse_price, parse_area,
    detect_property_type, parse_address_components,
)

# Listing URL: mogi.vn/{district}/{mua-*|thue-*}/{title-id}
_LISTING_URL_RE = re.compile(r"mogi\.vn/[^/]+/(mua-|thue-|cho-thue-)[^?#]+-id\d+")


class MogiScraper(BaseScraper):
    SOURCE = "mogi"
    BASE_URL = "https://mogi.vn"

    # (listing_type, url_path)
    CATEGORIES = [
        ("ban",  "/ha-noi/mua-nha"),
        ("ban",  "/ha-noi/mua-dat"),
        ("thue", "/ha-noi/thue-nha"),
    ]

    def __init__(self, max_pages: int = 5):
        super().__init__()
        self._max_pages = max_pages

    # ── Required by ABC ────────────────────────────────
    def get_max_pages(self) -> int:
        return self._max_pages

    def get_listing_urls(self, page: int) -> list[str]:
        return []  # không dùng - crawl() được override

    def parse_listing(self, url: str) -> dict | None:
        return self._parse_detail(url)

    # ── Override main crawl loop ───────────────────────
    def crawl(self) -> list[dict]:
        self._init_driver()
        all_listings = []

        try:
            for listing_type, path in self.CATEGORIES:
                print(f"[{self.SOURCE}] === {listing_type.upper()} | {path} ===")
                for page in range(1, self._max_pages + 1):
                    base = self.BASE_URL + path
                    page_url = base if page == 1 else f"{base}?cp={page}"
                    print(f"[{self.SOURCE}] Page {page}: {page_url}")

                    urls = self._get_urls_from_page(page_url)
                    print(f"[{self.SOURCE}]   -> {len(urls)} links")
                    if not urls:
                        print(f"[{self.SOURCE}]   Hết data, dừng category này.")
                        break

                    for u in urls:
                        d = self._parse_detail(u)
                        if d:
                            d["listing_type"] = listing_type
                            all_listings.append(d)
                            print(f"[{self.SOURCE}]   ✓ {d.get('title', '')[:55]}")
                        self._random_delay()

        except KeyboardInterrupt:
            print(f"\n[{self.SOURCE}] Dừng bởi người dùng.")
        finally:
            self._close_driver()

        print(f"[{self.SOURCE}] Xong. Tổng: {len(all_listings)} listings.")
        return all_listings

    # ── List page ──────────────────────────────────────
    def _get_urls_from_page(self, url: str) -> list[str]:
        self._ensure_driver()
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a[href*='/mua-'], a[href*='/thue-']")
                )
            )
        except Exception:
            pass
        self._random_delay()

        try:
            page_source = self.driver.page_source
        except Exception as e:
            print(f"[{self.SOURCE}] Mất session khi đọc page source: {e}")
            return []

        soup = BeautifulSoup(page_source, "html.parser")
        seen: set[str] = set()
        links: list[str] = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = self.BASE_URL + href if href.startswith("/") else href
            if _LISTING_URL_RE.search(full) and full not in seen:
                seen.add(full)
                links.append(full)

        return links[:40]

    # ── Detail page ────────────────────────────────────
    def _parse_detail(self, url: str) -> dict | None:
        soup = self._get_soup(url)
        if not soup:
            return None
        try:
            title = self._text(soup, ["h1.prop-title", "h1.title", "h1"])
            if not title:
                return None

            listing_type = "thue" if "cho-thue" in url else "ban"
            property_type = detect_property_type(title, url)

            # ── Giá ──────────────────────────────────────
            price_text = ""
            for el in soup.select(".price"):
                t = el.get_text(strip=True)
                if t and ("tỷ" in t or "triệu" in t or "tr/" in t.lower()):
                    price_text = t
                    break
            price = parse_price(price_text)

            # ── Specs từ div.info-attr (label / value spans) ──────────
            area_text = bedrooms = bathrooms = floors = facade = None
            direction = legal_status = furniture = posted_at_spec = ""

            for attr in soup.select("div.info-attr"):
                spans = attr.find_all("span", recursive=False)
                if len(spans) < 2:
                    continue
                lbl = spans[0].get_text(strip=True).lower()
                val = spans[1].get_text(" ", strip=True)

                if "diện tích" in lbl and area_text is None:
                    area_text = val
                elif "phòng ngủ" in lbl:
                    m = re.search(r"\d+", val)
                    bedrooms = int(m.group()) if m else None
                elif any(k in lbl for k in ("nhà tắm", "toilet", "wc")):
                    m = re.search(r"\d+", val)
                    bathrooms = int(m.group()) if m else None
                elif "số tầng" in lbl or lbl == "tầng":
                    m = re.search(r"\d+", val)
                    floors = int(m.group()) if m else None
                elif "mặt tiền" in lbl or "ngang" in lbl:
                    m = re.search(r"[\d.,]+", val)
                    facade = float(m.group().replace(",", ".")) if m else None
                elif "hướng" in lbl:
                    direction = val
                elif "pháp lý" in lbl or "giấy tờ" in lbl:
                    legal_status = val
                elif "nội thất" in lbl:
                    furniture = val
                elif "ngày đăng" in lbl:
                    posted_at_spec = val

            area = parse_area(area_text or "")

            # ── Địa chỉ ──────────────────────────────────
            address = self._text(soup, [
                "div.prop-loccation",   # note: typo in mogi CSS
                "div[class*='address']",
                "div.prop-address",
                "address",
                "div[class*='location']",
            ])
            district = self._extract_district(address)
            addr = parse_address_components(address)

            # ── Tọa độ ───────────────────────────────────
            lat, lng = self._extract_coords(soup)

            # ── Mô tả ─────────────────────────────────────
            desc_el = soup.select_one(
                "div.prop-description, div[class*='description'], "
                "div.detail-content, div[class*='content']"
            )
            description = desc_el.get_text(" ", strip=True)[:500] if desc_el else ""

            # ── Ngày đăng ─────────────────────────────────
            posted_at = posted_at_spec  # lấy từ spec table

            # ── Ảnh ───────────────────────────────────────
            images = []
            for img in soup.select("#gallery img, div.top-media img, div[class*='gallery'] img"):
                src = img.get("src") or img.get("data-src", "")
                if src and "cloud.mogi" in src and src not in images:
                    images.append(src)
            images = images[:10]

            listing = Listing(
                source=self.SOURCE,
                url=url,
                title=title,
                listing_type=listing_type,
                property_type=property_type,
                price=price,
                price_text=price_text,
                price_per_m2=round(price / area, 0) if price and area else None,
                area=area,
                area_text=area_text or "",
                address=address,
                district=district,
                ward=addr.get("ward", ""),
                city="Hà Nội",
                street=addr.get("street", ""),
                house_number=addr.get("house_number", ""),
                alley=addr.get("alley", ""),
                lat=lat,
                lng=lng,
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
            print(f"[{self.SOURCE}] Parse error {url}: {e}")
            return None

    # ── Helpers ────────────────────────────────────────
    def _extract_district(self, address: str) -> str:
        m = re.search(r"(Quận|Huyện|quận|huyện)\s+[\w\s]+", address)
        return m.group(0).strip() if m else ""

    def _extract_coords(self, soup: BeautifulSoup) -> tuple:
        # JSON-LD schema.org
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.string or "")
                geo = data.get("geo") or {}
                lat = float(geo.get("latitude") or 0)
                lng = float(geo.get("longitude") or 0)
                # Validate: Hanoi bounding box (lat 20.5-21.5, lng 105.3-106.1)
                if 20.5 <= lat <= 21.5 and 105.3 <= lng <= 106.1:
                    return lat, lng
            except Exception:
                pass
        # Pattern trong JS inline — chỉ nhận tọa độ trong khu vực Hà Nội
        for s in soup.find_all("script"):
            text = s.string or ""
            m = re.search(
                r'"lat(?:itude)?"\s*:\s*(2[01]\.[0-9]+).*?"'
                r'l(?:ng|on(?:gitude)?)"\s*:\s*(10[5-6]\.[0-9]+)',
                text
            )
            if m:
                try:
                    return float(m.group(1)), float(m.group(2))
                except Exception:
                    pass
        return None, None

    def _text(self, soup: BeautifulSoup, selectors: list[str]) -> str:
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                return el.get_text(strip=True)
        return ""


def crawl(max_pages: int = 5) -> list[dict]:
    return MogiScraper(max_pages=max_pages).crawl()


if __name__ == "__main__":
    listings = crawl(max_pages=1)
    if listings:
        print("\n=== Sample listing ===")
        for k, v in listings[0].items():
            if k != "images":
                print(f"  {k}: {v}")
