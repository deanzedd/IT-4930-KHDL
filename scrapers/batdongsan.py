"""
scrapers/batdongsan.py - Crawl batdongsan.com.vn qua Selenium
Hỗ trợ: nhà ở và chung cư, bán và cho thuê tại Hà Nội
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


class BatDongSanScraper(BaseScraper):
    SOURCE = "batdongsan"
    BASE_URL = "https://batdongsan.com.vn"
    USE_UC = True  # bypass Cloudflare bot protection

    # (listing_type, url_path)
    CATEGORIES = [
        ("ban",  "/ban-nha-ha-noi"),
        ("ban",  "/ban-can-ho-chung-cu-ha-noi"),
        ("thue", "/cho-thue-nha-ha-noi"),
        ("thue", "/cho-thue-can-ho-chung-cu-ha-noi"),
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
                    page_url = self.BASE_URL + (path if page == 1 else f"{path}/p{page}")
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
        import time as _time
        self._ensure_driver()
        try:
            self.driver.get(url)
            # Đợi Cloudflare challenge qua (UC cần ~5-8s) rồi mới check listing
            _time.sleep(6)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a.js__product-link-for-unauth")
                )
            )
        except Exception:
            pass  # tiếp tục với HTML hiện tại
        # Scroll để trigger lazy load
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5)")
            _time.sleep(2)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            _time.sleep(2)
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

        # Selector chính: link listing cho unauthenticated
        for a in soup.select("a.js__product-link-for-unauth"):
            href = a.get("href", "")
            if href and href not in seen:
                seen.add(href)
                links.append(self.BASE_URL + href if href.startswith("/") else href)

        # Fallback: tìm href chứa /pr[số] (pattern URL listing của BĐS)
        if not links:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"/pr\d+", href) and href not in seen:
                    seen.add(href)
                    full = self.BASE_URL + href if href.startswith("/") else href
                    links.append(full)

        return links

    # ── Detail page ────────────────────────────────────
    def _parse_detail(self, url: str) -> dict | None:
        soup = self._get_soup(url)
        if not soup:
            return None
        try:
            title = self._text(soup, ["h1.re__pr-title", "h1"])
            if not title:
                return None

            listing_type = "thue" if "cho-thue" in url else "ban"
            property_type = detect_property_type(title, url)

            # Giá và diện tích từ section tóm tắt đầu trang
            price_text, area_text = self._parse_short_info(soup)
            price = parse_price(price_text)
            area = parse_area(area_text)

            # Thông số chi tiết (bảng spec)
            specs = self._parse_specs(soup)
            if area is None:
                area = parse_area(specs.get("area_text", ""))

            # Địa chỉ
            address = self._text(soup, [
                "span.re__pr-short-description",
                "div.re__pr-short-description",
                "div[class*='short-description']",
            ])
            district = self._extract_district(soup, address)
            addr = parse_address_components(address)

            # Tọa độ (JSON-LD hoặc JS data)
            lat, lng = self._extract_coords(soup)

            # Mô tả
            desc_el = soup.select_one(
                "div.re__detail-content, div[class*='detail-content']"
            )
            description = desc_el.get_text(" ", strip=True)[:500] if desc_el else ""

            # Ngày đăng
            posted_at = self._text(soup, [
                "span.re__pr-short-info--qc-time",
                "span[class*='time']",
                "div[class*='time'] > span",
            ])

            # Ảnh
            images = []
            for img in soup.select("img[class*='gallery'], div.re__pr-gallery img"):
                src = img.get("src") or img.get("data-src", "")
                if src and src not in images:
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
                area_text=area_text,
                address=address,
                district=district,
                ward=addr.get("ward", ""),
                city="Hà Nội",
                street=addr.get("street", ""),
                house_number=addr.get("house_number", ""),
                alley=addr.get("alley", ""),
                lat=lat,
                lng=lng,
                bedrooms=specs.get("bedrooms"),
                bathrooms=specs.get("bathrooms"),
                floors=specs.get("floors"),
                facade=specs.get("facade"),
                direction=specs.get("direction", ""),
                legal_status=specs.get("legal_status", ""),
                furniture=specs.get("furniture", ""),
                description=description,
                posted_at=posted_at,
                images=images,
            )
            return listing.to_dict() if listing.is_valid() else None

        except Exception as e:
            print(f"[{self.SOURCE}] Parse error {url}: {e}")
            return None

    def _parse_short_info(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Lấy giá và diện tích từ section tóm tắt đầu trang."""
        price_text = area_text = ""
        for item in soup.select("div.re__pr-short-info-item"):
            val_el = item.select_one("span.re__pr-short-info-item__value")
            lbl_el = item.select_one("span.re__pr-short-info-item__text")
            if not val_el:
                continue
            v = val_el.get_text(strip=True)
            l = lbl_el.get_text(strip=True).lower() if lbl_el else ""

            if ("tỷ" in v or "triệu" in v or "giá" in l) and not price_text:
                price_text = v
            elif (re.search(r"m[²2]", v) or "tích" in l) and not area_text:
                area_text = v

        # Fallback: scan toàn trang tìm text ngắn chứa tỷ/m²
        if not price_text:
            for el in soup.find_all(True):
                t = el.get_text(strip=True)
                if ("tỷ" in t or "triệu" in t) and 3 < len(t) < 25 and not el.find():
                    price_text = t
                    break
        if not area_text:
            for el in soup.find_all(True):
                t = el.get_text(strip=True)
                if re.search(r"\d+\s*m[²2]", t) and len(t) < 20 and not el.find():
                    area_text = t
                    break

        return price_text, area_text

    def _parse_specs(self, soup: BeautifulSoup) -> dict:
        """Parse bảng thông số kỹ thuật chi tiết."""
        specs: dict = {}
        for item in soup.select("div.re__pr-specs-content-item"):
            lbl_el = item.select_one("span.re__pr-specs-content-item-title")
            val_el = item.select_one("span.re__pr-specs-content-item-value")
            if not lbl_el or not val_el:
                continue
            lbl = lbl_el.get_text(strip=True).lower()
            val = val_el.get_text(strip=True)

            if "diện tích" in lbl:
                specs["area_text"] = val
            elif "phòng ngủ" in lbl:
                specs["bedrooms"] = self._int(val)
            elif any(k in lbl for k in ("vệ sinh", "toilet", "wc", "tắm")):
                specs["bathrooms"] = self._int(val)
            elif "số tầng" in lbl or ("tầng" in lbl and "số" in lbl):
                specs["floors"] = self._int(val)
            elif "mặt tiền" in lbl or "ngang" in lbl:
                specs["facade"] = self._float(val)
            elif "hướng" in lbl:
                specs["direction"] = val
            elif "pháp lý" in lbl or "giấy tờ" in lbl:
                specs["legal_status"] = val
            elif "nội thất" in lbl:
                specs["furniture"] = val

        return specs

    def _extract_district(self, soup: BeautifulSoup, address: str) -> str:
        for li in soup.select("ol.re__breadcrumb li"):
            t = li.get_text(strip=True)
            if re.search(r"(quận|huyện)\s", t, re.I):
                return t
        m = re.search(r"(Quận|Huyện|quận|huyện)\s+[\w\s]+", address)
        return m.group(0).strip() if m else ""

    def _extract_coords(self, soup: BeautifulSoup) -> tuple:
        # Thử JSON-LD schema.org
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.string or "")
                geo = data.get("geo") or {}
                lat = float(geo.get("latitude") or 0)
                lng = float(geo.get("longitude") or 0)
                if lat and lng:
                    return lat, lng
            except Exception:
                pass
        # Thử tìm pattern trong JS inline
        for s in soup.find_all("script"):
            text = s.string or ""
            m = re.search(
                r'"lat(?:itude)?"\s*:\s*([\d.]+).*?"l(?:ng|on(?:gitude)?)"\s*:\s*([\d.]+)',
                text
            )
            if m:
                try:
                    return float(m.group(1)), float(m.group(2))
                except Exception:
                    pass
        return None, None

    # ── Helpers ────────────────────────────────────────
    def _text(self, soup: BeautifulSoup, selectors: list[str]) -> str:
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                return el.get_text(strip=True)
        return ""

    def _int(self, text: str) -> int | None:
        m = re.search(r"\d+", str(text))
        return int(m.group()) if m else None

    def _float(self, text: str) -> float | None:
        m = re.search(r"[\d.]+", str(text))
        return float(m.group()) if m else None


def crawl(max_pages: int = 5) -> list[dict]:
    return BatDongSanScraper(max_pages=max_pages).crawl()


if __name__ == "__main__":
    listings = crawl(max_pages=1)
    if listings:
        print("\n=== Sample listing ===")
        for k, v in listings[0].items():
            if k != "images":
                print(f"  {k}: {v}")
