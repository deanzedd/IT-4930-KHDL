"""
scrapers/base.py - Abstract base class cho tất cả scrapers
"""
import os
import time
import random
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


class BaseScraper(ABC):
    SOURCE = ""  # override trong subclass
    USE_UC = False  # subclass set True để dùng undetected-chromedriver (bypass Cloudflare)

    def __init__(self):
        self.driver = None
        self.delay_min = float(os.getenv("DELAY_MIN", 2))
        self.delay_max = float(os.getenv("DELAY_MAX", 5))
        self.headless = os.getenv("HEADLESS", "true").lower() == "true"

    # ── Selenium / UC setup ────────────────────────
    def _init_driver(self):
        if self.USE_UC:
            self._init_uc_driver()
        else:
            self._init_selenium_driver()

    def _init_selenium_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)
        print(f"[{self.SOURCE}] Driver initialized (Selenium).")

    def _init_uc_driver(self):
        """Dùng undetected-chromedriver để bypass Cloudflare."""
        import undetected_chromedriver as uc
        import subprocess, re as _re
        # Tự detect phiên bản Chrome đang cài (thử HKCU trước, fallback HKLM)
        chrome_ver = None
        for hive in ("HKCU", "HKLM"):
            try:
                out = subprocess.check_output(
                    f'reg query "{hive}\\SOFTWARE\\Google\\Chrome\\BLBeacon" /v version',
                    shell=True, stderr=subprocess.DEVNULL
                ).decode()
                m = _re.search(r"(\d+)\.\d+\.\d+\.\d+", out)
                if m:
                    chrome_ver = int(m.group(1))
                    break
            except Exception:
                continue
        print(f"[{self.SOURCE}] Chrome version detected: {chrome_ver}")

        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = uc.Chrome(
            options=options,
            headless=False,
            version_main=chrome_ver,  # khớp đúng phiên bản Chrome đang cài
        )
        self.driver.set_page_load_timeout(45)
        print(f"[{self.SOURCE}] Driver initialized (undetected-chromedriver v{chrome_ver}).")

    def _close_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _ensure_driver(self):
        """Kiểm tra session còn sống không, nếu chết thì khởi tạo lại."""
        try:
            _ = self.driver.title  # ping session
        except Exception:
            print(f"[{self.SOURCE}] Session mất — khởi tạo lại driver...")
            self._close_driver()
            self._init_driver()

    def _get_soup(self, url: str) -> BeautifulSoup | None:
        """Mở URL bằng Selenium, trả về BeautifulSoup object"""
        self._ensure_driver()
        try:
            self.driver.get(url)
            self._random_delay()
            return BeautifulSoup(self.driver.page_source, "html.parser")
        except Exception as e:
            print(f"[{self.SOURCE}] Error loading {url}: {e}")
            return None

    def _random_delay(self):
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    # ── Interface cần implement ────────────────────
    @abstractmethod
    def get_listing_urls(self, page: int) -> list[str]:
        """Lấy danh sách URL từ trang listing (phân trang)"""
        pass

    @abstractmethod
    def parse_listing(self, url: str) -> dict | None:
        """Parse chi tiết 1 listing từ URL"""
        pass

    @abstractmethod
    def get_max_pages(self) -> int:
        """Số trang tối đa muốn crawl"""
        pass

    # ── Main crawl loop ────────────────────────────
    def crawl(self) -> list[dict]:
        """Chạy toàn bộ quá trình crawl, trả về list listings"""
        self._init_driver()
        all_listings = []

        try:
            max_pages = self.get_max_pages()
            print(f"[{self.SOURCE}] Crawling {max_pages} pages...")

            for page in range(1, max_pages + 1):
                print(f"[{self.SOURCE}] Page {page}/{max_pages}")
                urls = self.get_listing_urls(page)
                print(f"[{self.SOURCE}]   Found {len(urls)} listings")

                for url in urls:
                    listing = self.parse_listing(url)
                    if listing:
                        all_listings.append(listing)
                        print(f"[{self.SOURCE}]   ✓ {listing.get('title', '')[:50]}")
                    self._random_delay()

        except KeyboardInterrupt:
            print(f"\n[{self.SOURCE}] Crawl interrupted by user.")
        except Exception as e:
            print(f"[{self.SOURCE}] Unexpected error: {e}")
        finally:
            self._close_driver()

        print(f"[{self.SOURCE}] Done. Total: {len(all_listings)} listings.")
        return all_listings
