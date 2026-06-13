"""
models/listing.py - Schema chuẩn hóa dữ liệu BĐS từ nhiều nguồn
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Listing:
    # Bắt buộc
    source: str            # "batdongsan" | "nhatot" | "mogi"
    url: str
    title: str
    listing_type: str      # "ban" | "thue"
    property_type: str     # "nha_pho" | "chung_cu" | "dat" | "biet_thu" | "khac"

    # Giá
    price: Optional[float] = None        # VNĐ (đã convert)
    price_text: str = ""                 # giá gốc từ trang ("15 tỷ", "5 tr/m²"...)
    price_per_m2: Optional[float] = None

    # Diện tích
    area: Optional[float] = None         # m²
    area_text: str = ""

    # Địa điểm (tổng quát)
    address: str = ""
    district: str = ""
    ward: str = ""
    city: str = "Hà Nội"

    # Địa điểm (chi tiết - phục vụ dự đoán giá)
    street: str = ""          # tên đường/phố (vd: "Hoàng Ngân", "Nguyễn Văn Linh")
    house_number: str = ""    # số nhà (vd: "15", "15/2")
    alley: str = ""           # ngõ/hẻm/ngách (vd: "ngõ 20", "hẻm 25/3")

    # Tọa độ địa lý
    lat: Optional[float] = None
    lng: Optional[float] = None

    # Thông số nhà
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floors: Optional[int] = None
    facade: Optional[float] = None       # mặt tiền (m)
    direction: str = ""                  # hướng nhà
    legal_status: str = ""               # pháp lý (sổ đỏ, sổ hồng, hợp đồng...)
    furniture: str = ""                  # nội thất (đầy đủ, cơ bản, không có)
    year_built: Optional[int] = None     # năm xây dựng

    # Meta
    posted_at: str = ""
    contact: str = ""
    description: str = ""
    images: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_valid(self) -> bool:
        """Kiểm tra listing có đủ thông tin tối thiểu không"""
        return bool(self.url and self.title and self.price is not None)


# ───────────────────────────────────────────────
# Hàm parse giá tiền từ text tiếng Việt
# ───────────────────────────────────────────────
def parse_price(text: str) -> Optional[float]:
    """
    '15 tỷ'        → 15_000_000_000
    '500 triệu'    → 500_000_000
    '2.5 tỷ'       → 2_500_000_000
    '15,000,000'   → 15_000_000
    'Thỏa thuận'   → None
    """
    if not text:
        return None

    text = text.lower().replace(",", ".").strip()

    if any(kw in text for kw in ["thỏa thuận", "liên hệ", "thoả thuận"]):
        return None

    ty_match = re.search(r"([\d.]+)\s*tỷ", text)
    if ty_match:
        return float(ty_match.group(1)) * 1_000_000_000

    trieu_match = re.search(r"([\d.]+)\s*triệu", text)
    if trieu_match:
        return float(trieu_match.group(1)) * 1_000_000

    num_match = re.search(r"[\d.]+", text)
    if num_match:
        return float(num_match.group().replace(".", ""))

    return None


def parse_area(text: str) -> Optional[float]:
    """'100 m²' → 100.0"""
    if not text:
        return None
    match = re.search(r"([\d.,]+)\s*m", text.lower())
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def detect_property_type(title: str, url: str = "") -> str:
    text = (title + " " + url).lower()

    if any(kw in text for kw in ["chung cư", "căn hộ", "ccmn", "chung cu", "căn hộ chung cư"]):
        return "chung_cu"

    if any(kw in text for kw in ["biệt thự", "biet thu", "villa", "liền kề", "lien ke"]):
        return "biet_thu"

    if any(kw in text for kw in [
        "bán đất", "cho thuê đất", "lô đất", "thổ cư", "tho cu",
        "quyền sử dụng", "qsd", "đất nền", "dat nen",
    ]):
        return "dat"

    _nha_pho_kws = [
        "bán nhà", "ban nha", "cho thuê nhà", "nhà phố", "nhà mặt",
        "nhà riêng", "nhà ngõ", "nhà hẻm", "nhà ở", "nhà đẹp",
        "nhà cấp 4", "nhà dân xây", "nhà mới", "nhà cũ",
        "tầng thang máy", "thang máy",
        "nhà 2 tầng", "nhà 3 tầng", "nhà 4 tầng", "nhà 5 tầng",
        "nhà 6 tầng", "nhà 7 tầng", "nhà 8 tầng", "nhà 9 tầng",
        "bán toà", "tòa nhà",
        "phân lô", "phan lo", "lô góc", "lo goc",
        "mặt tiền", "mat tien", "mặt phố", "mat pho",
        "phòng ngủ", "phong ngu",
        "kinh doanh", "ô tô tránh", "oto tranh",
        "sổ đỏ", "so do", "sổ hồng",
    ]
    if any(kw in text for kw in _nha_pho_kws):
        return "nha_pho"

    _dat_kws = [
        "bán lô", "ban lo", "lô đất", "lo dat",
        "bán đất", "ban dat", "quy hoạch", "thổ cư", "tho cu",
        "đất nền", "dat nen", "đường gom", "mặt đường",
        "trục chính", "trục đường",
    ]
    if any(kw in text for kw in _dat_kws):
        return "dat"

    import re as _re
    # "nhà" đứng độc lập trong chuỗi
    if _re.search(r'(?:^|\s)nh[àa](?:\s|$)', text):
        return "nha_pho"

    # Có "X tầng" → nhà ở
    if _re.search(r'\d+\s*t[aầ]ng', text):
        return "nha_pho"

    # Fallback: "đất" xuất hiện bất kỳ
    if "đất" in text or "dat" in text:
        return "dat"

    return "khac"


def parse_address_components(address: str) -> dict:
    """
    Tách địa chỉ tiếng Việt thành các thành phần chi tiết.
    Ví dụ: "Số 15, ngõ 20, phố Hoàng Ngân, phường Trung Hòa, quận Cầu Giấy"
    → {"house_number": "15", "alley": "ngõ 20", "street": "Hoàng Ngân", "ward": "phường Trung Hòa"}
    """
    result = {"house_number": "", "alley": "", "street": "", "ward": ""}
    if not address:
        return result

    # Số nhà: "Số 15", "số 12/3", "12B"
    m = re.search(r'\bsố\s+([\d]+[A-Za-z]?(?:[/\-][\d]+)*)', address, re.IGNORECASE)
    if m:
        result["house_number"] = m.group(1)

    # Ngõ/hẻm/ngách + số (vd: "ngõ 20", "hẻm 25/3", "ngách 5")
    m = re.search(r'\b(ngõ|ngách|hẻm)\s+([\d]+(?:[/\-][\d]+)*)', address, re.IGNORECASE)
    if m:
        result["alley"] = f"{m.group(1).lower()} {m.group(2)}"

    # Phố/đường - dừng trước dấu phẩy hoặc từ hành chính tiếp theo
    _admin_boundary = r'(?:\s*,|\s+(?:phường|xã|quận|huyện|thị\s+trấn|thị\s+xã|tp\b|thành\s+phố)|$)'
    m = re.search(
        r'\b(?:phố|đường|ph\.|đ\.)\s+([^,\n]+?)' + _admin_boundary,
        address, re.IGNORECASE
    )
    if m:
        result["street"] = m.group(1).strip().rstrip(",").strip()

    # Phường/xã/thị trấn
    m = re.search(
        r'\b(phường|xã|thị\s+trấn)\s+([^,\n]+?)(?:\s*,|$)',
        address, re.IGNORECASE
    )
    if m:
        ward_type = re.sub(r'\s+', ' ', m.group(1).strip())
        ward_name = m.group(2).strip().rstrip(",").strip()
        result["ward"] = f"{ward_type} {ward_name}"

    return result
