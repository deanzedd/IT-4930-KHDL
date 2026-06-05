#!/usr/bin/env python3
"""
Agent điền dữ liệu bất động sản Hà Nội
Sử dụng Qwen2.5-3B-Instruct (vLLM) để điền các trường còn thiếu
từ title và description của bất động sản.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─── Target fields ───────────────────────────────────────────────────────────

NUMERIC_FIELDS = ["bedrooms_detail", "bathrooms_detail", "floors", "width_m", "depth_m"]
TEXT_FIELDS    = ["full_address"]
TARGET_FIELDS  = NUMERIC_FIELDS + TEXT_FIELDS


# ─── Prompt builder ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích bất động sản Việt Nam.
Nhiệm vụ của bạn là trích xuất thông tin từ tiêu đề và mô tả bất động sản.
Chỉ trả về JSON, không giải thích thêm.
Nếu không tìm được thông tin, trả về null cho trường đó."""

def build_user_prompt(title: str, description: str, missing_fields: list[str]) -> str:
    fields_desc = []
    for f in missing_fields:
        if f == "bedrooms_detail":
            fields_desc.append(f'  "{f}": <số phòng ngủ, kiểu số thực, ví dụ: 3 hoặc null>')
        elif f == "bathrooms_detail":
            fields_desc.append(f'  "{f}": <số phòng tắm/vệ sinh, kiểu số thực, ví dụ: 2 hoặc null>')
        elif f == "floors":
            fields_desc.append(f'  "{f}": <số tầng, kiểu số thực, ví dụ: 4 hoặc null>')
        elif f == "width_m":
            fields_desc.append(f'  "{f}": <chiều rộng mặt tiền tính bằng mét, kiểu số thực, ví dụ: 4.5 hoặc null>')
        elif f == "depth_m":
            fields_desc.append(f'  "{f}": <chiều sâu/chiều dài tính bằng mét, kiểu số thực, ví dụ: 12.0 hoặc null>')
        elif f == "full_address":
            fields_desc.append(f'  "{f}": <địa chỉ đầy đủ (số nhà, phố, phường, quận, thành phố), ví dụ: "12 Đường Láng, Phường Láng Hạ, Quận Đống Đa, Hà Nội" hoặc null>')
        elif f == "area":
            fields_desc.append(f'  "{f}": <diện tích sử dụng đất tính bằng mét vuông, kiểu số thực, ví dụ: 100.0 hoặc null>')

    fields_json = "{\n" + ",\n".join(fields_desc) + "\n}"

    return f"""Tiêu đề bất động sản: {title}

Mô tả: {description[:1500] if description else "(không có mô tả)"}

Hãy trích xuất các thông tin sau từ tiêu đề và mô tả trên. Chỉ trả về JSON:
{fields_json}"""


# ─── Checkpoint ──────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, checkpoint_path: str):
        self.path = checkpoint_path
        self.processed: set[int] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.processed = set(data.get("processed_indices", []))
                print(f"[Checkpoint] Loaded {len(self.processed)} previously processed rows.")
            except Exception:
                self.processed = set()

    def save(self, index: int):
        self.processed.add(index)
        # Write periodically (every 10 saves) to reduce I/O
        if len(self.processed) % 10 == 0:
            self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump({"processed_indices": list(self.processed)}, f)

    def finalize(self):
        self._flush()

    def is_done(self, index: int) -> bool:
        return index in self.processed


# ─── LLM Client ──────────────────────────────────────────────────────────────

async def call_llm(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    title: str,
    description: str,
    missing_fields: list[str],
    max_retries: int = 3,
    timeout: int = 60,
) -> dict[str, Any]:
    """Gọi vLLM API và trả về dict các giá trị được điền."""
    user_prompt = build_user_prompt(title, description, missing_fields)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                return parse_json_response(raw, missing_fields)
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return {f: None for f in missing_fields}
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return {f: None for f in missing_fields}

    return {f: None for f in missing_fields}


def parse_json_response(raw: str, missing_fields: list[str]) -> dict[str, Any]:
    """Parse JSON từ LLM response, xử lý các trường hợp LLM trả về text thừa."""
    # Thử extract JSON block nếu LLM trả về thêm text
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {f: None for f in missing_fields}

    cleaned: dict[str, Any] = {}
    for field in missing_fields:
        val = result.get(field)
        if val is None or val == "" or val == "null":
            cleaned[field] = None
        elif field in NUMERIC_FIELDS:
            try:
                cleaned[field] = float(val)
            except (ValueError, TypeError):
                cleaned[field] = None
        else:
            cleaned[field] = str(val).strip() if val else None

    return cleaned


# ─── Semaphore-controlled worker ─────────────────────────────────────────────

async def process_row(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    idx: int,
    row: pd.Series,
    checkpoint: Checkpoint,
    logger: logging.Logger,
) -> tuple[int, dict[str, Any] | None]:
    """Xử lý một hàng: kiểm tra trường rỗng → gọi LLM → trả về kết quả."""
    if checkpoint.is_done(idx):
        return idx, None  # Đã xử lý, skip

    # Xác định trường nào đang thiếu
    missing = [f for f in TARGET_FIELDS if pd.isna(row.get(f)) or row.get(f) == ""]
    if not missing:
        checkpoint.save(idx)
        return idx, {}  # Không thiếu gì, skip

    title = str(row.get("title", "")) or ""
    desc  = str(row.get("description", "")) or ""

    async with sem:
        result = await call_llm(session, api_url, model_name, title, desc, missing)

    filled_count = sum(1 for v in result.values() if v is not None)
    logger.info(f"Row {idx}: filled {filled_count}/{len(missing)} fields {missing}")

    checkpoint.save(idx)
    return idx, result


# ─── Main pipeline ───────────────────────────────────────────────────────────

async def run_agent(args: argparse.Namespace):
    log_path = str(Path(args.output).parent / "agent_run.log")
    checkpoint_path = str(Path(args.output).parent / "checkpoint.json")
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Agent khởi động")
    logger.info(f"  Input : {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  API   : {args.api_url}")
    logger.info(f"  Workers: {args.workers}")

    # Load CSV
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Load checkpoint
    checkpoint = Checkpoint(checkpoint_path)

    # Lấy model name từ API
    model_name = args.model
    if not model_name:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{args.api_url}/v1/models") as r:
                data = await r.json()
                model_name = data["data"][0]["id"]
    logger.info(f"  Model : {model_name}")

    # Xác định hàng cần xử lý
    needs_processing = []
    for idx, row in df.iterrows():
        if checkpoint.is_done(idx):
            continue
        missing = [f for f in TARGET_FIELDS if pd.isna(row.get(f)) or row.get(f) == ""]
        if missing:
            needs_processing.append(idx)

    logger.info(f"Rows cần xử lý: {len(needs_processing)} / {len(df)}")

    if not needs_processing:
        logger.info("Không có hàng nào cần xử lý. Kết thúc.")
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        return

    # Semaphore giới hạn concurrency
    sem = asyncio.Semaphore(args.workers)

    connector = aiohttp.TCPConnector(limit=args.workers + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_row(sem, session, args.api_url, model_name, idx, df.iloc[idx], checkpoint, logger)
            for idx in needs_processing
        ]

        results = []
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Filling rows"):
            result = await coro
            results.append(result)

    # Áp dụng kết quả vào DataFrame
    filled_total = 0
    for idx, result in results:
        if result is None or result == {}:
            continue
        for field, value in result.items():
            if value is not None:
                df.at[idx, field] = value
                filled_total += 1

    logger.info(f"Tổng số giá trị đã điền: {filled_total}")

    # Lưu output
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu file output: {args.output}")

    checkpoint.finalize()
    logger.info("Agent hoàn thành!")

    # Thống kê sau khi chạy
    df_out = pd.read_csv(args.output)
    logger.info("─── Null counts sau khi điền ───")
    for f in TARGET_FIELDS:
        if f in df_out.columns:
            null_count = df_out[f].isna().sum()
            logger.info(f"  {f}: {null_count} null còn lại")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent điền dữ liệu bất động sản dùng Qwen2.5-3B qua vLLM"
    )
    parser.add_argument(
        "--input",
        default="/mnt/disk1/theanh/IT-4930-KHDL/data/hanoi_final_filtered_features.csv",
        help="Đường dẫn file CSV đầu vào",
    )
    parser.add_argument(
        "--output",
        default="/mnt/disk1/theanh/IT-4930-KHDL/data/hanoi_final_filled.csv",
        help="Đường dẫn file CSV đầu ra",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="URL của vLLM server",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Tên model (để trống để tự detect từ API)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số request song song tối đa (mặc định: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="(Dự phòng, không dùng trực tiếp) Kích thước batch",
    )

    args = parser.parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
