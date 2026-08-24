from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pypdf import PdfReader


class PTPFormatError(ValueError):
    pass


TICKER_LINE = re.compile(r"^([A-Z][A-Z0-9.]{0,9})\s+(?:[A-Z]{2})?[A-Z0-9]{8,12}\s+(.+)$")


def parse_ptp_pdf(content: bytes) -> dict[str, str]:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise PTPFormatError("無法讀取 PTP PDF。") from exc
    entries: dict[str, str] = {}
    for page in reader.pages:
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            match = TICKER_LINE.match(line)
            if match:
                entries[match.group(1)] = match.group(2).strip()
    if not entries:
        raise PTPFormatError("PDF 中沒有辨識到 PTP ticker；請確認檔案格式。")
    return entries


def screen_portfolio(
    portfolio: list[dict[str, Any]], ptp_entries: dict[str, str]
) -> list[dict[str, Any]]:
    matches = []
    for asset in portfolio:
        symbol = str(asset.get("symbol", "")).upper()
        if symbol in ptp_entries:
            matches.append(
                {
                    "Ticker": symbol,
                    "Allocation %": float(asset.get("allocation", 0)),
                    "PTP security": ptp_entries[symbol],
                }
            )
    return matches
