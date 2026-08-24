from __future__ import annotations

import re
from typing import Any

SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def normalize_symbol(value: str) -> str:
    return value.strip().upper()


def validate_portfolio(portfolio: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    total = sum(float(item.get("allocation", 0)) for item in portfolio)
    if abs(total - 100) > 0.001:
        errors.append(f"目前分配比例總計 {total:g}%，必須為 100%。")

    symbols: list[str] = []
    for item in portfolio:
        symbol = normalize_symbol(str(item.get("symbol", "")))
        allocation = float(item.get("allocation", 0))
        if allocation <= 0:
            continue
        if not symbol:
            errors.append("配置比例大於 0 的資產必須填寫股票代碼。")
        elif symbol != "CASH" and not SYMBOL_PATTERN.fullmatch(symbol):
            errors.append(f"股票代碼 {symbol!r} 格式不正確。")
        if symbol in symbols:
            errors.append(f"股票代碼 {symbol} 重複，請合併配置比例。")
        symbols.append(symbol)
    if len([s for s in symbols if s != "CASH"]) > 25:
        errors.append("單次最多分析 25 項風險資產。")
    return list(dict.fromkeys(errors))
