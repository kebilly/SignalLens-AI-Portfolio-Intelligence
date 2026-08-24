from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd


class HoldingsFormatError(ValueError):
    pass


COUNTRY_PREFIXES = {
    "TW": "Taiwan",
    "CN": "China",
    "HK": "Hong Kong",
    "IN": "India",
    "KR": "South Korea",
    "BR": "Brazil",
    "ZA": "South Africa",
    "SA": "Saudi Arabia",
    "MX": "Mexico",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "TH": "Thailand",
    "PL": "Poland",
    "TR": "Turkey",
    "AE": "United Arab Emirates",
    "QA": "Qatar",
    "KW": "Kuwait",
}


def parse_holdings(content: bytes, fallback_name: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldingsFormatError("ETF 檔案必須是 UTF-8 JSON 陣列。") from exc
    if not isinstance(payload, list) or not payload:
        raise HoldingsFormatError("ETF holdings 必須是非空白 JSON 陣列。")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        try:
            weight = float(item.get("weightPercentage", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise HoldingsFormatError(
                f"第 {index + 1} 筆 weightPercentage 無法解析。"
            ) from exc
        name = str(item.get("name") or "Unknown").strip()
        asset = str(item.get("asset") or "").strip().upper()
        isin = str(item.get("isin") or "").strip().upper()
        cusip = str(item.get("securityCusip") or "").strip().upper()
        identifier = isin or cusip or asset or name.upper()
        aliases = list(dict.fromkeys(value for value in (isin, cusip, asset) if value))
        rows.append(
            {
                "identifier": identifier,
                "aliases": aliases,
                "asset": asset or "N/A",
                "name": name,
                "isin": isin,
                "weight": max(weight, 0),
                "country": infer_country(isin, asset),
            }
        )
    if not rows:
        raise HoldingsFormatError("ETF 檔案沒有可用的 holdings 資料。")
    symbol = str(payload[0].get("symbol") or fallback_name.rsplit(".", 1)[0]).upper()
    return symbol, sorted(rows, key=lambda row: row["weight"], reverse=True)


def infer_country(isin: str, asset: str) -> str:
    if len(isin) >= 2 and isin[:2] in COUNTRY_PREFIXES:
        return COUNTRY_PREFIXES[isin[:2]]
    suffixes = {
        ".TW": "Taiwan",
        ".HK": "Hong Kong",
        ".KS": "South Korea",
        ".KQ": "South Korea",
    }
    for suffix, country in suffixes.items():
        if asset.endswith(suffix):
            return country
    return "Unclassified"


def comparison_table(
    left_symbol: str,
    left: list[dict[str, Any]],
    right_symbol: str,
    right: list[dict[str, Any]],
    limit: int = 20,
) -> pd.DataFrame:
    left_aliases = _alias_map(left)
    right_aliases = _alias_map(right)
    selected = left[:limit] + right[:limit]
    seen: set[tuple[str, str]] = set()
    result = []
    for row in selected:
        a = _find_match(row, left_aliases) or {}
        b = _find_match(row, right_aliases) or {}
        pair = (str(a.get("identifier", "")), str(b.get("identifier", "")))
        if pair in seen:
            continue
        seen.add(pair)
        left_weight = float(a.get("weight", 0))
        right_weight = float(b.get("weight", 0))
        result.append(
            {
                "Holding": a.get("name") or b.get("name"),
                "Ticker": a.get("asset") or b.get("asset"),
                f"{left_symbol} %": left_weight,
                f"{right_symbol} %": right_weight,
                "Difference (pp)": left_weight - right_weight,
            }
        )
    return pd.DataFrame(result).sort_values(
        by=[f"{left_symbol} %", f"{right_symbol} %"], ascending=False
    )


def country_exposure(rows: list[dict[str, Any]]) -> pd.DataFrame:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[row["country"]] += float(row["weight"])
    return pd.DataFrame(
        [{"Country": country, "Weight %": weight} for country, weight in totals.items()]
    ).sort_values("Weight %", ascending=False)


def overlap_metrics(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, float]:
    right_aliases = _alias_map(right)
    matched_right: set[str] = set()
    overlap_weight = 0.0
    common_count = 0
    for row in left:
        match = _find_match(row, right_aliases)
        if not match or match["identifier"] in matched_right:
            continue
        matched_right.add(match["identifier"])
        common_count += 1
        overlap_weight += min(float(row["weight"]), float(match["weight"]))
    return {
        "common_holdings": float(common_count),
        "overlap_weight": overlap_weight,
        "left_top10": sum(row["weight"] for row in left[:10]),
        "right_top10": sum(row["weight"] for row in right[:10]),
    }


def _alias_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for alias in row.get("aliases", [row["identifier"]]):
            result.setdefault(alias, row)
    return result


def _find_match(
    row: dict[str, Any], alias_map: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    for alias in row.get("aliases", [row["identifier"]]):
        if alias in alias_map:
            return alias_map[alias]
    return None
