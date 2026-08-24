from __future__ import annotations

import requests

from portfolio_app.sentiment import parse_news_feed
from portfolio_app.services import ExternalServiceError, _session
from portfolio_app.validation import normalize_symbol


class AlphaVantageClient:
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key.strip()
        self.timeout = timeout
        # Alpha Vantage requires query-parameter authorization. Disable
        # automatic retries so transport warnings cannot repeat a keyed URL.
        self.session = _session(retry_gets=False)

    def news_sentiment(self, symbol: str, limit: int = 50) -> list[dict]:
        params: dict[str, str | int] = {
            "function": "NEWS_SENTIMENT",
            "tickers": normalize_symbol(symbol),
            "limit": min(max(int(limit), 1), 200),
            "apikey": self.api_key,
        }
        try:
            response = self.session.get(
                self.URL,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ExternalServiceError(
                "新聞資料服務目前無法連線，請稍後再試。",
                f"Alpha Vantage transport error: {type(exc).__name__}",
            ) from exc
        if response.status_code != 200:
            raise ExternalServiceError(
                f"新聞資料服務暫時無法完成請求（HTTP {response.status_code}）。",
                f"Alpha Vantage HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalServiceError("新聞資料格式異常。", "Alpha Vantage invalid JSON") from exc
        message = str(payload.get("Information") or payload.get("Note") or "")
        if message:
            raise ExternalServiceError(
                "Alpha Vantage 已達用量限制，或目前帳戶無法使用 NEWS_SENTIMENT。",
                f"Alpha Vantage provider message: {message[:200]}",
            )
        if "feed" not in payload:
            raise ExternalServiceError(
                "找不到此股票的新聞情緒資料。",
                f"Alpha Vantage missing feed; keys={list(payload)[:8]}",
            )
        return parse_news_feed(payload, symbol)

    def daily_prices(self, symbol: str) -> list[dict]:
        """Return normalized OHLCV rows in ascending date order."""
        try:
            response = self.session.get(
                self.URL,
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": normalize_symbol(symbol),
                    "outputsize": "compact",
                    "apikey": self.api_key,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ExternalServiceError(
                "Alpha Vantage 股價服務目前無法連線。",
                f"Alpha Vantage price transport error: {type(exc).__name__}",
            ) from exc
        if response.status_code != 200:
            raise ExternalServiceError(
                f"Alpha Vantage 股價服務暫時無法完成請求（HTTP {response.status_code}）。",
                f"Alpha Vantage price HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                "Alpha Vantage 股價資料格式異常。", "Alpha Vantage price invalid JSON"
            ) from exc
        message = str(
            payload.get("Information") or payload.get("Note") or payload.get("Error Message") or ""
        )
        if message:
            raise ExternalServiceError(
                "Alpha Vantage 已達用量限制，或股票代碼無效。",
                f"Alpha Vantage price provider message: {message[:200]}",
            )
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise ExternalServiceError(
                "找不到此股票的每日價格資料。",
                f"Alpha Vantage missing daily series; keys={list(payload)[:8]}",
            )
        rows = []
        for date, values in series.items():
            try:
                rows.append(
                    {
                        "date": date,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(float(values["5. volume"])),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(rows, key=lambda row: row["date"])
