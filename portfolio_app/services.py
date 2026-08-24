from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from portfolio_app.config import Settings

logger = logging.getLogger(__name__)


class ExternalServiceError(RuntimeError):
    def __init__(self, public_message: str, log_message: str):
        super().__init__(log_message)
        self.public_message = public_message
        self.log_message = log_message


@dataclass(frozen=True)
class AIAnalysisResult:
    """AI 文字回應及其完成狀態，供 UI 清楚提示是否曾自動續寫。"""

    text: str
    provider: str
    model: str
    completed: bool = True
    continued: bool = False
    finish_reason: str | None = None


_CONTINUATION_PROMPT = (
    "上一段回應因輸出長度限制而中斷。請直接從未完成處繼續，只輸出尚未完成的內容；"
    "不要重複標題、前文或已完成段落，並完成原要求的所有章節與免責聲明。"
)
_INCOMPLETE_NOTICE = "\n\n> ⚠️ AI 服務再次達到輸出上限，以上報告可能仍不完整。"


def _merge_continuation(first: str, continuation: str) -> str:
    """Join a continuation while removing a small exact overlap at the boundary."""
    first, continuation = first.rstrip(), continuation.lstrip()
    if not continuation:
        return first
    if continuation.startswith(first):
        return continuation
    maximum = min(len(first), len(continuation), 500)
    for size in range(maximum, 19, -1):
        if first[-size:] == continuation[:size]:
            return first + continuation[size:]
    return f"{first}\n\n{continuation}"


def _session(*, retry_gets: bool = True) -> requests.Session:
    retry = Retry(
        total=3 if retry_gets else 0,
        connect=3 if retry_gets else 0,
        read=2 if retry_gets else 0,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        # 僅重試冪等 GET。AI POST 可能已計費，逾時後自動重送會造成重複成本。
        allowed_methods=frozenset({"GET"}) if retry_gets else frozenset(),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "code-gym-portfolio-risk/2.0"})
    return session


class FMPClient:
    BASE_URL = "https://financialmodelingprep.com"

    def __init__(self, settings: Settings):
        self.api_key = settings.fmp_api_key
        self.timeout = settings.request_timeout_seconds
        self.session = _session()
        # FMP supports header authorization. Keeping the key out of the URL
        # prevents proxy, browser and retry logs from recording it.
        self.session.headers.update({"apikey": self.api_key})

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = self.session.get(
                f"{self.BASE_URL}{path}", params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ExternalServiceError(
                "市場資料服務目前無法連線，請稍後再試。",
                f"FMP transport error: {type(exc).__name__}",
            ) from exc
        if response.status_code in (401, 403):
            raise ExternalServiceError(
                "市場資料服務的伺服器設定無效，請聯絡系統管理者。",
                f"FMP authentication failure: HTTP {response.status_code}",
            )
        if response.status_code != 200:
            raise ExternalServiceError(
                "市場資料服務暫時無法完成請求。",
                f"FMP API failure: HTTP {response.status_code}",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                "市場資料格式異常，請稍後再試。", "FMP returned invalid JSON"
            ) from exc

    def historical(self, symbol: str) -> list[dict[str, Any]]:
        end = date.today()
        start = end - timedelta(days=370)
        data = self._get(
            "/stable/historical-price-eod/full",
            {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
        )
        # Stable API 回傳陣列；保留 dict 相容處理，避免供應商過渡期格式差異。
        history = data if isinstance(data, list) else data.get("historical", []) if isinstance(data, dict) else []
        valid = [row for row in history if isinstance(row, dict) and row.get("close")]
        if len(valid) < 20:
            raise ExternalServiceError(
                f"{symbol} 的歷史價格資料不足，無法可靠計算風險。",
                f"Insufficient history for {symbol}: {len(valid)} rows",
            )
        return valid

    def price_history(self, symbol: str) -> list[dict[str, Any]]:
        """Return normalized ascending OHLCV rows for charts and quantitative analysis."""
        rows = self.historical(symbol)
        normalized = []
        for row in reversed(rows):
            normalized.append(
                {
                    "date": str(row.get("date", "")),
                    "open": float(row.get("open", row["close"])),
                    "high": float(row.get("high", row["close"])),
                    "low": float(row.get("low", row["close"])),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
            )
        return normalized

    def company(self, symbol: str) -> dict[str, Any]:
        data = self._get("/stable/profile", {"symbol": symbol})
        if isinstance(data, list) and data:
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            return {}
        return {
            "company_name": item.get("companyName") or symbol,
            "beta": _optional_float(item.get("beta")),
            "sector": item.get("sector") or "Unknown",
            "industry": item.get("industry") or "Unknown",
            "exchange": item.get("exchangeShortName") or "Unknown",
        }

    def asset_summary(self, symbol: str, include_beta: bool) -> dict[str, Any]:
        history = self.historical(symbol)
        company = self.company(symbol) if include_beta else {}
        closes = np.array([float(row["close"]) for row in reversed(history)], dtype=float)
        returns = np.diff(closes) / closes[:-1]
        annual_volatility = float(np.std(returns, ddof=1) * np.sqrt(252) * 100)
        annual_return = float((closes[-1] / closes[max(0, len(closes) - 252)] - 1) * 100)
        peaks = np.maximum.accumulate(closes)
        max_drawdown = float(np.min(closes / peaks - 1) * 100)
        beta = company.get("beta")
        return {
            "symbol": symbol,
            "company_name": company.get("company_name", symbol),
            "asset_type": "Stock",
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "volatility_level": _volatility_level(annual_volatility),
            "beta": beta,
            "beta_risk_level": _beta_level(beta),
            "sector": company.get("sector", "Unknown"),
            "industry": company.get("industry", "Unknown"),
            "exchange": company.get("exchange", "Unknown"),
            "daily_returns": [float(value) for value in returns[-252:]],
        }

    @staticmethod
    def cash_summary() -> dict[str, Any]:
        return {
            "symbol": "CASH",
            "company_name": "Cash Position",
            "asset_type": "Cash",
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "max_drawdown": 0.0,
            "volatility_level": "Very Low",
            "beta": None,
            "beta_risk_level": "無市場風險",
            "sector": "Cash",
            "industry": "Cash",
            "exchange": "N/A",
        }


class PerplexityClient:
    URL = "https://api.perplexity.ai/v1/sonar"

    def __init__(self, settings: Settings, model: str | None = None):
        self.api_key = settings.perplexity_api_key
        self.model = model or settings.perplexity_model
        self.timeout = settings.perplexity_timeout_seconds
        self.session = _session()

    def _request(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, str | None]:
        request_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = self.session.post(
                self.URL,
                headers=headers,
                json=request_payload,
                timeout=(5, self.timeout),
            )
        except requests.RequestException as exc:
            if isinstance(exc, requests.Timeout):
                raise ExternalServiceError(
                    f"AI 分析超過 {self.timeout} 秒仍未完成。可改用 sonar 模型，或稍後再試。",
                    f"Perplexity timeout after {self.timeout}s",
                ) from exc
            raise ExternalServiceError(
                "AI 分析服務目前無法連線，請稍後再試。",
                f"Perplexity transport error: {type(exc).__name__}",
            ) from exc
        if response.status_code in (401, 403):
            raise ExternalServiceError(
                "Perplexity API Key 無效或沒有存取權限，請檢查伺服器設定。",
                f"Perplexity authentication failure: HTTP {response.status_code}",
            )
        if response.status_code == 402:
            raise ExternalServiceError(
                "Perplexity API 帳戶餘額不足或尚未啟用 API 計費。",
                "Perplexity billing failure: HTTP 402",
            )
        if response.status_code == 429:
            raise ExternalServiceError(
                "Perplexity API 已達頻率或用量限制，請稍後再試。",
                "Perplexity rate limit: HTTP 429",
            )
        if response.status_code == 400:
            raise ExternalServiceError(
                f"Perplexity 不接受目前的模型或請求設定（模型：{self.model}）。",
                f"Perplexity invalid request: HTTP 400 model={self.model}",
            )
        if response.status_code != 200:
            raise ExternalServiceError(
                f"AI 分析服務暫時無法完成請求（HTTP {response.status_code}）。",
                f"Perplexity API failure: HTTP {response.status_code}",
            )
        try:
            payload = response.json()
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ExternalServiceError(
                "AI 分析服務回傳格式異常。", "Perplexity returned an invalid response"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ExternalServiceError(
                "AI 分析服務未回傳有效內容。", "Perplexity returned empty content"
            )
        return content.strip(), str(finish_reason) if finish_reason is not None else None

    def analyze(self, system_message: str, user_prompt: str) -> AIAnalysisResult:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        first, finish_reason = self._request(messages, 4500)
        if (finish_reason or "").lower() not in {"length", "max_tokens"}:
            return AIAnalysisResult(first, "Perplexity", self.model, finish_reason=finish_reason)

        continuation, second_reason = self._request(
            messages
            + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": _CONTINUATION_PROMPT},
            ],
            3500,
        )
        completed = (second_reason or "").lower() not in {"length", "max_tokens"}
        merged = _merge_continuation(first, continuation)
        if not completed:
            merged += _INCOMPLETE_NOTICE
        return AIAnalysisResult(
            merged,
            "Perplexity",
            self.model,
            completed=completed,
            continued=True,
            finish_reason=second_reason,
        )


class OpenAIClient:
    """Minimal Responses API client used by the unified news workflow."""

    URL = "https://api.openai.com/v1/responses"

    def __init__(self, settings: Settings, model: str | None = None):
        self.api_key = settings.openai_api_key
        self.model = model or settings.openai_model
        self.timeout = settings.perplexity_timeout_seconds
        self.session = _session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def _request(self, system_message: str, input_data: Any, max_output_tokens: int) -> dict[str, Any]:
        try:
            response = self.session.post(
                self.URL,
                json={
                    "model": self.model,
                    "instructions": system_message,
                    "input": input_data,
                    "max_output_tokens": max_output_tokens,
                    "store": False,
                },
                timeout=(5, self.timeout),
            )
        except requests.Timeout as exc:
            raise ExternalServiceError(
                "OpenAI 分析逾時，請縮小新聞數量後再試。",
                f"OpenAI timeout after {self.timeout}s",
            ) from exc
        except requests.RequestException as exc:
            raise ExternalServiceError(
                "OpenAI 分析服務目前無法連線。",
                f"OpenAI transport error: {type(exc).__name__}",
            ) from exc
        if response.status_code in (401, 403):
            raise ExternalServiceError(
                "OpenAI API Key 無效或沒有模型存取權限。",
                f"OpenAI authentication failure: HTTP {response.status_code}",
            )
        if response.status_code == 429:
            raise ExternalServiceError(
                "OpenAI API 已達頻率或用量限制。", "OpenAI rate limit: HTTP 429"
            )
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"OpenAI 不接受目前的請求設定（HTTP {response.status_code}）。",
                f"OpenAI API failure: HTTP {response.status_code} model={self.model}",
            )
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise ExternalServiceError(
                "OpenAI 分析服務回傳格式異常。", "OpenAI returned invalid JSON"
            ) from exc

    @staticmethod
    def _parse(payload: dict[str, Any]) -> tuple[str, bool, str | None]:
        try:
            chunks = [
                content["text"]
                for item in payload.get("output", [])
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text" and content.get("text")
            ]
            output = "\n".join(chunks).strip()
            status = str(payload.get("status") or "")
            details = payload.get("incomplete_details") or {}
            reason = details.get("reason") if isinstance(details, dict) else None
        except (TypeError, ValueError, KeyError) as exc:
            raise ExternalServiceError(
                "OpenAI 分析服務回傳格式異常。", "OpenAI returned invalid JSON"
            ) from exc
        if not output:
            raise ExternalServiceError(
                "OpenAI 分析服務未回傳有效內容。", "OpenAI returned empty output"
            )
        incomplete = status == "incomplete" or reason == "max_output_tokens"
        return output, not incomplete, str(reason) if reason is not None else None

    def analyze(self, system_message: str, user_prompt: str) -> AIAnalysisResult:
        first_payload = self._request(system_message, user_prompt, 5000)
        first, completed, reason = self._parse(first_payload)
        if completed:
            return AIAnalysisResult(first, "OpenAI", self.model, finish_reason=reason)

        continuation_input = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": first},
            {"role": "user", "content": _CONTINUATION_PROMPT},
        ]
        second_payload = self._request(system_message, continuation_input, 4000)
        continuation, completed, second_reason = self._parse(second_payload)
        merged = _merge_continuation(first, continuation)
        if not completed:
            merged += _INCOMPLETE_NOTICE
        return AIAnalysisResult(
            merged,
            "OpenAI",
            self.model,
            completed=completed,
            continued=True,
            finish_reason=second_reason,
        )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _volatility_level(value: float) -> str:
    if value < 20:
        return "Low"
    if value < 35:
        return "Medium"
    return "High"


def _beta_level(beta: float | None) -> str:
    if beta is None:
        return "無 Beta 數據"
    if beta > 1.2:
        return "高系統性風險"
    if beta > 0.8:
        return "中等系統性風險"
    return "低系統性風險"
