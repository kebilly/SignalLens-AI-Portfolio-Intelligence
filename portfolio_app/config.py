from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(RuntimeError):
    pass


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    return value


@dataclass(frozen=True)
class Settings:
    fmp_api_key: str = ""
    perplexity_api_key: str = ""
    alpha_vantage_api_key: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    perplexity_model: str = "sonar-reasoning-pro"
    request_timeout_seconds: int = 20
    perplexity_timeout_seconds: int = 120

    @classmethod
    def load(cls) -> Settings:
        fmp = _secret("FMP_API_KEY")
        perplexity = _secret("PERPLEXITY_API_KEY")
        timeout = int(_secret("REQUEST_TIMEOUT_SECONDS") or "20")
        timeout = min(max(timeout, 5), 60)
        ai_timeout = int(_secret("PERPLEXITY_TIMEOUT_SECONDS") or "120")
        ai_timeout = min(max(ai_timeout, 30), 300)
        return cls(
            fmp_api_key=fmp,
            perplexity_api_key=perplexity,
            alpha_vantage_api_key=_secret("ALPHA_VANTAGE_API_KEY"),
            openai_api_key=_secret("OPENAI_API_KEY"),
            openai_model=_secret("OPENAI_MODEL") or "gpt-5-mini",
            perplexity_model=_secret("PERPLEXITY_MODEL") or "sonar-reasoning-pro",
            request_timeout_seconds=timeout,
            perplexity_timeout_seconds=ai_timeout,
        )

    @property
    def sentiment_enabled(self) -> bool:
        return bool(self.alpha_vantage_api_key)

    @property
    def portfolio_enabled(self) -> bool:
        return bool(self.fmp_api_key)

    @property
    def ai_enabled(self) -> bool:
        return bool(self.openai_api_key or self.perplexity_api_key)
