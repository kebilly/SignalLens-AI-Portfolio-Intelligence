from __future__ import annotations

from datetime import UTC, datetime, timedelta


def demo_news(symbol: str = "AAPL") -> list[dict]:
    base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    samples = [
        ("Product demand remains resilient in latest channel checks", 0.31, 0.78, "Technology"),
        ("Analysts debate valuation after recent share-price advance", -0.08, 0.71, "Financial Markets"),
        ("Supplier investment highlights long-term capacity plans", 0.22, 0.63, "Manufacturing"),
        ("Regulatory scrutiny creates uncertainty for large platforms", -0.29, 0.69, "Economy - Monetary"),
        ("Quarterly services revenue reaches a new high", 0.42, 0.91, "Earnings"),
        ("Broader technology sector closes mixed amid rate concerns", -0.17, 0.44, "Financial Markets"),
        ("Company expands accessibility features across devices", 0.18, 0.82, "Technology"),
        ("Foreign-exchange headwinds weigh on international sales", -0.21, 0.76, "Economy - Macro"),
    ]
    result = []
    for index, (title, score, relevance, topic) in enumerate(samples):
        result.append(
            {
                "title": f"[Demo] {title}",
                "summary": (
                    f"Synthetic portfolio demonstration article for {symbol}. "
                    "This content is generated locally and is not real market news."
                ),
                "url": "",
                "source": ["Demo Wire", "Sample Markets", "Research Fixture"][index % 3],
                "published_at": base - timedelta(hours=index * 14),
                "ticker_score": score,
                "ticker_label": _label(score),
                "overall_score": score * 0.8,
                "overall_label": _label(score * 0.8),
                "relevance": relevance,
                "relevance_label": "Demo relevance",
                "topics": [
                    {"topic": topic, "relevance": 0.8, "weighted_relevance": 0.8 * relevance}
                ],
            }
        )
    return result


def demo_asset_summary(symbol: str) -> dict:
    """Deterministic synthetic metrics for an API-free product demonstration."""
    seed = sum(ord(character) for character in symbol)
    daily_returns = [((index * (seed % 11 + 3)) % 17 - 8) / 1000 for index in range(90)]
    return {
        "symbol": symbol,
        "company_name": f"{symbol} Demo Asset",
        "asset_type": "Equity",
        "annual_return": 4.0 + seed % 14,
        "annual_volatility": 16.0 + seed % 19,
        "max_drawdown": -(12.0 + seed % 24),
        "beta": 0.75 + (seed % 70) / 100,
        "sector": ["Information Technology", "Health Care", "Industrials"][seed % 3],
        "industry": "Synthetic demonstration data",
        "daily_returns": daily_returns,
    }


def _label(score: float) -> str:
    from portfolio_app.sentiment import classify_sentiment

    return classify_sentiment(score)
