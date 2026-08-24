from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from portfolio_app.validation import normalize_symbol

SENTIMENT_ORDER = [
    "Bearish",
    "Somewhat-Bearish",
    "Neutral",
    "Somewhat-Bullish",
    "Bullish",
]

SENTIMENT_COLORS = {
    "Bearish": "#C0392B",
    "Somewhat-Bearish": "#E67E73",
    "Neutral": "#95A5A6",
    "Somewhat-Bullish": "#6FCF97",
    "Bullish": "#219653",
}

SENTIMENT_ZH = {
    "Bearish": "看跌",
    "Somewhat-Bearish": "偏看跌",
    "Neutral": "中性",
    "Somewhat-Bullish": "偏看漲",
    "Bullish": "看漲",
    "No data": "無資料",
}


def sentiment_label_zh(label: str) -> str:
    return SENTIMENT_ZH.get(label, label)


def classify_sentiment(score: float) -> str:
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.35:
        return "Somewhat-Bullish"
    return "Bullish"


def classify_relevance(score: float) -> str:
    if score >= 0.8:
        return "Highly relevant"
    if score >= 0.5:
        return "Relevant"
    if score >= 0.3:
        return "Moderately relevant"
    return "Low relevance"


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    for pattern in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_news_feed(payload: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
    target = normalize_symbol(symbol)
    parsed: list[dict[str, Any]] = []
    feed = payload.get("feed", []) if isinstance(payload, dict) else []
    for item in feed:
        if not isinstance(item, dict):
            continue
        ticker_row = next(
            (
                row
                for row in item.get("ticker_sentiment", [])
                if normalize_symbol(str(row.get("ticker", ""))) == target
            ),
            None,
        )
        if not ticker_row:
            continue
        ticker_score = _bounded_float(ticker_row.get("ticker_sentiment_score"), -1, 1)
        overall_score = _bounded_float(item.get("overall_sentiment_score"), -1, 1)
        relevance = _bounded_float(ticker_row.get("relevance_score"), 0, 1)
        topics = []
        for topic in item.get("topics", []):
            if not isinstance(topic, dict):
                continue
            topic_relevance = _bounded_float(topic.get("relevance_score"), 0, 1)
            if topic_relevance >= 0.2:
                topics.append(
                    {
                        "topic": str(topic.get("topic") or "Other")[:80],
                        "relevance": topic_relevance,
                        "weighted_relevance": topic_relevance * relevance,
                    }
                )
        parsed.append(
            {
                "title": str(item.get("title") or "Untitled")[:500],
                "summary": str(item.get("summary") or "")[:4000],
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "Unknown")[:120],
                "published_at": parse_timestamp(str(item.get("time_published") or "")),
                "ticker_score": ticker_score,
                "ticker_label": classify_sentiment(ticker_score),
                "overall_score": overall_score,
                "overall_label": classify_sentiment(overall_score),
                "relevance": relevance,
                "relevance_label": classify_relevance(relevance),
                "topics": sorted(topics, key=lambda row: row["weighted_relevance"], reverse=True),
            }
        )
    return sorted(
        parsed,
        key=lambda row: (row["published_at"] or datetime.min.replace(tzinfo=UTC)),
        reverse=True,
    )


def calculate_sentiment_stats(news: list[dict[str, Any]]) -> dict[str, Any]:
    if not news:
        return {
            "count": 0,
            "average_ticker_score": 0.0,
            "average_overall_score": 0.0,
            "average_relevance": 0.0,
            "dominant_label": "No data",
            "dispersion": 0.0,
            "distribution": {label: 0 for label in SENTIMENT_ORDER},
            "overall_distribution": {label: 0 for label in SENTIMENT_ORDER},
        }
    ticker_scores = [float(item["ticker_score"]) for item in news]
    distribution = Counter(item["ticker_label"] for item in news)
    overall_distribution = Counter(
        item.get("overall_label", classify_sentiment(float(item.get("overall_score", 0))))
        for item in news
    )
    return {
        "count": len(news),
        "average_ticker_score": float(np.mean(ticker_scores)),
        "average_overall_score": float(np.mean([item["overall_score"] for item in news])),
        "average_relevance": float(np.mean([item["relevance"] for item in news])),
        "dominant_label": distribution.most_common(1)[0][0],
        "dispersion": float(np.std(ticker_scores, ddof=1)) if len(news) > 1 else 0.0,
        "distribution": {label: distribution.get(label, 0) for label in SENTIMENT_ORDER},
        "overall_distribution": {
            label: overall_distribution.get(label, 0) for label in SENTIMENT_ORDER
        },
    }


def sentiment_price_correlation(
    news: list[dict[str, Any]], prices: list[dict[str, Any]]
) -> dict[str, float | int | None]:
    """Align daily news tone with the following available trading-day return."""
    sentiment = daily_sentiment(news)
    if sentiment.empty or len(prices) < 3:
        return {"correlation": None, "observations": 0}
    price_frame = pd.DataFrame(prices).copy()
    price_frame["Date"] = pd.to_datetime(price_frame["date"]).dt.date
    price_frame = price_frame.sort_values("Date")
    price_frame["Next return"] = price_frame["close"].astype(float).pct_change().shift(-1)
    merged = sentiment.merge(price_frame[["Date", "Next return"]], on="Date", how="inner").dropna()
    if len(merged) < 3 or merged["Ticker sentiment"].nunique() < 2:
        return {"correlation": None, "observations": int(len(merged))}
    return {
        "correlation": float(merged["Ticker sentiment"].corr(merged["Next return"])),
        "observations": int(len(merged)),
    }


def source_distribution(news: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["source"] for item in news)
    return pd.DataFrame(
        [{"Source": name, "Articles": count} for name, count in counts.most_common()]
    )


def topic_distribution(news: list[dict[str, Any]]) -> pd.DataFrame:
    totals: dict[str, float] = defaultdict(float)
    for item in news:
        for topic in item["topics"]:
            totals[topic["topic"]] += float(topic["weighted_relevance"])
    return pd.DataFrame(
        [
            {"Topic": topic, "Weighted relevance": value}
            for topic, value in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        ]
    )


def daily_sentiment(news: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Date": item["published_at"].date(),
            "Ticker sentiment": item["ticker_score"],
            "Relevance": item["relevance"],
        }
        for item in news
        if item["published_at"] is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker sentiment", "Relevance", "Articles"])
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("Date", as_index=False).agg(
        {"Ticker sentiment": "mean", "Relevance": "mean"}
    )
    grouped["Articles"] = frame.groupby("Date").size().values
    return grouped.sort_values("Date")


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, lower), upper)
