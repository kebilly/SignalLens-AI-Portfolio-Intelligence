import unittest
from datetime import UTC, datetime
from io import BytesIO

from pypdf import PdfReader

from portfolio_app.analysis import (
    aggregate_portfolio_sentiment,
    build_deterministic_integrated_report,
    build_integrated_prompt,
)
from portfolio_app.reports import portfolio_report_pdf
from portfolio_app.sentiment import sentiment_price_correlation


class IntegrationTests(unittest.TestCase):
    def test_portfolio_sentiment_uses_covered_weight_only(self):
        assets = [
            {"symbol": "AAA", "allocation": 60, "asset_type": "Equity"},
            {"symbol": "BBB", "allocation": 30, "asset_type": "Equity"},
            {"symbol": "CASH", "allocation": 10, "asset_type": "Cash"},
        ]
        results = {
            "AAA": {"stats": {"average_ticker_score": 0.2, "count": 4}},
            "BBB": {"stats": {"average_ticker_score": -0.2, "count": 6}},
        }
        summary = aggregate_portfolio_sentiment(assets, results)
        self.assertAlmostEqual(summary["weighted_ticker_score"], 0.0666667, places=5)
        self.assertEqual(summary["covered_weight"], 90)
        self.assertEqual(summary["article_count"], 10)

    def test_integrated_prompt_keeps_scales_separate(self):
        risk = {
            "profile": {"risk_profile": "Balanced", "risk_score": 50},
            "portfolio_risk": 42,
            "quant": {},
            "assets": [{"symbol": "AAA", "allocation": 100, "asset_type": "Equity"}],
        }
        sentiment = {
            "AAA": {
                "stats": {"average_ticker_score": 0.1, "count": 2},
                "price_context": {},
            }
        }
        system, prompt = build_integrated_prompt(risk, sentiment)
        self.assertIn("不得相加", system)
        self.assertIn("<portfolio-risk>", prompt)
        self.assertIn("<portfolio-news-summary>", prompt)

    def test_offline_report_restores_full_sections(self):
        risk = {
            "profile": {"risk_profile": "平衡型", "risk_score": 50},
            "portfolio_risk": 42,
            "quant": {"annualized_volatility": 12, "historical_var_95": 1,
                      "historical_cvar_95": 1.5, "herfindahl_index": 1,
                      "effective_positions": 1},
            "assets": [{"symbol": "AAA", "allocation": 100, "asset_type": "Equity"}],
        }
        sentiment = {"AAA": {"stats": {"average_ticker_score": 0.1,
            "average_overall_score": 0.05, "average_relevance": 0.8,
            "dominant_label": "Neutral", "count": 8}}}
        report = build_deterministic_integrated_report(risk, sentiment)
        for heading in ("執行摘要", "價格風險與集中度", "持倉新聞情緒", "風險適配性", "資料限制"):
            self.assertIn(heading, report)
        self.assertGreater(len(report), 700)

    def test_sentiment_price_correlation_aligns_dates(self):
        news = [
            {"published_at": datetime(2026, 1, day, tzinfo=UTC), "ticker_score": score, "relevance": 0.8}
            for day, score in ((1, -0.5), (2, 0.0), (3, 0.5))
        ]
        prices = [
            {"date": f"2026-01-0{day}", "close": close}
            for day, close in ((1, 100), (2, 99), (3, 99), (4, 101))
        ]
        result = sentiment_price_correlation(news, prices)
        self.assertEqual(result["observations"], 3)
        self.assertIsNotNone(result["correlation"])

    def test_pdf_can_include_sentiment_section(self):
        profile = {"risk_profile": "Balanced", "risk_score": 50, "capital": 100000}
        assets = [{"symbol": "AAA", "allocation": 100, "annual_volatility": 20,
                   "annual_return": 5, "max_drawdown": -15, "beta": 1, "sector": "Technology"}]
        rows = [{"symbol": "AAA", "count": 8, "ticker_score": 0.1,
                 "overall_score": 0.05, "relevance": 0.8, "dominant_label": "Neutral"}]
        content = portfolio_report_pdf(profile, assets, 40, "## 摘要\n整合分析", rows)
        self.assertGreaterEqual(len(PdfReader(BytesIO(content)).pages), 1)


if __name__ == "__main__":
    unittest.main()
