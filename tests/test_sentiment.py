import unittest

from portfolio_app.sentiment import (
    calculate_sentiment_stats,
    classify_relevance,
    classify_sentiment,
    parse_news_feed,
    parse_timestamp,
)


class SentimentTests(unittest.TestCase):
    def test_sentiment_boundaries(self):
        cases = [
            (-0.35, "Bearish"),
            (-0.349, "Somewhat-Bearish"),
            (-0.15, "Somewhat-Bearish"),
            (-0.149, "Neutral"),
            (0.149, "Neutral"),
            (0.15, "Somewhat-Bullish"),
            (0.349, "Somewhat-Bullish"),
            (0.35, "Bullish"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(classify_sentiment(score), expected)

    def test_relevance_boundaries(self):
        self.assertEqual(classify_relevance(0.8), "Highly relevant")
        self.assertEqual(classify_relevance(0.5), "Relevant")
        self.assertEqual(classify_relevance(0.3), "Moderately relevant")
        self.assertEqual(classify_relevance(0.29), "Low relevance")

    def test_timestamp_parser(self):
        parsed = parse_timestamp("20251004T173200")
        self.assertEqual(parsed.year, 2025)
        self.assertEqual(parsed.minute, 32)
        self.assertIsNone(parse_timestamp("not-a-date"))

    def test_parser_filters_other_tickers_and_handles_missing_fields(self):
        payload = {
            "feed": [
                {
                    "title": "AAPL article",
                    "time_published": "20251004T173200",
                    "ticker_sentiment": [
                        {"ticker": "MSFT", "ticker_sentiment_score": "0.9", "relevance_score": "1"},
                        {"ticker": "AAPL", "ticker_sentiment_score": "0.4", "relevance_score": "0.7"},
                    ],
                },
                {"title": "Unrelated", "ticker_sentiment": [{"ticker": "TSLA"}]},
            ]
        }
        rows = parse_news_feed(payload, "aapl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker_label"], "Bullish")
        self.assertEqual(rows[0]["source"], "Unknown")

    def test_stats_are_computed_before_display_ranking(self):
        news = [
            {"ticker_score": 0.4, "overall_score": 0.2, "relevance": 0.9, "ticker_label": "Bullish"},
            {"ticker_score": -0.4, "overall_score": -0.2, "relevance": 0.1, "ticker_label": "Bearish"},
        ]
        stats = calculate_sentiment_stats(news)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["average_ticker_score"], 0)
        self.assertEqual(stats["distribution"]["Bearish"], 1)
        self.assertEqual(stats["distribution"]["Bullish"], 1)

    def test_empty_stats(self):
        stats = calculate_sentiment_stats([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["dominant_label"], "No data")
