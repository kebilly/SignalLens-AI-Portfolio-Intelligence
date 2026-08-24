import unittest

import requests

from portfolio_app.config import Settings
from portfolio_app.sentiment_provider import AlphaVantageClient
from portfolio_app.services import ExternalServiceError, FMPClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


class FakeGetSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}

    def get(self, *args, **kwargs):
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class MarketProviderErrorTests(unittest.TestCase):
    def test_fmp_authentication_error_is_sanitized(self):
        client = FMPClient(Settings(fmp_api_key="secret-fmp-key"))
        client.session = FakeGetSession(FakeResponse({}, 401))

        with self.assertRaises(ExternalServiceError) as raised:
            client._get("/stable/profile", {"symbol": "AAPL"})

        self.assertIn("伺服器設定無效", raised.exception.public_message)
        self.assertNotIn("secret-fmp-key", raised.exception.public_message)

    def test_fmp_timeout_and_server_failure_are_actionable(self):
        client = FMPClient(Settings(fmp_api_key="test"))
        client.session = FakeGetSession(requests.Timeout("simulated"))
        with self.assertRaises(ExternalServiceError) as timeout:
            client._get("/stable/profile", {"symbol": "AAPL"})
        self.assertIn("無法連線", timeout.exception.public_message)

        client.session = FakeGetSession(FakeResponse({}, 500))
        with self.assertRaises(ExternalServiceError) as server_error:
            client._get("/stable/profile", {"symbol": "AAPL"})
        self.assertIn("暫時無法完成", server_error.exception.public_message)

    def test_fmp_rejects_invalid_json_and_short_history(self):
        client = FMPClient(Settings(fmp_api_key="test"))
        client.session = FakeGetSession(FakeResponse(ValueError("invalid")))
        with self.assertRaises(ExternalServiceError) as invalid_json:
            client._get("/stable/profile", {"symbol": "AAPL"})
        self.assertIn("格式異常", invalid_json.exception.public_message)

        client.session = FakeGetSession(FakeResponse([{"date": "2026-01-01", "close": 100}]))
        with self.assertRaises(ExternalServiceError) as short_history:
            client.historical("AAPL")
        self.assertIn("資料不足", short_history.exception.public_message)

    def test_alpha_vantage_news_errors_are_sanitized(self):
        client = AlphaVantageClient("secret-alpha-key")
        client.session = FakeGetSession(requests.Timeout("simulated"))
        with self.assertRaises(ExternalServiceError) as timeout:
            client.news_sentiment("AAPL")
        self.assertIn("無法連線", timeout.exception.public_message)

        client.session = FakeGetSession(FakeResponse({}, 500))
        with self.assertRaises(ExternalServiceError) as server_error:
            client.news_sentiment("AAPL")
        self.assertIn("HTTP 500", server_error.exception.public_message)
        self.assertNotIn("secret-alpha-key", server_error.exception.public_message)

    def test_alpha_vantage_provider_limit_and_missing_feed(self):
        client = AlphaVantageClient("test")
        client.session = FakeGetSession(FakeResponse({"Note": "rate limit"}))
        with self.assertRaises(ExternalServiceError) as rate_limit:
            client.news_sentiment("AAPL")
        self.assertIn("用量限制", rate_limit.exception.public_message)

        client.session = FakeGetSession(FakeResponse({"items": "0"}))
        with self.assertRaises(ExternalServiceError) as missing:
            client.news_sentiment("AAPL")
        self.assertIn("找不到", missing.exception.public_message)

    def test_alpha_vantage_daily_prices_skip_malformed_rows(self):
        client = AlphaVantageClient("test")
        client.session = FakeGetSession(
            FakeResponse(
                {
                    "Time Series (Daily)": {
                        "2026-01-02": {
                            "1. open": "100",
                            "2. high": "103",
                            "3. low": "99",
                            "4. close": "102",
                            "5. volume": "1200",
                        },
                        "2026-01-01": {"1. open": "invalid"},
                    }
                }
            )
        )

        rows = client.daily_prices("AAPL")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], "2026-01-02")
        self.assertEqual(rows[0]["close"], 102.0)
