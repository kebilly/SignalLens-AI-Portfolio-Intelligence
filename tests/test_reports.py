import unittest
from io import BytesIO

from pypdf import PdfReader

from portfolio_app.reports import portfolio_report_pdf


class ReportTests(unittest.TestCase):
    def test_portfolio_pdf_is_readable(self):
        profile = {
            "risk_profile": "Balanced",
            "risk_score": 55,
            "capital": 100000,
            "scores": {},
            "include_beta": True,
        }
        assets = [
            {
                "symbol": "AAPL",
                "allocation": 80,
                "annual_volatility": 25,
                "annual_return": 12,
                "max_drawdown": -18,
                "beta": 1.1,
                "sector": "Information Technology",
            },
            {
                "symbol": "CASH",
                "allocation": 20,
                "annual_volatility": 0,
                "annual_return": 0,
                "max_drawdown": 0,
                "beta": None,
                "sector": "Cash",
            },
        ]
        content = portfolio_report_pdf(profile, assets, 48.5, "## 摘要\n歷史數據顯示組合具有中度風險。")
        self.assertTrue(content.startswith(b"%PDF"))
        self.assertGreaterEqual(len(PdfReader(BytesIO(content)).pages), 1)
