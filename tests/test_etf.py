import json
import unittest

from portfolio_app.etf import comparison_table, overlap_metrics, parse_holdings


class ETFTests(unittest.TestCase):
    def setUp(self):
        self.left = json.dumps(
            [
                {"symbol": "AAA", "asset": "2330", "name": "TSMC", "isin": "TW0002330008", "weightPercentage": 9},
                {"symbol": "AAA", "asset": "700.HK", "name": "Tencent", "isin": "KYG875721634", "securityCusip": "G87572163", "weightPercentage": 5},
            ]
        ).encode()
        self.right = json.dumps(
            [
                {"symbol": "BBB", "asset": "2330.TW", "name": "Taiwan Semiconductor", "isin": "TW0002330008", "weightPercentage": 8},
                {"symbol": "BBB", "asset": "RELIANCE", "name": "Reliance", "isin": "INE002A01018", "weightPercentage": 4},
                {"symbol": "BBB", "asset": "0700", "name": "Tencent Holdings", "securityCusip": "G87572163", "weightPercentage": 4.5},
            ]
        ).encode()

    def test_isin_aligns_different_tickers(self):
        a_symbol, a = parse_holdings(self.left, "a.txt")
        b_symbol, b = parse_holdings(self.right, "b.txt")
        table = comparison_table(a_symbol, a, b_symbol, b)
        tsmc = table[table["Holding"] == "TSMC"].iloc[0]
        self.assertEqual(tsmc["AAA %"], 9)
        self.assertEqual(tsmc["BBB %"], 8)

    def test_overlap_weight(self):
        _, a = parse_holdings(self.left, "a.txt")
        _, b = parse_holdings(self.right, "b.txt")
        metrics = overlap_metrics(a, b)
        self.assertEqual(metrics["common_holdings"], 2)
        self.assertEqual(metrics["overlap_weight"], 12.5)
