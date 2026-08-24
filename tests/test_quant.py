import unittest

from portfolio_app.quant import (
    concentration_hhi,
    effective_positions,
    quantitative_metrics,
    risk_contributions,
)


class QuantTests(unittest.TestCase):
    def test_equal_weight_hhi(self):
        assets = [{"allocation": 50}, {"allocation": 50}]
        self.assertAlmostEqual(concentration_hhi(assets), 0.5)
        self.assertAlmostEqual(effective_positions(assets), 2)

    def test_concentrated_portfolio_has_one_effective_position(self):
        self.assertEqual(effective_positions([{"allocation": 100}]), 1)

    def test_metrics_handle_no_returns(self):
        metrics = quantitative_metrics([{"asset_type": "Cash", "allocation": 100}])
        self.assertIsNone(metrics["annualized_volatility"])
        self.assertEqual(metrics["effective_positions"], 1)

    def test_metrics_with_two_assets(self):
        returns_a = [0.01, -0.01] * 20
        returns_b = [0.005, -0.002] * 20
        assets = [
            {"symbol": "A", "asset_type": "Stock", "allocation": 60, "daily_returns": returns_a},
            {"symbol": "B", "asset_type": "Stock", "allocation": 40, "daily_returns": returns_b},
        ]
        metrics = quantitative_metrics(assets)
        self.assertGreater(metrics["annualized_volatility"], 0)
        self.assertGreaterEqual(metrics["historical_var_95"], 0)
        self.assertGreaterEqual(metrics["historical_cvar_95"] + 1e-12, metrics["historical_var_95"])

    def test_risk_contributions_sum_to_100(self):
        returns_a = [0.01, -0.01] * 20
        returns_b = [0.005, -0.002] * 20
        rows = risk_contributions([
            {"symbol": "A", "asset_type": "Stock", "allocation": 50, "daily_returns": returns_a},
            {"symbol": "B", "asset_type": "Stock", "allocation": 50, "daily_returns": returns_b},
        ])
        self.assertAlmostEqual(sum(row["risk_contribution"] for row in rows), 100)

    def test_short_history_returns_no_quant_metrics(self):
        metrics = quantitative_metrics([
            {"symbol": "A", "asset_type": "Stock", "allocation": 100, "daily_returns": [0.01] * 5}
        ])
        self.assertIsNone(metrics["annualized_volatility"])
