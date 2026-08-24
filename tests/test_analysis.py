import unittest

from portfolio_app.analysis import (
    asset_risk_score,
    calculate_portfolio_risk,
    calculate_risk_profile_score,
    risk_band,
)


class AnalysisTests(unittest.TestCase):
    def test_profile_score_uses_documented_weights(self):
        score = calculate_risk_profile_score(
            {
                "financial_status": 80,
                "investment_experience": 60,
                "investment_goal": 50,
                "risk_tolerance": 40,
            }
        )
        self.assertEqual(score, 56)

    def test_asset_risk_formula_with_beta(self):
        self.assertEqual(asset_risk_score({"annual_volatility": 20, "beta": 1.0}), 50)

    def test_cash_does_not_add_market_risk(self):
        assets = [
            {"asset_type": "Stock", "annual_volatility": 20, "beta": 1, "allocation": 80},
            {"asset_type": "Cash", "annual_volatility": 0, "beta": None, "allocation": 20},
        ]
        self.assertEqual(calculate_portfolio_risk(assets), 40)

    def test_risk_bands_have_boundaries(self):
        self.assertEqual(risk_band(0), "極低")
        self.assertEqual(risk_band(40), "中等")
        self.assertEqual(risk_band(80), "極高")
