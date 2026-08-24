import unittest

from portfolio_app.validation import normalize_symbol, validate_portfolio


class ValidationTests(unittest.TestCase):
    def test_symbol_normalization(self):
        self.assertEqual(normalize_symbol(" brk.b "), "BRK.B")

    def test_valid_portfolio(self):
        self.assertEqual(
            validate_portfolio(
                [{"symbol": "AAPL", "allocation": 80}, {"symbol": "CASH", "allocation": 20}]
            ),
            [],
        )

    def test_rejects_invalid_total_duplicate_and_symbol(self):
        errors = validate_portfolio(
            [
                {"symbol": "BAD SYMBOL", "allocation": 30},
                {"symbol": "BAD SYMBOL", "allocation": 30},
            ]
        )
        self.assertTrue(any("必須為 100" in error for error in errors))
        self.assertTrue(any("格式不正確" in error for error in errors))
        self.assertTrue(any("重複" in error for error in errors))
