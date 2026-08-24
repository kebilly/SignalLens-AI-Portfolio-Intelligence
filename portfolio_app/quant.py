from __future__ import annotations

from typing import Any

import numpy as np


def portfolio_return_matrix(assets: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    risky = [
        asset
        for asset in assets
        if asset.get("asset_type") != "Cash" and asset.get("daily_returns")
    ]
    if not risky:
        return np.empty((0, 0)), np.empty(0)
    length = min(len(asset["daily_returns"]) for asset in risky)
    if length < 20:
        return np.empty((0, 0)), np.empty(0)
    matrix = np.column_stack(
        [np.asarray(asset["daily_returns"][-length:], dtype=float) for asset in risky]
    )
    weights = np.asarray([float(asset.get("allocation", 0)) / 100 for asset in risky])
    return matrix, weights


def quantitative_metrics(assets: list[dict[str, Any]]) -> dict[str, float | None]:
    matrix, weights = portfolio_return_matrix(assets)
    if matrix.size == 0:
        return {
            "annualized_volatility": None,
            "historical_var_95": None,
            "historical_cvar_95": None,
            "herfindahl_index": concentration_hhi(assets),
            "effective_positions": effective_positions(assets),
        }
    daily = matrix @ weights
    var_threshold = float(np.quantile(daily, 0.05))
    tail = daily[daily <= var_threshold]
    return {
        "annualized_volatility": float(np.std(daily, ddof=1) * np.sqrt(252) * 100),
        "historical_var_95": abs(var_threshold) * 100,
        "historical_cvar_95": abs(float(np.mean(tail))) * 100 if tail.size else None,
        "herfindahl_index": concentration_hhi(assets),
        "effective_positions": effective_positions(assets),
    }


def concentration_hhi(assets: list[dict[str, Any]]) -> float:
    weights = [float(asset.get("allocation", 0)) / 100 for asset in assets]
    return sum(weight * weight for weight in weights)


def effective_positions(assets: list[dict[str, Any]]) -> float:
    hhi = concentration_hhi(assets)
    return 1 / hhi if hhi else 0.0


def risk_contributions(assets: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    risky = [
        asset
        for asset in assets
        if asset.get("asset_type") != "Cash" and asset.get("daily_returns")
    ]
    matrix, weights = portfolio_return_matrix(assets)
    if matrix.size == 0:
        return []
    covariance = np.cov(matrix, rowvar=False) * 252
    covariance = np.atleast_2d(covariance)
    variance = float(weights @ covariance @ weights)
    if variance <= 0:
        return []
    marginal = covariance @ weights
    contributions = weights * marginal / variance
    return [
        {"symbol": str(asset["symbol"]), "risk_contribution": float(value * 100)}
        for asset, value in zip(risky, contributions, strict=True)
    ]
