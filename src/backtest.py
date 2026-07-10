"""
backtest.py
------------
Backtests a chosen portfolio (fixed weights, rebalanced daily — a
simplification; no transaction costs or rebalancing bands are modeled).

If real historical prices are available (data_loader "live" mode), computes
actual cumulative performance, annualized return, volatility, Sharpe ratio,
and maximum drawdown. If only fallback assumptions are available, simulates
a illustrative price path via Geometric Brownian Motion using those
assumptions — clearly labeled as SIMULATED, not a real backtest.
"""

import numpy as np
import pandas as pd


def backtest_from_prices(weights, prices, risk_free_rate=0.02):
    """Real backtest using actual historical prices (DataFrame, columns=assets order)."""
    daily_returns = prices.pct_change().dropna()
    port_daily_returns = daily_returns.values @ np.array(weights)
    cumulative = (1 + port_daily_returns).cumprod()

    n_years = len(port_daily_returns) / 252
    ann_return = cumulative[-1] ** (1 / n_years) - 1
    ann_vol = port_daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "mode": "live",
        "cumulative": cumulative,
        "dates": daily_returns.index,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def backtest_simulated(weights, exp_returns, cov, years=10, seed=42, risk_free_rate=0.02):
    """Illustrative simulated backtest via correlated GBM, used only when
    real price history is not available."""
    rng = np.random.default_rng(seed)
    n_days = years * 252
    dt = 1 / 252

    mean_daily = np.array(exp_returns) * dt
    cov_daily = np.array(cov) * dt

    daily_asset_returns = rng.multivariate_normal(mean_daily, cov_daily, size=n_days)
    port_daily_returns = daily_asset_returns @ np.array(weights)
    cumulative = (1 + port_daily_returns).cumprod()

    ann_return = cumulative[-1] ** (1 / years) - 1
    ann_vol = port_daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

    return {
        "mode": "simulated",
        "cumulative": cumulative,
        "dates": dates,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
