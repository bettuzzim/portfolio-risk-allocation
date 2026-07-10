"""
data_loader.py
--------------
Loads historical price data for four proxy asset classes via yfinance and
derives annualized expected returns, volatility, and the correlation matrix.

If no internet connection is available (or Yahoo Finance is unreachable),
falls back to illustrative long-run capital market assumptions, clearly
flagged as such. This makes the pipeline runnable end-to-end in any
environment while being explicit about which mode produced the numbers.
"""

import numpy as np
import pandas as pd

TICKERS = {
    "Global Equity": "ACWI",       # iShares MSCI ACWI ETF
    "IG Bonds": "AGG",              # iShares Core U.S. Aggregate Bond ETF
    "HY Bonds": "HYG",               # iShares iBoxx $ High Yield Corp Bond ETF
    "Cash / T-Bills": "BIL",         # SPDR Bloomberg 1-3 Month T-Bill ETF
}

# Illustrative long-run capital market assumptions (annualized).
# Used ONLY as a fallback when live data cannot be retrieved.
FALLBACK_RETURNS = np.array([0.075, 0.032, 0.048, 0.020])
FALLBACK_VOL = np.array([0.170, 0.055, 0.090, 0.008])
FALLBACK_CORR = np.array([
    [1.00, 0.15, 0.45, 0.00],
    [0.15, 1.00, 0.55, 0.05],
    [0.45, 0.55, 1.00, 0.02],
    [0.00, 0.05, 0.02, 1.00],
])


def load_market_data(start="2015-01-01", end=None):
    """
    Attempts to download real historical price data via yfinance and derive
    annualized statistics. Returns a dict with:
        assets, exp_returns, volatility, corr, cov, mode, prices (or None)
    `mode` is either "live" or "fallback".
    """
    try:
        import yfinance as yf

        tickers = list(TICKERS.values())
        # threads=False avoids a known yfinance issue where concurrent
        # downloads write to its internal SQLite cache at the same time,
        # raising "OperationalError: database is locked".
        data = yf.download(tickers, start=start, end=end, progress=False,
                            threads=False)["Close"]

        if data.empty or data.isna().all().all():
            raise ValueError("Empty data returned by yfinance.")

        data = data.dropna()
        daily_returns = data.pct_change().dropna()

        exp_returns = daily_returns.mean().values * 252
        volatility = daily_returns.std().values * np.sqrt(252)
        corr = daily_returns.corr().values

        assets = list(TICKERS.keys())
        cov = np.outer(volatility, volatility) * corr

        print(f"[data_loader] Live data loaded from Yahoo Finance "
              f"({data.index[0].date()} to {data.index[-1].date()}).")

        return {
            "assets": assets,
            "exp_returns": exp_returns,
            "volatility": volatility,
            "corr": corr,
            "cov": cov,
            "mode": "live",
            "prices": data,
        }

    except Exception as e:
        print(f"[data_loader] WARNING: could not retrieve live market data "
              f"({type(e).__name__}: {e}).")
        print("[data_loader] Falling back to illustrative capital market "
              "assumptions. Figures are NOT historical data.")

        assets = list(TICKERS.keys())
        cov = np.outer(FALLBACK_VOL, FALLBACK_VOL) * FALLBACK_CORR

        return {
            "assets": assets,
            "exp_returns": FALLBACK_RETURNS,
            "volatility": FALLBACK_VOL,
            "corr": FALLBACK_CORR,
            "cov": cov,
            "mode": "fallback",
            "prices": None,
        }


if __name__ == "__main__":
    result = load_market_data()
    print("\nMode:", result["mode"])
    for a, r, v in zip(result["assets"], result["exp_returns"], result["volatility"]):
        print(f"  {a:<18s} return={r*100:5.2f}%   vol={v*100:5.2f}%")
