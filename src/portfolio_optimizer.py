"""
portfolio_optimizer.py
------------------------
Mean-variance (Markowitz) portfolio optimization: builds the efficient
frontier and three model portfolios corresponding to the three risk
profiles used by risk_profiling.py.

Limitations (see README for full discussion):
- Assumes returns are normally distributed and that historical/assumed
  moments are stable going forward — both are simplifications.
- Mean-variance optimization is known to be highly sensitive to the input
  expected returns (Michaud, 1989); small changes can produce large shifts
  in optimal weights.
- Does not capture tail risk, liquidity constraints, or transaction costs.
"""

import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, assets, exp_returns, cov, risk_free_rate=0.02):
        self.assets = assets
        self.exp_returns = np.array(exp_returns)
        self.cov = np.array(cov)
        self.n = len(assets)
        self.rf = risk_free_rate
        self.bounds = tuple((0.0, 1.0) for _ in range(self.n))
        self.base_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def port_return(self, w):
        return w @ self.exp_returns

    def port_vol(self, w):
        return np.sqrt(w @ self.cov @ w)

    def neg_sharpe(self, w):
        return -(self.port_return(w) - self.rf) / self.port_vol(w)

    def _equal_weights(self):
        return np.repeat(1 / self.n, self.n)

    def min_volatility_portfolio(self):
        res = minimize(self.port_vol, x0=self._equal_weights(), method="SLSQP",
                        bounds=self.bounds, constraints=self.base_constraint)
        return res.x

    def max_sharpe_portfolio(self):
        res = minimize(self.neg_sharpe, x0=self._equal_weights(), method="SLSQP",
                        bounds=self.bounds, constraints=self.base_constraint)
        return res.x

    def max_return_portfolio(self, max_vol=0.20):
        cons = (
            self.base_constraint,
            {"type": "ineq", "fun": lambda w: max_vol - self.port_vol(w)},
        )
        res = minimize(lambda w: -self.port_return(w), x0=self._equal_weights(),
                        method="SLSQP", bounds=self.bounds, constraints=cons)
        return res.x

    def model_portfolios(self, aggressive_max_vol=0.20):
        return {
            "Conservative": self.min_volatility_portfolio(),
            "Balanced": self.max_sharpe_portfolio(),
            "Aggressive": self.max_return_portfolio(max_vol=aggressive_max_vol),
        }

    def efficient_frontier(self, n_points=40):
        target_returns = np.linspace(
            self.exp_returns.min() + 0.002, self.exp_returns.max() - 0.002, n_points
        )
        frontier_vol = []
        for tr in target_returns:
            cons = (
                self.base_constraint,
                {"type": "eq", "fun": lambda w, tr=tr: self.port_return(w) - tr},
            )
            res = minimize(self.port_vol, x0=self._equal_weights(), method="SLSQP",
                            bounds=self.bounds, constraints=cons)
            frontier_vol.append(res.fun)
        return target_returns, np.array(frontier_vol)

    def summary(self, w):
        return {
            "return": self.port_return(w),
            "volatility": self.port_vol(w),
            "sharpe": (self.port_return(w) - self.rf) / self.port_vol(w),
            "weights": dict(zip(self.assets, w)),
        }
