"""
main.py
--------
End-to-end pipeline:
  1. Load market data (live via yfinance, or fallback assumptions if offline)
  2. Run the risk-profiling questionnaire (or accept a fixed profile via CLI)
  3. Build the efficient frontier and the three model portfolios
  4. Backtest the portfolio matching the client's risk profile
  5. Save charts to outputs/ and print a summary report

Usage:
  python main.py                  # interactive questionnaire
  python main.py --profile Balanced   # skip questionnaire, use a fixed profile
"""

import argparse
import os
import matplotlib.pyplot as plt

from src.data_loader import load_market_data
from src.risk_profiling import run_questionnaire
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtest import backtest_from_prices, backtest_simulated

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {"Conservative": "#4a7c59", "Balanced": "#b3862f", "Aggressive": "#8b2e2e"}


def plot_efficient_frontier(optimizer, portfolios, save_path):
    target_returns, frontier_vol = optimizer.efficient_frontier()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(frontier_vol * 100, target_returns * 100, color="#1a2b4a", lw=2,
            label="Efficient frontier")

    for name, w in portfolios.items():
        ax.scatter(optimizer.port_vol(w) * 100, optimizer.port_return(w) * 100,
                   s=90, color=COLORS[name], zorder=5, label=name)

    ax.set_xlabel("Annual volatility (%)")
    ax.set_ylabel("Expected annual return (%)")
    ax.set_title("Efficient Frontier — Model Portfolios by Risk Profile",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_backtest(result, profile_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["dates"], result["cumulative"], color=COLORS.get(profile_name, "#1a2b4a"), lw=1.6)
    ax.set_title(f"{'Simulated' if result['mode'] == 'simulated' else 'Historical'} "
                 f"Cumulative Performance — {profile_name} Portfolio",
                 fontsize=11, weight="bold")
    ax.set_ylabel("Growth of 1 unit invested")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["Conservative", "Balanced", "Aggressive"],
                        default=None, help="Skip the questionnaire and force a profile.")
    args = parser.parse_args()

    print("\n### STEP 1 — Market data ###")
    data = load_market_data()

    print("\n### STEP 2 — Risk profiling ###")
    if args.profile:
        profile = args.profile
        print(f"Profile fixed via CLI argument: {profile}")
    else:
        profile, score, answers = run_questionnaire(interactive=True)
        print(f"Score: {score}/9 -> Profile: {profile}")

    print("\n### STEP 3 — Portfolio optimization ###")
    optimizer = PortfolioOptimizer(data["assets"], data["exp_returns"], data["cov"])
    portfolios = optimizer.model_portfolios()

    for name, w in portfolios.items():
        s = optimizer.summary(w)
        print(f"\n{name}")
        print(f"  Expected return: {s['return']*100:.2f}%   "
              f"Volatility: {s['volatility']*100:.2f}%   Sharpe: {s['sharpe']:.2f}")
        for asset, weight in s["weights"].items():
            print(f"    {asset:<18s} {weight*100:5.1f}%")

    frontier_path = os.path.join(OUTPUT_DIR, "efficient_frontier.png")
    plot_efficient_frontier(optimizer, portfolios, frontier_path)
    print(f"\nSaved: {frontier_path}")

    print("\n### STEP 4 — Backtest ###")
    chosen_weights = portfolios[profile]

    if data["mode"] == "live":
        result = backtest_from_prices(chosen_weights, data["prices"])
    else:
        result = backtest_simulated(chosen_weights, data["exp_returns"], data["cov"])

    print(f"Mode: {result['mode']}")
    print(f"Annualized return: {result['annualized_return']*100:.2f}%")
    print(f"Annualized volatility: {result['annualized_vol']*100:.2f}%")
    print(f"Sharpe ratio: {result['sharpe']:.2f}")
    print(f"Max drawdown: {result['max_drawdown']*100:.2f}%")

    backtest_path = os.path.join(OUTPUT_DIR, f"backtest_{profile.lower()}.png")
    plot_backtest(result, profile, backtest_path)
    print(f"Saved: {backtest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
