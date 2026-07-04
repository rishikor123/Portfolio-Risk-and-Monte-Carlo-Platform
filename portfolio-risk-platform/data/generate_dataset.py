"""
generate_dataset.py
====================
Generates data/prices.csv, a daily adjusted-close price panel for
config.TICKERS + config.BENCHMARK.

Two modes:
  1. Synthetic (default): correlated multivariate Geometric Brownian Motion,
     calibrated to the annualized return/vol/correlation targets in config.py.
     Deterministic given --seed, so the bundled dataset is reproducible.
  2. Live (--live): pulls real adjusted-close prices via yfinance. Requires
     internet access and `pip install yfinance`.

Usage:
    python generate_dataset.py                     # synthetic, default dates/seed
    python generate_dataset.py --seed 7             # different synthetic draw
    python generate_dataset.py --live --start 2021-01-01 --end 2025-12-31
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (TICKERS, BENCHMARK, ASSET_ORDER, CALIBRATION, CORRELATION_MATRIX,
                     TRADING_DAYS_PER_YEAR, DEFAULT_HISTORY_START, DEFAULT_HISTORY_END)


def nearest_psd(a):
    """Project a symmetric matrix to the nearest positive-semidefinite matrix
    (clips negative eigenvalues to a small positive floor). Guards against
    a hand-specified correlation matrix that is not quite PSD due to
    rounding of pairwise correlations."""
    a = (a + a.T) / 2
    eigval, eigvec = np.linalg.eigh(a)
    eigval_clipped = np.clip(eigval, 1e-8, None)
    a_psd = eigvec @ np.diag(eigval_clipped) @ eigvec.T
    d = np.sqrt(np.diag(a_psd))
    a_psd = a_psd / np.outer(d, d)
    np.fill_diagonal(a_psd, 1.0)
    return a_psd


def generate_synthetic(start=DEFAULT_HISTORY_START, end=DEFAULT_HISTORY_END, seed=42):
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n_assets = len(ASSET_ORDER)

    corr = nearest_psd(np.array(CORRELATION_MATRIX))
    vols = np.array([CALIBRATION[t]["sigma"] for t in ASSET_ORDER])
    mus = np.array([CALIBRATION[t]["mu"] for t in ASSET_ORDER])

    cov_annual = np.outer(vols, vols) * corr
    cov_daily = cov_annual / TRADING_DAYS_PER_YEAR
    mu_daily = mus / TRADING_DAYS_PER_YEAR  # simple daily drift approximation

    L = np.linalg.cholesky(cov_daily)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_days, n_assets))
    correlated_shocks = z @ L.T  # each row ~ N(0, cov_daily)

    # GBM log-return simulation: dlnP = (mu - 0.5*sigma^2)*dt + shock
    drift_adj = mu_daily - 0.5 * (vols ** 2) / TRADING_DAYS_PER_YEAR
    log_returns = drift_adj + correlated_shocks

    log_prices = np.cumsum(log_returns, axis=0)
    start_prices = {
        "NVDA": 130.0, "MSFT": 420.0, "GS": 480.0, "JPM": 210.0,
        "TLT": 95.0, "GLD": 190.0, "SPY": 590.0,
    }
    prices = np.zeros_like(log_prices)
    for j, t in enumerate(ASSET_ORDER):
        prices[:, j] = start_prices[t] * np.exp(log_prices[:, j])

    df = pd.DataFrame(prices, index=dates, columns=ASSET_ORDER)
    df.index.name = "Date"
    return df


def generate_live(start, end):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Live mode requires yfinance: pip install yfinance")
    tickers = ASSET_ORDER
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw = raw[tickers]  # enforce column order
    raw.index.name = "Date"
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Pull real data via yfinance instead of synthetic")
    p.add_argument("--start", default=DEFAULT_HISTORY_START)
    p.add_argument("--end", default=DEFAULT_HISTORY_END)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "prices.csv"))
    args = p.parse_args()

    if args.live:
        print(f"Pulling live data for {ASSET_ORDER} from {args.start} to {args.end} via yfinance...")
        df = generate_live(args.start, args.end)
    else:
        print(f"Generating calibrated synthetic dataset for {ASSET_ORDER}, "
              f"{args.start} to {args.end}, seed={args.seed}...")
        df = generate_synthetic(args.start, args.end, seed=args.seed)

    df.to_csv(args.out)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {args.out}")
    print(df.tail())


if __name__ == "__main__":
    main()
