"""
config.py
=========
Single source of truth for the portfolio definition and the calibration
statistics used to generate the bundled sample dataset (data/prices.csv).

WHY A CALIBRATED SYNTHETIC DATASET INSTEAD OF LIVE DATA?
This repo is designed to run against real historical prices via `yfinance`
(see data/generate_dataset.py --live). The sandbox this was originally built
in has no outbound network access to market-data providers, so the bundled
data/prices.csv is a SYNTHETIC daily price history (correlated multivariate
GBM) calibrated to real, sourced annualized return/volatility/correlation
statistics for each ticker -- NOT actual historical closing prices. This lets
the entire pipeline (every metric, the efficient frontier, both Monte Carlo
engines, the dashboard) run end-to-end out of the box with no API keys and
no internet dependency, while still producing realistic, defensible output.

Swap in real data any time: `python data/generate_dataset.py --live
--start 2021-01-01 --end 2025-12-31` pulls actual adjusted close prices via
yfinance and overwrites data/prices.csv with the exact same schema.
"""

TICKERS = ["NVDA", "MSFT", "GS", "JPM", "TLT", "GLD"]
BENCHMARK = "SPY"

ASSET_NAMES = {
    "NVDA": "NVIDIA Corp. (semiconductors / AI infrastructure)",
    "MSFT": "Microsoft Corp. (mega-cap tech / AI infrastructure)",
    "GS":   "Goldman Sachs Group (financials / investment banking)",
    "JPM":  "JPMorgan Chase & Co. (financials / diversified bank)",
    "TLT":  "iShares 20+ Year Treasury Bond ETF (rate duration / diversifier)",
    "GLD":  "SPDR Gold Shares (commodities / diversifier)",
    "SPY":  "SPDR S&P 500 ETF Trust (benchmark)",
}

# Target portfolio weights (must sum to 1.0). A growth-tilted, moderately
# diversified allocation: two high-growth tech/AI names, two financials,
# one rate-duration diversifier, one commodity diversifier.
WEIGHTS = {
    "NVDA": 0.20,
    "MSFT": 0.20,
    "GS":   0.15,
    "JPM":  0.15,
    "TLT":  0.15,
    "GLD":  0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

RISK_FREE_RATE_ANNUAL = 0.045  # ~T-bill rate assumption, used in Sharpe/Sortino

# ---------------------------------------------------------------------------
# Calibration statistics for the synthetic dataset generator.
# Annualized expected return and volatility per asset, informed by sourced
# 5-10yr historical figures (see README "Data & Methodology"), moderated
# toward more conservative forward-looking assumptions where recent realized
# performance was an extreme outlier (esp. NVDA's AI-supercycle run).
# Sources (approximate, pulled ~July 2026): portfolioslab.com, financecharts.com
# per-ticker performance pages. These are CALIBRATION TARGETS for a synthetic
# series, not a claim about actual historical returns -- see data/README note.
# ---------------------------------------------------------------------------
CALIBRATION = {
    #           exp. annual return   annual volatility
    "NVDA": dict(mu=0.40, sigma=0.48),
    "MSFT": dict(mu=0.20, sigma=0.28),
    "GS":   dict(mu=0.17, sigma=0.28),
    "JPM":  dict(mu=0.18, sigma=0.25),
    "TLT":  dict(mu=0.03, sigma=0.14),
    "GLD":  dict(mu=0.12, sigma=0.15),
    "SPY":  dict(mu=0.11, sigma=0.16),
}

# Correlation matrix (order follows TICKERS + [BENCHMARK]), symmetric.
# Approximate values informed by sourced pairwise correlations (JPM-SPY 0.55,
# TLT-SPY -0.20 to -0.25, GLD-TLT 0.15-0.30, MSFT-SPY ~0.5-0.7 typical for
# mega-cap tech) and reasonable priors for pairs not directly sourced
# (NVDA-MSFT tech-sector correlation, JPM-GS bank-sector correlation).
ASSET_ORDER = TICKERS + [BENCHMARK]  # NVDA, MSFT, GS, JPM, TLT, GLD, SPY

CORRELATION_MATRIX = [
    # NVDA  MSFT   GS    JPM   TLT   GLD   SPY
    [1.00, 0.55, 0.35, 0.30, -0.15, 0.05, 0.65],   # NVDA
    [0.55, 1.00, 0.40, 0.35, -0.10, 0.05, 0.68],   # MSFT
    [0.35, 0.40, 1.00, 0.82, -0.20, 0.05, 0.65],   # GS
    [0.30, 0.35, 0.82, 1.00, -0.20, 0.05, 0.62],   # JPM
    [-0.15, -0.10, -0.20, -0.20, 1.00, 0.20, -0.22],  # TLT
    [0.05, 0.05, 0.05, 0.05, 0.20, 1.00, 0.08],    # GLD
    [0.65, 0.68, 0.65, 0.62, -0.22, 0.08, 1.00],   # SPY
]

TRADING_DAYS_PER_YEAR = 252
DEFAULT_HISTORY_START = "2021-01-01"
DEFAULT_HISTORY_END = "2025-12-31"
