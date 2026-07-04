# Portfolio Risk & Monte Carlo Platform

An interactive quantitative risk-analytics platform for a multi-asset portfolio built as a portfolio
project demonstrating financial risk modeling, simulation, and dashboard-building skills for
quant/risk analyst, portfolio analyst, and asset management roles.

> Portfolio / educational project, not investment advice. See "Data & Methodology" below, the bundled
> dataset is a **calibrated synthetic** price history, not real historical prices.

## What's in here

| Deliverable | Location | Description |
|---|---|---|
| Portfolio return & volatility, Sharpe, Sortino, beta, alpha | `src/metrics.py` | Core risk/return metric functions |
| Maximum drawdown | `src/metrics.py` | Drawdown magnitude + peak/trough/recovery dates |
| VaR & CVaR | `src/metrics.py` | Both historical (empirical) and parametric (Gaussian) methods |
| Correlation analysis | `src/metrics.py`, dashboard `Correlation` tab | Pairwise correlation matrix + heatmap |
| Efficient frontier | `src/efficient_frontier.py` | Markowitz mean-variance optimization (SLSQP), min-variance & max-Sharpe portfolios, random portfolio cloud |
| Monte Carlo simulation | `src/simulation.py` | Aggregated GBM engine |
| Historical bootstrap | `src/simulation.py` | Block-resampling of actual historical daily returns |
| Correlated multivariate simulation | `src/simulation.py` | Cholesky-decomposition joint simulation of every asset, preserving the full covariance structure |
| Interactive dashboard | `dashboard/app.py` | Streamlit app — live weight sliders, all metrics/charts recalculate in real time |
| Validation suite | `tests/test_metrics.py` | 15 tests: known-answer checks + cross-validation across methods |

## Portfolio

| Ticker | Role | Target weight |
|---|---|---:|
| NVDA | Semiconductors / AI infrastructure | 20% |
| MSFT | Mega-cap tech / AI infrastructure | 20% |
| GS | Financials / investment banking | 15% |
| JPM | Financials / diversified bank | 15% |
| TLT | Rate-duration diversifier (20+yr Treasuries) | 15% |
| GLD | Commodity diversifier (gold) | 15% |

Benchmark: **SPY** (S&P 500), used for beta/alpha.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_metrics.py          # confirm everything is working (15/15 should pass)
streamlit run dashboard/app.py        # launch the interactive dashboard
```

The dashboard opens in your browser with live weight sliders — every metric, the efficient frontier,
the correlation heatmap, and all three Monte Carlo engines recalculate as you drag them.

## Data & Methodology

**Why a calibrated synthetic dataset instead of real historical prices?** This repo is built to run
against real data via `yfinance`, but was originally developed in a sandboxed environment with no
outbound access to market-data providers. `data/prices.csv` is therefore a **synthetic** daily price
history — generated via correlated multivariate Geometric Brownian Motion — calibrated to real, sourced
annualized return/volatility/correlation statistics for each ticker (see `data/config.py` for exact
targets and sources). This lets every piece of the pipeline run end-to-end, deterministically, with zero
API keys and zero internet dependency, while still producing realistic and internally consistent output
(realized volatility and correlations in the generated sample match calibration targets closely; realized
*returns* over any single 5-year draw will naturally deviate from long-run targets — that's expected
sampling variation, not a bug, and mirrors how real 5-year windows deviate from long-run CAGR too).

**To use real data instead:**
```bash
pip install yfinance
python data/generate_dataset.py --live --start 2021-01-01 --end 2025-12-31
```
This overwrites `data/prices.csv` with actual adjusted-close prices in the exact same schema — nothing
else in the repo needs to change.

## Methodology notes on the three simulation engines

- **GBM (aggregated):** treats the whole portfolio as one asset with its own historical mean/vol. Fast,
  simple, but assumes iid normal returns and can't capture a scenario where correlations shift (e.g., a
  "everything sells off together" tail event) differently than the historical average correlation would
  suggest.
- **Historical bootstrap:** resamples contiguous *blocks* (not single days) of the portfolio's own
  historical daily returns, which partially preserves volatility clustering and autocorrelation. Makes no
  distributional assumption, but implicitly assumes the future statistically resembles the sampled
  history.
- **Correlated multivariate simulation:** simulates every individual asset jointly via Cholesky
  decomposition of the full covariance matrix, then re-aggregates through the portfolio weights each
  simulated day. This is the most realistic of the three — it's the only engine where a simulated draw in
  which, say, NVDA and MSFT crash together produces a materially different (and correctly worse) portfolio
  outcome than an "average correlation" assumption would predict. `tests/test_metrics.py` includes an
  explicit test proving this engine's diversification benefit is real (lower terminal-value dispersion
  than an artificially perfectly-correlated version of the same portfolio).

All three converge to similar mean outcomes over a long horizon and many simulations (law of large
numbers) — see `test_three_simulation_engines_converge_to_similar_means` — which is itself a useful
validation that none of the three has a hidden bug producing systematically different results.

## Repository structure

```
portfolio-risk-platform/
├── README.md
├── requirements.txt
├── data/
│   ├── config.py              <- portfolio weights, calibration stats, sources
│   ├── generate_dataset.py    <- synthetic or live (--live) dataset generator
│   └── prices.csv             <- bundled sample dataset (see Data & Methodology)
├── src/
│   ├── metrics.py             <- return/vol/Sharpe/Sortino/beta/alpha/drawdown/VaR/CVaR
│   ├── efficient_frontier.py  <- Markowitz optimization
│   └── simulation.py          <- GBM, historical bootstrap, correlated multivariate simulation
├── dashboard/
│   └── app.py                 <- Streamlit interactive dashboard
└── tests/
    └── test_metrics.py        <- 15-test validation suite (known-answer + cross-validation)
```

## Known limitations

- Bundled dataset is synthetic (calibrated, not actual historical prices) — see Data & Methodology.
- Efficient frontier and simulations use long-only, fully-invested constraints (no shorting, no leverage)
  — a reasonable retail-portfolio default; relax the `bounds` in `efficient_frontier.py` for a version
  that allows shorting.
- Correlations and volatilities are assumed constant over the simulation horizon (no GARCH-style
  volatility clustering or regime-switching) — a standard simplifying assumption for this scope of
  project, flagged here rather than silently glossed over.
- Risk-free rate is a single flat assumption (4.5%), not a term structure.

## Suggested next extensions

- Add a GARCH(1,1) volatility model to let the Monte Carlo engines simulate time-varying volatility
  instead of a constant annualized figure
- Add a rolling (e.g., 1-year window) Sharpe/beta/correlation view to the dashboard to show how risk
  characteristics evolve over time, not just as a single full-sample number
- Extend `efficient_frontier.py` with a shorting-allowed and a leverage-constrained variant for comparison
- Add factor-model attribution (Fama-French 3/5-factor) alongside the single-factor CAPM beta/alpha
