"""
simulation.py
=============
Three forward-looking simulation engines for portfolio value paths:

1. monte_carlo_gbm       - single-asset-equivalent Geometric Brownian Motion
                            using the PORTFOLIO's own aggregated mu/sigma.
                            Fast, simple, assumes returns are iid normal and
                            ignores how the portfolio's risk is actually built
                            up from correlated components.
2. historical_bootstrap  - resamples the portfolio's own historical daily
                            return series (block bootstrap by default) to
                            build forward paths. Makes no distributional
                            assumption, but implicitly assumes the future
                            will look statistically like the historical window.
3. correlated_multivariate_simulation - simulates every ASSET forward
                            (not just the aggregated portfolio), preserving
                            the full covariance structure via Cholesky
                            decomposition, then re-aggregates through the
                            portfolio weights each day. The most realistic of
                            the three because it lets correlations shift the
                            portfolio's effective diversification over time
                            (e.g., a simulated draw where NVDA and MSFT both
                            crash together hits the portfolio harder than the
                            other two methods would capture).

All three return a (n_sims x horizon_days) array of simulated portfolio
VALUES (starting from initial_value), so they can be compared apples-to-apples.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def monte_carlo_gbm(initial_value, mu_annual, sigma_annual, horizon_days, n_sims, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    shock_scale = sigma_annual * np.sqrt(dt)
    z = rng.standard_normal(size=(n_sims, horizon_days))
    log_returns = drift + shock_scale * z
    log_paths = np.cumsum(log_returns, axis=1)
    values = initial_value * np.exp(log_paths)
    return np.hstack([np.full((n_sims, 1), initial_value), values])


def historical_bootstrap(daily_returns: pd.Series, initial_value, horizon_days, n_sims,
                          block_size=20, seed=42):
    """Block bootstrap: resample contiguous blocks of historical daily returns
    (rather than iid single days) to partially preserve short-term
    autocorrelation and volatility clustering, then stitch blocks together
    until the horizon is filled."""
    rng = np.random.default_rng(seed)
    returns = daily_returns.dropna().values
    n_hist = len(returns)
    values = np.zeros((n_sims, horizon_days + 1))
    values[:, 0] = initial_value

    for i in range(n_sims):
        path_returns = []
        while len(path_returns) < horizon_days:
            start = rng.integers(0, n_hist - block_size)
            path_returns.extend(returns[start:start + block_size])
        path_returns = np.array(path_returns[:horizon_days])
        values[i, 1:] = initial_value * np.cumprod(1 + path_returns)
    return values


def correlated_multivariate_simulation(mu_annual: pd.Series, cov_annual: pd.DataFrame,
                                        weights: dict, initial_value, horizon_days,
                                        n_sims, seed=42):
    """Simulate every asset jointly via Cholesky-correlated GBM shocks, then
    aggregate to a portfolio value path through the (fixed, no-rebalancing-
    drift-adjustment) target weights applied to each asset's simulated
    cumulative return."""
    rng = np.random.default_rng(seed)
    assets = list(mu_annual.index)
    n_assets = len(assets)
    w = np.array([weights.get(a, 0.0) for a in assets])

    dt = 1 / TRADING_DAYS
    mu = mu_annual.values
    cov_daily = cov_annual.values * dt
    L = np.linalg.cholesky(cov_daily)
    drift = (mu - 0.5 * np.diag(cov_annual.values)) * dt

    portfolio_values = np.zeros((n_sims, horizon_days + 1))
    portfolio_values[:, 0] = initial_value

    for i in range(n_sims):
        z = rng.standard_normal(size=(horizon_days, n_assets))
        shocks = z @ L.T
        log_returns = drift + shocks
        log_paths = np.cumsum(log_returns, axis=0)
        asset_relative_value = np.exp(log_paths)  # (horizon_days, n_assets), each asset's cum growth factor
        # Portfolio value = sum over assets of (initial $ allocated to that asset) * growth factor
        dollar_alloc = initial_value * w
        port_path = asset_relative_value @ dollar_alloc
        portfolio_values[i, 1:] = port_path

    return portfolio_values


def simulation_summary(paths: np.ndarray, initial_value: float) -> dict:
    """Common summary stats for any (n_sims x horizon+1) value-path array."""
    terminal = paths[:, -1]
    terminal_returns = terminal / initial_value - 1
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_max - 1
    worst_drawdowns = drawdowns.min(axis=1)

    return {
        "mean_terminal_value": terminal.mean(),
        "median_terminal_value": np.median(terminal),
        "std_terminal_value": terminal.std(),
        "p5_terminal_value": np.percentile(terminal, 5),
        "p25_terminal_value": np.percentile(terminal, 25),
        "p75_terminal_value": np.percentile(terminal, 75),
        "p95_terminal_value": np.percentile(terminal, 95),
        "mean_terminal_return": terminal_returns.mean(),
        "prob_loss": (terminal_returns < 0).mean(),
        "mean_worst_drawdown": worst_drawdowns.mean(),
        "p5_worst_drawdown": np.percentile(worst_drawdowns, 5),
    }
