"""
efficient_frontier.py
======================
Markowitz mean-variance optimization: computes the efficient frontier via
constrained quadratic optimization (scipy.optimize.minimize, SLSQP), plus a
random-portfolio cloud for visualization, the max-Sharpe ("tangency")
portfolio, and the global minimum-variance portfolio.

Long-only, fully-invested constraint (weights sum to 1, each weight in
[0, 1]) is used throughout -- a reasonable default for a retail/long-only
portfolio context; relax `bounds` for a version that allows shorting.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def annualize_mean_cov(daily_returns: pd.DataFrame):
    mu = daily_returns.mean() * TRADING_DAYS
    cov = daily_returns.cov() * TRADING_DAYS
    return mu, cov


def portfolio_perf(weights, mu, cov):
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(weights @ cov.values @ weights))
    return ret, vol


def min_variance_portfolio(mu, cov):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def objective(w):
        return w @ cov.values @ w

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(res.x, index=mu.index)


def max_sharpe_portfolio(mu, cov, risk_free_annual):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def neg_sharpe(w):
        ret, vol = portfolio_perf(w, mu, cov)
        return -(ret - risk_free_annual) / vol if vol > 0 else 1e6

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(res.x, index=mu.index)


def efficient_frontier(mu, cov, n_points=50):
    """Trace the frontier by minimizing variance for a grid of target returns
    spanning [min-variance-portfolio return, max single-asset return]."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))

    mvp = min_variance_portfolio(mu, cov)
    mvp_ret, _ = portfolio_perf(mvp.values, mu, cov)
    target_returns = np.linspace(mvp_ret, mu.max(), n_points)

    frontier = []
    x0 = np.repeat(1 / n, n)
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, mu) - target},
        )

        def objective(w):
            return w @ cov.values @ w

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            ret, vol = portfolio_perf(res.x, mu, cov)
            frontier.append(dict(target_return=target, ret=ret, vol=vol, weights=res.x))
    return pd.DataFrame(frontier)


def random_portfolio_cloud(mu, cov, n_portfolios=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mu)
    results = np.zeros((n_portfolios, 3))
    all_weights = np.zeros((n_portfolios, n))
    for i in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        ret, vol = portfolio_perf(w, mu, cov)
        sharpe = ret / vol if vol > 0 else np.nan
        results[i] = [ret, vol, sharpe]
        all_weights[i] = w
    df = pd.DataFrame(results, columns=["ret", "vol", "sharpe"])
    for j, name in enumerate(mu.index):
        df[f"w_{name}"] = all_weights[:, j]
    return df
