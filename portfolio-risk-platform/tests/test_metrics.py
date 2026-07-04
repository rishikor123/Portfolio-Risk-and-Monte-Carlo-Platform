"""
test_metrics.py
================
Validation suite for the risk platform. Two kinds of checks:

1. Known-answer tests: hand-constructed series where the correct metric
   value can be computed by hand (or via an independent formula), so we're
   not just checking "the code runs" but "the code is right."
2. Cross-validation / consistency checks: e.g., the three simulation engines
   should converge to similar summary statistics (law of large numbers), and
   the efficient frontier's minimum-variance point should actually have the
   lowest volatility of any portfolio on the frontier.

Runs standalone with plain Python (no pytest required), or via `pytest` if
installed -- every function name starts with `test_` either way.

Usage:
    python test_metrics.py
    # or
    pytest test_metrics.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import pandas as pd
import metrics as m
import efficient_frontier as ef
import simulation as sim
from config import TICKERS, BENCHMARK, WEIGHTS, RISK_FREE_RATE_ANNUAL

TOL = 1e-6


def _assert_close(actual, expected, tol=1e-4, msg=""):
    assert abs(actual - expected) < tol, f"{msg}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 1. Known-answer tests
# ---------------------------------------------------------------------------

def test_annualized_return_known_case():
    # 252 trading days of a constant 0.01% daily return should compound to
    # a known annualized figure: (1.0001)^252 - 1
    daily = pd.Series([0.0001] * 252)
    expected = (1.0001) ** 252 - 1
    _assert_close(m.annualized_return(daily), expected, tol=1e-8)


def test_annualized_volatility_known_case():
    # Constant daily std of 0.01 should annualize to 0.01 * sqrt(252)
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0, 0.01, 5000))
    expected = 0.01 * np.sqrt(252)
    _assert_close(m.annualized_volatility(daily), expected, tol=3e-3)


def test_beta_of_asset_vs_itself_is_one():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    _assert_close(m.beta(r, r), 1.0, tol=1e-8)


def test_beta_known_linear_relationship():
    # Construct benchmark returns, then an asset that is EXACTLY
    # 1.5x the benchmark plus idiosyncratic noise with zero mean and
    # zero correlation to the benchmark -> beta should recover ~1.5
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0.0004, 0.01, 5000))
    noise = pd.Series(rng.normal(0, 0.002, 5000))
    asset = 1.5 * bench + noise
    _assert_close(m.beta(asset, bench), 1.5, tol=0.05)


def test_max_drawdown_known_case():
    # Prices go 100 -> 150 -> 75 -> 90. Drawdown from peak 150 to trough 75
    # is exactly -50%.
    prices = pd.Series([100, 150, 75, 90])
    returns = prices.pct_change().dropna()
    result = m.max_drawdown(returns)
    _assert_close(result["max_drawdown"], -0.5, tol=1e-8)


def test_historical_var_matches_percentile_definition():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.02, 10000))
    var95 = m.historical_var(r, confidence=0.95, horizon_days=1)
    # by definition, ~5% of returns should be worse than -var95
    frac_worse = (r < -var95).mean()
    assert abs(frac_worse - 0.05) < 0.01, f"expected ~5% of returns below -VaR, got {frac_worse:.3%}"


def test_cvar_worse_than_var():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.standard_t(df=4, size=10000) * 0.01)  # fat-tailed
    var95 = m.historical_var(r, 0.95, 1)
    cvar95 = m.historical_cvar(r, 0.95, 1)
    assert cvar95 >= var95, "CVaR (average tail loss) must be >= VaR (tail threshold) by definition"


def test_parametric_var_close_to_historical_for_normal_data():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0003, 0.015, 20000))
    hist = m.historical_var(r, 0.95, 1)
    param = m.parametric_var(r, 0.95, 1)
    _assert_close(hist, param, tol=0.002, msg="Historical vs parametric VaR should be close for ~normal data")


def test_sortino_ge_related_to_sharpe_for_negatively_skewed_returns():
    # For left-skewed (crash-prone) return series, Sortino penalizes downside
    # more than Sharpe's symmetric std dev would -- Sortino should be LOWER
    # than Sharpe in that case (harsher assessment of downside-heavy risk).
    rng = np.random.default_rng(6)
    up_days = rng.normal(0.0015, 0.005, 900)
    crash_days = rng.normal(-0.03, 0.01, 100)
    r = pd.Series(np.concatenate([up_days, crash_days]))
    sharpe = m.sharpe_ratio(r, RISK_FREE_RATE_ANNUAL)
    sortino = m.sortino_ratio(r, RISK_FREE_RATE_ANNUAL)
    assert sortino < sharpe, f"Expected Sortino ({sortino:.3f}) < Sharpe ({sharpe:.3f}) for downside-skewed returns"


# ---------------------------------------------------------------------------
# 2. Cross-validation / consistency checks using the real bundled dataset
# ---------------------------------------------------------------------------

def _load_data():
    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.csv'),
                          index_col=0, parse_dates=True)
    rets = prices.pct_change().dropna()
    return rets[TICKERS], rets[BENCHMARK]


def test_portfolio_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < TOL


def test_correlation_matrix_is_valid():
    asset_rets, _ = _load_data()
    corr = asset_rets.corr()
    eigvals = np.linalg.eigvalsh(corr.values)
    assert (eigvals >= -1e-8).all(), "Correlation matrix must be positive semi-definite"
    assert np.allclose(np.diag(corr.values), 1.0), "Correlation matrix diagonal must be 1.0"


def test_min_variance_portfolio_has_lowest_vol_on_frontier():
    asset_rets, _ = _load_data()
    mu, cov = ef.annualize_mean_cov(asset_rets)
    mvp = ef.min_variance_portfolio(mu, cov)
    mvp_ret, mvp_vol = ef.portfolio_perf(mvp.values, mu, cov)
    frontier = ef.efficient_frontier(mu, cov, n_points=25)
    assert mvp_vol <= frontier["vol"].min() + 1e-3, "MVP should have the lowest volatility on the frontier"


def test_max_sharpe_portfolio_beats_equal_weight():
    asset_rets, _ = _load_data()
    mu, cov = ef.annualize_mean_cov(asset_rets)
    msp = ef.max_sharpe_portfolio(mu, cov, RISK_FREE_RATE_ANNUAL)
    msp_ret, msp_vol = ef.portfolio_perf(msp.values, mu, cov)
    msp_sharpe = (msp_ret - RISK_FREE_RATE_ANNUAL) / msp_vol

    n = len(mu)
    eq_w = np.repeat(1 / n, n)
    eq_ret, eq_vol = ef.portfolio_perf(eq_w, mu, cov)
    eq_sharpe = (eq_ret - RISK_FREE_RATE_ANNUAL) / eq_vol

    assert msp_sharpe >= eq_sharpe - 1e-6, "Max-Sharpe portfolio should have Sharpe >= equal-weight by construction"


def test_three_simulation_engines_converge_to_similar_means():
    """The GBM, bootstrap, and correlated-multivariate engines make different
    assumptions, so they won't match exactly -- but over a 1-year horizon
    with enough simulations, their mean terminal returns should be within a
    reasonable band of each other (they're all approximating the same
    underlying portfolio), and none should be wildly divergent."""
    asset_rets, bench_rets = _load_data()
    port_rets = m.portfolio_returns(asset_rets, WEIGHTS)
    initial_value = 100_000
    n_sims = 3000
    horizon = 252

    paths_gbm = sim.monte_carlo_gbm(initial_value, m.annualized_return(port_rets),
                                     m.annualized_volatility(port_rets), horizon, n_sims)
    paths_boot = sim.historical_bootstrap(port_rets, initial_value, horizon, n_sims)
    mu_annual = asset_rets.mean() * 252
    cov_annual = asset_rets.cov() * 252
    paths_mv = sim.correlated_multivariate_simulation(mu_annual, cov_annual, WEIGHTS,
                                                        initial_value, horizon, n_sims)

    means = [sim.simulation_summary(p, initial_value)["mean_terminal_return"]
             for p in (paths_gbm, paths_boot, paths_mv)]
    spread = max(means) - min(means)
    assert spread < 0.10, f"Simulation engines diverged more than expected: {means} (spread {spread:.3f})"


def test_multivariate_simulation_respects_diversification():
    """A portfolio simulated with the TRUE (diversifying) correlation
    structure should show a lower terminal-value std dev than a naive
    simulation that (incorrectly) treats all assets as perfectly correlated
    (i.e., ignoring diversification) -- this validates that the Cholesky
    step is actually doing something, not just adding noise."""
    asset_rets, _ = _load_data()
    mu_annual = asset_rets.mean() * 252
    cov_annual = asset_rets.cov() * 252
    initial_value = 100_000
    n_sims = 3000
    horizon = 126

    paths_real_corr = sim.correlated_multivariate_simulation(
        mu_annual, cov_annual, WEIGHTS, initial_value, horizon, n_sims, seed=1)

    # Build a "perfectly correlated" covariance matrix with the same
    # individual variances (i.e., correlation = 1.0 everywhere)
    vols = np.sqrt(np.diag(cov_annual.values))
    perfect_corr_cov = np.outer(vols, vols) + np.eye(len(vols)) * 1e-6
    perfect_corr_cov = pd.DataFrame(perfect_corr_cov, index=cov_annual.index, columns=cov_annual.columns)
    paths_perfect_corr = sim.correlated_multivariate_simulation(
        mu_annual, perfect_corr_cov, WEIGHTS, initial_value, horizon, n_sims, seed=1)

    std_real = paths_real_corr[:, -1].std()
    std_perfect = paths_perfect_corr[:, -1].std()
    assert std_real < std_perfect, (
        f"Diversified simulation (std={std_real:.0f}) should have lower terminal-value dispersion "
        f"than the perfectly-correlated case (std={std_perfect:.0f})")


# ---------------------------------------------------------------------------
# Test runner (works without pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_fns = [(name, obj) for name, obj in list(globals().items())
                if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(test_fns)} tests")
    sys.exit(1 if failed else 0)
