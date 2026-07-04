"""
metrics.py
==========
Core portfolio risk/return metric functions. Pure functions operating on
pandas Series/DataFrames of simple (not log) daily returns, so they can be
unit-tested independently of the data source (synthetic or live).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted daily portfolio return series from a DataFrame of asset returns."""
    w = pd.Series(weights)
    w = w.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns.mul(w, axis=1).sum(axis=1)


def annualized_return(daily_returns: pd.Series) -> float:
    """Geometric (CAGR-style) annualized return from a daily return series."""
    compounded = (1 + daily_returns).prod()
    n_years = len(daily_returns) / TRADING_DAYS
    return compounded ** (1 / n_years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float) -> float:
    excess_ann_return = annualized_return(daily_returns) - risk_free_annual
    vol = annualized_volatility(daily_returns)
    return excess_ann_return / vol if vol > 0 else np.nan


def sortino_ratio(daily_returns: pd.Series, risk_free_annual: float) -> float:
    """Like Sharpe, but penalizes only downside deviation. Standard definition:
    downside deviation = sqrt( sum(min(r - MAR, 0)^2) / N ), i.e. divided by
    the TOTAL number of observations (with 0 contributed by non-downside
    days), not just the count of downside days -- a common implementation
    mistake that inflates the metric when downside days are rare."""
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    downside_sq = np.minimum(daily_returns - rf_daily, 0) ** 2
    downside_dev_annual = np.sqrt(downside_sq.mean()) * np.sqrt(TRADING_DAYS)
    excess_ann_return = annualized_return(daily_returns) - risk_free_annual
    return excess_ann_return / downside_dev_annual if downside_dev_annual and downside_dev_annual > 0 else np.nan


def beta(asset_or_portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_or_portfolio_returns, benchmark_returns], axis=1).dropna()
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return cov / var if var > 0 else np.nan


def alpha_annualized(portfolio_returns_: pd.Series, benchmark_returns: pd.Series,
                      risk_free_annual: float) -> float:
    """CAPM (Jensen's) alpha: excess return not explained by beta-adjusted
    market exposure."""
    b = beta(portfolio_returns_, benchmark_returns)
    port_ann = annualized_return(portfolio_returns_)
    bench_ann = annualized_return(benchmark_returns)
    expected = risk_free_annual + b * (bench_ann - risk_free_annual)
    return port_ann - expected


def max_drawdown(daily_returns: pd.Series) -> dict:
    """Returns max drawdown magnitude plus the peak/trough/recovery dates."""
    wealth = (1 + daily_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    trough_date = drawdown.idxmin()
    mdd = drawdown.loc[trough_date]
    peak_date = wealth.loc[:trough_date].idxmax()
    post_trough = wealth.loc[trough_date:]
    recovered = post_trough[post_trough >= running_max.loc[trough_date]]
    recovery_date = recovered.index[0] if len(recovered) else None
    return dict(max_drawdown=mdd, peak_date=peak_date, trough_date=trough_date,
                recovery_date=recovery_date, drawdown_series=drawdown)


def historical_var(daily_returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Historical (empirical) Value at Risk, as a POSITIVE number representing
    a loss magnitude, scaled to horizon_days via sqrt-of-time."""
    q = 1 - confidence
    daily_var = -np.percentile(daily_returns, q * 100)
    return daily_var * np.sqrt(horizon_days)


def historical_cvar(daily_returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Historical Conditional VaR (Expected Shortfall): average loss in the
    tail beyond the VaR threshold."""
    q = 1 - confidence
    threshold = np.percentile(daily_returns, q * 100)
    tail_losses = daily_returns[daily_returns <= threshold]
    daily_cvar = -tail_losses.mean() if len(tail_losses) else np.nan
    return daily_cvar * np.sqrt(horizon_days)


def parametric_var(daily_returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Variance-covariance (Gaussian) VaR, as a POSITIVE loss magnitude."""
    from scipy.stats import norm
    mu, sigma = daily_returns.mean(), daily_returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    daily_var = -(mu + z * sigma)
    return daily_var * np.sqrt(horizon_days)


def parametric_cvar(daily_returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Gaussian CVaR (closed form: mu + sigma * phi(z)/(1-confidence))."""
    from scipy.stats import norm
    mu, sigma = daily_returns.mean(), daily_returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    daily_cvar = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return daily_cvar * np.sqrt(horizon_days)


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.corr()


def summary_table(daily_returns: pd.Series, benchmark_returns: pd.Series, risk_free_annual: float) -> dict:
    """One-stop summary of every headline metric for a return series."""
    mdd = max_drawdown(daily_returns)
    return {
        "Annualized Return": annualized_return(daily_returns),
        "Annualized Volatility": annualized_volatility(daily_returns),
        "Sharpe Ratio": sharpe_ratio(daily_returns, risk_free_annual),
        "Sortino Ratio": sortino_ratio(daily_returns, risk_free_annual),
        "Beta (vs. benchmark)": beta(daily_returns, benchmark_returns),
        "Alpha (annualized, vs. benchmark)": alpha_annualized(daily_returns, benchmark_returns, risk_free_annual),
        "Max Drawdown": mdd["max_drawdown"],
        "Max Drawdown Peak Date": mdd["peak_date"],
        "Max Drawdown Trough Date": mdd["trough_date"],
        "Max Drawdown Recovery Date": mdd["recovery_date"],
        "Historical VaR (95%, 1-day)": historical_var(daily_returns, 0.95, 1),
        "Historical CVaR (95%, 1-day)": historical_cvar(daily_returns, 0.95, 1),
        "Parametric VaR (95%, 1-day)": parametric_var(daily_returns, 0.95, 1),
        "Parametric CVaR (95%, 1-day)": parametric_cvar(daily_returns, 0.95, 1),
        "Historical VaR (99%, 1-day)": historical_var(daily_returns, 0.99, 1),
        "Historical CVaR (99%, 1-day)": historical_cvar(daily_returns, 0.99, 1),
    }
