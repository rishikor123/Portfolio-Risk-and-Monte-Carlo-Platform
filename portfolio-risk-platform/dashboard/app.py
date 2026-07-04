"""
app.py
======
Interactive Streamlit dashboard for the Portfolio Risk & Monte Carlo Platform.

Run with:
    streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import metrics as m
import efficient_frontier as ef
import simulation as sim
from config import TICKERS, BENCHMARK, ASSET_NAMES, WEIGHTS, RISK_FREE_RATE_ANNUAL

st.set_page_config(page_title="Portfolio Risk & Monte Carlo Platform", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.csv')


@st.cache_data
def load_data():
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns


prices, returns = load_data()
asset_returns = returns[TICKERS]
bench_returns = returns[BENCHMARK]

st.title("📊 Portfolio Risk & Monte Carlo Platform")
st.caption("NVDA · MSFT · GS · JPM · TLT · GLD  |  Benchmark: SPY  |  "
           "Data: calibrated synthetic daily prices (see README) — swap in live data via "
           "`data/generate_dataset.py --live`")

# --------------------------------------------------------------------------
# Sidebar: interactive weight sliders
# --------------------------------------------------------------------------
st.sidebar.header("Portfolio Weights")
st.sidebar.caption("Adjust and the whole dashboard recalculates live. Auto-normalized to sum to 100%.")
weights_input = {}
for t in TICKERS:
    weights_input[t] = st.sidebar.slider(f"{t}", 0.0, 1.0, WEIGHTS[t], 0.01)

total_w = sum(weights_input.values())
weights = {t: w / total_w for t, w in weights_input.items()} if total_w > 0 else WEIGHTS
st.sidebar.write(f"Normalized weights: {', '.join(f'{t} {w:.1%}' for t, w in weights.items())}")

st.sidebar.header("Risk Settings")
risk_free = st.sidebar.number_input("Risk-free rate (annual)", value=RISK_FREE_RATE_ANNUAL, step=0.005, format="%.3f")
var_confidence = st.sidebar.select_slider("VaR / CVaR confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95)

st.sidebar.header("Simulation Settings")
horizon_years = st.sidebar.slider("Simulation horizon (years)", 1, 10, 1)
n_sims = st.sidebar.select_slider("Number of simulations", options=[500, 1000, 2500, 5000, 10000], value=2500)
initial_value = st.sidebar.number_input("Initial portfolio value ($)", value=100_000, step=10_000)

port_returns = m.portfolio_returns(asset_returns, weights)
horizon_days = horizon_years * 252

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_risk, tab_frontier, tab_corr, tab_mc = st.tabs(
    ["Overview", "Risk Metrics", "Efficient Frontier", "Correlation", "Monte Carlo"])

# ---- Overview tab ----
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    ann_ret = m.annualized_return(port_returns)
    ann_vol = m.annualized_volatility(port_returns)
    sharpe = m.sharpe_ratio(port_returns, risk_free)
    sortino = m.sortino_ratio(port_returns, risk_free)
    col1.metric("Annualized Return", f"{ann_ret:.1%}")
    col2.metric("Annualized Volatility", f"{ann_vol:.1%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Sortino Ratio", f"{sortino:.2f}")

    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, name=BENCHMARK, line=dict(dash="dot")))
    fig.update_layout(title="Cumulative Growth of $1", yaxis_title="Growth of $1", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current Allocation")
    fig_pie = px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("Asset universe"):
        for t in TICKERS + [BENCHMARK]:
            st.write(f"**{t}** — {ASSET_NAMES[t]}")

# ---- Risk Metrics tab ----
with tab_risk:
    st.subheader("Drawdown")
    dd_info = m.max_drawdown(port_returns)
    dd_series = dd_info["drawdown_series"]
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, fill="tozeroy", name="Drawdown"))
    fig_dd.update_layout(title=f"Drawdown Over Time  (Max: {dd_info['max_drawdown']:.1%}, "
                                f"trough {dd_info['trough_date'].date()})",
                          yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Max Drawdown", f"{dd_info['max_drawdown']:.1%}")
        st.write(f"Peak: {dd_info['peak_date'].date()}  →  Trough: {dd_info['trough_date'].date()}")
        recov = dd_info['recovery_date']
        st.write(f"Recovered by: {recov.date() if recov is not None else 'Not yet recovered in sample'}")
        st.metric("Beta vs. SPY", f"{m.beta(port_returns, bench_returns):.2f}")
        st.metric("Annualized Alpha vs. SPY", f"{m.alpha_annualized(port_returns, bench_returns, risk_free):.1%}")

    with c2:
        st.write(f"**VaR / CVaR at {var_confidence:.0%} confidence, 1-day horizon**")
        hvar = m.historical_var(port_returns, var_confidence, 1)
        hcvar = m.historical_cvar(port_returns, var_confidence, 1)
        pvar = m.parametric_var(port_returns, var_confidence, 1)
        pcvar = m.parametric_cvar(port_returns, var_confidence, 1)
        risk_df = pd.DataFrame({
            "Method": ["Historical", "Parametric (Gaussian)"],
            "VaR": [f"{hvar:.2%}", f"{pvar:.2%}"],
            "CVaR (Expected Shortfall)": [f"{hcvar:.2%}", f"{pcvar:.2%}"],
        })
        st.table(risk_df.set_index("Method"))
        st.caption(f"On ${initial_value:,.0f}: Historical VaR = ${hvar*initial_value:,.0f}, "
                   f"Historical CVaR = ${hcvar*initial_value:,.0f}")

    st.subheader("Return Distribution")
    fig_hist = px.histogram(port_returns, nbins=80, title="Daily Return Distribution")
    fig_hist.add_vline(x=-hvar, line_color="red", annotation_text=f"-VaR ({var_confidence:.0%})")
    st.plotly_chart(fig_hist, use_container_width=True)

# ---- Efficient Frontier tab ----
with tab_frontier:
    st.subheader("Markowitz Efficient Frontier")
    mu, cov = ef.annualize_mean_cov(asset_returns)

    with st.spinner("Optimizing..."):
        cloud = ef.random_portfolio_cloud(mu, cov, n_portfolios=3000)
        frontier = ef.efficient_frontier(mu, cov, n_points=40)
        mvp = ef.min_variance_portfolio(mu, cov)
        msp = ef.max_sharpe_portfolio(mu, cov, risk_free)

    mvp_ret, mvp_vol = ef.portfolio_perf(mvp.values, mu, cov)
    msp_ret, msp_vol = ef.portfolio_perf(msp.values, mu, cov)
    cur_ret, cur_vol = ef.portfolio_perf(np.array([weights.get(t, 0) for t in mu.index]), mu, cov)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(x=cloud["vol"], y=cloud["ret"], mode="markers",
                                 marker=dict(size=4, color=cloud["sharpe"], colorscale="Viridis",
                                             showscale=True, colorbar=dict(title="Sharpe")),
                                 name="Random portfolios"))
    fig_ef.add_trace(go.Scatter(x=frontier["vol"], y=frontier["ret"], mode="lines",
                                 line=dict(color="black", width=3), name="Efficient frontier"))
    fig_ef.add_trace(go.Scatter(x=[mvp_vol], y=[mvp_ret], mode="markers",
                                 marker=dict(size=14, symbol="star", color="blue"), name="Min variance"))
    fig_ef.add_trace(go.Scatter(x=[msp_vol], y=[msp_ret], mode="markers",
                                 marker=dict(size=14, symbol="star", color="red"), name="Max Sharpe"))
    fig_ef.add_trace(go.Scatter(x=[cur_vol], y=[cur_ret], mode="markers",
                                 marker=dict(size=14, symbol="diamond", color="orange"),
                                 name="Your current weights"))
    fig_ef.update_layout(title="Risk / Return of 3,000 Random Portfolios vs. Efficient Frontier",
                          xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=550)
    st.plotly_chart(fig_ef, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Minimum Variance Portfolio**")
        st.dataframe(mvp.rename("Weight").apply(lambda x: f"{x:.1%}"))
    with c2:
        st.write("**Max Sharpe (Tangency) Portfolio**")
        st.dataframe(msp.rename("Weight").apply(lambda x: f"{x:.1%}"))
    with c3:
        st.write("**Your Current Weights**")
        st.dataframe(pd.Series(weights).rename("Weight").apply(lambda x: f"{x:.1%}"))

# ---- Correlation tab ----
with tab_corr:
    st.subheader("Asset Correlation Matrix")
    corr = asset_returns.join(bench_returns).corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Pairwise Daily Return Correlation")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("GS/JPM show the highest pairwise correlation (same sector); TLT is the primary "
               "diversifier (negative correlation to equities); GLD has near-zero correlation "
               "to everything, the classic 'diversifier of last resort' profile.")

# ---- Monte Carlo tab ----
with tab_mc:
    st.subheader(f"Forward Simulation — {horizon_years}-Year Horizon, {n_sims:,} Simulations")
    method = st.radio("Simulation engine", ["Correlated Multivariate (recommended)", "GBM (aggregated)",
                                             "Historical Bootstrap"], horizontal=True)

    with st.spinner("Simulating..."):
        if method.startswith("Correlated"):
            mu_annual = asset_returns.mean() * 252
            cov_annual = asset_returns.cov() * 252
            paths = sim.correlated_multivariate_simulation(mu_annual, cov_annual, weights,
                                                             initial_value, horizon_days, n_sims)
        elif method.startswith("GBM"):
            paths = sim.monte_carlo_gbm(initial_value, m.annualized_return(port_returns),
                                         m.annualized_volatility(port_returns), horizon_days, n_sims)
        else:
            paths = sim.historical_bootstrap(port_returns, initial_value, horizon_days, n_sims)

    summary = sim.simulation_summary(paths, initial_value)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median terminal value", f"${summary['median_terminal_value']:,.0f}")
    c2.metric("Mean terminal return", f"{summary['mean_terminal_return']:.1%}")
    c3.metric("P(loss)", f"{summary['prob_loss']:.1%}")
    c4.metric("P5 worst drawdown", f"{summary['p5_worst_drawdown']:.1%}")

    n_show = min(300, paths.shape[0])
    fig_paths = go.Figure()
    x = list(range(paths.shape[1]))
    for i in range(n_show):
        fig_paths.add_trace(go.Scatter(x=x, y=paths[i], mode="lines", line=dict(width=0.5),
                                        opacity=0.15, showlegend=False, hoverinfo="skip"))
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    fig_paths.add_trace(go.Scatter(x=x, y=p50, line=dict(color="black", width=3), name="Median"))
    fig_paths.add_trace(go.Scatter(x=x, y=p95, line=dict(color="green", width=2, dash="dash"), name="P95"))
    fig_paths.add_trace(go.Scatter(x=x, y=p5, line=dict(color="red", width=2, dash="dash"), name="P5"))
    fig_paths.update_layout(title=f"{n_show} Sample Simulated Paths (of {n_sims:,} total)",
                             xaxis_title="Trading days ahead", yaxis_title="Portfolio value ($)", height=500)
    st.plotly_chart(fig_paths, use_container_width=True)

    fig_term = px.histogram(paths[:, -1], nbins=80, title="Distribution of Terminal Portfolio Value")
    fig_term.add_vline(x=initial_value, line_color="black", annotation_text="Initial value")
    st.plotly_chart(fig_term, use_container_width=True)

    st.caption("**Correlated Multivariate** simulates every asset jointly via Cholesky decomposition, "
               "preserving the full covariance structure — the most realistic engine. **GBM** simulates "
               "only the portfolio's aggregated mean/vol (fast, but ignores how correlations could shift "
               "diversification in a given draw). **Historical Bootstrap** resamples blocks of the "
               "portfolio's own historical daily returns, making no distributional assumption at all.")

st.divider()
st.caption("Portfolio Risk & Monte Carlo Platform — portfolio/educational project, not investment advice.")
