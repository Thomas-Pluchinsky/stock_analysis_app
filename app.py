# app.py
# -------------------------------------------------------
# Multi-Stock Comparison & Analysis Dashboard
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import math
from scipy import stats

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Comparison Dashboard", layout="wide")
st.title("Stock Comparison & Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker_input = st.sidebar.text_input(
    "Stock Tickers (2–5, comma-separated)",
    value="AAPL, MSFT, GOOG",
    help="Enter 2 to 5 ticker symbols separated by commas.",
)

raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
seen = set()
tickers = []
for t in raw_tickers:
    if t not in seen:
        seen.add(t)
        tickers.append(t)

if len(tickers) < 2:
    st.sidebar.error("Please enter at least 2 ticker symbols.")
    st.stop()
if len(tickers) > 5:
    st.sidebar.error("Please enter no more than 5 ticker symbols.")
    st.stop()

default_start = date.today() - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("Date range must span at least 1 year.")
    st.stop()

ma_window = st.sidebar.slider(
    "Moving Average Window (days)", min_value=5, max_value=200, value=50, step=5
)
vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)", min_value=10, max_value=120, value=30, step=5
)

with st.sidebar.expander("About / Methodology"):
    st.markdown(
        """
**What this app does**

This dashboard compares multiple stocks across price trends, return
distributions, risk measures, and correlation / diversification analysis.

**Key assumptions**
- **Returns:** Simple (arithmetic) daily returns via `pct_change()`.
- **Annualization:** 252 trading days per year. Mean daily return × 252
  for annualized return; daily σ × √252 for annualized volatility.
- **Cumulative wealth:** (1 + r).cumprod() applied to simple returns,
  scaled to a $10,000 starting investment.
- **Two-asset portfolio:** Classical mean-variance formula using
  annualized returns and the annualized covariance matrix.

**Data source:** Yahoo Finance via the `yfinance` library.
Adjusted close prices are used throughout to account for dividends
and stock splits.
    """
    )

# -- Data download ----------------------------------------
BENCHMARK = "^GSPC"


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(tickers_tuple, benchmark, start, end):
    all_symbols = list(tickers_tuple) + [benchmark]
    failed = []
    frames = {}

    for sym in all_symbols:
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df.empty or len(df) < 20:
                failed.append(sym)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[sym] = df["Close"]
        except Exception:
            failed.append(sym)

    if not frames:
        return pd.DataFrame(), failed, []

    prices = pd.DataFrame(frames)
    threshold = len(prices) * 0.05
    cols_to_drop = [c for c in prices.columns if prices[c].isna().sum() > threshold]
    failed.extend([c for c in cols_to_drop if c != benchmark])
    prices = prices.drop(columns=cols_to_drop, errors="ignore")
    prices = prices.dropna(how="all").ffill().dropna()

    successful = [c for c in prices.columns if c != benchmark and c not in failed]
    return prices, failed, successful


with st.spinner("Fetching market data..."):
    prices, failed_tickers, valid_tickers = load_data(
        tuple(tickers), BENCHMARK, start_date, end_date
    )

failed_user_tickers = [t for t in failed_tickers if t != BENCHMARK]
if failed_user_tickers:
    st.warning(
        f"Could not retrieve sufficient data for: **{', '.join(failed_user_tickers)}**. "
        "These tickers have been excluded. Check the symbols and try again."
    )

if BENCHMARK in failed_tickers:
    st.warning(
        "Could not download S&P 500 benchmark data. Benchmark comparisons will be unavailable."
    )
    has_benchmark = False
else:
    has_benchmark = BENCHMARK in prices.columns

if len(valid_tickers) < 2:
    st.error(
        "Need at least 2 valid tickers with sufficient overlapping data. "
        "Please adjust your inputs in the sidebar."
    )
    st.stop()

st.caption(
    f"Showing **{len(valid_tickers)}** stocks from "
    f"**{prices.index[0].strftime('%Y-%m-%d')}** to "
    f"**{prices.index[-1].strftime('%Y-%m-%d')}** "
    f"({len(prices)} trading days)"
)

stock_cols = valid_tickers + ([BENCHMARK] if has_benchmark else [])
returns = prices[stock_cols].pct_change().dropna()

# -- Tabs -------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "Price & Returns",
        "Risk & Distribution",
        "Correlation & Diversification",
    ]
)

# =========================================================
# TAB 1 — Price & Return Analysis  (§2.2)
# =========================================================
with tab1:
    st.subheader("Adjusted Close Prices")
    selected_price = st.multiselect(
        "Select stocks to display",
        options=valid_tickers,
        default=valid_tickers,
        key="price_select",
    )
    if selected_price:
        fig_price = go.Figure()
        for sym in selected_price:
            fig_price.add_trace(
                go.Scatter(x=prices.index, y=prices[sym], mode="lines", name=sym)
            )
        fig_price.update_layout(
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Select at least one stock above.")

    st.divider()

    st.subheader("Summary Statistics")

    def compute_stats(r):
        return {
            "Ann. Mean Return": r.mean() * 252,
            "Ann. Volatility": r.std() * math.sqrt(252),
            "Skewness": r.skew(),
            "Excess Kurtosis": r.kurtosis(),
            "Min Daily Return": r.min(),
            "Max Daily Return": r.max(),
        }

    stats_rows = {}
    for sym in valid_tickers:
        stats_rows[sym] = compute_stats(returns[sym])
    if has_benchmark:
        stats_rows["S&P 500"] = compute_stats(returns[BENCHMARK])

    stats_df = pd.DataFrame(stats_rows).T
    fmt_df = stats_df.copy()
    for col in ["Ann. Mean Return", "Ann. Volatility", "Min Daily Return", "Max Daily Return"]:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.2%}")
    for col in ["Skewness", "Excess Kurtosis"]:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.3f}")

    st.dataframe(fmt_df, use_container_width=True)

    st.divider()

    st.subheader("Growth of $10,000 Investment")

    wealth = (1 + returns).cumprod() * 10_000
    ew_returns = returns[valid_tickers].mean(axis=1)
    ew_wealth = (1 + ew_returns).cumprod() * 10_000

    fig_wealth = go.Figure()
    for sym in valid_tickers:
        fig_wealth.add_trace(
            go.Scatter(x=wealth.index, y=wealth[sym], mode="lines", name=sym)
        )
    if has_benchmark:
        fig_wealth.add_trace(
            go.Scatter(
                x=wealth.index,
                y=wealth[BENCHMARK],
                mode="lines",
                name="S&P 500",
                line=dict(dash="dash", color="gray"),
            )
        )
    fig_wealth.add_trace(
        go.Scatter(
            x=ew_wealth.index,
            y=ew_wealth,
            mode="lines",
            name="Equal-Weight Portfolio",
            line=dict(dash="dot", width=2.5, color="black"),
        )
    )
    fig_wealth.update_layout(
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

# =========================================================
# TAB 2 — Risk & Distribution Analysis  (§2.3)
# =========================================================
with tab2:
    st.subheader("Rolling Annualized Volatility")

    rolling_vol = returns[valid_tickers].rolling(window=vol_window).std() * math.sqrt(252)

    fig_rvol = go.Figure()
    for sym in valid_tickers:
        fig_rvol.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol[sym], mode="lines", name=sym)
        )
    fig_rvol.update_layout(
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_rvol, use_container_width=True)

    st.divider()

    st.subheader("Return Distribution Analysis")

    dist_stock = st.selectbox(
        "Select a stock for distribution analysis", valid_tickers, key="dist_stock"
    )
    r_series = returns[dist_stock].dropna()

    hist_tab, qq_tab = st.tabs(["Histogram", "Q-Q Plot"])

    with hist_tab:
        mu_fit, sigma_fit = stats.norm.fit(r_series)
        x_range = np.linspace(float(r_series.min()), float(r_series.max()), 200)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=r_series,
                nbinsx=60,
                marker_color="mediumpurple",
                opacity=0.75,
                name="Daily Returns",
                histnorm="probability density",
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, mu_fit, sigma_fit),
                mode="lines",
                name="Fitted Normal",
                line=dict(color="red", width=2),
            )
        )
        fig_hist.update_layout(
            title=f"{dist_stock} Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with qq_tab:
        (osm, osr), (slope, intercept, _) = stats.probplot(r_series, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                name="Sample Quantiles",
                marker=dict(size=4, color="mediumpurple"),
            )
        )
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        fig_qq.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Normal Reference",
                line=dict(color="red", dash="dash"),
            )
        )
        fig_qq.update_layout(
            title=f"{dist_stock} Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    jb_stat, jb_pvalue = stats.jarque_bera(r_series)
    verdict = (
        "Fail to reject normality (p ≥ 0.05)"
        if jb_pvalue >= 0.05
        else "Rejects normality (p < 0.05)"
    )
    st.caption(
        f"**Jarque-Bera test for {dist_stock}:** statistic = {jb_stat:.2f}, "
        f"p-value = {jb_pvalue:.4f} — {verdict}"
    )

    st.divider()

    st.subheader("Daily Return Distributions — All Stocks")

    box_data = returns[valid_tickers].melt(var_name="Stock", value_name="Daily Return")
    fig_box = px.box(
        box_data, x="Stock", y="Daily Return", color="Stock", template="plotly_white"
    )
    fig_box.update_layout(height=400, showlegend=False, yaxis_tickformat=".1%")
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 3 — Correlation & Diversification  (§2.4)
# =========================================================
with tab3:
    # -- Correlation heatmap ------------------------------
    st.subheader("Pairwise Correlation Matrix")

    corr_matrix = returns[valid_tickers].corr()

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar_title="Correlation",
        )
    )
    fig_heat.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # -- Scatter plot of two stocks -----------------------
    st.subheader("Return Scatter Plot")

    col_s1, col_s2 = st.columns(2)
    scatter_a = col_s1.selectbox("Stock A", valid_tickers, index=0, key="scatter_a")
    scatter_b = col_s2.selectbox(
        "Stock B",
        valid_tickers,
        index=min(1, len(valid_tickers) - 1),
        key="scatter_b",
    )

    if scatter_a != scatter_b:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=returns[scatter_a],
                y=returns[scatter_b],
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="steelblue"),
                name="Daily Returns",
            )
        )
        fig_scatter.update_layout(
            xaxis_title=f"{scatter_a} Daily Return",
            yaxis_title=f"{scatter_b} Daily Return",
            template="plotly_white",
            height=400,
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Select two different stocks for the scatter plot.")

    st.divider()

    # -- Rolling correlation ------------------------------
    st.subheader("Rolling Correlation")

    col_r1, col_r2, col_r3 = st.columns(3)
    roll_a = col_r1.selectbox("Stock A", valid_tickers, index=0, key="roll_a")
    roll_b = col_r2.selectbox(
        "Stock B",
        valid_tickers,
        index=min(1, len(valid_tickers) - 1),
        key="roll_b",
    )
    roll_win = col_r3.slider("Window (days)", 20, 120, 60, step=5, key="roll_corr_win")

    if roll_a != roll_b:
        rolling_corr = returns[roll_a].rolling(window=roll_win).corr(returns[roll_b])
        fig_rcorr = go.Figure()
        fig_rcorr.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode="lines",
                name=f"{roll_a} / {roll_b}",
                line=dict(color="darkorange"),
            )
        )
        fig_rcorr.update_layout(
            yaxis_title="Correlation",
            xaxis_title="Date",
            yaxis_range=[-1, 1],
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_rcorr, use_container_width=True)
    else:
        st.info("Select two different stocks.")

    st.divider()

    # -- Two-asset portfolio explorer ---------------------
    st.subheader("Two-Asset Portfolio Explorer")

    st.markdown(
        "> **What this shows:** When two stocks are not perfectly correlated "
        "(ρ < 1), combining them in a portfolio can produce **lower volatility "
        "than either stock alone**. The curve below plots portfolio volatility "
        "against the weight on Stock A. The characteristic dip below both "
        "individual volatilities is the **diversification benefit** — and it "
        "grows stronger as correlation decreases."
    )

    col_p1, col_p2 = st.columns(2)
    port_a = col_p1.selectbox("Stock A", valid_tickers, index=0, key="port_a")
    port_b = col_p2.selectbox(
        "Stock B",
        valid_tickers,
        index=min(1, len(valid_tickers) - 1),
        key="port_b",
    )

    if port_a != port_b:
        weight_a = (
            st.slider(f"Weight on {port_a} (%)", 0, 100, 50, step=1, key="port_w")
            / 100.0
        )

        # Annualized metrics
        ann_ret_a = returns[port_a].mean() * 252
        ann_ret_b = returns[port_b].mean() * 252
        cov_matrix = returns[[port_a, port_b]].cov() * 252
        var_a = cov_matrix.loc[port_a, port_a]
        var_b = cov_matrix.loc[port_b, port_b]
        cov_ab = cov_matrix.loc[port_a, port_b]
        corr_ab = returns[port_a].corr(returns[port_b])

        # Current portfolio stats
        port_ret = weight_a * ann_ret_a + (1 - weight_a) * ann_ret_b
        port_var = (
            weight_a**2 * var_a
            + (1 - weight_a) ** 2 * var_b
            + 2 * weight_a * (1 - weight_a) * cov_ab
        )
        port_vol = math.sqrt(port_var)

        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio Ann. Return", f"{port_ret:.2%}")
        m2.metric("Portfolio Ann. Volatility", f"{port_vol:.2%}")
        m3.metric(f"Correlation ({port_a}, {port_b})", f"{corr_ab:.3f}")

        # Plot volatility curve across all weights
        weights = np.linspace(0, 1, 101)
        vols = np.sqrt(
            weights**2 * var_a
            + (1 - weights) ** 2 * var_b
            + 2 * weights * (1 - weights) * cov_ab
        )

        fig_port = go.Figure()
        fig_port.add_trace(
            go.Scatter(
                x=weights * 100,
                y=vols,
                mode="lines",
                name="Portfolio Volatility",
                line=dict(color="teal", width=2),
            )
        )
        fig_port.add_trace(
            go.Scatter(
                x=[weight_a * 100],
                y=[port_vol],
                mode="markers",
                name="Current Weight",
                marker=dict(size=12, color="crimson", symbol="diamond"),
            )
        )
        fig_port.add_trace(
            go.Scatter(
                x=[100, 0],
                y=[math.sqrt(var_a), math.sqrt(var_b)],
                mode="markers+text",
                text=[port_a, port_b],
                textposition="top center",
                marker=dict(size=9, color="gray"),
                name="Individual Stocks",
            )
        )
        fig_port.update_layout(
            xaxis_title=f"Weight on {port_a} (%)",
            yaxis_title="Annualized Volatility",
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_port, use_container_width=True)
    else:
        st.info("Select two different stocks for the portfolio explorer.")