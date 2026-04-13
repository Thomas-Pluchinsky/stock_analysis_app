# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math
import numpy as np
from scipy import stats

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Let the user pick a moving-average window
ma_window = st.sidebar.slider(
    "Moving Average Window (days)", min_value=5, max_value=200, value=50, step=5
)

# Risk-free rate for Sharpe ratio calculation
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1
) / 100

# Rolling volatility window
vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)", min_value=10, max_value=120, value=30, step=5
)

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily data from Yahoo Finance for a given date range."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

# -- Main logic -------------------------------------------
if ticker:
    try:
        df = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    if df.empty:
        st.error(
            f"No data found for **{ticker}**. "
            "Check the ticker symbol and try again."
        )
        st.stop()

    # Flatten any multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -- Compute a derived column -------------------------
    df["Daily Return"] = df["Close"].pct_change()
    df[f"{ma_window}-Day MA"] = df["Close"].rolling(window=ma_window).mean()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=vol_window).std() * math.sqrt(252)
    if ma_window > len(df):
        st.warning(
            f"The selected {ma_window}-day window is longer than the "
            f"available data ({len(df)} trading days). The moving average "
            "line won't appear — try a shorter window or a wider date range."
        )

    # -- Key metrics --------------------------------------
    latest_close = float(df["Close"].iloc[-1])
    total_return = float(df["Cumulative Return"].iloc[-1])
    avg_daily_ret = float(df["Daily Return"].mean())
    volatility = float(df["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)
    ann_return = avg_daily_ret * 252
    sharpe = (ann_return - risk_free_rate) / ann_volatility
    skewness = float(df["Daily Return"].skew())
    kurtosis = float(df["Daily Return"].kurtosis())
    max_close = float(df["Close"].max())
    min_close = float(df["Close"].min())

    st.subheader(f"{ticker} — Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${latest_close:,.2f}")
    col2.metric("Total Return", f"{total_return:.2%}")
    col3.metric("Annualized Return", f"{ann_return:.2%}")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Annualized Volatility (sigma)", f"{ann_volatility:.2%}")
    col6.metric("Skewness", f"{skewness:.2f}")
    col7.metric("Excess Kurtosis", f"{kurtosis:.2f}")
    col8.metric("Avg Daily Return", f"{avg_daily_ret:.4%}")

    col9, col10, _, _ = st.columns(4)
    col9.metric("Period High", f"${max_close:,.2f}")
    col10.metric("Period Low", f"${min_close:,.2f}")

    st.divider()

    # -- Price chart --------------------------------------
    st.subheader("Price & Moving Average")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Close Price",
            line=dict(width=1.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f"{ma_window}-Day MA"],
            mode="lines", name=f"{ma_window}-Day MA",
            line=dict(width=2, dash="dash")
        )
    )
    fig.update_layout(
        yaxis_title="Price (USD)", xaxis_title="Date",
        template="plotly_white", height=450
    )
    st.plotly_chart(fig, width="stretch")
    
    # -- Volume chart -------------------------------------
    st.subheader("Daily Trading Volume")

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color="steelblue", opacity=0.7)
    )
    fig_vol.update_layout(
        yaxis_title="Shares Traded", xaxis_title="Date",
        template="plotly_white", height=350
    )
    st.plotly_chart(fig_vol, width="stretch")

    # -- Daily returns distribution -----------------------
    st.subheader("Distribution of Daily Returns")

    returns_clean = df["Daily Return"].dropna()

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=returns_clean, nbinsx=60,
            marker_color="mediumpurple", opacity=0.75,
            name="Daily Returns", histnorm="probability density"
        )
    )

    # Overlay a fitted normal distribution curve
    x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
    mu = float(returns_clean.mean())
    sigma = float(returns_clean.std())
    fig_hist.add_trace(
        go.Scatter(
            x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
            mode="lines", name="Normal Distribution",
            line=dict(color="red", width=2)
        )
    )

    fig_hist.update_layout(
        xaxis_title="Daily Return", yaxis_title="Density",
        template="plotly_white", height=350
    )
    st.plotly_chart(fig_hist, width="stretch")

    # Display normality test results
    jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
    st.caption(
        f"**Jarque-Bera test:** statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f} — "
        f"{'Fail to reject normality (p > 0.05)' if jb_pvalue > 0.05 else 'Reject normality (p <= 0.05)'}"
    )
    
    # -- Cumulative return chart --------------------------
    st.subheader("Cumulative Return Over Time")

    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=df.index, y=df["Cumulative Return"],
            mode="lines", name="Cumulative Return",
            fill="tozeroy", line=dict(color="teal")
        )
    )
    fig_cum.update_layout(
        yaxis_title="Cumulative Return", yaxis_tickformat=".0%",
        xaxis_title="Date", template="plotly_white", height=400
    )
    st.plotly_chart(fig_cum, width="stretch")

    # -- Rolling volatility chart -------------------------
    st.subheader("Rolling Annualized Volatility")

    fig_roll_vol = go.Figure()
    fig_roll_vol.add_trace(
        go.Scatter(
            x=df.index, y=df["Rolling Volatility"],
            mode="lines", name=f"{vol_window}-Day Rolling Vol",
            line=dict(color="crimson", width=1.5)
        )
    )
    fig_roll_vol.update_layout(
        yaxis_title="Annualized Volatility", yaxis_tickformat=".0%",
        xaxis_title="Date", template="plotly_white", height=400
    )
    st.plotly_chart(fig_roll_vol, width="stretch")
    
    # -- Raw data (expandable) ----------------------------
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(60), width="stretch")



else:
    st.info("Enter a stock ticker in the sidebar to get started.")