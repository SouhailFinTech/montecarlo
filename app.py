import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="MonteCarloCo - Stock Forecast", layout="wide")

# Header
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight:bold;
}
.small-font {
    font-size:16px;
    color:#888;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📈 MonteCarloCo – Stock Price Forecast</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Built with Python | hrich.souhail5@gmail.com</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/souhail78/MonteCarloCo/main/logo.png",  width=200)
    st.header("⚙️ Settings")
    ticker = st.text_input("Enter Ticker", "AAPL")
    days = st.slider("Days to Simulate", 10, 500, 252)
    simulations = st.slider("Number of Simulations", 10, 1000, 100)
    run_button = st.button("Run Simulation")

# Main App
if run_button:
    # Download data
    data = yf.download(ticker, period="5y")['Adj Close']
    log_returns = np.log(1 + data.pct_change().dropna())
    
    u = log_returns.mean()
    var = log_returns.var()
    stdev = log_returns.std()
    drift = u - 0.5 * var
    
    # Generate returns
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, simulations)))
    
    # Price list
    price_list = np.zeros_like(daily_returns)
    price_list[0] = data.iloc[-1]
    
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    final_prices = price_list[-1]
    mean_price = round(final_prices.mean(), 2)
    upper_bound = round(np.percentile(final_prices, 95), 2)
    lower_bound = round(np.percentile(final_prices, 5), 2)
    
    # Plotly Chart
    fig = go.Figure()

    for i in range(simulations):
        fig.add_trace(go.Scatter(x=np.arange(days), y=price_list[:, i],
                                 mode='lines',
                                 name=f"Path {i+1}",
                                 line=dict(width=1, color='navy'),
                                 opacity=0.4,
                                 showlegend=False))

    fig.add_trace(go.Scatter(x=np.arange(days), y=np.mean(price_list, axis=1),
                  mode='lines',
                  name='Mean Path',
                  line=dict(color='gold', width=3)))

    fig.update_layout(title=f"{ticker} – {simulations} Monte Carlo Paths",
                      xaxis_title="Day",
                      yaxis_title="Simulated Price ($)",
                      template="plotly_dark",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Price", f"${mean_price}")
    col2.metric("Upper Bound (95%)", f"${upper_bound}")
    col3.metric("Lower Bound (5%)", f"${lower_bound}")

    # Export Button
    df_export = pd.DataFrame(price_list)
    csv = df_export.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Simulated Paths (CSV)",
        data=csv,
        file_name=f"{ticker}_monte_carlo_simulation.csv",
        mime="text/csv"
    )

    # Summary Text
    st.markdown(f"""
    ### 📊 Summary

    Based on the last 5 years of historical performance, we simulated **{simulations} possible future paths** over **{days} trading days**.

    - **Current Price**: ${round(data.iloc[-1], 2)}
    - **Expected Price after {days} days**: ${mean_price}
    - **95% Confidence Interval**: [{lower_bound}, {upper_bound}]
    - **Upside Target (95%)**: {(upper_bound / data.iloc[-1] - 1) * 100:.2f}% gain
    - **Downside Risk (5%)**: {(data.iloc[-1] / lower_bound - 1) * 100 - 100:.2f}% loss
    """)

else:
    st.info("📌 Enter a ticker symbol and click 'Run Simulation' to get started.")