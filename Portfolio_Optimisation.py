import streamlit as st
import pandas as pd
import os
from src.pyopt import PyOpt


# set os environment for backwards compatibility

os.environ["IS_STREAMLIT"] = "true"
opt = PyOpt()

st.set_page_config(
    page_title="Portfolio Optimisation",
    page_icon="📈",
    layout="wide",
)

st.title("Portfolio Optimisation")


st.write(
    """

This app optimises a portfolio of stocks and crytos based on the Sharpe Ratio and Volatility.

Input your stocks and crypto names and the app will return the optimal portfolio weights.

"""
)

opt.period = st.selectbox(
    label="Select the data to look back on",
    options=["6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    index=1,
    help="As the granularity is set to daily, the minimum period is 6 months."
    + "\nThis may be changed in the future.",
)

opt.RFR = st.number_input(
    label="Risk Free Rate", value=2.0, help="Risk Free Rate in percent", format="%f"
)

crypto = st.text_input(
    label="Enter your crypto tickers separated by a comma (,)." " Non case sensitive.",
    placeholder="btc, eth, xrp, bnb",
    help="""By default, add tickers use the -USD pair.
                       So there's no need to add -USD to the ticker.
                       This also means crypto pairs are not supported.
                       """,
)

crypto_split = [c.strip() for c in crypto.split(",")]
opt.add_crypto(crypto_split)

stocks = st.text_input(
    label="Enter your stock tickers separated by a comma (,)" " Non case sensitive.",
    placeholder="voo, vwra.l, qqq, gbug, spbo",
    help="You might need to google for the ticker name.",
)
stocks_split = [s.strip() for s in stocks.split(",")]
opt.add_stocks(stocks_split)

if stocks or crypto:
    crypto_split = [c.strip() for c in crypto.split(",")]
    opt.add_crypto(crypto_split)
    stocks_split = [s.strip() for s in stocks.split(",")]
    opt.add_stocks(stocks_split)

    opt.maxweights = st.slider(
        label="Max weightage of each asset",
        min_value=100 / len(stocks_split + crypto_split),
        max_value=100.0,
        value=100.0,
        step=1.0,
        help="Max allocation per asset in percent",
    )
    if st.button("Optimise"):
        col1, col2 = st.columns(2)
        with col1:
            outputs = opt.run(method="fmin", minimize="sharpe")
            st.write("## OPTIMIZED SHARPE RATIO")
            st.write(f"Returns:\t {outputs['metrics'][0] * 100:.3f}%")
            st.write(f"Volatility:\t {outputs['metrics'][1]:.3f}")
            st.write(f"Sharpe Ratio:\t {outputs['metrics'][2]:.3f}")
            st.subheader("Weights:")
            st.dataframe(outputs["result"])
        with col2:
            outputs = opt.run(method="fmin", minimize="vol")
            st.write("## OPTIMIZED VOLATILITY")
            st.write(f"Returns:\t {outputs['metrics'][0] * 100:.3f}%")
            st.write(f"Volatility:\t {outputs['metrics'][1]:.3f}")
            st.write(f"Sharpe Ratio:\t {outputs['metrics'][2]:.3f}")
            st.subheader("Weights:")
            st.dataframe(outputs["result"])

else:
    st.write("Looks like your portfolio is still empty. Add some stocks or crypto!")

with st.expander("Issues"):
    st.write(
        """
    It doesn't seem to handle negative sharpe ratios well yet.
    """
    )
