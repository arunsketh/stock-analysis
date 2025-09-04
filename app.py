import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- CSV Data Management for Portfolio ---
CSV_FILE = 'portfolio.csv'

def load_portfolio():
    """Loads the portfolio from a CSV file. If it doesn't exist, creates an empty one."""
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Symbol"])
    else:
        df = pd.DataFrame(columns=["Symbol"])
        df.to_csv(CSV_FILE, index=False)
        return df

def save_portfolio(df):
    """Saves the entire portfolio DataFrame to the CSV file, overwriting it."""
    df.to_csv(CSV_FILE, index=False)

# --- yfinance Data Fetching ---
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1d")
        if history.empty:
            st.error(f"Could not get price history for {symbol}. It might be an invalid ticker.")
            return None, None, None, None, None
        current_price = history['Close'].iloc[-1]
        target_price = info.get('targetMeanPrice')
        eps = info.get('trailingEps')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        return current_price, target_price, eps, market_cap, pe_ratio
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None, None, None, None

# --- Main Analysis Function ---
def perform_analysis(symbols):
    data = []
    for symbol in symbols:
        current_price, target_price, eps, market_cap, pe_ratio = get_stock_data(symbol)
        if current_price is not None:
            upside = ((target_price - current_price) / current_price) if target_price and current_price != 0 else 0
            data.append({
                "Stock Symbol": symbol,
                "Current Price": current_price,
                "Analyst Target": target_price,
                "Upside": upside,
                "EPS": eps,
                "Market Cap": market_cap,
                "P/E Ratio": pe_ratio
            })

    if not data:
        return None

    df = pd.DataFrame(data)
    df['Upside Rank'] = df['Upside'].rank(ascending=False, method='min', na_option='bottom')
    df['EPS Rank'] = df['EPS'].rank(ascending=False, method='min', na_option='bottom')
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, method='min', na_option='bottom')
    df['Rank Product'] = df['Upside Rank'] * df['EPS Rank'] * df['P/E Rank']
    df['Overall Rank'] = df['Rank Product'].rank(ascending=True, method='min')
    df = df.sort_values(by='Overall Rank').reset_index(drop=True)
    return df

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    st.sidebar.header("My Portfolio")

    # Add stock form
    with st.sidebar.form("add_stock_form", clear_on_submit=True):
        new_symbol = st.text_input("Enter Stock Symbol", max_chars=10)
        if st.form_submit_button("Add to Portfolio") and new_symbol:
            symbol_upper = new_symbol.strip().upper()
            if symbol_upper not in st.session_state.portfolio["Symbol"].values:
                new_entry = pd.DataFrame({"Symbol": [symbol_upper]})
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry], ignore_index=True)
                save_portfolio(st.session_state.portfolio)
                st.sidebar.success(f"Added {symbol_upper}!")
                st.experimental_rerun()
            else:
                st.sidebar.warning(f"{symbol_upper} is already in the portfolio.")

    # Remove stock buttons with deferred rerun
    remove_symbol = None
    if not st.session_state.portfolio.empty:
        st.sidebar.subheader("Current Stocks")
        for index, row in st.session_state.portfolio.iterrows():
            symbol = row["Symbol"]
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(symbol)
            if col2.button("❌", key=f"remove_{symbol}", help=f"Remove {symbol}"):
                remove_symbol = symbol
                break

    if remove_symbol is not None:
        st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Symbol"] != remove_symbol]
        save_portfolio(st.session_state.portfolio)
        st.experimental_rerun()

    # Main analysis section
    st.header("Financial Ranking & Analysis")

    default_stocks = ",".join(st.session_state.portfolio["Symbol"].values) if not st.session_state.portfolio.empty else "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META"
    if 'symbols_input' not in st.session_state:
        st.session_state.symbols_input = default_stocks

    st.session_state.symbols_input = st.text_input(
        "Enter stock symbols to analyze:",
        value=st.session_state.symbols_input
    )

    if st.button("Get Stock Analysis"):
        symbols_to_analyze = [s.strip().upper() for s in st.session_state.symbols_input.split(",") if s.strip()]
        if symbols_to_analyze:
            analysis_df = perform_analysis(symbols_to_analyze)
            if analysis_df is not None:
                def format_market_cap(cap):
                    if pd.isnull(cap):
                        return 'N/A'
                    cap = float(cap)
                    return f'${cap/1e12:.2f}T' if cap >= 1e12 else f'${cap/1e9:.2f}B'

                column_order = [
                    "Stock Symbol", "Market Cap", "Current Price", "Analyst Target",
                    "Upside", "EPS", "P/E Ratio", "Overall Rank"
                ]

                format_rules = {
                    'Overall Rank': '{:.0f}',
                    'Current Price': '${:,.2f}',
                    'Analyst Target': lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A',
                    'Upside': '{:,.2%}',
                    'Market Cap': format_market_cap,
                    'EPS': lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A',
                    'P/E Ratio': lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A'
                }

                styler = analysis_df[column_order].style.format(format_rules)
                styler = styler.background_gradient(cmap='RdYlGn', subset=['Upside'])
                styler = styler.background_gradient(cmap='RdYlGn', subset=['EPS'])
                styler = styler.background_gradient(cmap='RdYlGn_r', subset=['P/E Ratio'])

                st.dataframe(styler, use_container_width=True)
            else:
                st.info("No data to display for the entered symbols.")
        else:
            st.warning("Please enter at least one stock symbol to analyze.")

    with st.expander("🎓 Learn About the Metrics"):
        st.markdown("""
        ### Analyst Upside Potential
        - **What it is:** The percentage difference between the current stock price and the average target price set by market analysts.
        - **Significance:** A high upside suggests that analysts believe the stock is currently undervalued and has significant room to grow.

        ### EPS (Earnings Per Share)
        - **What it is:** A company's profit divided by its outstanding common stock shares.
        - **Significance:** EPS is a key indicator of a company's profitability. A higher EPS generally indicates better financial health.

        ### P/E (Price-to-Earnings) Ratio
        - **What it is:** The ratio of a company's stock price to its earnings per share.
        - **Significance:** A **low P/E ratio** can indicate that a stock is undervalued. A **high P/E ratio** can mean the stock is overvalued or that investors expect high future growth. We rank lower P/E ratios as better.
        """)

if __name__ == "__main__":
    main()
