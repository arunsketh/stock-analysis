# app.py

import streamlit as st
import pandas as pd
import yfinance as yf # Replaces requests and BeautifulSoup
import os

# --- CSV Data Management ---
CSV_FILE = 'portfolio.csv'

def load_portfolio():
    """Loads the portfolio from a CSV file."""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"])

def save_portfolio(df):
    """Saves the portfolio to a CSV file."""
    df.to_csv(CSV_FILE, index=False)

# --- NEW: yfinance Data Fetching ---
@st.cache_data(ttl=600) # Cache data for 10 minutes to avoid repeated API calls
def get_stock_data(symbol):
    """
    Fetches the current price and analyst target price for a given stock symbol
    using the yfinance library.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current price
        history = ticker.history(period="1d")
        if history.empty:
            st.error(f"Could not get price history for {symbol}. It might be delisted or an invalid ticker.")
            return None, None
        current_price = history['Close'].iloc[-1]

        # Get analyst target price
        target_price = ticker.info.get('targetMeanPrice')
        
        # If target price is not available, mention it
        if not target_price:
             st.warning(f"Analyst target price is not available for {symbol}.")

        return current_price, target_price
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from yfinance: {e}")
        return None, None

# --- Streamlit App (UI code is mostly the same) ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis & Portfolio Tracker")

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    st.sidebar.header("Your Portfolio")
    # ... (Sidebar form and display code remains unchanged) ...
    with st.sidebar.form("add_stock_form", clear_on_submit=True):
        st.subheader("Add a New Stock Purchase")
        new_symbol = st.text_input("Stock Symbol (e.g., AAPL)")
        new_shares = st.number_input("Number of Shares", min_value=0.01, step=0.01, format="%.2f")
        new_price = st.number_input("Purchase Price ($)", min_value=0.01, step=0.01, format="%.2f")
        add_stock = st.form_submit_button("Add to Portfolio")

        if add_stock and new_symbol:
            new_entry = pd.DataFrame({
                "Symbol": [new_symbol.upper()], "Shares": [new_shares], "Purchase Price": [new_price]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry], ignore_index=True)
            save_portfolio(st.session_state.portfolio)
            st.sidebar.success(f"Added {new_shares} shares of {new_symbol.upper()}!")

    st.sidebar.subheader("Current Portfolio")
    if not st.session_state.portfolio.empty:
        st.sidebar.dataframe(st.session_state.portfolio)
        if st.sidebar.button("Clear Portfolio"):
            st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"])
            save_portfolio(st.session_state.portfolio)
            st.sidebar.success("Portfolio cleared.")
            st.rerun()
            
    st.header("Analyst Target Price Comparison")
    default_stocks = "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META"
    stock_symbols_input = st.text_input("Enter stock symbols separated by commas:", default_stocks)
    symbols = [s.strip().upper() for s in stock_symbols_input.split(",") if s.strip()]

    if st.button("Get Stock Analysis"):
        if not symbols:
            st.warning("Please enter at least one stock symbol.")
        else:
            data = []
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols):
                current_price, target_price = get_stock_data(symbol)
                
                # Only add to table if we got data
                if current_price is not None:
                    upside = ((target_price - current_price) / current_price) if target_price and current_price != 0 else 0
                    data.append({
                        "Stock Symbol": symbol,
                        "Current Price": current_price,
                        "Analyst Target Price": target_price,
                        "Upside": upside
                    })
                progress_bar.progress((i + 1) / len(symbols))

            if data:
                df = pd.DataFrame(data)
                df_sorted = df.sort_values(by="Upside", ascending=False).reset_index(drop=True)
                df_display = df_sorted.copy()
                df_display['Current Price'] = df_display['Current Price'].map('${:,.2f}'.format)
                df_display['Analyst Target Price'] = df_display['Analyst Target Price'].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A')
                df_display['Upside'] = df_display['Upside'].map(lambda x: f"{x:.2%}" if pd.notnull(x) and x != 0 else 'N/A')
                st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()
