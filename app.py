# app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- CSV Data Management for Watchlist ---
CSV_FILE = 'watchlist.csv'

def load_watchlist():
    """Loads the watchlist from a CSV file. If it doesn't exist, creates an empty one."""
    if os.path.exists(CSV_FILE):
        # Handle empty file case
        try:
            return pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Symbol"])
    else:
        df = pd.DataFrame(columns=["Symbol"])
        df.to_csv(CSV_FILE, index=False)
        return df

def save_watchlist(df):
    """Saves the entire watchlist DataFrame to the CSV file, overwriting it."""
    df.to_csv(CSV_FILE, index=False)

# --- yfinance Data Fetching ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_stock_data(symbol):
    """Fetches stock data using the yfinance library."""
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1d")
        if history.empty:
            st.error(f"Could not get price history for {symbol}. It might be an invalid ticker.")
            return None, None
        current_price = history['Close'].iloc[-1]
        target_price = ticker.info.get('targetMeanPrice')
        
        if not target_price:
             st.warning(f"Analyst target price is not available for {symbol}.")
        return current_price, target_price
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")

    # Load watchlist from CSV into session_state
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # --- Sidebar for Watchlist Management ---
    st.sidebar.header("My Watchlist")

    # --- Form to add a new stock ---
    with st.sidebar.form("add_stock_form", clear_on_submit=True):
        new_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)")
        add_stock = st.form_submit_button("Add to Watchlist")

        if add_stock and new_symbol:
            symbol_upper = new_symbol.strip().upper()
            if symbol_upper not in st.session_state.watchlist["Symbol"].values:
                new_entry = pd.DataFrame({"Symbol": [symbol_upper]})
                st.session_state.watchlist = pd.concat([st.session_state.watchlist, new_entry], ignore_index=True)
                save_watchlist(st.session_state.watchlist)
                st.sidebar.success(f"Added {symbol_upper} to your watchlist!")
            else:
                st.sidebar.warning(f"{symbol_upper} is already in your watchlist.")
    
    # --- Display and Remove stocks from watchlist ---
    if not st.session_state.watchlist.empty:
        st.sidebar.subheader("Current Stocks")
        for index, row in st.session_state.watchlist.iterrows():
            symbol = row["Symbol"]
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(symbol)
            with col2:
                if st.button("❌", key=f"remove_{symbol}", help=f"Remove {symbol}"):
                    st.session_state.watchlist = st.session_state.watchlist[st.session_state.watchlist["Symbol"] != symbol]
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()
        
        if st.sidebar.button("Remove All Stocks"):
            st.session_state.watchlist = pd.DataFrame(columns=["Symbol"])
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    # --- Main Page for Stock Analysis ---
    st.header("Analyst Target Price Comparison")
    
    # Use watchlist for analysis if available
    default_stocks = ", ".join(st.session_state.watchlist["Symbol"].tolist()) if not st.session_state.watchlist.empty else "AAPL, MSFT, GOOGL, NVDA"
    stock_symbols_input = st.text_input("Enter stock symbols separated by commas:", default_stocks)
    symbols_to_analyze = [s.strip().upper() for s in stock_symbols_input.split(",") if s.strip()]

    if st.button("Get Stock Analysis"):
        if not symbols_to_analyze:
            st.warning("Please enter at least one stock symbol.")
        else:
            data = []
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols_to_analyze):
                current_price, target_price = get_stock_data(symbol)
                if current_price is not None:
                    upside = ((target_price - current_price) / current_price) if target_price and current_price != 0 else 0
                    data.append({
                        "Stock Symbol": symbol,
                        "Current Price": current_price,
                        "Analyst Target Price": target_price,
                        "Upside": upside
                    })
                progress_bar.progress((i + 1) / len(symbols_to_analyze))

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
