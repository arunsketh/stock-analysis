# app.py

import streamlit as st
import pandas as pd
import yfinance as yf

# --- URL-based Watchlist Management ---

def load_watchlist_from_url():
    """Loads the watchlist from the URL query parameters."""
    params = st.query_params.get("watchlist", "")
    symbols = params.split(",") if params else []
    return pd.DataFrame({"Symbol": symbols})

def save_watchlist_to_url(df):
    """Saves the watchlist to the URL query parameters."""
    symbols = df["Symbol"].tolist()
    st.query_params["watchlist"] = ",".join(symbols)

# --- yfinance Data Fetching (Unchanged) ---
@st.cache_data(ttl=600)
def get_stock_data(symbol):
    """Fetches stock data using the yfinance library."""
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1d")
        if history.empty: return None, None
        current_price = history['Close'].iloc[-1]
        target_price = ticker.info.get('targetMeanPrice')
        return current_price, target_price
    except Exception:
        return None, None

# --- Main Analysis Function (Unchanged) ---
def perform_analysis(symbols):
    """Takes a list of symbols and returns a formatted DataFrame with their analysis."""
    # ... (This function remains the same as the previous version) ...
    data = []
    st.header("Analyst Target Price Comparison")
    progress_bar = st.progress(0)
    for i, symbol in enumerate(symbols):
        current_price, target_price = get_stock_data(symbol)
        if current_price is not None:
            upside = ((target_price - current_price) / current_price) if target_price and current_price != 0 else 0
            data.append({"Stock Symbol": symbol, "Current Price": current_price, "Analyst Target Price": target_price, "Upside": upside})
        progress_bar.progress((i + 1) / len(symbols))
    progress_bar.empty()
    if data:
        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by="Upside", ascending=False).reset_index(drop=True)
        df_display = df_sorted.copy()
        df_display['Current Price'] = df_display['Current Price'].map('${:,.2f}'.format)
        df_display['Analyst Target Price'] = df_display['Analyst Target Price'].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A')
        df_display['Upside'] = df_display['Upside'].map(lambda x: f"{x:.2%}" if pd.notnull(x) and x != 0 else 'N/A')
        return df_display
    return None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")

    # Load watchlist from URL into session_state
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist_from_url()

    # --- Sidebar for Watchlist Management ---
    st.sidebar.header("My Watchlist")

    with st.sidebar.form("add_stock_form", clear_on_submit=True):
        new_symbol = st.text_input("Enter Stock Symbol")
        if st.form_submit_button("Add to Watchlist") and new_symbol:
            symbol_upper = new_symbol.strip().upper()
            if symbol_upper not in st.session_state.watchlist["Symbol"].values:
                new_entry = pd.DataFrame({"Symbol": [symbol_upper]})
                st.session_state.watchlist = pd.concat([st.session_state.watchlist, new_entry], ignore_index=True)
                save_watchlist_to_url(st.session_state.watchlist)
                st.sidebar.success(f"Added {symbol_upper}!")
            else:
                st.sidebar.warning(f"{symbol_upper} is already in the watchlist.")
    
    if not st.session_state.watchlist.empty:
        st.sidebar.subheader("Current Stocks")
        for index, row in st.session_state.watchlist.iterrows():
            symbol = row["Symbol"]
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(symbol)
            if col2.button("❌", key=f"remove_{symbol}", help=f"Remove {symbol}"):
                st.session_state.watchlist = st.session_state.watchlist[st.session_state.watchlist["Symbol"] != symbol]
                save_watchlist_to_url(st.session_state.watchlist)
                st.rerun()

    # --- Main Page for Stock Analysis ---
    # Default stocks if the URL is empty
    default_stocks_if_empty = "AAPL, MSFT, GOOGL"
    
    # Use watchlist symbols from URL for analysis
    watchlist_symbols = st.session_state.watchlist["Symbol"].tolist()
    symbols_to_analyze = watchlist_symbols if watchlist_symbols else default_stocks_if_empty.split(",")
    
    st.header("Analyst Target Price Comparison")
    analysis_df = perform_analysis(symbols_to_analyze)
    if analysis_df is not None:
        st.dataframe(analysis_df, use_container_width=True)

if __name__ == "__main__":
    main()
