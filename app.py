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

# --- yfinance Data Fetching (Updated) ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_stock_data(symbol):
    """Fetches stock data including new metrics using the yfinance library."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1d")
        
        if history.empty:
            st.error(f"Could not get price history for {symbol}. It might be an invalid ticker.")
            return None, None, None, None, None
            
        # Standard data
        current_price = history['Close'].iloc[-1]
        target_price = info.get('targetMeanPrice')
        
        # New metrics for ranking
        eps = info.get('trailingEps')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        
        if not target_price:
            st.warning(f"Analyst target price is not available for {symbol}.")
            
        return current_price, target_price, eps, market_cap, pe_ratio
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None, None, None, None

# --- Main Analysis Function (Updated) ---
def perform_analysis(symbols):
    """Takes a list of symbols and returns a formatted DataFrame with analysis and rankings."""
    data = []
    st.header("Analyst Target Price & Financial Ranking")
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(symbols):
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
        progress_bar.progress((i + 1) / len(symbols))
    progress_bar.empty()

    if not data:
        return None
        
    df = pd.DataFrame(data)
    
    # --- Create Rankings ---
    # Rank by EPS (higher is better)
    df['EPS Rank'] = df['EPS'].rank(ascending=False, method='min', na_option='bottom')
    # Rank by Market Cap (higher is better)
    df['Market Cap Rank'] = df['Market Cap'].rank(ascending=False, method='min', na_option='bottom')
    # Rank by P/E Ratio (lower is better)
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, method='min', na_option='bottom')
    
    # Calculate Combined Rank (lower sum is better)
    df['Combined Score'] = df['EPS Rank'] + df['Market Cap Rank'] + df['P/E Rank']
    df['Overall Rank'] = df['Combined Score'].rank(ascending=True, method='min')
    
    # Sort by the final overall rank
    df = df.sort_values(by='Overall Rank')
    
    # --- Format for Display ---
    df_display = df.copy()
    
    # Helper function to format large numbers
    def format_market_cap(cap):
        if pd.isnull(cap): return 'N/A'
        cap = float(cap)
        if cap >= 1e12:
            return f'${cap/1e12:.2f}T'
        elif cap >= 1e9:
            return f'${cap/1e9:.2f}B'
        elif cap >= 1e6:
            return f'${cap/1e6:.2f}M'
        return f'${cap:,.0f}'

    df_display['Overall Rank'] = df_display['Overall Rank'].astype(int)
    df_display['Current Price'] = df_display['Current Price'].map('${:,.2f}'.format)
    df_display['Analyst Target'] = df_display['Analyst Target'].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A')
    df_display['Upside'] = df_display['Upside'].map(lambda x: f"{x:.2%}" if pd.notnull(x) and x != 0 else 'N/A')
    df_display['EPS'] = df_display['EPS'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A')
    df_display['Market Cap'] = df_display['Market Cap'].map(format_market_cap)
    df_display['P/E Ratio'] = df_display['P/E Ratio'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A')

    # Reorder columns for the final view
    final_columns = ['Overall Rank', 'Stock Symbol', 'Current Price', 'Analyst Target', 'Upside', 
                     'Market Cap', 'EPS', 'P/E Ratio']
    
    return df_display[final_columns]

# --- Streamlit App (Unchanged) ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    st.sidebar.header("My Watchlist")

    with st.sidebar.form("add_stock_form", clear_on_submit=True):
        new_symbol = st.text_input("Enter Stock Symbol")
        add_stock = st.form_submit_button("Add to Watchlist")
        if add_stock and new_symbol:
            symbol_upper = new_symbol.strip().upper()
            if symbol_upper not in st.session_state.watchlist["Symbol"].values:
                new_entry = pd.DataFrame({"Symbol": [symbol_upper]})
                st.session_state.watchlist = pd.concat([st.session_state.watchlist, new_entry], ignore_index=True)
                save_watchlist(st.session_state.watchlist)
                st.sidebar.success(f"Added {symbol_upper}!")
                st.rerun()
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
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        
        if st.sidebar.button("Remove All Stocks"):
            st.session_state.watchlist = pd.DataFrame(columns=["Symbol"])
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    default_stocks = "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META"
    
    if 'symbols_input' not in st.session_state:
        st.session_state.symbols_input = default_stocks

    st.session_state.symbols_input = st.text_input(
        "Enter stock symbols to analyze:",
        value=st.session_state.symbols_input
    )
    
    st.button("Get Stock Analysis")
    
    symbols_to_analyze = [s.strip().upper() for s in st.session_state.symbols_input.split(",") if s.strip()]

    if symbols_to_analyze:
        analysis_df = perform_analysis(symbols_to_analyze)
        if analysis_df is not None:
            st.dataframe(analysis_df, use_container_width=True)
        else:
            st.info("No data to display for the entered symbols.")
    else:
        st.warning("Please enter at least one stock symbol to analyze.")

if __name__ == "__main__":
    main()
