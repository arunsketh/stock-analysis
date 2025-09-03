# app.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
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

# --- Web Scraping ---
def get_stock_data(symbol):
    """
    Fetches the current price and analyst target price for a given stock symbol
    from stockanalysis.com.
    """
    url = f"https://stockanalysis.com/stocks/{symbol}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        current_price_element = soup.find('div', class_='text-4xl')
        current_price = float(current_price_element.text.strip().replace(',', '')) if current_price_element else None

        target_price_element = soup.find(lambda tag: '1y Price Target' in tag.text)
        target_price = None
        if target_price_element:
            parent = target_price_element.find_parent('div')
            price_element = parent.find('div', class_='text-2xl')
            if price_element:
                target_price = float(price_element.text.strip().replace('$', '').replace(',', ''))

        return current_price, target_price
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis & Portfolio Tracker")

    # Load portfolio from CSV into session_state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    # --- Sidebar for Portfolio Management ---
    st.sidebar.header("Your Portfolio")
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
            save_portfolio(st.session_state.portfolio) # Save on change
            st.sidebar.success(f"Added {new_shares} shares of {new_symbol.upper()}!")

    st.sidebar.subheader("Current Portfolio")
    if not st.session_state.portfolio.empty:
        st.sidebar.dataframe(st.session_state.portfolio)
        if st.sidebar.button("Clear Portfolio"):
            st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"])
            save_portfolio(st.session_state.portfolio) # Save on change
            st.sidebar.success("Portfolio cleared.")
            st.rerun()

    # --- Main Page for Stock Analysis ---
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
                if current_price and target_price:
                    upside = ((target_price - current_price) / current_price) if current_price != 0 else 0
                    data.append({
                        "Stock Symbol": symbol,
                        "Current Price": current_price,
                        "Analyst Target Price": target_price,
                        "Upside": upside
                    })
                else:
                    st.error(f"Could not retrieve data for {symbol}.")
                progress_bar.progress((i + 1) / len(symbols))

            if data:
                df = pd.DataFrame(data)
                df_sorted = df.sort_values(by="Upside", ascending=False).reset_index(drop=True)
                df_display = df_sorted.copy()
                df_display['Current Price'] = df_display['Current Price'].map('${:,.2f}'.format)
                df_display['Analyst Target Price'] = df_display['Analyst Target Price'].map('${:,.2f}'.format)
                df_display['Upside'] = df_display['Upside'].map('{:.2%}'.format)
                st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()
