import streamlit as st
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional

# --- Configuration & Constants ---

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")

stocks = "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META, M&M.NS, NATIONALUM.NS,ZYDUSLIFE.BO,ITC.NS, CAMS.NS "

# Dictionary for mapping currency codes to symbols for cleaner display
CURRENCY_SYMBOLS: Dict[str, str] = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
    'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥'
}

# --- Data Fetching & Processing ---

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches key financial data for a single stock symbol from Yahoo Finance.
    Returns a dictionary of data or None if the ticker is invalid.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Ensure essential data points exist, otherwise the ticker might be invalid
        if not info or 'currentPrice' not in info:
            st.warning(f"Could not retrieve valid data for {symbol}. It may be an incorrect ticker.")
            return None

        return {
            "Stock Symbol": symbol,
            "Current Price": info.get('currentPrice'),
            "Analyst Target": info.get('targetMeanPrice'),
            "EPS": info.get('trailingEps'),
            "Market Cap": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "Currency": info.get('currency', 'USD')
        }
    except Exception as e:
        st.error(f"An error occurred while fetching data for {symbol}: {e}")
        return None

def process_stocks(symbols: List[str]) -> Optional[pd.DataFrame]:
    """
    Orchestrates fetching data for a list of symbols and calculates base metrics.
    Returns an unranked DataFrame or None if no data is available.
    """
    data = []
    progress_bar = st.progress(0, text="Initializing data fetch...")

    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols), text=f"Fetching data for {symbol}...")
        stock_data = get_stock_data(symbol)
        if stock_data:
            data.append(stock_data)

    progress_bar.empty()

    if not data:
        st.error("No valid data could be retrieved for any of the entered symbols.")
        return None

    df = pd.DataFrame(data)
    # Calculate upside potential, which is needed before ranking
    df['Upside'] = (df['Analyst Target'] - df['Current Price']) / df['Current Price']
    return df

# --- Ranking Logic ---

def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and adds ranking columns to it.
    """
    # Create individual ranks (higher is better for Upside/EPS, lower is better for P/E)
    df['Upside Rank'] = df['Upside'].rank(ascending=False, na_option='bottom')
    df['EPS Rank'] = df['EPS'].rank(ascending=False, na_option='bottom')
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, na_option='bottom')

    # Calculate a composite rank score
    df['Overall Rank'] = (df['Upside Rank'] + df['EPS Rank'] + df['P/E Rank']).rank(ascending=True, method='min')

    return df.sort_values(by='Overall Rank').reset_index(drop=True)

# --- Display Logic ---

def display_styled_table(df: pd.DataFrame):
    """
    Applies styling and formatting to the DataFrame for presentation in Streamlit.
    """
    display_df = df.copy()

    # --- Define desired column widths in pixels ---
    column_widths = {
        "Overall Rank": 80,
        "Stock Symbol": 120,
        "Market Cap": 130,
        "Current Price": 130,
        "Analyst Target": 130,
        "Upside": 110,
        "EPS": 100,
        "P/E Ratio": 100,
    }

    # --- Pre-format columns that need row-specific context (like currency) ---
    def format_currency_columns(row):
        symbol = CURRENCY_SYMBOLS.get(row['Currency'], row['Currency'])
        cap = row['Market Cap']
        if pd.notnull(cap):
            if cap >= 1e12: row['Market Cap'] = f"{symbol}{cap/1e12:,.2f}T"
            elif cap >= 1e9: row['Market Cap'] = f"{symbol}{cap/1e9:,.2f}B"
            elif cap >= 1e6: row['Market Cap'] = f"{symbol}{cap/1e6:,.2f}M"
            else: row['Market Cap'] = f"{symbol}{cap:,.2f}"
        else:
            row['Market Cap'] = "N/A"

        price = row['Current Price']
        row['Current Price'] = f"{symbol}{price:,.2f}" if pd.notnull(price) else "N/A"
        target = row['Analyst Target']
        row['Analyst Target'] = f"{symbol}{target:,.2f}" if pd.notnull(target) else "N/A"
        return row

    display_df = display_df.apply(format_currency_columns, axis=1)

    # --- Define column order and create the Styler ---
    column_order = [
        "Overall Rank", "Stock Symbol", "Market Cap", "Current Price",
        "Analyst Target", "Upside", "EPS", "P/E Ratio"
    ]
    styler = display_df[column_order].style

    # --- Apply styles and formats ---
    styler.background_gradient(cmap='RdYlGn', subset=['Upside', 'EPS'])
    styler.background_gradient(cmap='RdYlGn_r', subset=['P/E Ratio'])

    styler.format({
        'Overall Rank': '{:,.0f}',
        'Upside': '{:,.2%}',
        'EPS': '{:,.2f}',
        'P/E Ratio': '{:,.1f}x',
    }, na_rep="N/A")

    # --- Generate CSS for column widths and apply to the table ---
    styles = []
    for col_name, width in column_widths.items():
        col_idx = display_df[column_order].columns.get_loc(col_name)
        styles.append({'selector': f'th.col_heading.level0.col{col_idx}', 'props': [('width', f'{width}px')]})

    styler.set_table_styles(styles)

    # --- Final Touches ---
    styler.hide()
    table_height = (len(df.index) + 1) * 35 + 3
    st.dataframe(styler, use_container_width=True, height=table_height)

# --- Main Application ---

def main():
    """Main function to run the Streamlit app."""
    st.title("📈 Stock Analysis Dashboard")
    st.markdown("Enter stock symbols to get a ranked analysis based on key financial metrics. Stocks are ranked separately for US and International groups.")

    default_stocks = stocks

    symbols_input = st.text_input(
        "Enter stock symbols (comma-separated):",
        value=default_stocks,
        help="Use Yahoo Finance tickers (e.g., 'AAPL' for Apple, 'TCS.NS' for TCS India)."
    )

    symbols_to_analyze = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if symbols_to_analyze:
        unranked_df = process_stocks(symbols_to_analyze)

        if unranked_df is not None and not unranked_df.empty:
            # Split stocks into US and International *before* ranking
            us_stocks_df = unranked_df[~unranked_df['Stock Symbol'].str.contains(r'\.')].copy()
            intl_stocks_df = unranked_df[unranked_df['Stock Symbol'].str.contains(r'\.')].copy()

            if not us_stocks_df.empty:
                # Rank the US stocks group
                ranked_us_df = rank_dataframe(us_stocks_df)
                st.subheader("🇺🇸 US Stocks (Ranked Separately)")
                display_styled_table(ranked_us_df)

            if not intl_stocks_df.empty:
                # Rank the International stocks group
                ranked_intl_df = rank_dataframe(intl_stocks_df)
                st.subheader("🌍 International Stocks (Ranked Separately)")
                display_styled_table(ranked_intl_df)
    else:
        st.info("Please enter at least one stock symbol to begin analysis.")

    # --- Explainer Section ---
    with st.expander("🎓 Learn About the Metrics Used for Ranking"):
        st.markdown(r"""
        The **Overall Rank** is determined by combining the ranks of the following three key metrics within each group (US or International). A lower overall rank is better.

        ### 1. Analyst Upside Potential
        - **What it is:** The potential percentage increase from the stock's **current price** to the **average target price** set by market analysts.
        - **Why it matters:** A high upside suggests that analysts believe the stock is undervalued and has room to grow. A higher upside is ranked better.
        - **Formula:** $ \text{Upside} = \frac{\text{Analyst Target Price} - \text{Current Price}}{\text{Current Price}} $

        ### 2. EPS (Earnings Per Share)
        - **What it is:** The portion of a company's profit allocated to each outstanding share of common stock.
        - **Why it matters:** EPS is a core indicator of a company's profitability. A higher, positive EPS is a sign of good financial health and is ranked better.

        ### 3. P/E (Price-to-Earnings) Ratio
        - **What it is:** The ratio of the company's stock price to its earnings per share. It shows how much investors are willing to pay for each dollar of earnings.
        - **Why it matters:** A **low P/E ratio** can indicate that a stock is undervalued compared to its earnings. In this analysis, a lower P/E ratio is ranked better.
        - **Formula:** $ \text{P/E Ratio} = \frac{\text{Share Price}}{\text{Earnings per Share (EPS)}} $
        """)

if __name__ == "__main__":
    main()
