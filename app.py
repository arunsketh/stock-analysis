# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional
from io import BytesIO

# --- Configuration & Constants ---

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")

# Define the name of the file containing the default stock tickers
DEFAULT_STOCKS_FILE = "stocks.txt"

# Dictionary for mapping currency codes to symbols for cleaner display
CURRENCY_SYMBOLS: Dict[str, str] = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
    'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥', 'GBp': 'p'
}

# --- Data Loading & Processing ---

def load_initial_tickers(filename: str) -> str:
    """
    Loads tickers from a file, with a fallback to a default list if the file is not found.
    Returns a comma-separated string of tickers.
    """
    try:
        with open(filename, 'r') as f:
            # Reads tickers, assuming one per line, and joins them into a single string
            tickers = [line.strip() for line in f if line.strip()]
            return ", ".join(tickers)
    except FileNotFoundError:
        st.warning(f"'{filename}' not found. Using a default list of stocks.")
        return "AAPL, MSFT, GOOGL, NVDA, TSLA, VOD.L"

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches key financial data for a single stock symbol from Yahoo Finance.
    Returns a dictionary of data or None if the ticker is invalid.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # Validate that essential data ('currentPrice') is present
        if not info or 'currentPrice' not in info:
            st.warning(f"Could not retrieve valid data for '{symbol}'. It may be an incorrect ticker or delisted.")
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

def process_all_stocks(symbols: List[str]) -> Optional[pd.DataFrame]:
    """
    Orchestrates fetching data for a list of symbols and calculates initial metrics.
    Displays a progress bar in the UI. Returns a DataFrame or None.
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
    # Calculate the 'Upside' potential based on analyst targets
    df['Upside'] = (df['Analyst Target'] - df['Current Price']) / df['Current Price']
    return df

# --- Ranking & Formatting ---

def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of stock data and adds ranking columns to it.
    A lower rank is better.
    """
    df['Upside Rank'] = df['Upside'].rank(ascending=False, na_option='bottom')
    df['EPS Rank'] = df['EPS'].rank(ascending=False, na_option='bottom')
    # For P/E Ratio, a lower value is generally better (undervalued)
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, na_option='bottom')
    # Calculate overall rank and sort by it
    df['Overall Rank'] = (df['Upside Rank'] + df['EPS Rank'] + df['P/E Rank']).rank(ascending=True, method='min')
    return df.sort_values(by='Overall Rank').reset_index(drop=True)

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all string formatting for currency, market cap, and percentages.
    This separates data transformation from styling.
    """
    display_df = df.copy()

    # Helper function to format a single row
    def format_row(row):
        currency = row.get('Currency', 'USD')
        
        # Format Market Cap (e.g., $1.23T, £45.6B)
        cap = row['Market Cap']
        cap_symbol = '£' if currency == 'GBp' else CURRENCY_SYMBOLS.get(currency, currency)
        if pd.notnull(cap):
            if cap >= 1e12: row['Market Cap'] = f"{cap_symbol}{cap/1e12:,.2f}T"
            elif cap >= 1e9: row['Market Cap'] = f"{cap_symbol}{cap/1e9:,.2f}B"
            elif cap >= 1e6: row['Market Cap'] = f"{cap_symbol}{cap/1e6:,.2f}M"
            else: row['Market Cap'] = f"{cap_symbol}{cap:,.2f}"
        
        # Format Prices (e.g., $150.25, 234.50p)
        price_symbol = CURRENCY_SYMBOLS.get(currency, currency)
        for col in ['Current Price', 'Analyst Target']:
            val = row[col]
            if pd.notnull(val):
                if currency == 'GBp': # Pence format (e.g., 150p)
                    row[col] = f"{val:,.2f}{price_symbol}"
                else: # Standard currency format (e.g., $150.00)
                    row[col] = f"{price_symbol}{val:,.2f}"
        return row

    return display_df.apply(format_row, axis=1)

# --- UI Components ---

def display_styled_table(df: pd.DataFrame):
    """
    Takes a pre-formatted DataFrame and applies visual styles for Streamlit display.
    """
    column_order = [
        "Overall Rank", "Stock Symbol", "Market Cap", "Current Price",
        "Analyst Target", "Upside", "EPS", "P/E Ratio"
    ]
    # Ensure all required columns are present, adding missing ones as None
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    formatted_df = format_dataframe_for_display(df)

    styler = formatted_df[column_order].style
    
    # Apply color gradients to key metric columns
    styler.background_gradient(cmap='RdYlGn', subset=['Upside', 'EPS'])
    styler.background_gradient(cmap='RdYlGn_r', subset=['P/E Ratio'])
    
    # Apply number formatting for percentages, ratios, etc.
    styler.format({
        'Overall Rank': '{:,.0f}',
        'Upside': '{:,.2%}',
        'EPS': '{:,.2f}',
        'P/E Ratio': '{:,.1f}x',
    }, na_rep="N/A")
    
    # Hide the index column for a cleaner look
    styler.hide()
    
    # Calculate dynamic height for the table
    table_height = (len(df.index) + 1) * 35 + 3
    
    st.dataframe(styler, use_container_width=True, height=table_height)

def render_metrics_explanation():
    """Displays an expander with explanations of the ranking metrics."""
    with st.expander("🎓 Learn About the Metrics Used for Ranking"):
        st.markdown(r"""
        The **Overall Rank** is determined by combining the ranks of the following three key metrics. A lower overall rank is better.
        
        ### 1. Analyst Upside Potential
        - **Why it matters:** A high upside suggests that analysts believe the stock is undervalued and has room to grow. A higher upside is ranked better.
        - **Formula:** $\text{Upside} = \frac{\text{Analyst Target Price} - \text{Current Price}}{\text{Current Price}}$
        
        ### 2. EPS (Earnings Per Share)
        - **Why it matters:** EPS is a core indicator of a company's profitability. A higher, positive EPS is a sign of good financial health and is ranked better.
        
        ### 3. P/E (Price-to-Earnings) Ratio
        - **Why it matters:** A **low P/E ratio** can indicate that a stock is undervalued compared to its earnings. In this analysis, a lower P/E ratio is ranked better.
        """)

# --- Excel Export ---

def dfs_to_excel_bytes(df_dict: Dict[str, pd.DataFrame]) -> bytes:
    """
    Takes a dictionary of DataFrames and writes them to separate sheets in an Excel file in memory.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if not df.empty:
                # Use the formatted version for the Excel export for consistency
                formatted_df_for_excel = format_dataframe_for_display(df)
                formatted_df_for_excel.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.title("📈 Stock Analysis Dashboard")
    
    # Load default tickers from the file
    default_tickers_str = load_initial_tickers(DEFAULT_STOCKS_FILE)
    
    symbols_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value=default_tickers_str,
        help="Use Yahoo Finance tickers (e.g., 'AAPL', 'TCS.NS', 'VOD.L')."
    )
    
    # Clean and capitalize the input symbols
    symbols_to_analyze = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    if not symbols_to_analyze:
        st.info("Please enter at least one stock ticker to begin analysis.")
        return

    unranked_df = process_all_stocks(symbols_to_analyze)
    
    if unranked_df is not None and not unranked_df.empty:
        # Separate stocks into US and International based on the presence of a '.'
        us_stocks_df = unranked_df[~unranked_df['Stock Symbol'].str.contains(r'\.')].copy()
        intl_stocks_df = unranked_df[unranked_df['Stock Symbol'].str.contains(r'\.')].copy()
        
        ranked_dfs = {}
        if not us_stocks_df.empty:
            ranked_dfs["US Stocks"] = rank_dataframe(us_stocks_df)
        if not intl_stocks_df.empty:
            ranked_dfs["International Stocks"] = rank_dataframe(intl_stocks_df)
            
        # Display tables for each group
        if "US Stocks" in ranked_dfs:
            st.subheader("🗽 US Stocks (Ranked)")
            display_styled_table(ranked_dfs["US Stocks"])
        if "International Stocks" in ranked_dfs:
            st.subheader("🌍 International Stocks (Ranked)")
            display_styled_table(ranked_dfs["International Stocks"])
            
        # Add a download button if there is data to download
        if ranked_dfs:
            st.markdown("---")
            excel_bytes = dfs_to_excel_bytes(ranked_dfs)
            st.download_button(
                label="📥 Download Analysis as Excel",
                data=excel_bytes,
                file_name="stock_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    render_metrics_explanation()

if __name__ == "__main__":
    main()
