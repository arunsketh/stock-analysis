import streamlit as st
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional
from io import BytesIO

st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")
DEFAULT_STOCKS = "AAPL,MSFT,GOOGL,NVDA,PLTR,TSLA,META,M&M.NS,NATIONALUM.NS,ITC.NS,CAMS.NS,IEX.NS,VMM.NS"

CURRENCY_SYMBOLS: Dict[str, str] = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
    'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥', 'GBp': 'p'
}

@st.cache_data(ttl=3600)
def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or 'currentPrice' not in info:
            return None
        name = info.get('longName', symbol)
        url = f"https://finance.yahoo.com/quote/{symbol}"
        display_name = f"<a href='{url}' target='_blank'>{name} - {symbol}</a>"
        return {
            "Stock Name-Symbol": display_name,
            "Current Price": info.get('currentPrice'),
            "Analyst Target": info.get('targetMeanPrice'),
            "EPS": info.get('trailingEps'),
            "Market Cap": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "Currency": info.get('currency', 'USD')
        }
    except Exception as e:
        return None

def process_stocks(symbols: List[str]) -> Optional[pd.DataFrame]:
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
    df['Upside'] = (df['Analyst Target'] - df['Current Price']) / df['Current Price']
    return df

def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['Upside Rank'] = df['Upside'].rank(ascending=False, na_option='bottom')
    df['EPS Rank'] = df['EPS'].rank(ascending=False, na_option='bottom')
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, na_option='bottom')
    df['Overall Rank'] = (df['Upside Rank'] + df['EPS Rank'] + df['P/E Rank']).rank(ascending=True, method='min')
    return df.sort_values(by='Overall Rank').reset_index(drop=True)

def dfs_to_excel(df_dict: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

def format_html_table(df: pd.DataFrame) -> str:
    df = df.copy()
    def format_currency(row):
        currency = row['Currency']
        cap = row['Market Cap']
        cap_symbol = '£' if currency == 'GBp' else CURRENCY_SYMBOLS.get(currency, currency)
        if pd.notnull(cap):
            if cap >= 1e12: row['Market Cap'] = f"{cap_symbol}{cap/1e12:,.2f}T"
            elif cap >= 1e9: row['Market Cap'] = f"{cap_symbol}{cap/1e9:,.2f}B"
            elif cap >= 1e6: row['Market Cap'] = f"{cap_symbol}{cap/1e6:,.2f}M"
            else: row['Market Cap'] = f"{cap_symbol}{cap:,.2f}"
        else: row['Market Cap'] = "N/A"
        price_symbol = CURRENCY_SYMBOLS.get(currency, currency)
        price = row['Current Price']
        target = row['Analyst Target']
        if currency == 'GBp':
             row['Current Price'] = f"{price:,.2f}{price_symbol}" if pd.notnull(price) else "N/A"
             row['Analyst Target'] = f"{target:,.2f}{price_symbol}" if pd.notnull(target) else "N/A"
        else:
            row['Current Price'] = f"{price_symbol}{price:,.2f}" if pd.notnull(price) else "N/A"
            row['Analyst Target'] = f"{price_symbol}{target:,.2f}" if pd.notnull(target) else "N/A"
        return row
    df = df.apply(format_currency, axis=1)

    df['Upside'] = df['Upside'].apply(lambda v: f"{v:.2%}" if pd.notnull(v) else "N/A")
    df['EPS'] = df['EPS'].apply(lambda v: f"{v:,.2f}" if pd.notnull(v) else "N/A")
    df['P/E Ratio'] = df['P/E Ratio'].apply(lambda v: f"{v:,.1f}x" if pd.notnull(v) else "N/A")
    df['Overall Rank'] = df['Overall Rank'].apply(lambda v: f"{int(v)}" if pd.notnull(v) else "N/A")

    show_cols = ["Overall Rank", "Stock Name-Symbol", "Market Cap", "Current Price", "Analyst Target", "Upside", "EPS", "P/E Ratio"]
    return df[show_cols].to_html(escape=False, index=False)

def main():
    st.title("📈 Stock Analysis Dashboard")

    symbols_input = st.text_input(
        "Enter stock symbols (comma-separated):",
        value=DEFAULT_STOCKS,
        help="Use Yahoo Finance tickers (e.g., 'AAPL' for Apple, 'TCS.NS' for TCS India, 'VOD.L' for Vodafone UK)."
    )
    symbols_to_analyze = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if symbols_to_analyze:
        unranked_df = process_stocks(symbols_to_analyze)
        if unranked_df is not None and not unranked_df.empty:
            us_stocks_df = unranked_df[~unranked_df['Stock Name-Symbol'].str.contains(r'\.')].copy()
            intl_stocks_df = unranked_df[unranked_df['Stock Name-Symbol'].str.contains(r'\.')].copy()
            excel_dfs = {}

            if not us_stocks_df.empty:
                us_ranked = rank_dataframe(us_stocks_df.copy())
                st.subheader("🗽 US Stocks (Ranked Separately)")
                st.markdown(format_html_table(us_ranked), unsafe_allow_html=True)
                excel_dfs["US Stocks"] = us_ranked

            if not intl_stocks_df.empty:
                intl_ranked = rank_dataframe(intl_stocks_df.copy())
                st.subheader("🌍 International Stocks (Ranked Separately)")
                st.markdown(format_html_table(intl_ranked), unsafe_allow_html=True)
                excel_dfs["International Stocks"] = intl_ranked

            if excel_dfs:
                st.markdown("---") 
                excel_bytes = dfs_to_excel(excel_dfs)
                st.download_button(
                    label="📥 Download Analysis as Excel",
                    data=excel_bytes,
                    file_name="stock_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("Please enter at least one stock symbol to begin analysis.")

    with st.expander("🎓 Learn About the Metrics Used for Ranking"):
        st.markdown(r"""
The **Overall Rank** is determined by combining the ranks of the following three key metrics within each group (US or International). A lower overall rank is better.

### 1. Analyst Upside Potential
- **Why it matters:** A high upside suggests that analysts believe the stock is undervalued and has room to grow. A higher upside is ranked better.
- **Formula:** $\text{Upside} = \frac{\text{Analyst Target Price} - \text{Current Price}}{\text{Current Price}}$

### 2. EPS (Earnings Per Share)
- **Why it matters:** EPS is a core indicator of a company's profitability. A higher, positive EPS is a sign of good financial health and is ranked better.

### 3. P/E (Price-to-Earnings) Ratio
- **Why it matters:** A **low P/E ratio** can indicate that a stock is undervalued compared to its earnings. In this analysis, a lower P/E ratio is ranked better.
        """)

if __name__ == "__main__":
    main()
