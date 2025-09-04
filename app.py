# app.py

import streamlit as st
import pandas as pd
import yfinance as yf

# --- Currency Symbol Mapping ---
CURRENCY_SYMBOLS = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
    'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥'
}

# --- yfinance Data Fetching ---
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_stock_data(symbol):
    """Fetches stock data including currency and other metrics using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # Use a longer period to ensure data is fetched for less liquid stocks
        history = ticker.history(period="5d")

        if history.empty:
            st.warning(f"Could not get price history for {symbol}. It might be delisted or an invalid ticker.")
            return None, None, None, None, None, None

        current_price = history['Close'].iloc[-1]
        target_price = info.get('targetMeanPrice')
        eps = info.get('trailingEps')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        currency = info.get('currency', 'USD') # Default to USD if not found

        return current_price, target_price, eps, market_cap, pe_ratio, currency

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None, None, None, None, None

# --- Main Analysis Function ---
def perform_analysis(symbols):
    """Takes a list of symbols and returns a DataFrame with analysis and rankings."""
    data = []

    progress_bar = st.progress(0, text="Fetching stock data...")
    for i, symbol in enumerate(symbols):
        current_price, target_price, eps, market_cap, pe_ratio, currency = get_stock_data(symbol)
        if current_price is not None:
            upside = ((target_price - current_price) / current_price) if target_price and current_price != 0 else 0
            data.append({
                "Stock Symbol": symbol,
                "Current Price": current_price,
                "Analyst Target": target_price,
                "Upside": upside,
                "EPS": eps,
                "Market Cap": market_cap,
                "P/E Ratio": pe_ratio,
                "Currency": currency
            })
        progress_bar.progress((i + 1) / len(symbols), text=f"Fetching data for {symbol}...")

    progress_bar.empty()  # Clear the progress bar

    if not data:
        return None

    df = pd.DataFrame(data)

    # --- Create Rankings ---
    df['Upside Rank'] = df['Upside'].rank(ascending=False, method='min', na_option='bottom')
    df['EPS Rank'] = df['EPS'].rank(ascending=False, method='min', na_option='bottom')
    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, method='min', na_option='bottom')

    df['Rank Product'] = df['Upside Rank'] * df['EPS Rank'] * df['P/E Rank']
    df['Overall Rank'] = df['Rank Product'].rank(ascending=True, method='min')

    df = df.sort_values(by='Overall Rank').reset_index(drop=True)

    return df

# --- UI Function to Display Styled Table ---
def display_styled_table(df):
    """Takes a DataFrame, formats it with currency symbols, and displays it."""
    
    # --- Pre-format currency columns before styling ---
    
    # Helper to format any price column with the correct currency symbol
    def format_price(row, col_name):
        price = row[col_name]
        currency_code = row.get('Currency', 'USD')
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        if pd.isnull(price): return 'N/A'
        return f'{symbol}{price:,.2f}'

    # Helper to format Market Cap with currency symbol and size (M, B, T)
    def format_market_cap(row):
        cap = row['Market Cap']
        currency_code = row.get('Currency', 'USD')
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        if pd.isnull(cap): return 'N/A'
        cap = float(cap)
        if cap >= 1e12: return f'{symbol}{cap/1e12:.2f}T'
        if cap >= 1e9: return f'{symbol}{cap/1e9:.2f}B'
        if cap >= 1e6: return f'{symbol}{cap/1e6:.2f}M'
        return f'{symbol}{cap:,.2f}'

    df_display = df.copy()
    df_display['Market Cap'] = df_display.apply(format_market_cap, axis=1)
    df_display['Current Price'] = df_display.apply(lambda row: format_price(row, 'Current Price'), axis=1)
    df_display['Analyst Target'] = df_display.apply(lambda row: format_price(row, 'Analyst Target'), axis=1)
    
    # --- Define styling for the remaining numeric columns ---
    
    column_order = [
        "Stock Symbol", "Market Cap", "Current Price", "Analyst Target",
        "Upside", "EPS", "P/E Ratio", "Overall Rank"
    ]

    format_rules = {
        'Overall Rank': '{:.0f}',
        'Upside': '{:,.2%}',
        'EPS': lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A',
        'P/E Ratio': lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A'
    }

    styler = df_display[column_order].style.format(format_rules)

    # Apply color gradients to the original numeric data, not the formatted strings
    styler.background_gradient(cmap='RdYlGn', subset=['Upside'], data=df)
    styler.background_gradient(cmap='RdYlGn', subset=['EPS'], data=df)
    styler.background_gradient(cmap='RdYlGn_r', subset=['P/E Ratio'], data=df)

    # Calculate dynamic height
    table_height = (len(df.index) + 1) * 35 + 3
    st.dataframe(styler, use_container_width=True, height=table_height)


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")

    st.header("Financial Ranking & Analysis")

    default_stocks = "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META, M&M.NS, NATIONALUM.NS, ZYDUSLIFE.BO, ITC.NS, CAMS.NS, TESCO.L"

    symbols_input = st.text_input(
        "Enter stock symbols to analyze (comma-separated):",
        value=default_stocks
    )

    symbols_to_analyze = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if symbols_to_analyze:
        analysis_df = perform_analysis(symbols_to_analyze)

        if analysis_df is not None and not analysis_df.empty:
            # --- Split DataFrame into US and Non-US stocks ---
            us_stocks_df = analysis_df[~analysis_df['Stock Symbol'].str.contains(r'\.')].copy()
            non_us_stocks_df = analysis_df[analysis_df['Stock Symbol'].str.contains(r'\.')].copy()

            if not us_stocks_df.empty:
                st.subheader("🇺🇸 US Stocks")
                display_styled_table(us_stocks_df)

            if not non_us_stocks_df.empty:
                st.subheader("🌍 International Stocks")
                display_styled_table(non_us_stocks_df)
        else:
            st.info("No valid data could be retrieved for the entered symbols.")
    else:
        st.warning("Please enter at least one stock symbol to analyze.")

    with st.expander("🎓 Learn About the Metrics"):
        st.markdown(r"""
        ### Analyst Upside Potential
        - **What it is:** The percentage difference between the current stock price and the average target price set by market analysts. A higher percentage suggests more potential growth in the eyes of analysts.
        - **Formula:** $ \text{Upside} = \frac{\text{Analyst Target Price} - \text{Current Price}}{\text{Current Price}} $

        ### EPS (Earnings Per Share)
        - **What it is:** A company's profit allocated to each outstanding share of common stock. It is a primary indicator of a company's profitability.
        - **Formula:** $ \text{EPS} = \frac{\text{Net Income} - \text{Preferred Dividends}}{\text{Average Outstanding Shares}} $

        ### P/E (Price-to-Earnings) Ratio
        - **What it is:** The ratio of a company's stock price to its earnings per share. It shows what the market is willing to pay today for a stock based on its past or future earnings.
        - **Significance:** A **low P/E** can indicate that a stock is undervalued. A **high P/E** can mean the stock is overvalued or that investors expect high future growth.
        - **Formula:** $ \text{P/E Ratio} = \frac{\text{Market Value per Share}}{\text{Earnings per Share (EPS)}} $
        """)

if __name__ == "__main__":
    main()
