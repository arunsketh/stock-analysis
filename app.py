import streamlit as st

import pandas as pd

import yfinance as yf

from typing import List, Dict, Any, Optional

from io import BytesIO



# --- Configuration & Constants ---



# Set wide layout and page title

st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")



# DEFAULT STOCKS PARAMETER: Edit this list to change the default stocks.

DEFAULT_STOCKS = "AAPL, MSFT, GOOGL, NVDA, PLTR, TSLA, META, M&M.NS, NATIONALUM.NS,ITC.NS, CAMS.NS, IEX.NS, VMM.NS,"



# Dictionary for mapping currency codes to symbols for cleaner display

CURRENCY_SYMBOLS: Dict[str, str] = {

    'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',

    'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥', 'GBp': 'p'

}



# --- Data Fetching & Processing ---



@st.cache_data(ttl=3600)  # Cache data for 1 hour

def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:

    """

    Fetches key financial data for a single stock symbol from Yahoo Finance.

    Returns a dictionary of data or None if the ticker is invalid.

    """

    try:

        ticker = yf.Ticker(symbol)

        info = ticker.info



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

    df['Upside'] = (df['Analyst Target'] - df['Current Price']) / df['Current Price']

    return df



# --- Ranking Logic ---



def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """

    Takes a DataFrame and adds ranking columns to it.

    """

    df['Upside Rank'] = df['Upside'].rank(ascending=False, na_option='bottom')

    df['EPS Rank'] = df['EPS'].rank(ascending=False, na_option='bottom')

    df['P/E Rank'] = df['P/E Ratio'].rank(ascending=True, na_option='bottom')

    df['Overall Rank'] = (df['Upside Rank'] + df['EPS Rank'] + df['P/E Rank']).rank(ascending=True, method='min')

    return df.sort_values(by='Overall Rank').reset_index(drop=True)



# --- Excel Export ---



def dfs_to_excel(df_dict: Dict[str, pd.DataFrame]) -> bytes:

    """

    Takes a dictionary of DataFrames and writes them to an Excel file in memory.

    Returns the Excel file as bytes.

    """

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        for sheet_name, df in df_dict.items():

            if not df.empty:

                df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()



# --- Display Logic ---



def display_styled_table(df: pd.DataFrame):

    """

    Applies styling and formatting to the DataFrame for presentation in Streamlit.

    """

    display_df = df.copy()



    column_widths = {

        "Overall Rank": 80, "Stock Symbol": 120, "Market Cap": 130, "Current Price": 130,

        "Analyst Target": 130, "Upside": 110, "EPS": 100, "P/E Ratio": 100,

    }



    def format_currency_columns(row):

        currency = row['Currency']

        

        # Handle Market Cap: Special case for GBp stocks, as market cap is in GBP (£)

        cap = row['Market Cap']

        cap_symbol = '£' if currency == 'GBp' else CURRENCY_SYMBOLS.get(currency, currency)

        if pd.notnull(cap):

            if cap >= 1e12: row['Market Cap'] = f"{cap_symbol}{cap/1e12:,.2f}T"

            elif cap >= 1e9: row['Market Cap'] = f"{cap_symbol}{cap/1e9:,.2f}B"

            elif cap >= 1e6: row['Market Cap'] = f"{cap_symbol}{cap/1e6:,.2f}M"

            else: row['Market Cap'] = f"{cap_symbol}{cap:,.2f}"

        else: row['Market Cap'] = "N/A"

        

        # Handle Prices: Use the currency symbol directly (e.g., 'p' for GBp)

        price_symbol = CURRENCY_SYMBOLS.get(currency, currency)

        price = row['Current Price']

        target = row['Analyst Target']



        if currency == 'GBp': # Format pence with symbol at the end

             row['Current Price'] = f"{price:,.2f}{price_symbol}" if pd.notnull(price) else "N/A"

             row['Analyst Target'] = f"{target:,.2f}{price_symbol}" if pd.notnull(target) else "N/A"

        else: # Standard formatting for all other currencies

            row['Current Price'] = f"{price_symbol}{price:,.2f}" if pd.notnull(price) else "N/A"

            row['Analyst Target'] = f"{price_symbol}{target:,.2f}" if pd.notnull(target) else "N/A"

            

        return row



    display_df = display_df.apply(format_currency_columns, axis=1)



    column_order = [

        "Overall Rank", "Stock Symbol", "Market Cap", "Current Price",

        "Analyst Target", "Upside", "EPS", "P/E Ratio"

    ]

    styler = display_df[column_order].style



    styler.background_gradient(cmap='RdYlGn', subset=['Upside', 'EPS'])

    styler.background_gradient(cmap='RdYlGn_r', subset=['P/E Ratio'])

    styler.format({

        'Overall Rank': '{:,.0f}', 'Upside': '{:,.2%}', 'EPS': '{:,.2f}', 'P/E Ratio': '{:,.1f}x',

    }, na_rep="N/A")



    # CORRECTED: Generate CSS for column widths and apply to the table

    # Added !important to force the styles to apply

    styles = []

    for col_name, width in column_widths.items():

        col_idx = display_df[column_order].columns.get_loc(col_name)

        # Apply style to both header (th) and data cells (td)

        props = [

            ('width', f'{width}px !important'),

            ('min-width', f'{width}px !important'),

            ('max-width', f'{width}px !important')

        ]

        styles.append({'selector': f'th.col_heading.level0.col{col_idx}', 'props': props})

        styles.append({'selector': f'td.col{col_idx}', 'props': props})



    styler.set_table_styles(styles, overwrite=False)



    # --- Final Touches ---

    styler.hide()

    table_height = (len(df.index) + 1) * 35 + 3

    st.dataframe(styler, use_container_width=True, height=table_height)



# --- Main Application ---



def main():

    """Main function to run the Streamlit app."""

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

            us_stocks_df = unranked_df[~unranked_df['Stock Symbol'].str.contains(r'\.')].copy()

            intl_stocks_df = unranked_df[unranked_df['Stock Symbol'].str.contains(r'\.')].copy()



            excel_dfs = {}

            if not us_stocks_df.empty:

                excel_dfs["US Stocks"] = rank_dataframe(us_stocks_df.copy())

            if not intl_stocks_df.empty:

                excel_dfs["International Stocks"] = rank_dataframe(intl_stocks_df.copy())

            

            # Display ranked tables on the page

            if "US Stocks" in excel_dfs:

                st.subheader("🗽 US Stocks (Ranked Separately)")

                display_styled_table(excel_dfs["US Stocks"])



            if "International Stocks" in excel_dfs:

                st.subheader("🌍 International Stocks (Ranked Separately)")

                display_styled_table(excel_dfs["International Stocks"])



            # Place download button at the bottom

            if excel_dfs:

                st.markdown("---") # Add a horizontal rule for separation

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

        - **Formula:** $ \text{Upside} = \frac{\text{Analyst Target Price} - \text{Current Price}}{\text{Current Price}} $



        ### 2. EPS (Earnings Per Share)

        - **Why it matters:** EPS is a core indicator of a company's profitability. A higher, positive EPS is a sign of good financial health and is ranked better.



        ### 3. P/E (Price-to-Earnings) Ratio

        - **Why it matters:** A **low P/E ratio** can indicate that a stock is undervalued compared to its earnings. In this analysis, a lower P/E ratio is ranked better.

        """)



if __name__ == "__main__":

    main()
