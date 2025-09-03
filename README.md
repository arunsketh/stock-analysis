# README.md

# 📈 Stock Analysis & Portfolio Tracker

A Streamlit web application that fetches stock data from [StockAnalysis.com](https://stockanalysis.com/) to compare analyst target prices against current prices.

## Features

-   **Analyst Target Comparison**: Enter multiple stock tickers to see their potential upside.
-   **Persistent Portfolio**: Add your stock purchases to a simple portfolio tracker. Your data is automatically saved to a local `portfolio.csv` file and reloaded each time you open the app.
-   **Dynamic Data**: Fetches real-time stock data upon request.

## 🚀 How to Run Locally

1.  **Clone the repository and navigate into the directory.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
The first time you add a stock, the app will create a `portfolio.csv` file in the same folder to store your data.
