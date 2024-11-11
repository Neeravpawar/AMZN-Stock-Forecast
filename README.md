# Stock Market Analysis

Time series decomposition and technical analysis of stock market data using Python.

## Features
- Long-term market visualization (12-year period)
  * Candlestick charts with volume
  * 100, 200, and 500-day moving averages
- Two-year segment analysis
  * Six periods (2006-2017)
  * 10, 20, and 50-day moving averages
  * Individual candlestick charts
- Seasonal decomposition of time series data
  * Log transformation
  * Trend removal
  * Additive decomposition
- Stationarity testing using Augmented Dickey-Fuller test
- Autocorrelation and Partial Autocorrelation analysis
- Technical indicators (RSI, MFI, MACD, Bollinger Bands)

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with required variables:
   - DATA_FILE: path to your CSV data file
   - FIGURE_WIDTH: plot width (default: 12)
   - FIGURE_HEIGHT: plot height (default: 8)

## Input Data Format
The CSV file should contain:
- Date column: YYYY-MM-DD format
- OHLCV columns:
  * Open: Daily opening prices
  * High: Daily high prices
  * Low: Daily low prices
  * Close: Daily closing prices
  * Volume: Daily trading volume

## Usage
Run `python main.py` to:
1. Generate market visualization
   - Long-term analysis with major moving averages
   - Two-year segment analysis with short-term moving averages
2. Perform time series analysis
   - Log-additive decomposition
   - Stationarity testing
   - Autocorrelation analysis

## Output
- Market Analysis Plots
  * 12-year overview with long-term moving averages
  * Six 2-year period plots with short-term moving averages
- Time Series Analysis
  * Decomposition plots (trend, seasonal, residual)
  * Stationarity test results and statistics
  * ACF and PACF plots with confidence intervals