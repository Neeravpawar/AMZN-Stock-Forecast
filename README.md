# Stock Market Analysis

Time series decomposition analysis of stock market data using Python.

## Features
- Seasonal decomposition of time series data
- Stationarity testing using Augmented Dickey-Fuller test
- Autocorrelation and Partial Autocorrelation analysis
- Visualization of decomposition components

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
- Close: Daily closing prices

## Usage
Run `python main.py` to:
- Generate time series decomposition plots
- Test for stationarity
- Analyze autocorrelation patterns

## Output
- Decomposition plots (trend, seasonal, residual components)
- Stationarity test results
- ACF and PACF plots with significance levels