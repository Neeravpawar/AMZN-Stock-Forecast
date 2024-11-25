
# AMZN Stock Forecast

## Project Overview
This repository presents a comprehensive analysis and predictive modeling of Amazon (AMZN) stock prices, focusing on time series forecasting for swing trading applications. The project integrates data visualization, decomposition, autocorrelation analysis, and advanced time series modeling strategies.

## Objectives
- Analyze and visualize historical AMZN stock price data to uncover trends, seasonality, and patterns.
- Employ autocorrelation and partial autocorrelation analyses to identify temporal dependencies.
- Develop robust time series models, including RNN, LSTM, and Transformer architectures.
- Establish and test a predictive model suitable for swing trading applications.

## Data Overview
- **Structure**: Historical daily stock price and volume data in OHLCV (Open, High, Low, Close, Volume) format.
- **Continuity**: Daily measurements aligned with U.S. stock market trading days, excluding weekends and holidays, ensuring no missing data.
- **Visualization**:
  - Long-term market trends using candlestick charts with 100, 200, and 500-day moving averages.
  - Two-year segment analysis using weekly aggregated data with short-term moving averages (2-week, 4-week, 8-week).

## Methodology
### Data Preprocessing
- Log transformation and detrending of stock price data to stabilize variance.
- Seasonal decomposition into trend, seasonal, and residual components for clearer analysis.

### Stationarity and Autocorrelation
- **Stationarity Testing**: Augmented Dickey-Fuller test confirms stationarity of detrended data.
- **ACF and PACF Analysis**: Reveals short-term momentum, medium-term reversals, and long-term cyclical behavior.

### Modeling Strategy
1. **Rolling Window Approach**:
   - Training window: 504 trading days (~2 years).
   - Forecast horizon: 10-42 days (2 weeks to 2 months).
   - Step size: 21 trading days (~1 month).
2. **Data Partitioning**:
   - Training/Validation (70%), Buffer Zone (10%), and Test Data (20%) to prevent leakage and ensure robust evaluation.

## Baseline Model
- **AR(1) Model**:
  - Autoregressive model using the first lag.
  - Simple benchmark to compare advanced models.
  - Results:
    - AIC: -10670.980, BIC: -10647.822.
    - AR(1) coefficient: 0.9806 (p-value < 0.001).

## Planned Advanced Models
- LSTM and Transformer architectures to capture complex temporal dependencies.
- Feature engineering to include trends and seasonality for enhanced model accuracy.

## Usage
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Fl4yd/AMZN-Stock-Forecast.git
   cd AMZN-Stock-Forecast
   pip install -r requirements.txt
   ```
2. Prepare the dataset and ensure it follows the OHLCV format (Open, High, Low, Close, Volume).
3. Execute the scripts:
   - Use visualization scripts to analyze historical trends and moving averages.
   - Apply decomposition methods to identify trend, seasonality, and residuals.
   - Train predictive models for stock price forecasting using the specified frameworks.

## Key Features
- **Data Analysis**:
  - Long-term trend analysis using 100, 200, and 500-day moving averages.
  - Short-term segment analysis with 2-week, 4-week, and 8-week moving averages.
- **Time Series Decomposition**:
  - Log transformation for variance stabilization.
  - Seasonal decomposition to isolate key components for clearer insights.
- **Modeling**:
  - Rolling window implementation for robust training and validation.
  - Baseline AR(1) model for performance benchmarking.

## Contributors
- **Jere Arokivi** [@Fl4yd](https://github.com/Fl4yd)
- **Neerav Pawar**
- **Subin Khatiwada**

## Future Work
- Extend current models to include LSTM and Transformer architectures.
- Optimize hyperparameter tuning using cross-validation.
- Investigate ensemble learning techniques for enhanced accuracy.

---
