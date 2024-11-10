import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

def perform_seasonal_decomposition(df, period=252):
    """
    Perform seasonal decomposition on stock data using log transformation
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' price column and DateTimeIndex
        period (int): Number of periods for seasonal decomposition (default: 252 trading days)
    
    Returns:
        tuple: (decomposition object, log-transformed prices)
    """
    # Apply log transformation to closing prices
    log_prices = np.log(df['Close'])
    
    # Perform time series decomposition on log-transformed prices
    decomposition = seasonal_decompose(log_prices, 
                                     period=period,  
                                     model='additive')  # additive on log scale = multiplicative on original scale
    
    return decomposition, log_prices

def plot_decomposition(decomposition, log_prices, fig_width=12, fig_height=8):
    """
    Plot the seasonal decomposition components
    
    Args:
        decomposition: Seasonal decomposition object
        log_prices (pd.Series): Log-transformed price series
        fig_width (int): Figure width
        fig_height (int): Figure height
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(fig_width, fig_height*2))

    # Original (log scale)
    ax1.plot(log_prices.index, log_prices)
    ax1.set_title('Original Time Series (Log Scale)')

    # Trend
    ax2.plot(log_prices.index, decomposition.trend)
    ax2.set_title('Trend (Log Scale)')

    # Seasonal
    ax3.plot(log_prices.index, decomposition.seasonal)
    ax3.set_title('Seasonal Component (Log Scale)')

    # Residual
    ax4.plot(log_prices.index, decomposition.resid)
    ax4.set_title('Residual (Log Scale)')

    plt.tight_layout()
    plt.show()

def test_stationarity(data):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Args:
        data (pd.Series): Time series data to test
        
    Returns:
        tuple: (bool indicating if stationary, dict with test results)
    """
    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(data.dropna())
    
    # Organize results
    results = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4]
    }
    
    # Determine if stationary (using 5% significance level)
    is_stationary = adf_result[1] < 0.05
    
    # Print results
    print("\nStationarity Test Results for Non-Trend Components:")
    print(f"ADF Statistic: {results['ADF Statistic']:.4f}")
    print(f"p-value: {results['p-value']:.4f}")
    print("Critical Values:")
    for key, value in results['Critical Values'].items():
        print(f"\t{key}: {value:.4f}")
        
    # Print interpretation
    print("\nThe non-trend components are", 
          "stationary (reject null hypothesis)" if is_stationary 
          else "non-stationary (fail to reject null hypothesis)")
    
    return is_stationary, results

def analyze_autocorrelation(data, lags=260, figsize=(12, 6)):
    """
    Analyze and plot autocorrelation and partial autocorrelation of the data
    
    Args:
        data (pd.Series): Stationary time series data
        lags (int): Number of lags to calculate
        figsize (tuple): Figure size for plots
    
    Returns:
        tuple: (acf values, pacf values)
    """
    # Calculate ACF and PACF
    acf_values = acf(data.dropna(), nlags=lags)
    pacf_values = pacf(data.dropna(), nlags=lags)
    
    # Create confidence intervals (95%)
    confidence_interval = 1.96 / np.sqrt(len(data))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot ACF
    ax1.vlines(range(lags + 1), [0], acf_values, color='b', lw=2)
    ax1.axhline(y=0, color='k', linestyle='-')
    ax1.axhline(y=confidence_interval, color='r', linestyle='--')
    ax1.axhline(y=-confidence_interval, color='r', linestyle='--')
    ax1.set_title('Autocorrelation Function')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    
    # Plot PACF
    ax2.vlines(range(lags + 1), [0], pacf_values, color='b', lw=2)
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.axhline(y=confidence_interval, color='r', linestyle='--')
    ax2.axhline(y=-confidence_interval, color='r', linestyle='--')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')
    
    plt.tight_layout()
    plt.show()
    
    # Print significant lags
    print("\nSignificant autocorrelation lags (95% confidence):")
    significant_lags = np.where(np.abs(acf_values) > confidence_interval)[0]
    print(f"ACF: {significant_lags.tolist()}")
    
    significant_lags_pacf = np.where(np.abs(pacf_values) > confidence_interval)[0]
    print(f"PACF: {significant_lags_pacf.tolist()}")
    
    return acf_values, pacf_values

def main():
    """
    Main function to run the stock market analysis pipeline.
    
    Performs the following steps:
    1. Loads and validates environment variables
    2. Reads and preprocesses stock data
    3. Performs seasonal decomposition
    4. Tests for stationarity
    5. Analyzes autocorrelation if data is stationary
    
    Raises:
        ValueError: If environment variables or data format is invalid
        FileNotFoundError: If data file doesn't exist
        Exception: For other unexpected errors
    """
    try:
        # Load environment variables
        load_dotenv()

        # Get environment variables with validation
        data_file = os.getenv('DATA_FILE')
        if not data_file:
            raise ValueError("DATA_FILE environment variable is not set")
        
        fig_width = int(os.getenv('FIGURE_WIDTH', 12))
        fig_height = int(os.getenv('FIGURE_HEIGHT', 8))

        # Validate file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Read and prepare stock data
        df = pd.read_csv(data_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Perform decomposition
        decomposition, log_prices = perform_seasonal_decomposition(df)
        
        # Plot decomposition results
        plot_decomposition(decomposition, log_prices, fig_width, fig_height)

        # Combine seasonal and residual components
        non_trend_data = decomposition.seasonal + decomposition.resid
        
        # Test for stationarity
        is_stationary, stat_results = test_stationarity(non_trend_data)
        
        # If data is stationary, analyze autocorrelation
        if is_stationary:
            acf_values, pacf_values = analyze_autocorrelation(non_trend_data)
        else:
            print("\nSkipping autocorrelation analysis as data is non-stationary")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
