# Import necessary libraries for data manipulation, visualization, and time series analysis
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import mplfinance as mpf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

def plot_long_term_analysis(df):
    """
    Create candlestick plot for the entire 12-year period with long-term moving averages
    using monthly aggregated data
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and DateTimeIndex
    """
    # Resample data to monthly timeframe
    monthly_df = pd.DataFrame()
    monthly_df['Open'] = df['Open'].resample('M').first()
    monthly_df['High'] = df['High'].resample('M').max()
    monthly_df['Low'] = df['Low'].resample('M').min()
    monthly_df['Close'] = df['Close'].resample('M').last()
    monthly_df['Volume'] = df['Volume'].resample('M').sum()
    
    # Calculate moving averages (scaled from daily to monthly)
    # 100 days ≈ 3 months, 200 days ≈ 6 months, 500 days ≈ 15 months
    monthly_df['MA100'] = monthly_df['Close'].rolling(window=3).mean()   # 100-day MA
    monthly_df['MA200'] = monthly_df['Close'].rolling(window=6).mean()   # 200-day MA
    monthly_df['MA500'] = monthly_df['Close'].rolling(window=15).mean()  # 500-day MA
    
    # Create custom style
    mc = mpf.make_marketcolors(up='g', down='r',
                              edge='inherit',
                              wick='inherit',
                              volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--')
    
    # Define colors for consistency
    ma_colors = {
        '100-day MA': 'blue',
        '200-day MA': 'orange',
        '500-day MA': 'purple'
    }
    
    # Add Moving Average overlays
    add_plots = [
        mpf.make_addplot(monthly_df['MA100'], color=ma_colors['100-day MA'], label='100-day MA'),
        mpf.make_addplot(monthly_df['MA200'], color=ma_colors['200-day MA'], label='200-day MA'),
        mpf.make_addplot(monthly_df['MA500'], color=ma_colors['500-day MA'], label='500-day MA')
    ]
    
    # Create the plot
    fig, axes = mpf.plot(monthly_df, 
                        type='candle',
                        style=s,
                        addplot=add_plots,
                        volume=True,
                        title='Long-term Market Analysis (2006-2018) - Monthly',
                        figsize=(15, 10),
                        panel_ratios=(6,2),
                        returnfig=True,
                        volume_panel=1)
    
    # Clear existing legend and add colored legend
    axes[0].legend_.remove()
    axes[0].legend([
        plt.Line2D([], [], color=ma_colors['100-day MA']),
        plt.Line2D([], [], color=ma_colors['200-day MA']),
        plt.Line2D([], [], color=ma_colors['500-day MA'])
    ], ['100-day MA', '200-day MA', '500-day MA'], 
    loc='upper left')
    plt.show()

def plot_two_year_segments(df):
    """
    Create separate plots for each 2-year segment with short-term moving averages
    using weekly aggregated data
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and DateTimeIndex
    """
    # Define time periods
    periods = [
        ('2006-2007', '2006-01-01', '2007-12-31'),
        ('2008-2009', '2008-01-01', '2009-12-31'),
        ('2010-2011', '2010-01-01', '2011-12-31'),
        ('2012-2013', '2012-01-01', '2013-12-31'),
        ('2014-2015', '2014-01-01', '2015-12-31'),
        ('2016-2017', '2016-01-01', '2017-12-31')
    ]
    
    # Create custom style
    mc = mpf.make_marketcolors(up='g', down='r',
                              edge='inherit',
                              wick='inherit',
                              volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--')
    
    # Define colors for consistency
    ma_colors = {
        '2-week MA': 'blue',
        '4-week MA': 'orange',
        '8-week MA': 'purple'
    }
    
    for idx, (period_name, start_date, end_date) in enumerate(periods):
        # Get data for the period and resample to weekly
        period_df = df[start_date:end_date].copy()
        weekly_df = pd.DataFrame()
        weekly_df['Open'] = period_df['Open'].resample('W').first()
        weekly_df['High'] = period_df['High'].resample('W').max()
        weekly_df['Low'] = period_df['Low'].resample('W').min()
        weekly_df['Close'] = period_df['Close'].resample('W').last()
        weekly_df['Volume'] = period_df['Volume'].resample('W').sum()
        
        # Calculate MAs for this period (2-week, 4-week, 8-week)
        weekly_df['MA10'] = weekly_df['Close'].rolling(window=2).mean()  # 10-day ≈ 2-week MA
        weekly_df['MA20'] = weekly_df['Close'].rolling(window=4).mean()  # 20-day ≈ 4-week MA
        weekly_df['MA50'] = weekly_df['Close'].rolling(window=8).mean()  # 50-day ≈ 8-week MA
        
        # Create addplot with MAs
        ap = [
            mpf.make_addplot(weekly_df['MA10'], color=ma_colors['2-week MA']),
            mpf.make_addplot(weekly_df['MA20'], color=ma_colors['4-week MA']),
            mpf.make_addplot(weekly_df['MA50'], color=ma_colors['8-week MA'])
        ]
        
        # Plot candlesticks and MAs
        fig, axes = mpf.plot(weekly_df,
                           type='candle',
                           style=s,
                           title=f'Market Analysis {period_name} - Weekly',
                           volume=True,
                           addplot=ap,
                           figratio=(3,2),
                           figscale=1.5,
                           panel_ratios=(6,2),
                           returnfig=True,
                           volume_panel=1)
        
        axes[0].legend([
            plt.Line2D([], [], color=ma_colors['2-week MA']),
            plt.Line2D([], [], color=ma_colors['4-week MA']),
            plt.Line2D([], [], color=ma_colors['8-week MA'])
        ], ['2-week MA', '4-week MA', '8-week MA'], 
        loc='upper left')
        plt.show()

def plot_decomposition(decomposition, log_prices, figsize=(12, 10)):
    """
    Plot time series decomposition components and seasonal cycle in separate figures
    
    Args:
        decomposition: Seasonal decomposition object
        log_prices: Original log-transformed prices
        figsize (tuple): Figure size for the plot
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    
    # Plot original data
    ax1.plot(log_prices)
    ax1.set_title('Original Log-Transformed Data')
    ax1.grid(True)
    
    # Plot trend component
    ax2.plot(decomposition.trend)
    ax2.set_title('Trend Component')
    ax2.grid(True)
    
    # Plot seasonal component
    ax3.plot(decomposition.seasonal)
    ax3.set_title('Seasonal Component')
    ax3.grid(True)
    
    # Plot residual component
    ax4.plot(decomposition.resid)
    ax4.set_title('Residual Component')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Second figure: One cycle of seasonal pattern
    plt.figure(figsize=(12, 6))
    seasonal_data = decomposition.seasonal
    cycle_length = 252  # One year of trading days
    one_cycle = seasonal_data[:cycle_length]
    
    plt.plot(range(len(one_cycle)), one_cycle)
    plt.title('One Cycle of Seasonal Pattern (1 Year)')
    plt.xlabel('Trading Days')
    plt.ylabel('Seasonal Effect')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def perform_seasonal_decomposition(df, period=252, plot=True):
    """
    Perform seasonal decomposition on stock data using log transformation
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC price columns and DateTimeIndex
        period (int): Number of periods for seasonal decomposition (default: 252 trading days)
        plot (bool): Whether to plot the decomposition components (default: True)
    
    Returns:
        tuple: (decomposition object, log_transformed prices, non_trend_df)
    """
    # Apply log transformation to closing prices
    log_prices = np.log(df['Close'])
    
    # Perform time series decomposition on log-transformed close prices
    decomposition = seasonal_decompose(log_prices, 
                                    period=period,  
                                    model='additive')
    
    # Calculate RMSE for residuals
    residuals = decomposition.resid
    
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"\nResidual Component RMSE: {rmse:.6f}")
    
    # Plot decomposition components if requested
    if plot:
        plot_decomposition(decomposition, log_prices)
    
    # Create non-trend dataframe
    price_columns = ['Open', 'High', 'Low', 'Close']
    non_trend_df = pd.DataFrame(index=df.index)
    
    # Remove trend from all price columns using close trend
    for col in price_columns:
        log_col = np.log(df[col])
        non_trend_df[col] = np.exp(log_col - decomposition.trend)
    
    # Add volume as is (no detrending needed)
    if 'Volume' in df.columns:
        non_trend_df['Volume'] = df['Volume']
    
    return decomposition, log_prices, non_trend_df

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
    
    print("\nThe series is", 
          "stationary" if is_stationary else "non-stationary")
    
    return is_stationary, results

def analyze_autocorrelation(data, lags=260, figsize=(12, 6)):
    """
    Analyze and plot autocorrelation and partial autocorrelation
    
    Args:
        data (pd.Series): Stationary time series data
        lags (int): Number of lags to calculate (default: 260 trading days)
        figsize (tuple): Figure size for plots (default: (12, 6))
    
    Returns:
        tuple: (acf values, pacf values)
    """
    # Calculate ACF and PACF
    acf_values = acf(data.dropna(), nlags=lags)
    pacf_values = pacf(data.dropna(), nlags=lags, method='ywm')
    
    # Create confidence intervals (95%)
    confidence_interval = 1.96 / np.sqrt(len(data))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot ACF
    ax1.vlines(range(lags + 1), [0], acf_values)
    ax1.axhline(y=0, color='k')
    ax1.axhline(y=confidence_interval, color='r', linestyle='--')
    ax1.axhline(y=-confidence_interval, color='r', linestyle='--')
    ax1.set_title('Autocorrelation Function')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    
    # Plot PACF
    ax2.vlines(range(lags + 1), [0], pacf_values)
    ax2.axhline(y=0, color='k')
    ax2.axhline(y=confidence_interval, color='r', linestyle='--')
    ax2.axhline(y=-confidence_interval, color='r', linestyle='--')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')
    
    plt.tight_layout()
    plt.show()
    
    return acf_values, pacf_values

def train_and_forecast_ar(df, train_size=0.8):
    """
    Train ARIMA model and make predictions on detrended data
    
    Args:
        df (pd.DataFrame): DataFrame with Close prices
        train_size (float): Proportion of data to use for training (default: 0.8)
    
    Returns:
        tuple: (predictions, train_data, test_data)
    """
    # Apply log transformation to remove exponential trend
    log_prices = np.log(df['Close'])
    
    # Calculate and remove trend
    X = np.arange(len(log_prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, log_prices)
    trend = model.predict(X)
    detrended_prices = log_prices - trend
    
    # Calculate split point
    split_idx = int(len(detrended_prices) * train_size)
    
    # Split data into train and test sets
    train_data = detrended_prices[:split_idx]
    test_data = detrended_prices[split_idx:]
    
    # Train ARIMA model on detrended data (p=1, d=0, q=0 for AR(1) equivalent)
    arima_model = ARIMA(train_data, order=(1, 0, 0))
    model_fit = arima_model.fit()
    print(model_fit.summary())
    # Make predictions
    detrended_predictions = model_fit.forecast(steps=len(test_data))
    
    # Add trend back to predictions
    predictions = np.exp(detrended_predictions + trend[split_idx:])
    
    # Plot training data and forecast
    plt.figure(figsize=(15, 6))
    plt.plot(df.index[:split_idx], df['Close'][:split_idx], label='Training Data', color='blue')
    plt.plot(df.index[split_idx:], df['Close'][split_idx:], label='Actual Test Data', color='green')
    plt.plot(df.index[split_idx:], predictions, label='Forecast', color='red', linestyle='--')
    plt.axvline(x=df.index[split_idx], color='gray', linestyle='--', label='Train/Test Split')
    plt.title('ARIMA(1,0,0) Model Training Data and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return predictions, train_data, test_data

def main():
    """
    Main function to run the stock market analysis pipeline.
    
    Process:
    1. Load and validate environment variables
    2. Read and preprocess stock data
    3. Generate market visualization plots
    4. Perform seasonal decomposition
    5. Test for stationarity
    6. Analyze autocorrelation patterns
    
    Raises:
        ValueError: If environment variables or data format is invalid
        FileNotFoundError: If data file doesn't exist
    """
    try:
        # Load environment variables
        load_dotenv()

        # Get environment variables
        data_file = os.getenv('DATA_FILE')
        if not data_file:
            raise ValueError("DATA_FILE environment variable is not set")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Read and prepare stock data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Generate market analysis plots
        print("Generating long-term market analysis plot...")
        #plot_long_term_analysis(df)
        
        print("Generating two-year segment analysis plots...")
        #plot_two_year_segments(df)

        # Perform time series analysis
        #decomposition, log_prices, non_trend_df = perform_seasonal_decomposition(df)
        
        # Test for stationarity
        #is_stationary, stat_results = test_stationarity(non_trend_df['Close'])
        
        # Analyze autocorrelation if data is stationary
        #if is_stationary:
            #acf_values, pacf_values = analyze_autocorrelation(non_trend_df['Close'])
        #else:
            #print("\nSkipping autocorrelation analysis as data is non-stationary")

        # Run baseline model

        if df.isnull().values.any():
            print('Dropping NaN values from dataframe')
            df = df.dropna()
        
        # Train and forecast using AR model
        predictions, train_data, test_data = train_and_forecast_ar(df, train_size=0.8)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
