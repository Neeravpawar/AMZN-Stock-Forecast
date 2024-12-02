import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

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
