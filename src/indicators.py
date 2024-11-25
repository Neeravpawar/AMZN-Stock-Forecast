import numpy as np
import pandas as pd

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_mfi(df, period=14):
    """
    Calculate Money Flow Index (MFI)
    """
    # Typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Money flow
    money_flow = typical_price * df['Volume']
    
    # Positive and negative money flow
    delta = typical_price.diff()
    positive_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=df.index)
    negative_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=df.index)
    
    # Calculate money flow ratio and MFI
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    """
    # Calculate EMAs
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': macd_histogram
    })

def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    """
    # Calculate middle band (SMA)
    middle_band = data.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = data.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return pd.DataFrame({
        'Middle': middle_band,
        'Upper': upper_band,
        'Lower': lower_band
    }) 