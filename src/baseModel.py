import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

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
