import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from src.visualizations import plot_decomposition
from src.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_mfi
from datetime import datetime
from torch.utils.data import DataLoader
from statsmodels.tsa.stattools import adfuller
class TimeSeriesDataset(Dataset):
    """
    Custom Dataset class for time series data with lagged features
    """
    def __init__(self, sequences, targets):
        """
        Args:
            sequences (np.ndarray): Array of shape [n_samples, seq_length, n_features]
            targets (np.ndarray): Array of shape [n_samples, n_features]
        """
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.targets[idx]))


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

def prepare_data_for_modeling(df):
    """
    Prepare data for modeling by applying log transformation and normalization
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and DateTimeIndex
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed and normalized values
    """
    model_df = df.copy()
    
    # Apply logarithmic transformation to 'Open', 'High', 'Low', 'Close'
    for col in ['Open', 'High', 'Low', 'Close']:
        model_df[col] = np.log(model_df[col])
    
    # Calculate technical indicators (using log-transformed prices)
    model_df['RSI'] = calculate_rsi(model_df['Close'])
    model_df['MFI'] = calculate_mfi(model_df)
    macd_df = calculate_macd(model_df['Close'])
    bollinger_df = calculate_bollinger_bands(model_df['Close'])
    
    # Merge MACD and Bollinger Bands into model_df
    model_df = pd.concat([model_df, macd_df, bollinger_df], axis=1)
    
    # Drop initial rows with NaN values due to indicator calculations
    model_df.dropna(inplace=True)
    
    # --------------------- Add Temporal Features ---------------------
    # Calculate day of the year
    model_df['day_of_year'] = model_df.index.dayofyear
    # Calculate day of the quarter
    def calculate_day_of_quarter(date):
        quarter = (date.month - 1) // 3 + 1
        first_day_of_quarter = datetime(date.year, 3 * quarter - 2, 1)
        return (date - first_day_of_quarter).days + 1  # +1 to make it 1-based
    
    model_df['day_of_quarter'] = model_df.index.to_pydatetime()
    model_df['day_of_quarter'] = model_df['day_of_quarter'].apply(calculate_day_of_quarter)
    
    # Sin and Cos transformations for day_of_year
    model_df['day_of_year_sin'] = np.sin(2 * np.pi * model_df['day_of_year'] / 365)
    model_df['day_of_year_cos'] = np.cos(2 * np.pi * model_df['day_of_year'] / 365)
    
    # Sin and Cos transformations for day_of_quarter
    model_df['day_of_quarter_sin'] = np.sin(2 * np.pi * model_df['day_of_quarter'] / 90)  # Approx. 90 days per quarter
    model_df['day_of_quarter_cos'] = np.cos(2 * np.pi * model_df['day_of_quarter'] / 90)
    
    # Drop the original temporal features if not needed
    model_df.drop(['day_of_year', 'day_of_quarter'], axis=1, inplace=True)
    # -------------------------------------------------------------------
    
    # Columns to normalize (log-transformed prices, volume, indicators, and temporal features)
    columns_to_transform = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MFI',
                            'MACD', 'Signal', 'Histogram', 'Middle', 'Upper', 'Lower',
                            'day_of_year_sin', 'day_of_year_cos',
                            'day_of_quarter_sin', 'day_of_quarter_cos']
    
    # Store original values
    model_df.attrs['original_values'] = {
        col: model_df[col].copy() for col in columns_to_transform
    }
    
    # Min-Max Scaling
    for col in columns_to_transform:
        min_val = model_df[col].min()
        max_val = model_df[col].max()
        model_df[f'{col}_norm'] = (model_df[col] - min_val) / (max_val - min_val)
    
    # Store scaling parameters
    model_df.attrs['norm_params'] = {
        col: {'min': model_df[col].min(), 'max': model_df[col].max()}
        for col in columns_to_transform
    }

    # Add time index
    model_df['TimeIndex'] = np.arange(len(model_df))
    model_df['TimeIndex_norm'] = model_df['TimeIndex'] / model_df['TimeIndex'].max()
    
    return model_df

def prepare_data_loaders(model_df, seq_length=100, pred_len=10, batch_size=32, feature_columns=None):
    """
    Prepare train and test DataLoaders for time series modeling with multi-step forecasting.
    """
    if feature_columns is None:
        # Default feature columns include normalized price data
        feature_columns = ['Open_norm', 'High_norm', 'Low_norm', 'Close_norm']
    
    # Ensure that all specified feature columns are in the DataFrame
    missing_features = [col for col in feature_columns if col not in model_df.columns]
    if missing_features:
        raise ValueError(f"Feature columns {missing_features} not found in DataFrame.")
    
    # Adjust total_samples to account for pred_len
    total_samples = len(model_df) - seq_length - pred_len + 1
    sequences = np.zeros((total_samples, seq_length, len(feature_columns)))
    targets = np.zeros((total_samples, pred_len, len(feature_columns)))
    
    # Generate sequences and targets
    for idx in range(total_samples):
        sequences[idx] = model_df[feature_columns].iloc[idx : idx + seq_length].values
        targets[idx] = model_df[feature_columns].iloc[idx + seq_length : idx + seq_length + pred_len].values
    
    # Calculate split points
    train_size = int(total_samples * 0.85)
    test_size = total_samples - train_size
    
    # Split data without buffer zone
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    test_dataset = TimeSeriesDataset(test_sequences, test_targets)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\nDataset Summary:")
    print(f"Total sequences generated: {total_samples}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature dimension: {len(feature_columns)}")
    print(f"Sequence length: {seq_length}")
    print(f"Prediction length: {pred_len}")
    print(f"Batch size: {batch_size}")

    train_end_date = model_df.index[train_size + seq_length - 1]
    test_end_date = model_df.index[total_samples + seq_length - 1]
    
    print(f"\nDate Ranges:")
    print(f"Training: up to {train_end_date.strftime('%Y-%m-%d')}")
    print(f"Testing: {train_end_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")
    
    return train_loader, test_loader, train_dataset, test_dataset, train_size

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
