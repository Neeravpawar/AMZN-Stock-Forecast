# Import necessary libraries for data manipulation, visualization, and time series analysis
import os
from dotenv import load_dotenv
import pandas as pd
from src import *
import torch


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
    7. Baseline model training and forecasting
    8. Prepare data for modeling
    9. Prepare data loaders
    10. Cross-validation for hyperparameter tuning
    11. Final model training and evaluation
    12. Plotting results
    13. Save model and results
    
    Raises:
        ValueError: If environment variables or data format is invalid
        FileNotFoundError: If data file doesn't exist
    """
    try:
        # ------------------------- Prepare environment and data -------------------------
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

        # ------------------------- Generate market analysis plots -------------------------
        #print("Generating long-term market analysis plot...")
        #plot_long_term_analysis(df)
        
        #print("Generating two-year segment analysis plots...")
        #plot_two_year_segments(df)


        # ------------------------- Perform time series and statistical analysis -------------------------
        # Perform time series analysis
        #decomposition, log_prices, non_trend_df = perform_seasonal_decomposition(df)
        
        # Test for stationarity
        #is_stationary, stat_results = test_stationarity(non_trend_df['Close'])
        
        # Analyze autocorrelation if data is stationary
        #if is_stationary:
            #acf_values, pacf_values = analyze_autocorrelation(non_trend_df['Close'])
        #else:
            #print("\nSkipping autocorrelation analysis as data is non-stationary")

        # ------------------------- Start modelling -------------------------
        
        # Train and forecast using AR model
        #predictions, train_data, test_data = train_and_forecast_ar(df, train_size=0.8)

        # Prepare data for modeling
        print("Preparing data for modeling...")
        model_df = prepare_data_for_modeling(df)

        # Initialize training parameters
        training_params = {
            'n_splits': 4,           # Number of cross-validation folds
            'num_epochs': 50,        # Number of epochs per fold
            'batch_size': 64,        # Batch size
            'patience': 12,           # Early stopping patience
            'min_delta': 0.0,      # Minimum improvement required
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
        # Select features for modelling
        feature_columns = ['Close_norm','RSI_norm','MACD_norm','Signal_norm','MFI_norm','TimeIndex_norm',
                           'Upper_norm','Middle_norm','Lower_norm','day_of_year_sin', 
                           'day_of_year_cos','day_of_quarter_sin', 'day_of_quarter_cos','Signal_norm']
        # Initialize model parameters
        model_params = {
            'input_dim': len(feature_columns),    # Number of features (OHLC normalized)
            'output_dim': len(feature_columns),   # Prediction dimension
            'seq_len': 252,                       # Input sequence length
            'label_len': 21,                       # Number of labels to predict
            'pred_len': 42,                        # Prediction sequence length
            'd_model': 128,                       # Model dimension
            'n_heads': 2,                         # Number of attention heads
            'e_layers': 4,                        # Number of encoder layers
            'd_layers': 3,                        # Number of decoder layers
            'd_ff': 512,                          # Dimension of FCN
            'moving_avg': 91,                     # Moving average kernel size
            'dropout': 0.15,                       # Dropout rate
            'activation': 'gelu',                 # Activation function
            'output_attention': False             # Whether to output attention weights
        }

        # Prepare data loaders
        train_loader, test_loader, train_dataset, test_dataset, test_start_idx = prepare_data_loaders(
            model_df,
            seq_length=model_params['seq_len'],
            pred_len=model_params['pred_len'],
            batch_size=training_params['batch_size'],
            feature_columns = feature_columns
        )

        sequences_tensor = torch.FloatTensor(train_dataset.sequences)
        targets_tensor = torch.FloatTensor(train_dataset.targets)
        
        print("\nData Preparation Summary:")
        print(f"Total sequences: {len(sequences_tensor)}")
        print(f"Sequence shape: {sequences_tensor.shape}")
        print(f"Target shape: {targets_tensor.shape}")
        
        
        print("\nTraining Configuration:")
        print(f"Device: {training_params['device']}")
        print(f"Number of CV folds: {training_params['n_splits']}")
        print(f"Epochs per fold: {training_params['num_epochs']}")
        print(f"Early stopping patience: {training_params['patience']}")
        print(f"Minimum delta: {training_params['min_delta']}")
        print(f"Batch size: {training_params['batch_size']}")

        # ------------------------- Perform cross-validation -------------------------
        cv_results = cross_validate(
           model_params=model_params,
          data=sequences_tensor,
           targets=targets_tensor,
           n_splits=training_params['n_splits'],
           num_epochs=training_params['num_epochs'],
           batch_size=training_params['batch_size'],
           patience=training_params['patience'],
           min_delta=training_params['min_delta'],
           device=training_params['device']
        )
        # 
        # Print final results
        mean_scores = cv_results.get_mean_scores()
        print("\nFinal Cross-validation Results:")
        print(f"Mean Train Loss: {mean_scores['mean_train_loss']:.4f} ± {mean_scores['std_train_loss']:.4f}")
        print(f"Mean Val Loss: {mean_scores['mean_val_loss']:.4f} ± {mean_scores['std_val_loss']:.4f}")
        

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
