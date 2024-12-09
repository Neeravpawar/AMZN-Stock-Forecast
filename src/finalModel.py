import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from torch.utils.data import DataLoader
from src.crossValidation import EarlyStopping
from src.adopt import ADOPT
from torch.optim.lr_scheduler import ReduceLROnPlateau

@torch.no_grad()
def evaluate_model(model, test_loader, device, norm_params, feature_columns, feature_to_plot='Close_norm'):
    """
    Evaluate the model on the test set, perform inverse transformations, and collect predictions and true labels in original scale.
    
    Args:
        model (nn.Module): Trained Autoformer model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to perform computation on.
        norm_params (Dict[str, Dict[str, float]]): Normalization parameters for inverse transformation.
        feature_columns (List[str]): List of feature column names.
        feature_to_plot (str): The feature to inverse transform and plot.
    
    Returns:
        test_loss (float): Mean Squared Error on the test set.
        all_predictions_original (np.ndarray): Predicted values in original scale.
        all_targets_original (np.ndarray): True target values in original scale.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    # Identify the index of the feature_to_plot
    try:
        feature_idx = feature_columns.index(feature_to_plot)
    except ValueError:
        raise ValueError(f"Feature '{feature_to_plot}' not found in feature_columns.")
    
    original_feature = feature_to_plot.replace('_norm', '')
    if original_feature not in norm_params:
        raise ValueError(f"Normalization parameters for '{original_feature}' not found.")
    
    min_val = norm_params[original_feature]['min']
    max_val = norm_params[original_feature]['max']
    
    for batch_data, batch_targets in test_loader:
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_predictions.append(outputs.cpu())
        all_targets.append(batch_targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()        
    
    test_loss = total_loss / num_batches
    
    # Inverse transform only the feature_to_plot
    # Extract the relevant feature from predictions and targets
    pred_normalized = all_predictions[:, :, feature_idx]
    true_normalized = all_targets[:, :, feature_idx]    
    
    # Denormalize
    pred_log = pred_normalized * (max_val - min_val) + min_val
    true_log = true_normalized * (max_val - min_val) + min_val
    
    # Exponentiate to get original scale
    pred_original = np.exp(pred_log)
    true_original = np.exp(true_log)
    
    return test_loss, pred_original, true_original

def train_final_model(model, train_loader, validation_loader, optimizer, scheduler, criterion, num_epochs, device, patience=7, min_delta=0.001):
    """
    Train the final model for the forecast.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        criterion (nn.Module): Loss function.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        patience (int): Early stopping patience.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
    """
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    optimizer = ADOPT(
        model.parameters(),
        lr=7e-3,
        betas=(0.9, 0.9999),
        eps=1e-6,
        weight_decay=0.02,
        decouple=True
    )
    criterion = nn.MSELoss()
        
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4, verbose=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_losses = []

        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_losses = []

        with torch.no_grad():
            for batch_data, batch_targets in validation_loader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_data)
                val_loss = criterion(outputs, batch_targets)
                epoch_val_losses.append(val_loss.item())

        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Print epoch progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping check
        if early_stopping(model, avg_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best weights if enabled
    early_stopping.restore_weights(model)

    return train_losses, val_losses


def plot_predictions(predictions_original, true_values_original, model_df, test_start_idx, seq_length, pred_len, feature_columns, feature_to_plot='Close_norm'):
    """
    Plot the predicted values against the true values in the original scale, including additional training data before the test set.
    
    Args:
        predictions_original (np.ndarray): Predicted values in original scale. Shape: [n_samples, pred_len]
        true_values_original (np.ndarray): True target values in original scale. Shape: [n_samples, pred_len]
        model_df (pd.DataFrame): The preprocessed DataFrame with original data.
        test_start_idx (int): Index where the test set starts.
        seq_length (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        feature_columns (List[str]): List of feature column names.
        feature_to_plot (str): The feature to plot. Default is 'Close_norm'.
    """
    # Choose the feature to plot
    if feature_to_plot not in feature_columns:
        raise ValueError(f"Feature '{feature_to_plot}' not in feature_columns.")
    
    # Remove '_norm' suffix to get the original feature name
    original_feature = feature_to_plot.replace('_norm', '')
    
    # Get normalization parameters for the selected feature
    norm_params = model_df.attrs['norm_params'][original_feature]
    min_val, max_val = norm_params['min'], norm_params['max']
    
    # Since predictions_original and true_values_original are already in original scale, no need to inverse transform
    # Reconstruct the time series from predictions and true values
    num_samples = predictions_original.shape[0]
    total_pred_steps = num_samples + pred_len - 1
    
    pred_series = np.zeros(total_pred_steps)
    true_series = np.zeros(total_pred_steps)
    count_series = np.zeros(total_pred_steps)  # To handle overlapping counts
    
    for i in range(num_samples):
        pred_series[i:i + pred_len] += predictions_original[i]
        true_series[i:i + pred_len] += true_values_original[i]
        count_series[i:i + pred_len] += 1
    
    # Avoid division by zero
    count_series[count_series == 0] = 1
    pred_series /= count_series
    true_series /= count_series
    
    # Determine the amount of training data to include
    test_data_length = len(pred_series)
    additional_train_length = 2 * test_data_length
    
    # Ensure we don't go beyond the start of the data
    available_train_data = test_start_idx + seq_length
    start_idx = max(0, available_train_data - additional_train_length)
    
    # Extract the true values and dates for the additional training data
    train_series_original = np.exp(model_df['Close_norm'].iloc[start_idx : available_train_data].values * (max_val - min_val) + min_val)
    
    # Extract dates
    dates_train = model_df.index[start_idx : available_train_data]
    dates_test = model_df.index[available_train_data : available_train_data + len(pred_series)]
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(dates_train, train_series_original, label='Training Data', color='blue', alpha=0.7)
    plt.plot(dates_test, true_series, label='True Values', color='green', alpha=0.7)
    plt.plot(dates_test, pred_series, label='Predictions', color='red', alpha=0.7)
    
    # Add vertical line to separate training and test data
    plt.axvline(x=dates_test[0], color='gray', linestyle='--', alpha=0.5)
    plt.text(dates_test[0], plt.ylim()[0], 'Test Set Start', 
             rotation=90, verticalalignment='bottom')
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.title('Predicted vs. True Stock Prices')
    plt.legend()
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
