import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
from src.Autoformer import Autoformer
from src.adopt import ADOPT
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CrossValidationResults:
    """Store and analyze cross-validation results"""
    def __init__(self):
        self.fold_scores = []
        self.train_losses = []
        self.val_losses = []
    
    def add_fold(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.fold_scores.append({'train_loss': train_loss, 'val_loss': val_loss})
    
    def get_mean_scores(self):
        return {
            'mean_train_loss': np.mean(self.train_losses),
            'mean_val_loss': np.mean(self.val_losses),
            'std_train_loss': np.std(self.train_losses),
            'std_val_loss': np.std(self.val_losses)
        }
    
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        """
        Args:
            patience (int): How many epochs to wait before stopping after loss has stopped improving
            min_delta (float): Minimum change in loss to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop
    
    def restore_weights(self, model):
        """Restores the best weights if restore_best_weights is True"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def create_blocked_cv_datasets(data, targets, n_splits=5):
    """
    Create blocked cross-validation datasets with non-overlapping blocks.

    In each fold, the training data is the block immediately before the validation block.

    Args:
        data: Input sequences tensor
        targets: Target values tensor
        n_splits: Number of CV folds

    Returns:
        List of (train_data, train_targets, val_data, val_targets) for each fold
    """
    total_samples = len(data)
    block_size = total_samples // (n_splits + 1)
    cv_splits = []

    for i in range(n_splits):
        train_start = i * block_size
        train_end = train_start + block_size

        val_start = train_end
        val_end = val_start + block_size

        # Ensure indices do not go beyond data length
        if val_end > total_samples:
            val_end = total_samples

        train_data = data[train_start:train_end]
        train_targets = targets[train_start:train_end]

        val_data = data[val_start:val_end]
        val_targets = targets[val_start:val_end]

        cv_splits.append((train_data, train_targets, val_data, val_targets))

        # Break if we've reached the end of the data
        if val_end == total_samples:
            break

    return cv_splits

def train_validate_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, device, patience=7, min_delta=0.001):
    """
    Train and validate a model for one fold of cross-validation.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (Optimizer): The optimizer for training
        scheduler (LRScheduler): Learning rate scheduler
        criterion (Loss): Loss function
        num_epochs (int): Maximum number of epochs to train
        device (torch.device): Device to run the training on
        patience (int, optional): Patience for early stopping. Defaults to 7
        min_delta (float, optional): Minimum improvement required for early stopping. Defaults to 0.001

    Returns:
        tuple: Best training loss and validation loss achieved during training
    """
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
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
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_data)
                val_loss = criterion(outputs, batch_targets)
                epoch_val_losses.append(val_loss.item())
        
        # Calculate average losses for this epoch
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(model, avg_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Restore best weights if enabled
    early_stopping.restore_weights(model)
    
    # Return the best validation loss and corresponding training loss
    best_epoch = np.argmin(val_losses)
    return train_losses[best_epoch], val_losses[best_epoch]


def cross_validate(model_params, data, targets, n_splits=5, num_epochs=10, batch_size=32, patience=7, min_delta=0.001, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Perform k-fold cross-validation on a model.

    Args:
        model_params (dict): Parameters for initializing the Autoformer model
        data (torch.Tensor): Input data tensor
        targets (torch.Tensor): Target values tensor
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5
        num_epochs (int, optional): Maximum number of epochs per fold. Defaults to 10
        batch_size (int, optional): Batch size for training. Defaults to 32
        patience (int, optional): Patience for early stopping. Defaults to 7
        min_delta (float, optional): Minimum improvement required for early stopping. Defaults to 0.001
        device (torch.device, optional): Device to run training on. Defaults to GPU if available, else CPU

    Returns:
        CrossValidationResults: Object containing training and validation metrics for all folds
    """
    cv_results = CrossValidationResults()
    cv_splits = create_blocked_cv_datasets(data, targets, n_splits)
    
    for fold, (train_data, train_targets, val_data, val_targets) in enumerate(cv_splits):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        # Create data loaders for this fold
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize a new model for this fold
        model = Autoformer(**model_params).to(device)
        optimizer = ADOPT(
            model.parameters(),
            lr=2e-3,
            betas=(0.9, 0.9999),
            eps=1e-6,
            weight_decay=0.01,
            decouple=True
        )
        # Define the scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4, verbose=True)
        criterion = nn.MSELoss()
        
        # Train and validate on this fold with early stopping and scheduler
        train_loss, val_loss = train_validate_fold(
            model, train_loader, val_loader, optimizer, scheduler, criterion, 
            num_epochs, device, patience, min_delta
        )
        
        cv_results.add_fold(train_loss, val_loss)
        
        print(f"Fold {fold + 1} - Best Train Loss: {train_loss:.4f}, Best Val Loss: {val_loss:.4f}")
    
    # Print summary statistics
    mean_scores = cv_results.get_mean_scores()
    print("\nCross-validation Results:")
    print(f"Mean Train Loss: {mean_scores['mean_train_loss']:.4f} ± {mean_scores['std_train_loss']:.4f}")
    print(f"Mean Val Loss: {mean_scores['mean_val_loss']:.4f} ± {mean_scores['std_val_loss']:.4f}")
    
    return cv_results