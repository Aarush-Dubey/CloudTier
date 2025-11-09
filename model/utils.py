"""
Utility functions for metrics, plotting, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from typing import Dict, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   target_names: List[str] = None) -> Dict[str, float]:
    """
    Compute MAE and RMSE for each target.
    
    Args:
        y_true: Ground truth of shape (n_samples, n_targets)
        y_pred: Predictions of shape (n_samples, n_targets)
        target_names: List of target variable names
    
    Returns:
        Dictionary of metrics
    """
    if target_names is None:
        target_names = ['reads', 'writes', 'egress']
    
    metrics = {}
    
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        
        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_rmse'] = rmse
    
    # Overall metrics
    metrics['overall_mae'] = mean_absolute_error(y_true, y_pred)
    metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return metrics


def compute_relative_error(y_true: np.ndarray, y_pred: np.ndarray, 
                          target_names: List[str] = None) -> Dict[str, float]:
    """
    Compute relative percentage error for each target.
    
    Args:
        y_true: Ground truth of shape (n_samples, n_targets)
        y_pred: Predictions of shape (n_samples, n_targets)
        target_names: List of target variable names
    
    Returns:
        Dictionary of relative errors
    """
    if target_names is None:
        target_names = ['reads', 'writes', 'egress']
    
    rel_errors = {}
    
    for i, name in enumerate(target_names):
        # Avoid division by zero
        mask = y_true[:, i] > 0
        if mask.sum() > 0:
            rel_error = np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i]) * 100
            rel_errors[f'{name}_mape'] = np.mean(rel_error)
        else:
            rel_errors[f'{name}_mape'] = 0.0
    
    return rel_errors


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         save_path: str = 'training_loss.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    target_names: List[str] = None, 
                    save_path: str = 'predictions.png',
                    n_samples: int = 100):
    """
    Plot predicted vs actual values for each target.
    
    Args:
        y_true: Ground truth of shape (n_samples, n_targets)
        y_pred: Predictions of shape (n_samples, n_targets)
        target_names: List of target variable names
        save_path: Path to save the plot
        n_samples: Number of samples to plot
    """
    if target_names is None:
        target_names = ['Reads', 'Writes', 'Egress']
    
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(15, 4))
    
    if n_targets == 1:
        axes = [axes]
    
    # Subsample if too many points
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true_plot[:, i], y_pred_plot[:, i], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true_plot[:, i].min(), y_pred_plot[:, i].min())
        max_val = max(y_true_plot[:, i].max(), y_pred_plot[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{name}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved predictions plot to {save_path}")


def plot_forecast_example(input_seq: np.ndarray, actual: np.ndarray, 
                         predicted: np.ndarray, target_names: List[str] = None,
                         save_path: str = 'forecast_example.png'):
    """
    Plot an example forecast showing input sequence and predictions.
    
    Args:
        input_seq: Input sequence of shape (window_size, n_features)
        actual: Actual values of shape (n_targets,)
        predicted: Predicted values of shape (n_targets,)
        target_names: List of target variable names
        save_path: Path to save the plot
    """
    if target_names is None:
        target_names = ['Reads', 'Writes', 'Egress']
    
    # Assume first few features are the log-transformed targets
    n_targets = len(target_names)
    
    fig, axes = plt.subplots(1, n_targets, figsize=(15, 4))
    
    if n_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        # Plot input history (first feature corresponds to reads, second to writes, etc.)
        if i < input_seq.shape[1]:
            history = input_seq[:, i]
            ax.plot(range(len(history)), history, 'b-', linewidth=2, label='History')
        
        # Plot actual and predicted at forecast point
        forecast_point = len(input_seq) + 1  # +2 hours horizon
        ax.scatter([forecast_point], [actual[i]], color='green', s=100, 
                  label='Actual', marker='o', zorder=5)
        ax.scatter([forecast_point], [predicted[i]], color='red', s=100, 
                  label='Predicted', marker='x', zorder=5)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{name} Forecast', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved forecast example to {save_path}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        self.monitor_op = np.less if mode == 'min' else np.greater
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.monitor_op(score, self.best_score - self.min_delta):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Validation improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epoch(s)")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        
        return False


if __name__ == "__main__":
    # Test metrics
    y_true = np.random.randn(100, 3) * 100 + 500
    y_pred = y_true + np.random.randn(100, 3) * 20
    
    metrics = compute_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    rel_errors = compute_relative_error(y_true, y_pred)
    print("Relative errors:", rel_errors)
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    for epoch in range(10):
        loss = 1.0 / (epoch + 1) + np.random.randn() * 0.01
        if early_stop(loss, epoch):
            break