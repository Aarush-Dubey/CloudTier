<<<<<<< HEAD
"""
Training script for LSTM forecaster with early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
import os

from model import LSTMForecaster, HuberLoss
from utils import compute_metrics, plot_training_history, plot_predictions, EarlyStopping


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, clip_grad: float = 1.0) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model: nn.Module, dataloader: DataLoader, 
            criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Validation", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    metrics = compute_metrics(targets, predictions)
    
    return avg_loss, metrics


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               num_epochs: int = 50, learning_rate: float = 1e-3,
               weight_decay: float = 1e-5, patience: int = 5,
               device: torch.device = None, save_dir: str = 'checkpoints') -> Dict:
    """
    Train the LSTM forecaster with early stopping.
    
    Args:
        model: LSTMForecaster model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min', verbose=True)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for up to {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Val MAE - Reads: {val_metrics['reads_mae']:.2f}, "
              f"Writes: {val_metrics['writes_mae']:.2f}, "
              f"Egress: {val_metrics['egress_mae']:.2f}")
        print(f"Val RMSE - Reads: {val_metrics['reads_rmse']:.2f}, "
              f"Writes: {val_metrics['writes_rmse']:.2f}, "
              f"Egress: {val_metrics['egress_rmse']:.2f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, 'lstm_forecaster.pt'))
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot training history
    plot_training_history(history['train_loss'], history['val_loss'], 
                         save_path=os.path.join(save_dir, 'training_loss.png'))
    
    # Generate validation predictions for visualization
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            val_predictions.append(predictions.cpu().numpy())
            val_targets.append(batch_y.numpy())
    
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    
    plot_predictions(val_targets, val_predictions, 
                    save_path=os.path.join(save_dir, 'validation_predictions.png'))
    
    return history


if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    from dataset import create_dataloaders
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(window_size=24, forecast_horizon=2)
    
    df = preprocessor.load_and_prepare_data("workload_data.csv")
    df_encoded = preprocessor.encode_features(df)
    
    train_df, val_df, test_df = preprocessor.create_splits(df_encoded)
    
    X_train, y_train = preprocessor.create_sequences(train_df, fit_scaler=True)
    X_val, y_val = preprocessor.create_sequences(val_df, fit_scaler=False)
    
    # Save scaler for inference
    preprocessor.save_scaler('checkpoints/scaler.pkl')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=256
    )
    
    # Initialize model
    input_size = X_train.shape[2]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_targets=3
    )
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=5,
        save_dir='checkpoints'
    )
    
    print("\nTraining completed successfully!")
=======
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OptimizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def prepare_data_reads(df, horizon=12, lookback=24):
    """Prepare sequences for reads prediction"""
    # Features for reads prediction (no write features)
    feature_cols = [
        'reads_1h', 'bytes_read_1h', 'hour_of_day', 'day_of_week',
        'reads_6h', 'reads_12h', 'reads_24h', 'reads_48h', 'reads_96h',
        'trend_6h', 'data_temperature_encoded', 'access_freq_24h'
    ]
    
    # Encode categorical
    df['data_temperature_encoded'] = df['data_temperature'].map({'hot': 2, 'warm': 1, 'cold': 0})
    
    X, y = [], []
    dataset_groups = df.groupby('dataset_id')
    
    for _, group in dataset_groups:
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(group) - lookback - horizon + 1):
            # Input sequence
            seq = group.iloc[i:i+lookback][feature_cols].values
            # Target: sum of reads over next horizon hours
            target = group.iloc[i+lookback:i+lookback+horizon]['reads_1h'].sum()
            
            X.append(seq)
            y.append(target)
    
    return np.array(X), np.array(y).reshape(-1, 1), feature_cols

def prepare_data_writes(df, horizon=12, lookback=24):
    """Prepare sequences for writes prediction"""
    # Features for writes prediction (no read features)
    feature_cols = [
        'writes_1h', 'bytes_read_1h', 'hour_of_day', 'day_of_week',
        'writes_6h', 'writes_12h', 'writes_24h', 'writes_48h', 'writes_96h',
        'trend_6h', 'data_temperature_encoded', 'access_freq_24h'
    ]
    
    # Encode categorical
    df['data_temperature_encoded'] = df['data_temperature'].map({'hot': 2, 'warm': 1, 'cold': 0})
    
    X, y = [], []
    dataset_groups = df.groupby('dataset_id')
    
    for _, group in dataset_groups:
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(group) - lookback - horizon + 1):
            # Input sequence
            seq = group.iloc[i:i+lookback][feature_cols].values
            # Target: sum of writes over next horizon hours
            target = group.iloc[i+lookback:i+lookback+horizon]['writes_1h'].sum()
            
            X.append(seq)
            y.append(target)
    
    return np.array(X), np.array(y).reshape(-1, 1), feature_cols

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and MAPE"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE: MAE / mean(actual) * 100
    # Avoid division by zero
    mean_actual = np.mean(np.abs(y_true))
    mape = (mae / mean_actual * 100) if mean_actual > 0 else 0
    
    return mae, rmse, mape

def evaluate_model(model, data_loader, scaler_y, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            
            # Inverse transform to original scale
            preds = scaler_y.inverse_transform(outputs)
            targets = scaler_y.inverse_transform(y_batch.numpy())
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mae, rmse, mape = calculate_metrics(all_targets, all_preds)
    
    return mae, rmse, mape

def train_model(model, train_loader, val_loader, scaler_y, epochs=100, patience=7, lr=0.001):
    """Train the model with early stopping and detailed metrics"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mape': [],
        'val_mape': []
    }
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<12} {'Val MAE':<12} {'Train RMSE':<12} {'Val RMSE':<12} {'Train MAPE':<12} {'Val MAPE':<12}")
    print("="*140)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_mae, train_rmse, train_mape = evaluate_model(model, train_loader, scaler_y, device)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics
        val_mae, val_rmse, val_mape = evaluate_model(model, val_loader, scaler_y, device)
        
        scheduler.step(val_loss)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_mae'].append(train_mae)
        training_history['val_mae'].append(val_mae)
        training_history['train_rmse'].append(train_rmse)
        training_history['val_rmse'].append(val_rmse)
        training_history['train_mape'].append(train_mape)
        training_history['val_mape'].append(val_mape)
        
        # Print metrics
        print(f"{epoch+1:<6} {train_loss:<12.4f} {val_loss:<12.4f} {train_mae:<12.2f} {val_mae:<12.2f} {train_rmse:<12.2f} {val_rmse:<12.2f} {train_mape:<12.2f}% {val_mape:<12.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss, training_history

def load_or_prepare_data(csv_path, cache_dir='cache'):
    """Load data from cache or prepare and cache it"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'processed_data.pkl')
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print("Cached data loaded successfully!")
        return cached_data['train_df'], cached_data['val_df'], cached_data['df']
    
    print("Loading and processing data...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.fillna(0)
    
    # Train/val split (80/20)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Cache the data
    print(f"Caching data to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_df': train_df,
            'val_df': val_df,
            'df': df
        }, f)
    print("Data cached successfully!")
    
    return train_df, val_df, df

def main():
    # Load data with caching
    print("Loading data...")
    train_df, val_df, df = load_or_prepare_data('training_data.csv')
    
    results = {}
    all_histories = {}
    
    # Train READS models for both horizons
    print("\n" + "="*140)
    print("TRAINING READS PREDICTION MODELS")
    print("="*140)
    
    for horizon in [12]:
        print(f"\n>>> Training Reads Model - Horizon: {horizon}h")
        
        cache_dir = 'cache'
        reads_cache_file = os.path.join(cache_dir, f'reads_data_h{horizon}.pkl')
        
        # Check for cached processed sequences
        if os.path.exists(reads_cache_file):
            print(f"Loading cached sequences from {reads_cache_file}...")
            with open(reads_cache_file, 'rb') as f:
                cached = pickle.load(f)
            X_train_scaled = cached['X_train_scaled']
            X_val_scaled = cached['X_val_scaled']
            y_train_scaled = cached['y_train_scaled']
            y_val_scaled = cached['y_val_scaled']
            scaler_X = cached['scaler_X']
            scaler_y = cached['scaler_y']
            feature_cols = cached['feature_cols']
            print("Cached sequences loaded!")
        else:
            # Prepare data
            X_train, y_train, feature_cols = prepare_data_reads(train_df, horizon=horizon)
            X_val, y_val, _ = prepare_data_reads(val_df, horizon=horizon)
            
            # Normalize
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled = scaler_y.transform(y_val)
            
            # Cache sequences
            print(f"Caching sequences to {reads_cache_file}...")
            with open(reads_cache_file, 'wb') as f:
                pickle.dump({
                    'X_train_scaled': X_train_scaled,
                    'X_val_scaled': X_val_scaled,
                    'y_train_scaled': y_train_scaled,
                    'y_val_scaled': y_val_scaled,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y,
                    'feature_cols': feature_cols
                }, f)
            print("Sequences cached!")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
        
        # Create model
        model = OptimizedLSTM(input_size=len(feature_cols), hidden_size=128, num_layers=2, dropout=0.2)
        model = model.to(device)
        
        # Train
        model, val_loss, history = train_model(model, train_loader, val_loader, scaler_y, epochs=100, patience=7)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols,
            'horizon': horizon,
            'training_history': history
        }, f'reads_model_h{horizon}.pth')
        
        results[f'reads_h{horizon}'] = {
            'val_loss': val_loss,
            'val_mae': history['val_mae'][-1],
            'val_rmse': history['val_rmse'][-1],
            'val_mape': history['val_mape'][-1]
        }
        all_histories[f'reads_h{horizon}'] = history
        print(f"\nModel saved: reads_model_h{horizon}.pth")
        print(f"Final Metrics - MAE: {history['val_mae'][-1]:.2f}, RMSE: {history['val_rmse'][-1]:.2f}, MAPE: {history['val_mape'][-1]:.2f}%")
    
    # Train WRITES models for both horizons
    print("\n" + "="*140)
    print("TRAINING WRITES PREDICTION MODELS")
    print("="*140)
    
    for horizon in [8, 12]:
        print(f"\n>>> Training Writes Model - Horizon: {horizon}h")
        
        writes_cache_file = os.path.join(cache_dir, f'writes_data_h{horizon}.pkl')
        
        # Check for cached processed sequences
        if os.path.exists(writes_cache_file):
            print(f"Loading cached sequences from {writes_cache_file}...")
            with open(writes_cache_file, 'rb') as f:
                cached = pickle.load(f)
            X_train_scaled = cached['X_train_scaled']
            X_val_scaled = cached['X_val_scaled']
            y_train_scaled = cached['y_train_scaled']
            y_val_scaled = cached['y_val_scaled']
            scaler_X = cached['scaler_X']
            scaler_y = cached['scaler_y']
            feature_cols = cached['feature_cols']
            print("Cached sequences loaded!")
        else:
            # Prepare data
            X_train, y_train, feature_cols = prepare_data_writes(train_df, horizon=horizon)
            X_val, y_val, _ = prepare_data_writes(val_df, horizon=horizon)
            
            # Normalize
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled = scaler_y.transform(y_val)
            
            # Cache sequences
            print(f"Caching sequences to {writes_cache_file}...")
            with open(writes_cache_file, 'wb') as f:
                pickle.dump({
                    'X_train_scaled': X_train_scaled,
                    'X_val_scaled': X_val_scaled,
                    'y_train_scaled': y_train_scaled,
                    'y_val_scaled': y_val_scaled,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y,
                    'feature_cols': feature_cols
                }, f)
            print("Sequences cached!")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
        
        # Create model
        model = OptimizedLSTM(input_size=len(feature_cols), hidden_size=128, num_layers=2, dropout=0.2)
        model = model.to(device)
        
        # Train
        model, val_loss, history = train_model(model, train_loader, val_loader, scaler_y, epochs=100, patience=7)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols,
            'horizon': horizon,
            'training_history': history
        }, f'writes_model_h{horizon}.pth')
        
        results[f'writes_h{horizon}'] = {
            'val_loss': val_loss,
            'val_mae': history['val_mae'][-1],
            'val_rmse': history['val_rmse'][-1],
            'val_mape': history['val_mape'][-1]
        }
        all_histories[f'writes_h{horizon}'] = history
        print(f"\nModel saved: writes_model_h{horizon}.pth")
        print(f"Final Metrics - MAE: {history['val_mae'][-1]:.2f}, RMSE: {history['val_rmse'][-1]:.2f}, MAPE: {history['val_mape'][-1]:.2f}%")
    
    # Summary
    print("\n" + "="*140)
    print("TRAINING SUMMARY")
    print("="*140)
    print(f"{'Model':<20} {'Val Loss':<15} {'MAE':<15} {'RMSE':<15} {'MAPE':<15}")
    print("-"*140)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['val_loss']:<15.4f} {metrics['val_mae']:<15.2f} {metrics['val_rmse']:<15.2f} {metrics['val_mape']:<15.2f}%")
    
    # Determine best horizons
    best_reads = min([('8h', results['reads_h8']), ('12h', results['reads_h12'])], key=lambda x: x[1]['val_mae'])
    best_writes = min([('8h', results['writes_h8']), ('12h', results['writes_h12'])], key=lambda x: x[1]['val_mae'])
    
    print(f"\n{'='*140}")
    print(f"Best Reads Model: {best_reads[0]} (MAE: {best_reads[1]['val_mae']:.2f}, MAPE: {best_reads[1]['val_mape']:.2f}%)")
    print(f"Best Writes Model: {best_writes[0]} (MAE: {best_writes[1]['val_mae']:.2f}, MAPE: {best_writes[1]['val_mape']:.2f}%)")
    
    # Save summary
    with open('training_summary.json', 'w') as f:
        json.dump({
            'results': results,
            'best_reads': best_reads[0],
            'best_writes': best_writes[0]
        }, f, indent=2)
    print(f"\nTraining summary saved to training_summary.json")

if __name__ == "__main__":
    main()
>>>>>>> a15b950 (MVP)
