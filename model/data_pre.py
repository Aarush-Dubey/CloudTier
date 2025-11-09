"""
Data preprocessing module for cloud storage workload forecasting.
Handles loading, encoding, normalization, and sequence windowing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import pickle


class DataPreprocessor:
    """Preprocesses telemetry data for LSTM training."""
    
    def __init__(self, window_size: int = 24, forecast_horizon: int = 2):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = ['reads_1h', 'writes_1h', 'remote_egress_bytes_1h']
        
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and perform initial preprocessing."""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by dataset_id and timestamp
        df = df.sort_values(['dataset_id', 'timestamp']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} rows, {df['dataset_id'].nunique()} unique datasets")
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding and transformations to features."""
        df = df.copy()
        
        # Encode data_temperature: cold=0, warm=1, hot=2
        temp_mapping = {'cold': 0, 'warm': 1, 'hot': 2}
        df['data_temperature'] = df['data_temperature'].map(temp_mapping)
        
        # Apply log1p to heavy-tailed columns
        log_cols = ['reads_1h', 'writes_1h', 'bytes_read_1h', 
                    'remote_egress_bytes_1h', 'access_freq_24h', 'trend_6h']
        for col in log_cols:
            df[f'{col}_log'] = np.log1p(df[col])
        
        # Add cyclic time encodings
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Define feature columns (after transformations)
        self.feature_columns = [
            'reads_1h_log', 'writes_1h_log', 'bytes_read_1h_log',
            'remote_egress_bytes_1h_log', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'trend_6h_log', 'data_temperature',
            'access_freq_24h_log'
        ]
        
        print(f"Encoded features. Total feature count: {len(self.feature_columns)}")
        return df
    
    def create_sequences(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for each dataset."""
        sequences_X = []
        sequences_y = []
        
        # Group by dataset
        grouped = df.groupby('dataset_id')
        
        print(f"Creating sequences with window_size={self.window_size}, horizon={self.forecast_horizon}...")
        
        for dataset_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Need at least window_size + forecast_horizon rows
            if len(group) < self.window_size + self.forecast_horizon:
                continue
            
            # Extract features and targets
            features = group[self.feature_columns].values
            
            # Target columns (original, not log-transformed for targets)
            targets = group[self.target_columns].values
            
            # Create sliding windows
            for i in range(len(group) - self.window_size - self.forecast_horizon + 1):
                # Input: window of size 24
                X_window = features[i:i + self.window_size]
                
                # Target: values at t + forecast_horizon
                y_target = targets[i + self.window_size + self.forecast_horizon - 1]
                
                sequences_X.append(X_window)
                sequences_y.append(y_target)
        
        X = np.array(sequences_X, dtype=np.float32)
        y = np.array(sequences_y, dtype=np.float32)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape} (samples, window_size, features)")
        print(f"y shape: {y.shape} (samples, targets)")
        
        # Normalize features
        if fit_scaler:
            # Flatten for fitting scaler
            X_flat = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X_flat)
            print("Fitted scaler on training data")
        
        # Transform
        X_normalized = np.zeros_like(X)
        for i in range(len(X)):
            X_normalized[i] = self.scaler.transform(X[i])
        
        return X_normalized, y
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test based on time."""
        # Get unique timestamps
        timestamps = sorted(df['timestamp'].unique())
        
        # Calculate split points (20/5/5 days)
        n_total = len(timestamps)
        train_end_idx = int(n_total * 20 / 30)
        val_end_idx = int(n_total * 25 / 30)
        
        train_cutoff = timestamps[train_end_idx - 1]
        val_cutoff = timestamps[val_end_idx - 1]
        
        train_df = df[df['timestamp'] <= train_cutoff].copy()
        val_df = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)].copy()
        test_df = df[df['timestamp'] > val_cutoff].copy()
        
        print(f"\nData splits:")
        print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
        print(f"Val:   {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
        print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
        
        return train_df, val_df, test_df
    
    def save_scaler(self, path: str):
        """Save the fitted scaler."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {path}")
    
    def load_scaler(self, path: str):
        """Load a fitted scaler."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Loaded scaler from {path}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    # Assumes CSV exists
    df = preprocessor.load_and_prepare_data("workload_data.csv")
    df_encoded = preprocessor.encode_features(df)
    
    train_df, val_df, test_df = preprocessor.create_splits(df_encoded)
    
    X_train, y_train = preprocessor.create_sequences(train_df, fit_scaler=True)
    X_val, y_val = preprocessor.create_sequences(val_df, fit_scaler=False)
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")