"""
Test inference script: Load model and predict at specific timestamps.
Compares predictions with actual ground truth values.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from model import LSTMForecaster
from data_pre import DataPreprocessor
from utils import compute_metrics, compute_relative_error, plot_forecast_example


class TimeSeriesInference:
    """Handles inference at specific timestamps."""
    
    def __init__(self, model_path: str, scaler_path: str, csv_path: str,
                 window_size: int = 24, forecast_horizon: int = 2):
        """
        Args:
            model_path: Path to saved model checkpoint
            scaler_path: Path to saved scaler
            csv_path: Path to CSV data
            window_size: Input window size
            forecast_horizon: Prediction horizon
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_names = ['reads_1h', 'writes_1h', 'remote_egress_bytes_1h']
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(window_size, forecast_horizon)
        self.preprocessor.load_scaler(scaler_path)
        
        # Load and prepare data
        print(f"Loading data from {csv_path}...")
        self.df = self.preprocessor.load_and_prepare_data(csv_path)
        self.df_encoded = self.preprocessor.encode_features(self.df)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        input_size = len(self.preprocessor.feature_columns)
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            num_targets=3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best validation loss: {checkpoint['val_loss']:.6f}\n")
    
    def predict_at_timestamp(self, dataset_id: str, target_timestamp: str,
                           verbose: bool = True) -> dict:
        """
        Predict workload at target_timestamp using 24-hour history.
        
        Args:
            dataset_id: Dataset identifier
            target_timestamp: Target timestamp string (e.g., "2025-01-01 12:00:00")
            verbose: Print detailed information
        
        Returns:
            Dictionary containing predictions, actuals, and metrics
        """
        target_dt = pd.to_datetime(target_timestamp)
        
        # Calculate prediction timestamp (target + forecast_horizon)
        prediction_dt = target_dt + timedelta(hours=self.forecast_horizon)
        
        if verbose:
            print(f"{'='*60}")
            print(f"Inference for dataset: {dataset_id}")
            print(f"Target window end: {target_dt}")
            print(f"Prediction timestamp: {prediction_dt}")
            print(f"{'='*60}\n")
        
        # Get data for this dataset
        dataset_data = self.df_encoded[
            self.df_encoded['dataset_id'] == dataset_id
        ].sort_values('timestamp').reset_index(drop=True)
        
        if len(dataset_data) == 0:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Find the target timestamp
        target_idx = dataset_data[dataset_data['timestamp'] == target_dt].index
        
        if len(target_idx) == 0:
            raise ValueError(f"Timestamp {target_dt} not found for dataset {dataset_id}")
        
        target_idx = target_idx[0]
        
        # Check if we have enough history
        if target_idx < self.window_size - 1:
            raise ValueError(f"Not enough history (need {self.window_size} hours)")
        
        # Extract input window: 24 hours ending at target_timestamp
        window_start_idx = target_idx - self.window_size + 1
        window_end_idx = target_idx + 1
        
        window_data = dataset_data.iloc[window_start_idx:window_end_idx]
        
        if verbose:
            print(f"Input window: {window_data['timestamp'].min()} to {window_data['timestamp'].max()}")
            print(f"Window has {len(window_data)} hours of data\n")
        
        # Extract features
        X_window = window_data[self.preprocessor.feature_columns].values
        
        # Normalize using fitted scaler
        X_normalized = self.preprocessor.scaler.transform(X_window)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_normalized).unsqueeze(0)  # (1, 24, F)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.squeeze(0).numpy()  # (3,)
        
        # Get actual values at prediction timestamp
        actual_idx = dataset_data[dataset_data['timestamp'] == prediction_dt].index
        
        if len(actual_idx) == 0:
            print(f"Warning: No actual data found at {prediction_dt}")
            actuals = None
        else:
            actual_idx = actual_idx[0]
            actuals = dataset_data.iloc[actual_idx][self.target_names].values
        
        # Print results
        if verbose:
            print("PREDICTIONS vs ACTUALS:")
            print(f"{'Target':<25} {'Predicted':>15} {'Actual':>15} {'Error':>15} {'Rel Error %':>15}")
            print("-" * 90)
            
            if actuals is not None:
                for i, name in enumerate(self.target_names):
                    pred = predictions[i]
                    actual = actuals[i]
                    error = pred - actual
                    rel_error = (abs(error) / actual * 100) if actual > 0 else 0
                    
                    print(f"{name:<25} {pred:>15.2f} {actual:>15.2f} {error:>15.2f} {rel_error:>14.2f}%")
                
                print("\nMETRICS:")
                mae = np.mean(np.abs(predictions - actuals))
                rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
                mape = np.mean(np.abs((predictions - actuals) / actuals) * 100)
                
                print(f"  MAE:  {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  MAPE: {mape:.2f}%\n")
            else:
                for i, name in enumerate(self.target_names):
                    print(f"{name:<25} {predictions[i]:>15.2f} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
                print()
        
        return {
            'dataset_id': dataset_id,
            'target_timestamp': target_dt,
            'prediction_timestamp': prediction_dt,
            'predictions': predictions,
            'actuals': actuals,
            'window_data': window_data
        }
    
    def evaluate_multiple_timestamps(self, dataset_ids: list = None, 
                                    n_timestamps: int = 10,
                                    save_path: str = 'test_results.csv') -> pd.DataFrame:
        """
        Evaluate model on multiple timestamps and datasets.
        
        Args:
            dataset_ids: List of dataset IDs to test (None = random sample)
            n_timestamps: Number of timestamps to test per dataset
            save_path: Path to save results CSV
        
        Returns:
            DataFrame with all predictions and actuals
        """
        print(f"\n{'='*60}")
        print("BATCH INFERENCE EVALUATION")
        print(f"{'='*60}\n")
        
        if dataset_ids is None:
            # Sample random datasets from test set
            test_datasets = self.df_encoded['dataset_id'].unique()
            dataset_ids = np.random.choice(test_datasets, 
                                          size=min(5, len(test_datasets)), 
                                          replace=False)
        
        all_predictions = []
        all_actuals = []
        results_data = []
        
        for dataset_id in dataset_ids:
            dataset_data = self.df_encoded[
                self.df_encoded['dataset_id'] == dataset_id
            ].sort_values('timestamp')
            
            # Get valid timestamps (with enough history and future data)
            valid_timestamps = dataset_data.iloc[
                self.window_size-1:-self.forecast_horizon
            ]['timestamp'].values
            
            # Sample timestamps
            if len(valid_timestamps) > n_timestamps:
                sample_timestamps = np.random.choice(valid_timestamps, 
                                                    n_timestamps, 
                                                    replace=False)
            else:
                sample_timestamps = valid_timestamps
            
            print(f"Testing dataset: {dataset_id} ({len(sample_timestamps)} timestamps)")
            
            for timestamp in sample_timestamps:
                try:
                    result = self.predict_at_timestamp(
                        dataset_id, 
                        str(timestamp), 
                        verbose=False
                    )
                    
                    if result['actuals'] is not None:
                        all_predictions.append(result['predictions'])
                        all_actuals.append(result['actuals'])
                        
                        # Store detailed results
                        for i, target in enumerate(self.target_names):
                            results_data.append({
                                'dataset_id': dataset_id,
                                'target_timestamp': result['target_timestamp'],
                                'prediction_timestamp': result['prediction_timestamp'],
                                'target': target,
                                'predicted': result['predictions'][i],
                                'actual': result['actuals'][i],
                                'error': result['predictions'][i] - result['actuals'][i],
                                'abs_error': abs(result['predictions'][i] - result['actuals'][i]),
                                'rel_error_pct': abs(result['predictions'][i] - result['actuals'][i]) / result['actuals'][i] * 100 if result['actuals'][i] > 0 else 0
                            })
                
                except Exception as e:
                    print(f"  Error at {timestamp}: {e}")
                    continue
        
        # Compute overall metrics
        if len(all_predictions) > 0:
            all_predictions = np.array(all_predictions)
            all_actuals = np.array(all_actuals)
            
            print(f"\n{'='*60}")
            print("OVERALL TEST METRICS")
            print(f"{'='*60}")
            print(f"Total predictions: {len(all_predictions)}\n")
            
            metrics = compute_metrics(all_actuals, all_predictions, self.target_names)
            rel_errors = compute_relative_error(all_actuals, all_predictions, self.target_names)
            
            for target in self.target_names:
                print(f"{target}:")
                print(f"  MAE:  {metrics[f'{target}_mae']:.2f}")
                print(f"  RMSE: {metrics[f'{target}_rmse']:.2f}")
                print(f"  MAPE: {rel_errors[f'{target}_mape']:.2f}%\n")
            
            print(f"Overall MAE:  {metrics['overall_mae']:.2f}")
            print(f"Overall RMSE: {metrics['overall_rmse']:.2f}\n")
            
            # Save results
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(save_path, index=False)
            print(f"Saved detailed results to {save_path}")
            
            return results_df
        else:
            print("No valid predictions made")
            return pd.DataFrame()


def main():
    """Main inference function."""
    
    # Paths
    model_path = 'checkpoints/lstm_forecaster.pt'
    scaler_path = 'checkpoints/scaler.pkl'
    csv_path = 'workload_data.csv'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return
    
    # Initialize inference engine
    inference = TimeSeriesInference(
        model_path=model_path,
        scaler_path=scaler_path,
        csv_path=csv_path,
        window_size=24,
        forecast_horizon=2
    )
    
    # Example 1: Single timestamp prediction
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Timestamp Prediction")
    print("="*60 + "\n")
    
    try:
        result = inference.predict_at_timestamp(
            dataset_id='ds_000000',
            target_timestamp='2025-01-25 12:00:00',
            verbose=True
        )
        
        # Visualize this example
        if result['actuals'] is not None:
            plot_forecast_example(
                result['window_data'][inference.preprocessor.feature_columns].values,
                result['actuals'],
                result['predictions'],
                target_names=['Reads', 'Writes', 'Egress'],
                save_path='checkpoints/forecast_example.png'
            )
    
    except Exception as e:
        print(f"Error in single prediction: {e}")
    
    # Example 2: Batch evaluation
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*60 + "\n")
    
    results_df = inference.evaluate_multiple_timestamps(
        dataset_ids=None,  # Random sample
        n_timestamps=10,
        save_path='checkpoints/test_results.csv'
    )
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()