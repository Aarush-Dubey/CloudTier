"""
Main entry point for cloud storage workload forecasting pipeline.
Handles training, evaluation, and inference.
"""

import argparse
import os
import sys


def train_pipeline(csv_path: str, save_dir: str = 'checkpoints'):
    """Run the complete training pipeline."""
    print("\n" + "="*70)
    print("CLOUD STORAGE WORKLOAD FORECASTING - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    from data_pre import DataPreprocessor
    from data import create_dataloaders
    from model import LSTMForecaster
    from train import train_model
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load and preprocess data
    print("Step 1: Data Preprocessing")
    print("-" * 70)
    preprocessor = DataPreprocessor(window_size=24, forecast_horizon=2)
    
    df = preprocessor.load_and_prepare_data(csv_path)
    df_encoded = preprocessor.encode_features(df)
    
    # 2. Create train/val/test splits
    print("\nStep 2: Data Splitting")
    print("-" * 70)
    train_df, val_df, test_df = preprocessor.create_splits(df_encoded)
    
    # 3. Create sequences
    print("\nStep 3: Sequence Creation")
    print("-" * 70)
    X_train, y_train = preprocessor.create_sequences(train_df, fit_scaler=True)
    X_val, y_val = preprocessor.create_sequences(val_df, fit_scaler=False)
    X_test, y_test = preprocessor.create_sequences(test_df, fit_scaler=False)
    
    # Save scaler
    preprocessor.save_scaler(os.path.join(save_dir, 'scaler.pkl'))
    
    # 4. Create dataloaders
    print("\nStep 4: DataLoader Creation")
    print("-" * 70)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=256
    )
    
    # 5. Initialize model
    print("\nStep 5: Model Initialization")
    print("-" * 70)
    input_size = X_train.shape[2]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_targets=3
    )
    
    print(f"Model architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: 128")
    print(f"  Number of layers: 2")
    print(f"  Dropout: 0.2")
    print(f"  Output targets: 3 (reads, writes, egress)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Train model
    print("\nStep 6: Model Training")
    print("-" * 70)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=5,
        save_dir=save_dir
    )
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nCheckpoint saved to: {os.path.join(save_dir, 'lstm_forecaster.pt')}")
    print(f"Scaler saved to: {os.path.join(save_dir, 'scaler.pkl')}")
    print(f"Training plots saved to: {save_dir}/")
    print("\nYou can now run inference with: python main.py --mode inference")


def inference_pipeline(csv_path: str, model_path: str, scaler_path: str,
                      dataset_id: str = 'ds_000000', 
                      timestamp: str = '2025-01-25 12:00:00'):
    """Run inference at a specific timestamp."""
    print("\n" + "="*70)
    print("CLOUD STORAGE WORKLOAD FORECASTING - INFERENCE")
    print("="*70 + "\n")
    
    from test import TimeSeriesInference
    
    # Initialize inference engine
    inference = TimeSeriesInference(
        model_path=model_path,
        scaler_path=scaler_path,
        csv_path=csv_path,
        window_size=24,
        forecast_horizon=2
    )
    
    # Run prediction
    result = inference.predict_at_timestamp(
        dataset_id=dataset_id,
        target_timestamp=timestamp,
        verbose=True
    )
    
    # Visualize
    if result['actuals'] is not None:
        from utils import plot_forecast_example
        plot_forecast_example(
            result['window_data'][inference.preprocessor.feature_columns].values,
            result['actuals'],
            result['predictions'],
            target_names=['Reads', 'Writes', 'Egress'],
            save_path='forecast_example.png'
        )
        print(f"\nVisualization saved to: forecast_example.png")


def evaluate_pipeline(csv_path: str, model_path: str, scaler_path: str,
                     n_datasets: int = 5, n_timestamps: int = 10):
    """Run comprehensive evaluation on test set."""
    print("\n" + "="*70)
    print("CLOUD STORAGE WORKLOAD FORECASTING - EVALUATION")
    print("="*70 + "\n")
    
    from test import TimeSeriesInference
    
    # Initialize inference engine
    inference = TimeSeriesInference(
        model_path=model_path,
        scaler_path=scaler_path,
        csv_path=csv_path,
        window_size=24,
        forecast_horizon=2
    )
    
    # Run batch evaluation
    results_df = inference.evaluate_multiple_timestamps(
        dataset_ids=None,  # Random sample
        n_timestamps=n_timestamps,
        save_path='test_results.csv'
    )
    
    if len(results_df) > 0:
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nDetailed results saved to: test_results.csv")
        print(f"Total predictions evaluated: {len(results_df) // 3}")  # 3 targets per prediction


def main():
    parser = argparse.ArgumentParser(
        description='LSTM-based Cloud Storage Workload Forecaster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py --mode train --csv workload_data.csv
  
  # Run inference at a specific timestamp
  python main.py --mode inference --dataset ds_000000 --timestamp "2025-01-25 12:00:00"
  
  # Evaluate on test set
  python main.py --mode evaluate --n_timestamps 20
        """
    )
    
    parser.add_argument('--mode', type=str, default = "train",
                       choices=['train', 'inference', 'evaluate'],
                       help='Operation mode')
    
    parser.add_argument('--csv', type=str, default='model/data/aggregated.csv',
                       help='Path to CSV data file')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save/load checkpoints')
    
    parser.add_argument('--model', type=str, default='checkpoints/lstm_forecaster.pt',
                       help='Path to model checkpoint')
    
    parser.add_argument('--scaler', type=str, default='checkpoints/scaler.pkl',
                       help='Path to scaler file')
    
    parser.add_argument('--dataset', type=str, default='ds_000000',
                       help='Dataset ID for inference')
    
    parser.add_argument('--timestamp', type=str, default='2025-01-25 12:00:00',
                       help='Target timestamp for inference')
    
    parser.add_argument('--n_timestamps', type=int, default=10,
                       help='Number of timestamps to evaluate per dataset')
    
    args = parser.parse_args()
    
    # Run the appropriate pipeline
    try:
        if args.mode == 'train':
            if not os.path.exists(args.csv):
                print(f"Error: CSV file not found at {args.csv}")
                sys.exit(1)
            train_pipeline(args.csv, args.save_dir)
        
        elif args.mode == 'inference':
            if not os.path.exists(args.model):
                print(f"Error: Model not found at {args.model}")
                print("Please run training first: python main.py --mode train")
                sys.exit(1)
            inference_pipeline(args.csv, args.model, args.scaler,
                             args.dataset, args.timestamp)
        
        elif args.mode == 'evaluate':
            if not os.path.exists(args.model):
                print(f"Error: Model not found at {args.model}")
                print("Please run training first: python main.py --mode train")
                sys.exit(1)
            evaluate_pipeline(args.csv, args.model, args.scaler,
                            n_timestamps=args.n_timestamps)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()