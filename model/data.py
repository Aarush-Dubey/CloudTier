"""
PyTorch Dataset class for sequence data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    """Dataset for time-series sequences with multi-output targets."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            y: Target values of shape (n_samples, n_targets)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        assert len(self.X) == len(self.y), "X and y must have same length"
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            input_seq: Tensor of shape (window_size, n_features)
            target_vec: Tensor of shape (n_targets,)
        """
        return self.X[idx], self.y[idx]


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=256, num_workers=2):
    """Create train and validation dataloaders."""
    
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    X = np.random.randn(1000, 24, 11).astype(np.float32)
    y = np.random.randn(1000, 3).astype(np.float32)
    
    dataset = SequenceDataset(X, y)
    print(f"Dataset size: {len(dataset)}")
    
    sample_x, sample_y = dataset[0]
    print(f"Sample X shape: {sample_x.shape}")
    print(f"Sample y shape: {sample_y.shape}")