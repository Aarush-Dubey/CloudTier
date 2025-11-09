"""
LSTM-based multi-output forecasting model.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM encoder with separate MLP heads for each target variable.
    Predicts reads, writes, and egress for cloud storage workloads.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, num_targets: int = 3):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_targets: Number of prediction targets (default: 3)
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_targets = num_targets
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Separate MLP heads for each target
        self.head_reads = self._create_mlp_head(hidden_size)
        self.head_writes = self._create_mlp_head(hidden_size)
        self.head_egress = self._create_mlp_head(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _create_mlp_head(self, hidden_size: int) -> nn.Module:
        """Create a 2-layer MLP head for one target."""
        return nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            predictions: Tensor of shape (batch_size, num_targets)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state from the top layer
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)
        
        # Pass through each head
        pred_reads = self.head_reads(last_hidden)    # (batch_size, 1)
        pred_writes = self.head_writes(last_hidden)  # (batch_size, 1)
        pred_egress = self.head_egress(last_hidden)  # (batch_size, 1)
        
        # Concatenate predictions
        predictions = torch.cat([pred_reads, pred_writes, pred_egress], dim=1)
        
        return predictions  # Shape: (batch_size, 3)


class HuberLoss(nn.Module):
    """Sum of Huber losses for multi-output regression."""
    
    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions of shape (batch_size, num_targets)
            target: Ground truth of shape (batch_size, num_targets)
        
        Returns:
            Total loss (sum of individual Huber losses)
        """
        total_loss = 0
        for i in range(pred.size(1)):
            loss_i = self.huber(pred[:, i], target[:, i])
            total_loss += loss_i
        return total_loss


if __name__ == "__main__":
    # Test model
    batch_size = 32
    seq_len = 24
    input_size = 11
    
    model = LSTMForecaster(input_size=input_size, hidden_size=128, num_layers=2)
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    predictions = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss
    criterion = HuberLoss()
    target = torch.randn(batch_size, 3)
    loss = criterion(predictions, target)
    print(f"Loss: {loss.item():.4f}")