"""Train the CloudTier LSTM demand forecasters (reads and writes, horizon = 8h).

This is the reconciled training script. History: the file previously contained an
unresolved merge conflict between two different training approaches. The retained path
is the single-target `OptimizedLSTM` that actually produced the shipped checkpoints
`reads_model_h8.pth` / `writes_model_h8.pth`; the other branch used a different
multi-target architecture and depended on modules that were never committed.

Two defects from that branch are fixed here:
  * Horizon mismatch — the reads model was trained at horizon 12 while the shipped
    artifact is `h8`. Both reads and writes now train at horizon 8 so the code, the
    checkpoint names, and the README agree. (Configurable via --horizon.)
  * The summary crashed with a KeyError because it compared `reads_h8`/`reads_h12`
    while only one horizon was trained. The summary now reports exactly what was trained.

The forecaster feeds the same feature shape the heuristic uses (recent/prior windows,
trend ratio, size, time-of-day, day-of-week), so it is a drop-in comparison for the
`shared/pricing.py:forecast_access` heuristic.

Data: pass a prepared CSV via --csv, or use --synthetic to generate a self-contained
synthetic trace so the script runs end-to-end with no external file. The synthetic
trace is for wiring/validation only; real Wikipedia-pageview training is Task 2.2. The
synthetic generator encodes a KNOWN rule (next-window demand tracks the recent window
scaled by a diurnal factor) which Task 4.5 uses to validate the SHAP explainer.

Usage:
    python -m model.train --synthetic --epochs 3 --out-dir /tmp/cloudtier-model   # smoke
    python -m model.train --csv training_data.csv                                 # real
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

LOOKBACK = 24
READS_FEATURES = [
    "reads_1h", "bytes_read_1h", "size_gb", "hour_of_day", "day_of_week",
    "reads_6h", "reads_12h", "reads_24h", "reads_48h", "reads_96h",
    "reads_trend_6h", "data_temperature_encoded", "reads_access_freq_24h",
]
WRITES_FEATURES = [
    "writes_1h", "bytes_read_1h", "size_gb", "hour_of_day", "day_of_week",
    "writes_6h", "writes_12h", "writes_24h", "writes_48h", "writes_96h",
    "writes_trend_6h", "data_temperature_encoded", "writes_access_freq_24h",
]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class OptimizedLSTM(nn.Module):
    """Single-target LSTM regressor: sequence of features -> scalar horizon demand."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last timestep
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)


# --- data ------------------------------------------------------------------

def _add_rolling_features(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add the recent/prior-window and trend features the heuristic also uses."""
    source_col = f"{metric}_1h"
    g = df.groupby("dataset_id")[source_col]
    for w in (6, 12, 24, 48, 96):
        df[f"{metric}_{w}h"] = g.transform(lambda s: s.rolling(w, min_periods=1).sum())
    recent = g.transform(lambda s: s.rolling(6, min_periods=1).mean())
    prior = g.transform(lambda s: s.shift(6).rolling(6, min_periods=1).mean())
    df[f"{metric}_trend_6h"] = (recent / prior.replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)
    df[f"{metric}_access_freq_24h"] = g.transform(lambda s: s.rolling(24, min_periods=1).mean())
    return df


def generate_synthetic_training_data(n_datasets=8, n_hours=180, seed=42) -> pd.DataFrame:
    """Self-contained synthetic trace with a KNOWN demand rule (see Task 4.5).

    Each dataset has a base rate and a diurnal multiplier; next-hour demand is the base
    rate times the hour-of-day factor plus noise. That rule is what SHAP should recover
    on the synthetic-trained model, validating the explainer before it is trusted on
    real data.
    """
    rng = np.random.default_rng(seed)
    temps = np.array(["hot", "warm", "cold"])
    rows = []
    for d in range(n_datasets):
        temp = temps[d % 3]
        base_reads = {"hot": 5000, "warm": 800, "cold": 20}[temp] * rng.uniform(0.6, 1.4)
        base_writes = base_reads * rng.uniform(0.05, 0.2)
        size_gb = rng.uniform(10, 5000)
        for h in range(n_hours):
            hour_of_day = h % 24
            day_of_week = (h // 24) % 7
            diurnal = 0.4 + 0.6 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi) ** 2
            weekday = 1.0 if day_of_week < 5 else 0.5
            reads = max(0, base_reads * diurnal * weekday * rng.uniform(0.8, 1.2))
            writes = max(0, base_writes * diurnal * rng.uniform(0.8, 1.2))
            rows.append(
                {
                    "dataset_id": f"ds_{d:04d}",
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=h),
                    "reads_1h": reads,
                    "writes_1h": writes,
                    "bytes_read_1h": reads * 1024.0,
                    "size_gb": size_gb,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "data_temperature": temp,
                }
            )
    df = pd.DataFrame(rows)
    df = _add_rolling_features(df, "reads")
    df = _add_rolling_features(df, "writes")
    return df


def split_by_dataset_time(df: pd.DataFrame, train_fraction: float = 0.8):
    train_parts, val_parts = [], []
    for _, group in df.sort_values(["dataset_id", "timestamp"]).groupby("dataset_id"):
        split_at = int(train_fraction * len(group))
        train_parts.append(group.iloc[:split_at])
        val_parts.append(group.iloc[split_at:])
    train_df = pd.concat(train_parts).sort_values(["dataset_id", "timestamp"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(["dataset_id", "timestamp"]).reset_index(drop=True)
    return train_df, val_df


def load_or_prepare_data(csv_path: str | None, synthetic: bool, synthetic_datasets: int, synthetic_hours: int):
    if synthetic or not csv_path or not os.path.exists(csv_path):
        if not synthetic and csv_path:
            print(f"{csv_path} not found; generating synthetic data (use real data in Task 2.2).")
        df = generate_synthetic_training_data(n_datasets=synthetic_datasets, n_hours=synthetic_hours)
    else:
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.fillna(0)
    for metric in ("reads", "writes"):
        if f"{metric}_96h" not in df.columns or f"{metric}_trend_6h" not in df.columns:
            df = _add_rolling_features(df.sort_values(["dataset_id", "timestamp"]), metric)
    df["data_temperature_encoded"] = df["data_temperature"].map({"hot": 2, "warm": 1, "cold": 0}).fillna(0)
    # Per-dataset time-ordered split with no sequence crossing the split boundary.
    df = df.sort_values(["dataset_id", "timestamp"]).reset_index(drop=True)
    train_df, val_df = split_by_dataset_time(df)
    return train_df, val_df, df


def prepare_sequences(df, feature_cols, target_col, horizon, lookback=LOOKBACK):
    X, y = [], []
    for _, group in df.groupby("dataset_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        vals = group[feature_cols].values
        target = group[target_col].values
        for i in range(len(group) - lookback - horizon + 1):
            X.append(vals[i : i + lookback])
            y.append(target[i + lookback : i + lookback + horizon].sum())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


# --- train / eval ----------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mean_actual = float(np.mean(np.abs(y_true)))
    mape = (mae / mean_actual * 100) if mean_actual > 0 else 0.0
    return mae, rmse, mape


def evaluate_model(model, loader, scaler_y, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch.to(device)).cpu().numpy()
            preds.append(scaler_y.inverse_transform(out))
            targets.append(scaler_y.inverse_transform(y_batch.numpy()))
    return calculate_metrics(np.vstack(targets), np.vstack(preds))


def train_model(model, train_loader, val_loader, scaler_y, device, epochs=100, patience=7, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_val_loss, patience_counter, best_state = float("inf"), 0, None
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_mape": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch.to(device)), y_batch.to(device)).item()
        val_loss /= max(1, len(val_loader))
        val_mae, val_rmse, val_mape = evaluate_model(model, val_loader, scaler_y, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_mape"].append(val_mape)
        print(f"epoch {epoch + 1:>3} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"MAE {val_mae:.2f} | RMSE {val_rmse:.2f} | MAPE {val_mape:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter, best_state = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss, history


def train_target(name, feature_cols, target_col, train_df, val_df, horizon, device, epochs, out_dir):
    print(f"\n=== training {name} model (horizon={horizon}h) ===")
    X_train, y_train = prepare_sequences(train_df, feature_cols, target_col, horizon)
    X_val, y_val = prepare_sequences(val_df, feature_cols, target_col, horizon)
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f"not enough data to build sequences for {name} (need > lookback+horizon per dataset)")

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)

    train_loader = DataLoader(TimeSeriesDataset(X_train_s, y_train_s), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val_s, y_val_s), batch_size=64, shuffle=False)

    model = OptimizedLSTM(input_size=len(feature_cols)).to(device)
    model, val_loss, history = train_model(model, train_loader, val_loader, scaler_y, device, epochs=epochs)
    val_mae, val_rmse, val_mape = evaluate_model(model, val_loader, scaler_y, device)

    out_path = os.path.join(out_dir, f"{name}_model_h{horizon}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "feature_cols": feature_cols,
            "horizon": horizon,
            "training_history": history,
        },
        out_path,
    )
    print(f"saved {out_path}  (val MAE {val_mae:.2f}, RMSE {val_rmse:.2f})")
    return {"val_loss": val_loss, "val_mae": val_mae, "val_rmse": val_rmse, "val_mape": val_mape}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CloudTier LSTM forecasters")
    parser.add_argument("--csv", default="training_data.csv", help="prepared training CSV")
    parser.add_argument("--synthetic", action="store_true", help="ignore --csv and use a synthetic trace")
    parser.add_argument("--synthetic-datasets", type=int, default=8)
    parser.add_argument("--synthetic-hours", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=8, help="forecast horizon in hours (shipped artifacts are h8)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out-dir", default=".", help="where to write *_model_h<H>.pth (default: repo model dir)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, val_df, _ = load_or_prepare_data(
        args.csv, args.synthetic, args.synthetic_datasets, args.synthetic_hours
    )
    results = {
        f"reads_h{args.horizon}": train_target(
            "reads", READS_FEATURES, "reads_1h", train_df, val_df, args.horizon, device, args.epochs, args.out_dir
        ),
        f"writes_h{args.horizon}": train_target(
            "writes", WRITES_FEATURES, "writes_1h", train_df, val_df, args.horizon, device, args.epochs, args.out_dir
        ),
    }

    print("\n=== summary ===")
    for name, m in results.items():
        print(f"{name:<12} val_loss {m['val_loss']:.4f} | MAE {m['val_mae']:.2f} | RMSE {m['val_rmse']:.2f} | MAPE {m['val_mape']:.1f}%")
    with open(os.path.join(args.out_dir, "training_summary.json"), "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
