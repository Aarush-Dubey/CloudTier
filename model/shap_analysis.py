"""Run SHAP analysis for the shipped CloudTier LSTM demand checkpoints.

This uses the existing synthetic-trained h8 checkpoints only. That makes the result a
validation of the explainer plumbing against known synthetic structure, not evidence
of real-world predictive power.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import shap
import torch

from model.train import OptimizedLSTM, generate_synthetic_training_data, prepare_sequences

warnings.filterwarnings("ignore")


def _alias_legacy_features(df, target):
    """Existing h8 checkpoints used legacy shared trend/frequency feature names."""
    df = df.copy()
    if "trend_6h" in df.columns and "access_freq_24h" in df.columns:
        return df
    df["trend_6h"] = df[f"{target}_trend_6h"]
    df["access_freq_24h"] = df[f"{target}_access_freq_24h"]
    return df


def _load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    model = OptimizedLSTM(input_size=len(feature_cols)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _scaled_sequences(df, checkpoint, target_col):
    feature_cols = checkpoint["feature_cols"]
    horizon = checkpoint["horizon"]
    X, _ = prepare_sequences(df, feature_cols, target_col, horizon)
    scaler = checkpoint["scaler_X"]
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return X_scaled.astype(np.float32)


def _summarize_shap(values, feature_cols):
    arr = np.asarray(values)
    if arr.ndim == 4:
        arr = arr[..., 0]
    mean_abs = np.abs(arr).mean(axis=(0, 1))
    mean_signed = arr.mean(axis=(0, 1))
    total = float(mean_abs.sum()) or 1.0
    rows = []
    for name, abs_value, signed_value in zip(feature_cols, mean_abs, mean_signed):
        rows.append(
            {
                "feature": name,
                "mean_abs_shap": float(abs_value),
                "share": float(abs_value / total),
                "mean_signed_shap": float(signed_value),
            }
        )
    return sorted(rows, key=lambda row: row["mean_abs_shap"], reverse=True)


def analyze_target(target, checkpoint_path, df, samples, background, nsamples, device):
    model, checkpoint = _load_model(checkpoint_path, device)
    target_df = _alias_legacy_features(df, target)
    X = _scaled_sequences(target_df, checkpoint, f"{target}_1h")
    if len(X) < background + samples:
        raise RuntimeError(f"Need at least {background + samples} sequences for {target}; got {len(X)}")

    background_x = torch.tensor(X[:background], dtype=torch.float32, device=device)
    sample_x = torch.tensor(X[background : background + samples], dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(model, background_x)
    shap_values = explainer.shap_values(sample_x, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    ranked = _summarize_shap(shap_values, checkpoint["feature_cols"])
    return {
        "checkpoint": str(checkpoint_path),
        "horizon": checkpoint["horizon"],
        "samples": samples,
        "background": background,
        "nsamples": nsamples,
        "top_features": ranked,
    }


def write_report(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shap_summary.json"
    md_path = out_dir / "shap_report.md"

    with json_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    lines = [
        "# SHAP Analysis — Existing Synthetic h8 Checkpoints",
        "",
        "Scope: existing `model/reads_model_h8.pth` and `model/writes_model_h8.pth`.",
        "Data: synthetic trace from `model.train.generate_synthetic_training_data`.",
        "Meaning: explainer sanity check only, not real predictive evidence.",
        "",
        "Synthetic rule: demand is driven by base rate/temperature plus hour-of-day",
        "diurnal shape and day-of-week effect, with recent rolling history carrying",
        "that signal into the forecast window.",
        "",
    ]
    for target, result in results.items():
        lines.extend(
            [
                f"## {target.title()} Model",
                "",
                f"- Checkpoint: `{result['checkpoint']}`",
                f"- Horizon: {result['horizon']}h",
                f"- SHAP samples/background: {result['samples']}/{result['background']}",
                f"- GradientExplainer nsamples: {result['nsamples']}",
                "",
                "| Rank | Feature | Mean abs SHAP | Share | Mean signed SHAP |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(result["top_features"][:10], start=1):
            lines.append(
                f"| {rank} | `{row['feature']}` | {row['mean_abs_shap']:.6f} | "
                f"{row['share']:.1%} | {row['mean_signed_shap']:.6f} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for CloudTier h8 LSTM checkpoints")
    parser.add_argument("--out-dir", default="model/shap_outputs")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--background", type=int, default=32)
    parser.add_argument("--nsamples", type=int, default=20)
    parser.add_argument("--synthetic-datasets", type=int, default=12)
    parser.add_argument("--synthetic-hours", type=int, default=240)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = generate_synthetic_training_data(
        n_datasets=args.synthetic_datasets,
        n_hours=args.synthetic_hours,
        seed=42,
    )
    df["data_temperature_encoded"] = df["data_temperature"].map({"hot": 2, "warm": 1, "cold": 0}).fillna(0)

    results = {}
    for target in ("reads", "writes"):
        checkpoint = Path("model") / f"{target}_model_h8.pth"
        results[target] = analyze_target(
            target, checkpoint, df, args.samples, args.background, args.nsamples, device
        )

    json_path, md_path = write_report(results, Path(args.out_dir))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for target, result in results.items():
        top = ", ".join(row["feature"] for row in result["top_features"][:5])
        print(f"{target}: {top}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
