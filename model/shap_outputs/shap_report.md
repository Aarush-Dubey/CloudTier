# SHAP Analysis — Existing Synthetic h8 Checkpoints

Scope: existing `model/reads_model_h8.pth` and `model/writes_model_h8.pth`.
Data: synthetic trace from `model.train.generate_synthetic_training_data`.
Meaning: explainer sanity check only, not real predictive evidence.

Synthetic rule: demand is driven by base rate/temperature plus hour-of-day
diurnal shape and day-of-week effect, with recent rolling history carrying
that signal into the forecast window.

## Reads Model

- Checkpoint: `model/reads_model_h8.pth`
- Horizon: 8h
- SHAP samples/background: 16/8
- GradientExplainer nsamples: 20

| Rank | Feature | Mean abs SHAP | Share | Mean signed SHAP |
| ---: | --- | ---: | ---: | ---: |
| 1 | `hour_of_day` | 0.003046 | 36.4% | -0.001037 |
| 2 | `reads_1h` | 0.001461 | 17.5% | 0.000132 |
| 3 | `day_of_week` | 0.001088 | 13.0% | -0.000265 |
| 4 | `bytes_read_1h` | 0.000777 | 9.3% | 0.000477 |
| 5 | `reads_48h` | 0.000716 | 8.6% | -0.000248 |
| 6 | `reads_24h` | 0.000368 | 4.4% | -0.000360 |
| 7 | `reads_96h` | 0.000364 | 4.4% | 0.000091 |
| 8 | `reads_6h` | 0.000360 | 4.3% | 0.000166 |
| 9 | `reads_12h` | 0.000142 | 1.7% | 0.000080 |
| 10 | `access_freq_24h` | 0.000047 | 0.6% | -0.000007 |

## Writes Model

- Checkpoint: `model/writes_model_h8.pth`
- Horizon: 8h
- SHAP samples/background: 16/8
- GradientExplainer nsamples: 20

| Rank | Feature | Mean abs SHAP | Share | Mean signed SHAP |
| ---: | --- | ---: | ---: | ---: |
| 1 | `hour_of_day` | 0.008611 | 25.9% | -0.002357 |
| 2 | `writes_24h` | 0.007341 | 22.1% | -0.007277 |
| 3 | `writes_48h` | 0.006942 | 20.9% | 0.006942 |
| 4 | `writes_96h` | 0.002437 | 7.3% | 0.002435 |
| 5 | `day_of_week` | 0.002154 | 6.5% | 0.002151 |
| 6 | `writes_1h` | 0.002057 | 6.2% | -0.000208 |
| 7 | `bytes_read_1h` | 0.001742 | 5.2% | 0.000075 |
| 8 | `writes_6h` | 0.001387 | 4.2% | 0.000014 |
| 9 | `writes_12h` | 0.000571 | 1.7% | -0.000258 |
| 10 | `access_freq_24h` | 0.000029 | 0.1% | 0.000010 |
