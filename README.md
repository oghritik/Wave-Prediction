# 🌊 Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU

> **NeurIPS 2026 Submission**  
> *Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU with Sliding Window and Correction Mechanism*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-orange?logo=pytorch)](https://lightning.ai/)
[![Data: CMEMS](https://img.shields.io/badge/Data-CMEMS%20IBI-0077B6)](https://marine.copernicus.eu/)
[![Bathymetry: GEBCO](https://img.shields.io/badge/Bathymetry-GEBCO%202022-1A6E38)](https://www.gebco.net/)

---

## 📌 Overview

This repository is the official code for our systematic benchmark of spatiotemporal deep learning architectures for **extreme ocean wave forecasting at Nazaré Canyon, Portugal** — Europe's largest submarine canyon and one of the world's most extreme wave environments.

The core question we answer: **does increasing spatial encoder complexity improve extreme wave prediction, or does bidirectional temporal modelling dominate?**

Our finding: **CNN+BiGRU — the simplest spatial encoder — outperforms ASPP and UNet++ variants by up to 23% RMSE**, demonstrating that bidirectional temporal modelling is the decisive factor when bathymetry is explicitly encoded as a static input channel.

> 📁 **The final implementation with all paper results lives in `V4/`.**  
> `V1/`, `V2/`, `V3/` are earlier development iterations kept for reference.

---

## 🏔️ Why Nazaré Canyon?

Nazaré Canyon (39.6°N, 9.1°W) stretches 230 km, reaches 5,000 m depth, and terminates metres from the Portuguese coastline. Incoming North Atlantic swell splits across the deep canyon channel and the shallower shelf, reconverging near shore through constructive interference. The result:

- **Routine winter maxima > 15 m** crest-trough wave heights (VCMX)
- **Documented extremes > 30 m**
- Home of the **WSL Tudor Big Wave Challenge**, attracting tens of thousands of spectators annually

Standard numerical models (WAVEWATCH III, SWAN) and open-ocean deep learning approaches systematically underestimate wave heights here by ignoring or flattening bathymetric structure. This work directly addresses that failure.

---

## ✨ Key Contributions

1. **First systematic benchmark** of spatiotemporal architectures on a bathymetry-critical extreme wave environment — four models trained identically on 35,017 hourly CMEMS IBI steps with explicit GEBCO 2022 bathymetry.

2. **Predict-three-retain-one recursive protocol** with internal drift correction — enables stable 168-hour forecasting, evaluated on a 696-step November 2023 window capturing a **17.18 m extreme event**.

3. **Architectural insight**: CNN+BiGRU (simplest spatial encoder) outperforms ASPP and UNet++ variants by up to 23% RMSE. Bidirectional temporal modelling dominates when bathymetry is explicitly encoded.

4. **Negative design guidance**: Large ASPP dilation rates are counterproductive on restricted coastal grids (72×90 at 0.027°) — receptive fields designed for high-resolution semantic segmentation exceed the domain's effective spatial extent.

---

## 📂 Repository Structure

```
Wave-Prediction/
│
├── V4/                                     # ✅ FINAL — all paper results here
│   │
│   ├── v6.ipynb                            # Architecture run: v6 (lookback 24h)
│   ├── v6_VCMX_only_accuracy.png           #   └─ VCMX accuracy plot
│   ├── v6_VCMX_only_error_evolution.png    #   └─ Error evolution over time
│   ├── v6_VCMX_only_predictions.png        #   └─ Predicted vs actual VCMX
│   ├── v6_sliding_window_30h_accuracy.png  #   └─ 30h sliding window accuracy
│   ├── v6_sliding_window_30h_errors.png    #   └─ 30h sliding window errors
│   ├── v6_sliding_window_30h_predictions.png  #   └─ 30h sliding window predictions
│   ├── v6_spatial_maps_hours_10_20_30.png  #   └─ Spatial wave field at t+10,20,30h
│   ├── v6_training_history_lookback_24.png #   └─ Training loss curve (24h lookback)
│   │
│   ├── v7.ipynb                            # Architecture run: v7 (lookback 12h)
│   ├── v7_VCMX_only_accuracy.png
│   ├── v7_VCMX_only_error_evolution.png
│   ├── v7_VCMX_only_predictions.png
│   ├── v7_sliding_window_30h_accuracy.png
│   ├── v7_sliding_window_30h_errors.png
│   ├── v7_sliding_window_30h_predictions.png
│   ├── v7_spatial_maps_hours_10_20_30.png
│   ├── v7_training_history_lookback_12.png #   └─ Training loss curve (12h lookback)
│   │
│   ├── v8.ipynb                            # Architecture run: v8 (lookback 72h) — final
│   ├── v8_VCMX_only_accuracy.png
│   ├── v8_VCMX_only_error_evolution.png
│   ├── v8_VCMX_only_predictions.png
│   ├── v8_sliding_window_30h_accuracy.png
│   ├── v8_sliding_window_30h_errors.png
│   ├── v8_sliding_window_30h_predictions.png
│   ├── v8_spatial_maps_hours_10_20_30.png
│   └── v8_training_history_lookback_72.png #   └─ Training loss curve (72h lookback)
│
├── V1/                          # 🔧 Early prototype
├── V2/                          # 🔧 Intermediate experiments
├── V3/                          # 🔧 Intermediate experiments
│
├── models/                      # Saved model checkpoints
├── Documents/                   # Paper drafts and supplementary material
├── paper_figures/               # Figures used in the paper
├── result images/               # Additional prediction visualisations
├── lightning_logs/              # PyTorch Lightning training logs
│
├── EDA.ipynb                    # Exploratory data analysis
├── feature_engineering.ipynb    # Feature selection & correlation analysis
├── model_comparision.ipynb      # Early model comparison notebook
├── model_comparison_refactored.ipynb  # Refactored final benchmark
├── ocean_visuals.ipynb          # Bathymetry & wave field visualisations
├── download.py                  # CMEMS data download script
├── research_experiment.py       # Main training script
└── req.txt                      # Python dependencies
```

### V4 Notebook Runs Explained

Each notebook (`v6`, `v7`, `v8`) inside `V4/` represents a complete training and evaluation run with a different **lookback window**, letting us ablate the effect of historical context length:

| Notebook | Lookback | Key output |
|---|---|---|
| `v6.ipynb` | 24 h | Baseline lookback comparison |
| `v7.ipynb` | 12 h | Short-context ablation |
| `v8.ipynb` | 72 h | Extended-context ablation |

Each run produces the same standardised set of 8 output figures:

| Figure suffix | What it shows |
|---|---|
| `_VCMX_only_accuracy.png` | Accuracy metrics for the target variable (VCMX) |
| `_VCMX_only_error_evolution.png` | How prediction error evolves across the forecast horizon |
| `_VCMX_only_predictions.png` | Predicted vs actual VCMX time series |
| `_sliding_window_30h_accuracy.png` | Accuracy under 30-hour recursive sliding window |
| `_sliding_window_30h_errors.png` | Error breakdown for the sliding window run |
| `_sliding_window_30h_predictions.png` | Sliding window predicted vs actual |
| `_spatial_maps_hours_10_20_30.png` | Spatial wave field snapshots at t+10h, t+20h, t+30h |
| `_training_history_lookback_XX.png` | Training & validation loss curves |

---

## 🏗️ Architecture Overview

All four models share the same design contract:

```
[10-channel spatial input at each timestep]
        ↓
[Spatial Encoder]  ← the controlled variable across architectures
        ↓
[Bidirectional ConvGRU temporal backbone]
        ↓
[1×1 Conv output head → 8-variable wave field]
```

**Input tensor:** `(B, 48, 10, 72, 90)` — batch × 48h lookback × 10 channels × 72×90 grid  
**Output:** Full 8-variable wave field for next 3 hours (sequence-to-one)

| Model | Spatial Encoder | Overall RMSE |
|---|---|---|
| ConvLSTM | Fused ConvLSTM cells | 0.3579 m |
| **CNN+BiGRU** ⭐ | Stride-2 CNN (64→32 filters) | **0.2965 m** |
| DeepLabV3+BiGRU | ASPP multi-scale pooling (d=1,6,12) | 0.3174 m |
| UNet+++BiGRU | DW-separable U-Net++ with dense skip connections | 0.3833 m |

> ⭐ Best performer. More parameters ≠ better results at this domain scale.

---

## 📊 Results

### Main Benchmark — November 2023 Extreme Event Window
*(696 steps, VCMX range 5–17 m, 168-hour sliding window with drift correction)*

| Model | Overall RMSE | Overall MAE | VCMX RMSE | VCMX MAE |
|---|---|---|---|---|
| ConvLSTM | 0.3579 m | 0.1613 m | 0.2839 m | 0.2213 m |
| **CNN+BiGRU** | **0.2965 m** | **0.1264 m** | **0.1832 m** | **0.1384 m** |
| DeepLabV3+BiGRU | 0.3174 m | 0.1551 m | 0.2797 m | 0.2035 m |
| UNet+++BiGRU | 0.3833 m | 0.1999 m | 0.3577 m | 0.2763 m |

### Against Baselines

| Model | Lookback | Horizon | VCMX RMSE | VCMX MAE |
|---|---|---|---|---|
| EarthFormer (single-pass) | 12 h | 14 h | 6.7475 m | 6.4283 m |
| Vanilla ConvLSTM (no sliding window) | 48 h | 168 h | 1.1960 m | 0.8211 m |
| **CNN+BiGRU (ours)** | **48 h** | **168 h** | **0.2965 m** | **0.1264 m** |

---

## 🗃️ Dataset

### Dynamic Wave Data — CMEMS IBI Reanalysis
- **Source:** [Copernicus Marine Service — IBI Ocean Wave Reanalysis](https://marine.copernicus.eu/)
- **Period:** January 2020 – December 2023 (35,017 hourly steps)
- **Domain:** 38.5°N–40.5°N, 11.0°W–8.5°W at 0.027° → **72×90 grid**
- **Format:** CF-1.8 compliant NetCDF-4

### Static Bathymetry — GEBCO 2022
- **Source:** [GEBCO 2025 Grid](https://www.gebco.net/)
- **Why:** Native CMEMS reports only ~1,000 m at Nazaré; true depth >5,000 m. GEBCO is bilinearly interpolated onto the CMEMS grid to preserve the canyon geometry critical for extreme-wave formation.

### Selected Features

| Variable | Units | Corr. w/ VCMX | Description |
|---|---|---|---|
| VHM0_SW1 | m | 0.89 | Primary swell significant wave height |
| VTM02 | s | 0.63 | Mean wave period (second moment) |
| VTM10 | s | 0.60 | Mean energy period |
| VTM01_SW1 | s | 0.57 | Primary swell mean period |
| VSDmag | m/s | 0.52 | Stokes drift magnitude |
| VHM0_WW | m | 0.46 | Wind-wave significant wave height |
| VTM01_WW | s | 0.40 | Wind-wave mean period |
| **VCMX** | **m** | — | **Target: Max crest-trough wave height** |
| GEBCO depth | m | — | Static bathymetric depth |
| Ocean mask | — | — | Binary ocean/land mask |

### Data Split *(strict chronological, no shuffling)*

| Split | Period | Steps |
|---|---|---|
| Train | Jan 2020 – Dec 2022 | 26,304 |
| Validation | Jan 2023 – Jun 2023 | 4,344 |
| Test | Jul 2023 – Dec 2023 | 4,392 |

Benchmark evaluation uses the **November 2023 sub-window** (696 steps) — peak storm month capturing the 17.18 m extreme event of November 4–5.

---

## 🔄 Sliding Window Forecasting & Drift Correction

Although models are trained to predict **3 hours ahead** from a **48-hour lookback**, we extend to stable **168-hour forecasts** via a recursive protocol:

### Predict-Three-Retain-One

```
At time t:
  Input:  X_t = {x_{t-47}, ..., x_t}
  Output: Ŷ_t = {ŷ_{t+1}, ŷ_{t+2}, ŷ_{t+3}}

  RETAIN ŷ_{t+1}  →  appended to next window as ground truth substitute
  USE ŷ_{t+2}, ŷ_{t+3}  →  internal drift monitoring only, then discard
```

### Internal Drift Correction

```python
Δ₁ = mean(|ŷ_{t+2} - ŷ_{t+1}|)   # step-1 forecast change
Δ₂ = mean(|ŷ_{t+3} - ŷ_{t+2}|)   # step-2 forecast change
D  = |Δ₂ - Δ₁|                    # internal drift magnitude
C  = 1.0 - (D × 0.5)              # correction factor ∈ (0, 1]
Ŷ_corrected = Ŷ × C               # attenuate unstable predictions
```

This suppresses unrealistic spikes and prevents autoregressive error accumulation — especially critical during the rapid 13.59 m → 16.14 m ramp-up of the November 4–5 storm event.

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/oghritik/Wave-Prediction.git
cd Wave-Prediction
pip install -r req.txt
```

Additional dependencies for training:
```bash
pip install torch pytorch-lightning copernicusmarine
```

### 2. Download Data

```bash
python download.py
```

Requires a free [Copernicus Marine Service](https://marine.copernicus.eu/) account:
```bash
export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
```

### 3. Explore the Data

```bash
jupyter notebook EDA.ipynb                    # variable distributions, correlations
jupyter notebook feature_engineering.ipynb    # feature selection
jupyter notebook ocean_visuals.ipynb          # bathymetry & spatial wave fields
```

### 4. Run the Final Experiments (V4)

Open the notebooks inside `V4/` in order of lookback ablation:

```bash
cd V4

# Short lookback (12h)
jupyter notebook v7.ipynb

# Medium lookback (24h)
jupyter notebook v6.ipynb

# Extended lookback (72h) — closest to paper results
jupyter notebook v8.ipynb
```

Each notebook runs the full pipeline: data loading → model training → 168-hour sliding window evaluation with drift correction → all 8 result figures.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Loss function | MSE |
| Early stopping patience | 5 epochs |
| Max epochs | 50 |
| Lookback window | 48 hours (paper) / 12, 24, 72h (ablations in V4) |
| Training horizon | 3 hours (sequence-to-one) |
| Evaluation horizon | 168 hours (recursive sliding window) |
| Batch size | 4 |
| Precision | 16-bit AMP |
| Framework | PyTorch Lightning + DDP |
| Hardware | 2× NVIDIA GTX 1080 Ti |

---

## 📋 Requirements

```
numpy
pandas
xarray
netCDF4
scikit-learn
tensorflow
matplotlib
seaborn
torch
pytorch-lightning
copernicusmarine
```

---

## 📜 Citation

If you use this code or findings in your work, please cite:

```bibtex
@inproceedings{routia2026waveprediction,
  title     = {Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU
               with Sliding Window and Correction Mechanism},
  author    = {Routia, Hritik and Patel, Parth and Kumar, Santosh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {IIIT Naya Raipur},
  url       = {https://github.com/oghritik/Wave-Prediction}
}
```

---

## 👥 Authors

| Name | Affiliation | Email |
|---|---|---|
| **Hritik Routia** | IIIT Naya Raipur | hritik23100@iiitnr.edu.in |
| **Parth Patel** | IIIT Naya Raipur | parth23100@iiitnr.edu.in |
| **Santosh Kumar** | IIIT Naya Raipur | santosh@iiitnr.edu.in |

---

## 🙏 Acknowledgements

- **Copernicus Marine Service (CMEMS)** — IBI Ocean Wave Analysis and Forecast product
- **GEBCO** — General Bathymetric Chart of the Oceans 2022 grid
- **IIIT Naya Raipur** — Institutional support and compute resources

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
