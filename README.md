# 🌊 Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU

> **Under Review at NeurIPS 2026**  
> *Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU with Sliding Window and Correction Mechanism*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange?logo=pytorch)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-purple)](https://github.com/oghritik/Wave-Prediction)

---

## 📌 Overview

This repository contains the official implementation of our NeurIPS 2026 paper benchmarking spatiotemporal deep learning architectures for **extreme ocean wave forecasting** at **Nazaré Canyon, Portugal** — Europe's largest submarine canyon and the world's most bathymetrically complex wave amplification site.

Our central finding: **bidirectional temporal modelling dominates spatial encoder complexity** when bathymetry is explicitly encoded. CNN+BiGRU — the simplest spatial encoder tested — outperforms ASPP multi-scale and UNet++ dense skip-connection encoders by up to **23% RMSE** on 168-hour forecasts.

We also introduce a **predict-three-retain-one recursive evaluation protocol** with internal drift correction, enabling stable 168-hour spatiotemporal wave forecasting without runaway error accumulation.

---

## 🏔️ Why Nazaré Canyon?

Nazaré Canyon (39.6°N, 9.1°W) is 230 km long, reaches 5,000 m depth, and terminates metres from the Portuguese coast. Incoming North Atlantic swell splits across the deep canyon channel and shallow shelf, reconverging near shore through constructive interference that routinely produces:

- **Winter maxima > 15 m** crest-trough wave heights (VCMX)
- **Documented extremes > 30 m**
- Events like the **WSL Tudor Big Wave Challenge**

Standard numerical models (WAVEWATCH III, SWAN) and open-ocean deep learning approaches systematically underestimate wave heights here due to ignoring or flattening bathymetric structure. This work addresses that gap.

---

## ✨ Key Contributions

1. **First systematic benchmark** of spatiotemporal deep learning architectures on a bathymetry-critical extreme wave environment — four architectures trained identically on 35,017 hourly CMEMS IBI steps with explicit GEBCO 2022 bathymetry.

2. **Predict-three-retain-one recursive protocol** with internal drift correction — enabling stable 168-hour forecasting evaluated on a 696-step November 2023 window capturing a **17.18 m extreme event**.

3. **Architectural finding**: CNN+BiGRU (simplest encoder) outperforms ASPP and UNet++ variants by up to 23% RMSE, demonstrating that bidirectional temporal modelling is the decisive factor when bathymetry is explicitly encoded as a static input channel.

4. **Negative design guidance**: ASPP dilation rates are counterproductive on restricted coastal ocean grids (72×90 at 0.027°), as receptive fields exceed the domain's effective spatial extent.

---

## 🏗️ Architectures Benchmarked

All four models share an identical design contract: **spatial encoder → BiGRU temporal backbone → 1×1 Conv output head**. They differ only in the spatial encoder.

| Version | Model | Spatial Encoder | Temporal Backbone |
|---------|-------|-----------------|-------------------|
| V1 | **ConvLSTM** | Convolutional LSTM cells (fused space-time) | — (integrated) |
| V2 | **CNN + BiGRU** ⭐ | Stride-2 CNN (2 layers: 64, 32 filters) | Bidirectional ConvGRU |
| V3 | **DeepLabV3 + BiGRU** | ASPP multi-scale pooling (d=1,6,12) | Bidirectional ConvGRU |
| V4 | **UNet++ + BiGRU** | DW-separable U-Net++ dense skip connections | Bidirectional ConvGRU |

> ⭐ Best performing architecture

**Input tensor:** `(B, 48, 10, 72, 90)` — batch × 48-hour lookback × 10 channels (8 dynamic + 2 static) × 72×90 spatial grid  
**Output:** 8-variable wave field for next 3 hours (sequence-to-one)

---

## 📊 Results

### Benchmark — November 2023 Extreme Event Window (696 steps, VCMX range 5–17 m)

| Model | Overall RMSE (m) | Overall MAE (m) | VCMX RMSE (m) | VCMX MAE (m) |
|---|---|---|---|---|
| ConvLSTM | 0.3579 | 0.1613 | 0.2839 | 0.2213 |
| **CNN+BiGRU** | **0.2965** | **0.1264** | **0.1832** | **0.1384** |
| DeepLabV3+BiGRU | 0.3174 | 0.1551 | 0.2797 | 0.2035 |
| UNet+++BiGRU | 0.3833 | 0.1999 | 0.3577 | 0.2763 |

### Baseline Comparison

| Model | Lookback | Horizon | VCMX RMSE (m) | VCMX MAE (m) |
|---|---|---|---|---|
| EarthFormer | 12 h | 14 h | 6.7475 | 6.4283 |
| Vanilla ConvLSTM (single-pass) | 48 h | 168 h | 1.1960 | 0.8211 |
| CNN+BiGRU (12h lookback) | 12 h | 168 h | 0.1832 | 0.1384 |
| **CNN+BiGRU (48h lookback)** | **48 h** | **168 h** | **0.2965** | **0.1264** |

---

## 📂 Repository Structure

```
Wave-Prediction/
│
├── V1/                          # ConvLSTM architecture
├── V2/                          # CNN + BiGRU (best model)
├── V3/                          # DeepLabV3 + BiGRU
├── V4/                          # UNet++ + BiGRU
│
├── models/                      # Saved model checkpoints
├── Documents/                   # Paper drafts and supplementary
├── paper_figures/               # Figures used in the paper
├── result images/               # Prediction output visualisations
├── lightning_logs/              # PyTorch Lightning training logs
│
├── EDA.ipynb                    # Exploratory data analysis
├── feature_engineering.ipynb    # Feature selection & correlation analysis
├── model_comparision.ipynb      # Initial model comparison notebook
├── model_comparison_refactored.ipynb  # Cleaned benchmark notebook
├── ocean_visuals.ipynb          # Bathymetry & wave field visualisations
├── download.py                  # CMEMS data download script
├── research_experiment.py       # Main training/evaluation script
├── req.txt                      # Python dependencies
└── README.md
```

---

## 🗃️ Dataset

### Dynamic Wave Data — CMEMS IBI Reanalysis
- **Source:** [Copernicus Marine Service (CMEMS) IBI Wave Analysis & Forecast](https://marine.copernicus.eu/)
- **Period:** January 2020 – December 2023 (35,017 hourly steps)
- **Domain:** 38.5°N–40.5°N, 11.0°W–8.5°W at 0.027° resolution → **72×90 grid**
- **Format:** CF-1.8 compliant NetCDF-4

### Static Bathymetry — GEBCO 2022
- **Source:** [GEBCO 2025 Grid](https://www.gebco.net/)
- **Processing:** Bilinearly interpolated from native 90×90 grid onto the CMEMS grid to preserve canyon geometry (true depth: ~5,000 m vs CMEMS native: ~1,000 m)

### Selected Input Features

| Variable | Units | Correlation w/ VCMX | Description |
|---|---|---|---|
| VHM0_SW1 | m | 0.89 | Primary swell significant wave height |
| VTM02 | s | 0.63 | Mean wave period (second moment) |
| VTM10 | s | 0.60 | Mean energy period |
| VTM01_SW1 | s | 0.57 | Primary swell mean period |
| VSDmag | m/s | 0.52 | Stokes drift magnitude |
| VHM0_WW | m | 0.46 | Wind-wave significant wave height |
| VTM01_WW | s | 0.40 | Wind-wave mean period |
| **VCMX** | **m** | **—** | **Target: Maximum crest-trough wave height** |
| GEBCO depth | m | — | Static bathymetric depth (GEBCO 2022) |
| Ocean mask | — | — | Binary ocean/land mask |

### Temporal Split (no shuffling — strict chronological)

| Split | Period | Steps |
|---|---|---|
| Train | Jan 2020 – Dec 2022 | 26,304 |
| Validation | Jan 2023 – Jun 2023 | 4,344 |
| Test | Jul 2023 – Dec 2023 | 4,392 |

---

## ⚙️ Sliding Window Forecasting & Drift Correction

### Predict-Three-Retain-One Protocol

Although trained to predict 3 hours ahead from a 48-hour lookback, **168-hour forecasts** are generated recursively:

1. At time `t`, the model receives `X_t = {x_{t-47}, ..., x_t}` and outputs `Ŷ_t = {ŷ_{t+1}, ŷ_{t+2}, ŷ_{t+3}}`
2. Only `ŷ_{t+1}` is **retained** as the official forecast and appended to the next input window
3. `ŷ_{t+2}` and `ŷ_{t+3}` form an **internal correction window** to monitor drift

### Dynamic Error Correction

Consecutive forecast deviations are monitored:

```
Δ₁ = mean(|ŷ_{t+2} − ŷ_{t+1}|)
Δ₂ = mean(|ŷ_{t+3} − ŷ_{t+2}|)
D  = |Δ₂ − Δ₁|
C  = 1.0 − (D × 0.5)
Ŷ_corrected = Ŷ × C
```

This attenuates unstable forecasts proportional to internal drift magnitude, suppressing unrealistic spikes over the 168-hour window.

---

## 🚀 Getting Started

### Prerequisites

```bash
git clone https://github.com/oghritik/Wave-Prediction.git
cd Wave-Prediction
pip install -r req.txt
```

Additional requirements for model training (not in `req.txt`):

```bash
pip install torch pytorch-lightning netCDF4 xarray copernicusmarine
```

### Data Download

```bash
python download.py
```

> You will need a free [Copernicus Marine Service](https://marine.copernicus.eu/) account. Set your credentials as environment variables `CMEMS_USER` and `CMEMS_PASSWORD` before running.

### Exploratory Analysis

```bash
jupyter notebook EDA.ipynb
jupyter notebook feature_engineering.ipynb
jupyter notebook ocean_visuals.ipynb
```

### Training

Navigate to the model version you want to train:

```bash
# Example: train CNN+BiGRU (V2 — best model)
cd V2
python train.py
```

Training uses **PyTorch Lightning with DDP** across 2× NVIDIA GTX 1080 Ti GPUs and 16-bit AMP. See `Table 2` in the paper for full hyperparameter configuration.

### Evaluation / Benchmark

```bash
jupyter notebook model_comparison_refactored.ipynb
```

This notebook runs the 168-hour sliding window evaluation with drift correction over the November 2023 extreme event window (696 steps).

---

## 🧪 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1×10⁻⁴ |
| Loss function | MSE |
| Early stopping patience | 5 epochs |
| Max epochs | 50 |
| Lookback window | 48 hours |
| Training horizon | 3 hours (direct) |
| Evaluation horizon | 168 hours (recursive sliding window) |
| Batch size | 4 |
| Precision | 16-bit AMP |
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
```

---

## 📜 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{routia2026waveprediction,
  title     = {Extended-Horizon Ocean Wave Forecasting via Hybrid CNN-BiGRU 
               with Sliding Window and Correction Mechanism},
  author    = {Routia, Hritik and Patel, Parth and Kumar, Santosh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  institution = {IIIT Naya Raipur}
}
```

---

## 👥 Authors

| Name | Institution | Contact |
|---|---|---|
| **Hritik Routia** | IIIT Naya Raipur | hritik23100@iiitnr.edu.in |
| **Parth Patel** | IIIT Naya Raipur | parth23100@iiitnr.edu.in |
| **Santosh Kumar** | IIIT Naya Raipur | santosh@iiitnr.edu.in |

---

## 🙏 Acknowledgements

- **CMEMS** — Copernicus Marine Environment Monitoring Service for the IBI Ocean Wave Reanalysis product
- **GEBCO** — General Bathymetric Chart of the Oceans for the 2022 high-resolution bathymetric grid
- **IIIT Naya Raipur** — Institutional support and compute resources

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
