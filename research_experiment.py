# =============================================================================
# RESEARCH EXPERIMENT: FocusedLoss vs MSE for Extreme Wave Event Prediction
# =============================================================================

# This script implements a rigorous scientific comparison between standard MSE 
# loss and our novel FocusedLoss function for predicting extreme wave events.

import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import pickle
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import warnings
import shutil
from torch.utils.data import Dataset, DataLoader
import glob
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =============================================================================
# MASTER CONFIGURATION - MODIFIED FOR RESEARCH EXPERIMENT
# =============================================================================

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")
if cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(0)}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# Core Parameters
LOOKBACK_HOURS = 96
# RESEARCH CHANGE 1: Reduced from 168 to 24 hours for stable baseline
FORECAST_HORIZON_HOURS = 24

# File Paths
RAW_DATA_PATH = r'/home/aidl/Wave-Prediction/dAtA/cmems_mod_ibi_wav_my_0.027deg_PT1H-i_multi-vars_11.00W-8.53W_38.50N-40.47N_2020-01-01-2023-12-30.nc'
STATIC_DATA_PATH = r'/home/aidl/Wave-Prediction/dAtA/cmems_GEBCO_resampled.nc'

BASE_DIR = r'/home/aidl/babe_prediction'
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, f'processed_data_lookback_{LOOKBACK_HOURS}_forecast_{FORECAST_HORIZON_HOURS}_static')
MODEL_SAVE_DIR = '/home/aidl/Wave-Prediction/models/'

# Feature Engineering
# RESEARCH CHANGE 2: Added VHM0 back as it's vital for VCMX prediction
VARS_TO_USE = ['VCMX', 'VHM0', 'VSDmag', 'VTM10', 'VTM02', 'VTM01_WW', 'VTM01_SW1', 'VMXL', 'VHM0_WW', 'VHM0_SW1']
TARGET_VAR = 'VCMX'
INPUT_CHANNELS = len(VARS_TO_USE) + 2  # time-varying + 2 static

# Training Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Research Parameters
EXTREME_THRESHOLD = 4.0  # VCMX threshold for extreme events
FOCUSED_WEIGHT = 50.0    # Penalty multiplier for extreme events

# System Configuration
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Research Experiment Configuration ---")
print(f"Lookback Period: {LOOKBACK_HOURS} hours")
print(f"Forecast Horizon: {FORECAST_HORIZON_HOURS} hours (CHANGED from 168)")
print(f"Input Channels: {INPUT_CHANNELS} ({len(VARS_TO_USE)} time-varying + 2 static)")
print(f"VHM0 included: {'VHM0' in VARS_TO_USE} (RESEARCH CHANGE)")
print(f"Extreme Event Threshold: {EXTREME_THRESHOLD}m")
print(f"FocusedLoss Weight: {FOCUSED_WEIGHT}x")
print(f"Using Device: {device}")
print("------------------------------------------")

# =============================================================================
# NOVEL FOCUSEDLOSS FUNCTION
# =============================================================================

class FocusedLoss(nn.Module):
    """
    A custom Weighted Mean Squared Error (WMSE) loss function that applies
    a heavy penalty to errors on high-value targets (extreme events).
    
    This loss function forces the model to prioritize accurate predictions
    of extreme wave events by applying significantly higher penalties
    to prediction errors when the true VCMX exceeds the threshold.
    
    Args:
        threshold (float): The value above which to apply the heavy penalty
        weight (float): The penalty multiplier for errors on extreme events
    """
    
    def __init__(self, threshold=4.0, weight=50.0):
        super(FocusedLoss, self).__init__()
        self.threshold = threshold
        self.weight = weight
        # Use 'none' reduction to get element-wise errors
        self.mse = nn.MSELoss(reduction='none')
        
        print(f"FocusedLoss initialized: threshold={self.threshold}m, weight={self.weight}x")
    
    def forward(self, y_pred, y_true):
        # 1. Calculate the standard, element-wise squared error
        element_wise_error = self.mse(y_pred, y_true)
        
        # 2. Create the weight map based on the *true* values
        # Weights are 1.0 by default
        # Weights become self.weight where y_true is >= threshold
        weights = torch.ones_like(y_true)
        weights = torch.where(y_true >= self.threshold, self.weight, weights)
        
        # 3. Apply the weights to the errors
        weighted_error = element_wise_error * weights
        
        # 4. Return the mean of the weighted errors
        return weighted_error.mean()

# Test the FocusedLoss function
print("Testing FocusedLoss function...")
test_loss = FocusedLoss(threshold=EXTREME_THRESHOLD, weight=FOCUSED_WEIGHT)

# Create test tensors
y_true_test = torch.tensor([[1.0, 2.0], [5.0, 6.0]])  # Mix of normal and extreme values
y_pred_test = torch.tensor([[1.1, 2.1], [4.5, 5.5]])  # Predictions with some error

loss_value = test_loss(y_pred_test, y_true_test)
print(f"Test loss value: {loss_value.item():.4f}")
print("✅ FocusedLoss function working correctly!")