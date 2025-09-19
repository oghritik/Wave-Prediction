# ==============================================================================
# train.py
#
# End-to-end script for training the ocean wave height prediction model.
# ==============================================================================

# --- Part 1: Imports and Definitions ---

import os
import time
import pickle
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

# --- PyTorch Dataset Class ---

class WaveDataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed spatio-temporal wave data.
    """
    def __init__(self, ds, feature_vars, target_var, scalers, static_data_tensor, lookback=48, forecast_horizon=168):
        self.ds = ds
        self.feature_vars = feature_vars
        self.target_var = target_var
        self.scalers = scalers
        self.static_data_tensor = static_data_tensor
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.n_sequences = len(self.ds['time']) - (self.lookback + self.forecast_horizon)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        seq_end_idx = idx + self.lookback + self.forecast_horizon
        data_slice = self.ds.isel(time=slice(idx, seq_end_idx))
        
        scaled_data = {}
        for var, scaler in self.scalers.items():
            if var not in data_slice.variables: continue
            scaled_data[var] = scaler.transform(data_slice[var].values.reshape(-1, 1)).reshape(data_slice[var].shape)

        features_list = [
            scaled_data[var][:self.lookback, np.newaxis, :, :] 
            for var in self.feature_vars
        ]
        
        static_data_with_time = np.tile(self.static_data_tensor[np.newaxis, ...], (self.lookback, 1, 1, 1))
        all_features = np.concatenate(features_list + [static_data_with_time], axis=1).astype(np.float32)
        y = scaled_data[self.target_var][self.lookback:].astype(np.float32)

        return torch.from_numpy(all_features), torch.from_numpy(y)

# --- Model Architecture ---

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.kernel_size, self.bias = kernel_size, bias
        self.padding = kernel_size[0] // 2
        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, self.kernel_size, padding=self.padding, bias=self.bias)
    
    def forward(self, x, h_c):
        h, c = h_c
        combined = torch.cat([x, h], dim=1)
        cc = self.conv(combined)
        i, f, o, g = torch.split(cc, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_n = f * c + i * g
        h_n = o * torch.tanh(c_n)
        return h_n, c_n

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()
        self.batch_first, self.num_layers = batch_first, num_layers
        hidden_dims = [hidden_dim] * num_layers if isinstance(hidden_dim, int) else hidden_dim
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, h_c=None):
        if not self.batch_first: x = x.permute(1, 0, 2, 3, 4)
        b, s_l, _, h, w = x.size()
        if h_c is None: h_c = self._init_hidden(b, h, w, x.device)
        
        cur_in = x
        for l_idx in range(self.num_layers):
            h, c = h_c[l_idx]
            output_inner = []
            for t in range(s_l):
                h, c = self.cell_list[l_idx](cur_in[:, t, :, :, :], [h, c])
                output_inner.append(h)
            cur_in = torch.stack(output_inner, dim=1)
        return cur_in, [h, c]

    def _init_hidden(self, b, h, w, d):
        return [cell.init_hidden(b, h, w, d) for cell in self.cell_list]

class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], kernel_size=(3, 3)):
        super(ConvLSTMNet, self).__init__()
        self.cl1 = ConvLSTM(input_dim, hidden_dims[0], kernel_size, 1, batch_first=True)
        self.cl2 = ConvLSTM(hidden_dims[0], hidden_dims[1], kernel_size, 1, batch_first=True)
        self.output_conv = nn.Conv2d(hidden_dims[1], 1, kernel_size=(1, 1), padding='same')

    def forward(self, x_seq):
        l1_o, _ = self.cl1(x_seq)
        l2_o, _ = self.cl2(l1_o)
        return self.output_conv(l2_o[:, -1, :, :, :])

# --- Training and Evaluation Functions ---

def weighted_mse_loss(output, target, weight_threshold=0.8):
    mse = (output - target)**2
    weights = torch.where(target > weight_threshold, 2.0, 1.0)
    return torch.mean(weights * mse)

def train_model(model, train_loader, val_loader, device, epochs, lr, model_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = weighted_mse_loss
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    patience = 5
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Starting Model Training ---")
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for X, y in train_pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                predicted_step = model(X)
                target_step = y[:, 0, :, :].unsqueeze(1)
                loss = loss_fn(predicted_step, target_step)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
        
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    predicted_step = model(X)
                    target_step = y[:, 0, :, :].unsqueeze(1)
                    loss = loss_fn(predicted_step, target_step)
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, model_path)
            print(f"✅ New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    print(f"\n✅ Training completed. Loading best model from {model_path}")
    best_model_state = torch.load(model_path)
    if isinstance(model, nn.DataParallel): model.module.load_state_dict(best_model_state)
    else: model.load_state_dict(best_model_state)
    
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print("✅ Training history plot saved to 'training_history.png'")

# ==============================================================================
# --- Part 2: Main Execution Block ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration ---
    PROCESSED_DATA_DIR = r'D:\babe_prediction' # IMPORTANT: Update if your path is different
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 4 # Tune this based on your system. Start with 4 or 8.
    MODEL_SAVE_PATH = 'best_convlstm_model.pth'
    
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Pre-processed Data ---
    print("--- Loading pre-processed data ---")
    with open(os.path.join(PROCESSED_DATA_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    static_data_tensor = np.load(os.path.join(PROCESSED_DATA_DIR, 'static_data_tensor.npy'))
    ds_train = xr.open_dataset(os.path.join(PROCESSED_DATA_DIR, 'ds_train.nc')).load()
    ds_val = xr.open_dataset(os.path.join(PROCESSED_DATA_DIR, 'ds_val.nc')).load()
    ds_test = xr.open_dataset(os.path.join(PROCESSED_DATA_DIR, 'ds_test.nc')).load()
    
    target_var = 'VCMX'
    initial_feature_vars = ['VSDmag', 'VTM10', 'VTM02', 'VTM01_WW', 'VTM01_SW1', 'VMXL', 'VHM0_WW', 'VHM0_SW1']
    print(f"✅ Data loaded. Using device: {device}")

    # --- 2. Create DataLoaders ---
    train_dataset = WaveDataset(ds_train, initial_feature_vars, target_var, scalers, static_data_tensor)
    val_dataset = WaveDataset(ds_val, initial_feature_vars, target_var, scalers, static_data_tensor)
    test_dataset = WaveDataset(ds_test, initial_feature_vars, target_var, scalers, static_data_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"DataLoaders created with {NUM_WORKERS} workers.")

    # --- 3. Create Model ---
    n_feature_channels = len(initial_feature_vars)
    n_static_channels = static_data_tensor.shape[0]
    input_channels = n_feature_channels + n_static_channels

    model = ConvLSTMNet(input_dim=input_channels)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training via DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)
    print(f"Model built and moved to {device}. Total Input Channels: {input_channels}")

    # --- 4. Start Training ---
    trained_model, training_history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE,
        model_path=MODEL_SAVE_PATH
    )

    # --- 5. Plot Training History ---
    plot_training_history(training_history)
    
    print("\n🎉🎉🎉 Training Pipeline Completed Successfully! 🎉🎉🎉")