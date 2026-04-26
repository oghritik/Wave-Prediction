# train.py

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import glob

# =============================================================================
# MASTER CONFIGURATION
# =============================================================================
LOOKBACK_HOURS = 96
FORECAST_HORIZON_HOURS = 168
BASE_DIR = r'D:\babe_prediction'
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, f'processed_data_lookback_{LOOKBACK_HOURS}_static')
MODEL_SAVE_DIR = 'models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'convlstm_lookback_{LOOKBACK_HOURS}_forecast_{FORECAST_HORIZON_HOURS}_static.pth')
VARS_TO_USE = ['VCMX','VSDmag', 'VTM10', 'VTM02', 'VTM01_WW', 'VTM01_SW1', 'VMXL', 'VHM0_WW', 'VHM0_SW1']
INPUT_CHANNELS = len(VARS_TO_USE) + 2
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0

# =============================================================================
# PYTORCH LIGHTNING DEFINITIONS
# =============================================================================

class PreprocessedWaveDataset(Dataset):
    def __init__(self, split_dir):
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, '*.pt')))
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])

class ConvLSTMCell(nn.Module):
    def __init__(self, i, h, k, b): super(ConvLSTMCell, self).__init__(); self.input_dim, self.hidden_dim, self.kernel_size, self.bias = i, h, k, b; self.padding = k[0] // 2; self.conv = nn.Conv2d(i + h, 4 * h, k, padding=self.padding, bias=b)
    def forward(self, x, h_c): h, c = h_c; combined = torch.cat([x, h], dim=1); cc = self.conv(combined); i, f, o, g = torch.split(cc, self.hidden_dim, dim=1); i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g); c_n = f * c + i * g; h_n = o * torch.tanh(c_n); return h_n, c_n
    def init_hidden(self, b, i, d): h, w = i; return (torch.zeros(b, self.hidden_dim, h, w, device=d), torch.zeros(b, self.hidden_dim, h, w, device=d))

class ConvLSTM(nn.Module):
    def __init__(self, i, h, k, n, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__(); self.batch_first, self.num_layers = batch_first, n; h_dims = [h] * n if isinstance(h, int) else h; c_list = [];
        for j in range(self.num_layers): c_i_dim = i if j == 0 else h_dims[j - 1]; c_list.append(ConvLSTMCell(c_i_dim, h_dims[j], k, bias)); self.cell_list = nn.ModuleList(c_list)
    def forward(self, x, h_c=None):
        b, s_l, _, h, w = x.size();
        if h_c is None: h_c = self._init_hidden(b, (h, w), x.device)
        cur_in = x
        for l_idx in range(self.num_layers):
            h, c = h_c[l_idx]; out_inner = []
            for t in range(s_l): h, c = self.cell_list[l_idx](cur_in[:, t, :, :, :], [h, c]); out_inner.append(h)
            cur_in = torch.stack(out_inner, dim=1)
        return cur_in, [h, c]
    def _init_hidden(self, b, i, d): return [cell.init_hidden(b, i, d) for cell in self.cell_list]

class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim, forecast_horizon, hidden_dims=[32, 16], kernel_size=(3, 3)):
        super(ConvLSTMNet, self).__init__()
        self.cl1 = ConvLSTM(input_dim, hidden_dims[0], kernel_size, 1, batch_first=True)
        self.cl2 = ConvLSTM(hidden_dims[0], hidden_dims[1], kernel_size, 1, batch_first=True)
        self.output_conv = nn.Conv2d(hidden_dims[1], forecast_horizon, kernel_size=(1, 1), padding='same')
    def forward(self, x_seq):
        l1_o, _ = self.cl1(x_seq); l2_o, _ = self.cl2(l1_o); return self.output_conv(l2_o[:, -1, :, :, :])

class WavePredDataModule(pl.LightningDataModule):
    def __init__(self, processed_data_dir, batch_size, num_workers):
        super().__init__()
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.train_dataset = PreprocessedWaveDataset(os.path.join(self.processed_data_dir, 'train'))
        self.val_dataset = PreprocessedWaveDataset(os.path.join(self.processed_data_dir, 'val'))
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)

class WavePredLightningModule(pl.LightningModule):
    def __init__(self, input_dim, forecast_horizon, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvLSTMNet(input_dim=input_dim, forecast_horizon=forecast_horizon)
        self.loss_fn = nn.MSELoss()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        X, y = batch
        predicted = self(X)
        loss = self.loss_fn(predicted, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        X, y = batch
        predicted = self(X)
        loss = self.loss_fn(predicted, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# =============================================================================
# MAIN TRAINING EXECUTION
# =============================================================================
def main():
    pl.seed_everything(42, workers=True)
    
    # ================== ADD THESE TWO LINES ==================
    # Manually set the address and port for inter-process communication
    # This is crucial for environments where auto-detection might fail (like Windows)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" # You can use any free port
    # =========================================================
    
    data_module = WavePredDataModule(PROCESSED_DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    lightning_model = WavePredLightningModule(INPUT_CHANNELS, FORECAST_HORIZON_HOURS, LEARNING_RATE)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=True, mode='min')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        filename=os.path.basename(MODEL_SAVE_PATH).replace('.pth', ''),
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=[0, 1],  # Use both GPUs
        strategy=DDPStrategy(process_group_backend='nccl'),  # NCCL is optimal for NVIDIA GPUs
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
        sync_batchnorm=True  # Synchronize batch norm across GPUs
    )

    print(f"\n--- Starting DDP training on GPUs 0 and 1 ---")
    trainer.fit(lightning_model, data_module)
    print(f"\n✅ Training complete. Best model saved to: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()