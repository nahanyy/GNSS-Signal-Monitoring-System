import os
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TRAIN_DIR = Path("data/train")      
OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "cnn_ae.pth"

LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

SEED = 42
USE_CHANNEL_ATTENTION = True
REUSE_TRAINED_AE = True

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global seed fixed to {seed}")


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class GNSSJsonDataset(Dataset):
    """
    Read JSON files from ONE directory and build an unlabeled dataset.
    Each JSON sample must have shape (32, 6) and will be transposed to (6, 32).
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"[ERROR] Training directory not found: {self.data_dir}")

        self.file_paths = sorted(glob(str(self.data_dir / "*.json")))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"[ERROR] No JSON files found in: {self.data_dir}")

        random.shuffle(self.file_paths)

        print("[INFO] Training directory:")
        print(f"       - {self.data_dir}")
        print(f"[INFO] Total training samples: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        arr = np.array(data, dtype=np.float32)
        if arr.shape != (32, 6):
            raise ValueError(f"[ERROR] Bad shape in {fpath}: {arr.shape}, expected (32, 6)")

        arr = arr.T  # (6, 32)
        x = torch.from_numpy(arr)
        return x, fpath

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class DualBranchCNN1DAE(nn.Module):
    def __init__(self, latent_dim: int = 64, use_channel_attention: bool = True):
        super().__init__()
        self.use_ca = use_channel_attention
        self.obs_enc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  
        )

        self.pvt_enc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
        )

        self.ca = ChannelAttention(in_channels=64, reduction=4) if self.use_ca else None

        self.enc_fc = nn.Linear(64 * 8, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 64 * 8)

        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.Conv1d(16, 6, kernel_size=3, stride=1, padding=1),
            nn.Identity(),
        )

    def encode(self, x):
        x_obs = x[:, 0:3, :]
        x_pvt = x[:, 3:6, :]
        h_obs = self.obs_enc_cnn(x_obs)
        h_pvt = self.pvt_enc_cnn(x_pvt)
        h = torch.cat([h_obs, h_pvt], dim=1)  # (B, 64, 8)
        if self.use_ca and self.ca is not None:
            h = self.ca(h)
        b, c, l = h.shape
        z = self.enc_fc(h.view(b, -1))
        return z

    def decode(self, z):
        b = z.shape[0]
        h = self.dec_fc(z).view(b, 64, 8)
        x_recon = self.dec_cnn(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_autoencoder() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if REUSE_TRAINED_AE and MODEL_PATH.exists():
        print(f"[INFO] Found existing AE model, reusing: {MODEL_PATH}")
        return MODEL_PATH

    dataset = GNSSJsonDataset(TRAIN_DIR)

    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = DualBranchCNN1DAE(
        latent_dim=LATENT_DIM,
        use_channel_attention=USE_CHANNEL_ATTENTION
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[EPOCH {epoch:03d}] recon_loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] AE saved to: {MODEL_PATH}")
    return MODEL_PATH

def main():
    set_global_seed(SEED)
    train_autoencoder()

if __name__ == "__main__":
    main()
