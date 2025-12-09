import os
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==================== Config ====================
DATA_DIR = "./data"                    # Read all json files from a single folder
OUTPUT_DIR = "./output/case5/k7"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_ae.pth")
LATENT_NPY = os.path.join(OUTPUT_DIR, "latent_features.npy")
KMEANS_LABEL_NPY = os.path.join(OUTPUT_DIR, "kmeans_labels_k7.npy")
PCA_FIG_PATH = os.path.join(OUTPUT_DIR, "kmeans_k7_PCA.png")
FILE_LIST_TXT = os.path.join(OUTPUT_DIR, "file_list.txt")

LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
N_CLUSTERS = 7
RANDOM_STATE = 42

USE_CHANNEL_ATTENTION = True
# =====================================================


# ==================== Dataset ====================
class GNSSJsonDataset(Dataset):
    """
    Load 32x6 GNSS feature matrices from json files.
    Transpose to (C=6, L=32) for Conv1D input.
    """
    def __init__(self, data_dir):
        self.file_paths = sorted(glob(os.path.join(data_dir, "*.json")))
        print(f"[INFO] Found {len(self.file_paths)} json files in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        with open(fpath, "r") as f:
            data = json.load(f)

        arr = np.array(data, dtype=np.float32)
        if arr.shape != (32, 6):
            raise ValueError(f"Invalid shape {arr.shape}, expected (32, 6)")

        arr = arr.T  # (6, 32)
        x = torch.from_numpy(arr)
        return x, fpath


# ==================== Channel Attention ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
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


# ==================== Dual-Branch CNN AutoEncoder ====================
class DualBranchCNN1DAE(nn.Module):
    def __init__(self, latent_dim=64, use_channel_attention=True):
        super().__init__()
        self.use_ca = use_channel_attention

        # Encoder branches
        self.obs_enc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.pvt_enc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        if self.use_ca:
            self.ca = ChannelAttention(64, reduction=4)
        else:
            self.ca = None

        self.enc_fc = nn.Linear(64 * 8, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 64 * 8)

        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),

            nn.Conv1d(16, 6, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x_obs = x[:, 0:3, :]
        x_pvt = x[:, 3:6, :]

        h_obs = self.obs_enc_cnn(x_obs)
        h_pvt = self.pvt_enc_cnn(x_pvt)

        h = torch.cat([h_obs, h_pvt], dim=1)

        if self.use_ca:
            h = self.ca(h)

        b = h.shape[0]
        z = self.enc_fc(h.view(b, -1))
        return z

    def decode(self, z):
        b = z.shape[0]
        h = self.dec_fc(z).view(b, 64, 8)
        x_recon = self.dec_cnn(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ==================== Training ====================
def train_autoencoder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = GNSSJsonDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = DualBranchCNN1DAE(latent_dim=LATENT_DIM,
                              use_channel_attention=USE_CHANNEL_ATTENTION).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[EPOCH {epoch:03d}] recon_loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Saved model to: {MODEL_PATH}")

    return MODEL_PATH


# ==================== Latent Extraction ====================
def extract_latent_features(model_path):
    dataset = GNSSJsonDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchCNN1DAE(latent_dim=LATENT_DIM,
                              use_channel_attention=USE_CHANNEL_ATTENTION).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_latents = []
    all_files = []

    with torch.no_grad():
        for x, files in loader:
            x = x.to(device)
            z = model.encode(x)
            all_latents.append(z.cpu().numpy())
            all_files.extend(files)

    latents = np.concatenate(all_latents, axis=0)
    np.save(LATENT_NPY, latents)
    print(f"[INFO] Saved latent features: {LATENT_NPY}")

    with open(FILE_LIST_TXT, "w") as f:
        for fp in all_files:
            f.write(fp + "\n")

    return latents


# ==================== Clustering + PCA ====================
def cluster_and_visualize(latents):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(latents)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(X_std)

    np.save(KMEANS_LABEL_NPY, labels)
    print("[INFO] Saved kmeans labels.")

    centers_path = os.path.join(OUTPUT_DIR, "kmeans_centers.npy")
    np.save(centers_path, kmeans.cluster_centers_)
    print("[INFO] Saved kmeans centers.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    plt.colorbar(sc, label="Cluster ID")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Projection of Latent Features (k={N_CLUSTERS})")
    plt.tight_layout()
    plt.savefig(PCA_FIG_PATH, dpi=300)
    plt.close()


# ==================== Main ====================
def main():
    model_path = train_autoencoder()
    latents = extract_latent_features(model_path)
    cluster_and_visualize(latents)


if __name__ == "__main__":
    main()
