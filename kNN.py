import os
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd

MODEL_PATH = Path("outputs/cnn_ae.pth")
TEST_DIRS = Path("data/test")

OUTPUT_DIR = Path("outputs/test_clustering")

LATENT_DIM = 128
BATCH_SIZE = 64

K_MIN = 2
K_MAX = 10

SEED = 42
RANDOM_STATE = 42

USE_CHANNEL_ATTENTION = True

REUSE_LATENT = True
LATENT_NPY = OUTPUT_DIR / "latent_features.npy"
FILE_LIST_TXT = OUTPUT_DIR / "file_list.txt"

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global seed fixed to {seed}")

class GNSSJsonDataset(Dataset):

    def __init__(self, data_dirs):
        self.file_paths = []
        for d in data_dirs:
            d = Path(d)
            files = sorted(glob(str(d / "*.json")))
            print(f"[INFO] Found {len(files)} JSON files in: {d}")
            self.file_paths.extend(files)

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"[ERROR] No JSON files found in TEST_DIRS: {data_dirs}")

        print(f"[INFO] Total mixed-test samples: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        arr = np.array(data, dtype=np.float32)  
        if arr.shape != (32, 6):
            raise ValueError(f"[ERROR] Bad shape in {fpath}: {arr.shape}, expected (32, 6)")

        arr = arr.T 
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
            nn.Sigmoid(),
        )

    def encode(self, x):
        x_obs = x[:, 0:3, :]
        x_pvt = x[:, 3:6, :]

        h_obs = self.obs_enc_cnn(x_obs)
        h_pvt = self.pvt_enc_cnn(x_pvt)

        h = torch.cat([h_obs, h_pvt], dim=1)

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

def extract_latent_features(model_path: Path) -> np.ndarray:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if REUSE_LATENT and LATENT_NPY.exists() and FILE_LIST_TXT.exists():
        print(f"[INFO] Reusing existing latent: {LATENT_NPY}")
        latents = np.load(LATENT_NPY)
        print(f"[INFO] latent shape={latents.shape}")
        return latents

    dataset = GNSSJsonDataset(TEST_DIRS)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = DualBranchCNN1DAE(
        latent_dim=LATENT_DIM,
        use_channel_attention=USE_CHANNEL_ATTENTION
    ).to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] MODEL_PATH not found: {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded AE weights: {model_path}")

    all_latents = []
    all_files = []

    with torch.no_grad():
        for x, fpaths in loader:
            x = x.to(device)
            z = model.encode(x)
            all_latents.append(z.cpu().numpy())
            all_files.extend(list(fpaths))

    latents = np.concatenate(all_latents, axis=0)
    np.save(LATENT_NPY, latents)

    with open(FILE_LIST_TXT, "w", encoding="utf-8") as f:
        for fp in all_files:
            f.write(str(fp) + "\n")

    print(f"[INFO] latent saved: {LATENT_NPY}, shape={latents.shape}")
    print(f"[INFO] file_list saved: {FILE_LIST_TXT}")
    return latents


#Silhouette scan for k
def scan_k_by_silhouette(latents: np.ndarray, k_min: int = 2, k_max: int = 10):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(latents)
    print("[INFO] Latent standardized (for silhouette scan)")

    results = []
    best_k = None
    best_score = -1e9

    for k in range(k_min, k_max + 1):
        if k < 2 or k >= X_std.shape[0]:
            print(f"[WARN] Skip k={k} (invalid given sample size)")
            continue

        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_std)

        if len(np.unique(labels)) < 2:
            print(f"[WARN] Skip k={k} (unique clusters < 2)")
            continue

        score = silhouette_score(X_std, labels, metric="euclidean")
        results.append({"k": k, "silhouette": float(score)})
        print(f"[Silhouette] k={k:2d}  score={score:.6f}")

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise RuntimeError("Silhouette scan failed: no valid k found.")

    df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    print(f"[INFO] best_k = {best_k}, best_score = {best_score:.6f}")

    csv_path = OUTPUT_DIR / f"silhouette_k{k_min}_to_k{k_max}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] silhouette CSV saved: {csv_path}")

    fig_path = OUTPUT_DIR / f"silhouette_k{k_min}_to_k{k_max}.png"
    plt.figure(figsize=(7, 4))
    plt.plot(df["k"], df["silhouette"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs. k (latent space, mixed test)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] silhouette plot saved: {fig_path}")

    return best_k, df


# KMeans(best_k) + PCA visualization
def cluster_and_visualize(latents: np.ndarray, n_clusters: int):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(latents)
    print("[INFO] Latent standardized (for KMeans/PCA)")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(X_std)

    label_npy = OUTPUT_DIR / f"kmeans_labels_k{n_clusters}.npy"
    centers_npy = OUTPUT_DIR / f"kmeans_centers_k{n_clusters}.npy"
    pca_fig = OUTPUT_DIR / f"kmeans_k{n_clusters}_PCA.png"

    np.save(label_npy, labels)
    np.save(centers_npy, kmeans.cluster_centers_)

    print(f"[INFO] KMeans finished: k={n_clusters}")
    print(f"[INFO] labels saved: {label_npy}")
    print(f"[INFO] centers saved: {centers_npy}")
    print(f"[INFO] cluster sizes: {np.bincount(labels)}")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    plt.colorbar(sc, label="Cluster ID")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"AE Latent KMeans (k={n_clusters}) PCA Projection (mixed test)")
    plt.tight_layout()
    plt.savefig(pca_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] PCA figure saved: {pca_fig}")

    return labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    latents = extract_latent_features(MODEL_PATH)
    best_k, _ = scan_k_by_silhouette(latents, k_min=K_MIN, k_max=K_MAX)
    cluster_and_visualize(latents, n_clusters=best_k)


if __name__ == "__main__":
    main()
