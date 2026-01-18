import os
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

AE_MODEL_PATH = Path("outputs/cnn_ae.pth")

TRAIN_DIR = Path("data/gmm_train")   
TEST_DIR  = Path("data/gmm_test")    

OUTPUT_DIR = Path("outputs/gmm_out")

LATENT_DIM = 128
BATCH_SIZE = 64
SEED = 42

# Must be consistent with AE training
USE_CHANNEL_ATTENTION = True

PCA_DIM = 32
PCA_WHITEN = False

GMM_COMPONENTS = 6          
COV_TYPE = "diag"          
REG_COVAR = 1e-4          
MAX_ITER = 500

REUSE_TRAIN_LATENT = False
REUSE_TEST_LATENT  = False
REUSE_PCA          = False
REUSE_SCALER       = False
REUSE_GMM          = False

# File names
TRAIN_LATENT_NPY = OUTPUT_DIR / "z_train.npy"
TEST_LATENT_NPY  = OUTPUT_DIR / "z_test.npy"
TRAIN_PCA_NPY    = OUTPUT_DIR / "z_train_pca.npy"
TEST_PCA_NPY     = OUTPUT_DIR / "z_test_pca.npy"

TRAIN_FILELIST   = OUTPUT_DIR / "train_file_list.txt"
TEST_FILELIST    = OUTPUT_DIR / "test_file_list.txt"

PCA_PKL    = OUTPUT_DIR / "pca.pkl"
SCALER_PKL = OUTPUT_DIR / "latent_scaler.pkl"
GMM_PKL    = OUTPUT_DIR / "gmm.pkl"


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
    """
    Each JSON: shape (32, 6) -> transpose to (6, 32)
    The dataset reads JSON files from ONE directory only.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"[ERROR] Data directory not found: {self.data_dir}")

        self.file_paths = sorted(glob(str(self.data_dir / "*.json")))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"[ERROR] No JSON files found in: {self.data_dir}")

        print(f"[INFO] Found {len(self.file_paths)} JSON files in: {self.data_dir}")

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
    """
    Encoder definition aligned with your AE training script.
    Decoder is kept for compatibility with weight loading (strict=True),
    but only encode() is used in this script.
    """
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

        h = torch.cat([h_obs, h_pvt], dim=1)  
        if self.use_ca and self.ca is not None:
            h = self.ca(h)

        b, c, l = h.shape
        z = self.enc_fc(h.view(b, -1))
        return z


def load_ae(model_path: Path, device: torch.device) -> DualBranchCNN1DAE:
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] AE weight file not found: {model_path}")

    model = DualBranchCNN1DAE(latent_dim=LATENT_DIM, use_channel_attention=USE_CHANNEL_ATTENTION).to(device)
    state = torch.load(model_path, map_location=device)

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] Loaded AE weights: {model_path}")
    return model


@torch.no_grad()
def extract_latent(
    model: DualBranchCNN1DAE,
    data_dir: Path,
    out_npy: Path,
    out_filelist: Path,
    reuse: bool = False
) -> np.ndarray:
    if reuse and out_npy.exists() and out_filelist.exists():
        z = np.load(out_npy)
        print(f"[INFO] Reusing latent: {out_npy}, shape={z.shape}")
        return z

    ds = GNSSJsonDataset(data_dir)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    device = next(model.parameters()).device
    zs = []
    files = []

    for x, fpaths in tqdm(loader, desc=f"Extract latent ({out_npy.name})"):
        x = x.to(device)
        z = model.encode(x).cpu().numpy()
        zs.append(z)
        files.extend(list(fpaths))

    z_all = np.concatenate(zs, axis=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, z_all)

    with open(out_filelist, "w", encoding="utf-8") as f:
        for fp in files:
            f.write(str(fp) + "\n")

    return z_all

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ae = load_ae(AE_MODEL_PATH, device)
    z_train = extract_latent(
        model=ae,
        data_dir=TRAIN_DIR,
        out_npy=TRAIN_LATENT_NPY,
        out_filelist=TRAIN_FILELIST,
        reuse=REUSE_TRAIN_LATENT
    )

    if REUSE_PCA and PCA_PKL.exists():
        pca = joblib.load(PCA_PKL)
        print(f"[INFO] Reusing PCA: {PCA_PKL}")
    else:
        pca = PCA(n_components=PCA_DIM, whiten=PCA_WHITEN, random_state=SEED)
        pca.fit(z_train)
        joblib.dump(pca, PCA_PKL)
        print(f"[INFO] Saved PCA: {PCA_PKL}")
        print(f"[SANITY] PCA explained_var_ratio sum = {float(np.sum(pca.explained_variance_ratio_)):.4f}")

    z_train_pca = pca.transform(z_train)
    np.save(TRAIN_PCA_NPY, z_train_pca)
    print(f"[INFO] Saved z_train_pca: {TRAIN_PCA_NPY}, shape={z_train_pca.shape}")

    if REUSE_SCALER and SCALER_PKL.exists():
        scaler = joblib.load(SCALER_PKL)
        print(f"[INFO] Reusing scaler: {SCALER_PKL}")
    else:
        scaler = StandardScaler()
        scaler.fit(z_train_pca)
        joblib.dump(scaler, SCALER_PKL)
        print(f"[INFO] Saved scaler: {SCALER_PKL}")

    z_train_std = scaler.transform(z_train_pca)

    if REUSE_GMM and GMM_PKL.exists():
        gmm = joblib.load(GMM_PKL)
        print(f"[INFO] Reusing GMM: {GMM_PKL}")
    else:
        gmm = GaussianMixture(
            n_components=GMM_COMPONENTS,
            covariance_type=COV_TYPE,
            reg_covar=REG_COVAR,
            max_iter=MAX_ITER,
            random_state=SEED,
            init_params="kmeans"
        )
        gmm.fit(z_train_std)
        joblib.dump(gmm, GMM_PKL)
        print(f"[INFO] Saved GMM: {GMM_PKL}")

    nll_train = -gmm.score_samples(z_train_std)
    print(f"[SANITY] nll_train min/med/max = {nll_train.min():.4f} / {np.median(nll_train):.4f} / {nll_train.max():.4f}")

    z_test = extract_latent(
        model=ae,
        data_dir=TEST_DIR,
        out_npy=TEST_LATENT_NPY,
        out_filelist=TEST_FILELIST,
        reuse=REUSE_TEST_LATENT
    )

    z_test_pca = pca.transform(z_test)
    np.save(TEST_PCA_NPY, z_test_pca)
    print(f"[INFO] Saved z_test_pca: {TEST_PCA_NPY}, shape={z_test_pca.shape}")

    z_test_std = scaler.transform(z_test_pca)

    resp = gmm.predict_proba(z_test_std)    
    nll = -gmm.score_samples(z_test_std)    

    resp_npy = OUTPUT_DIR / "test_gmm_resp.npy"
    nll_npy  = OUTPUT_DIR / "test_gmm_nll.npy"
    np.save(resp_npy, resp)
    np.save(nll_npy, nll)

    out_csv = OUTPUT_DIR / "test_gmm_scores.csv"
    df = pd.DataFrame({
        "idx": np.arange(len(nll)),
        "nll": nll,
        "nll_per_dim": nll / float(PCA_DIM),
        "max_resp": resp.max(axis=1),
        "argmax_comp": resp.argmax(axis=1),
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
