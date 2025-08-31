# database
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit

L1_FOLDER       = r"E:/L1"
L2_FOLDER       = r"E:/L1"
CKPT_PATH       = r"/vgg_dualband_best.pth"

CODEBOOK_PATH   = r"/codebook_train_only.npz"   
EMB_KB_PATH     = r"/embedding_kb_train_only.npz"  

K_CLUSTERS      = 50
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE    = 42
TEST_SIZE       = 0.9        
BATCH_SIZE_EVAL = 128       

SPOOF_START = datetime.strptime('2023-12-21 12-32-10', '%Y-%m-%d %H-%M-%S')
SPOOF_END   = datetime.strptime('2023-12-21 12-49-10', '%Y-%m-%d %H-%M-%S')

class CN0DualBandDataset(Dataset):

    def __init__(self, l1_dir, l2_dir):
        self.l1_dir = l1_dir
        self.l2_dir = l2_dir
        self.files = []
        self.labels = []

        l1_files = sorted([f for f in os.listdir(l1_dir) if f.endswith('.xlsx')])
        for f in l1_files:
            l2_path = os.path.join(l2_dir, f)
            if not os.path.exists(l2_path):
                continue
            time_str = f.replace('.xlsx', '')
            try:
                dt = datetime.strptime(time_str, '%Y-%m-%d %H-%M-%S')
            except:
                continue
            label = 'spoof' if (SPOOF_START <= dt <= SPOOF_END) else 'normal'
            self.files.append(f)
            self.labels.append(label)

        assert len(self.files) > 0, "No available .xlsx data found in L1_FOLDER."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        l1 = pd.read_excel(os.path.join(self.l1_dir, fname), header=None).values.astype(np.float32)
        l2 = pd.read_excel(os.path.join(self.l2_dir, fname), header=None).values.astype(np.float32)
        if l1.shape != (32, 30) or l2.shape != (32, 30):
            raise ValueError(f"{fname} size should be 32x30, but got L1={l1.shape}, L2={l2.shape}")

        x = np.stack([l1, l2], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.labels[idx]
        return x, y, fname

class VGGSpoofClassifierDual(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_feat=False):
        feat = self.features(x)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits

def stratified_split(files, labels, test_size=0.1, random_state=42):
    y = np.array([0 if l=='normal' else 1 for l in labels])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_all = np.arange(len(files))
    train_idx, val_idx = next(sss.split(idx_all, y))
    return train_idx.tolist(), val_idx.tolist()

def compute_log_odds(token_counts, total_counts, K, prior=0.1):
    classes = ['normal', 'jam', 'spoof']  
    for c in classes:
        token_counts.setdefault(c, Counter())
        total_counts.setdefault(c, 0)
    token_logodds = {}
    for t in range(K):
        token_logodds[t] = {}
        for c in classes:
            count_c = token_counts[c][t] + prior
            total_c = total_counts[c] + K * prior
            p_c = count_c / max(total_c, 1e-8)

            other = [oc for oc in classes if oc != c]
            count_bg = sum(token_counts[oc][t] for oc in other) + prior
            total_bg = sum(total_counts[oc] for oc in other) + K * prior
            p_bg = count_bg / max(total_bg, 1e-8)

            logit = np.log(p_c / (1 - p_c + 1e-12)) - np.log(p_bg / (1 - p_bg + 1e-12))
            token_logodds[t][c] = float(logit)
    return token_logodds

def main():
    device = torch.device(DEVICE)
    ds = CN0DualBandDataset(L1_FOLDER, L2_FOLDER)
    print(f"Total samples: {len(ds)}")

    train_idx, val_idx = stratified_split(ds.files, ds.labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    def count_labels(idxs):
        n_normal = sum(1 for i in idxs if ds.labels[i] == 'normal')
        n_spoof  = sum(1 for i in idxs if ds.labels[i] == 'spoof')
        return n_normal, n_spoof
    
    tr_normal, tr_spoof = count_labels(train_idx)
    va_normal, va_spoof = count_labels(val_idx)
    print(f"Train class counts -> normal: {tr_normal}, spoof: {tr_spoof}")
    print(f"Val   class counts -> normal: {va_normal}, spoof: {va_spoof}")

    model = VGGSpoofClassifierDual(in_channels=2, num_classes=2).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights: {CKPT_PATH}")

    all_scalars = []        
    emb_list = []          
    label_list = []       
    fname_list = []        

    with torch.no_grad():

        for start in range(0, len(train_idx), BATCH_SIZE_EVAL):
            batch_ids = train_idx[start:start+BATCH_SIZE_EVAL]
            xs, ys = [], []
            fnames = []
            for i in batch_ids:
                x, y, fname = ds[i]
                xs.append(x.numpy())
                ys.append(y)
                fnames.append(fname)
            x_tensor = torch.from_numpy(np.stack(xs, axis=0)).to(device)  
            logits, feat = model(x_tensor, return_feat=True)              
            embs = feat.squeeze(-1).squeeze(-1).cpu().numpy()           

            emb_list.extend([e.copy() for e in embs])
            label_list.extend(ys)
            fname_list.extend(fnames)
            all_scalars.append(embs.reshape(-1, 1))

    all_scalars = np.concatenate(all_scalars, axis=0)  
    print(f"KMeans fit on TRAIN scalars: {all_scalars.shape}")

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE)
    kmeans.fit(all_scalars)
    centroids = kmeans.cluster_centers_  # [K,1]
    print(f"Fitted KMeans (train only): K={K_CLUSTERS}")

    token_counts = defaultdict(Counter)
    total_counts = defaultdict(int)
    for emb, y in zip(emb_list, label_list):
        d = euclidean_distances(emb.reshape(-1,1), centroids)  # [256,K]
        toks = np.argmin(d, axis=1)
        token_counts[y].update(toks.tolist())
        total_counts[y] += len(toks)

    token_logodds = compute_log_odds(token_counts, total_counts, K_CLUSTERS, prior=0.1)

    cls_order = ['normal', 'jam', 'spoof']
    token_logodds_arr = np.zeros((K_CLUSTERS, 3), dtype=np.float32)
    for t in range(K_CLUSTERS):
        token_logodds_arr[t, 0] = token_logodds[t]['normal']
        token_logodds_arr[t, 1] = token_logodds[t]['jam']
        token_logodds_arr[t, 2] = token_logodds[t]['spoof']

    np.savez_compressed(
        CODEBOOK_PATH,
        centroids=centroids,
        token_logodds=token_logodds_arr,
        classes=np.array(cls_order),
        K=np.array([K_CLUSTERS]),
        total_counts_normal=np.array([total_counts['normal']]),
        total_counts_jam=np.array([total_counts['jam']]),
        total_counts_spoof=np.array([total_counts['spoof']]),
        split_info=np.array(['train_only']) 
    )
    print(f"✅ Saved train-only codebook: {CODEBOOK_PATH}")

    emb_arr   = np.stack(emb_list, axis=0).astype(np.float32)   
    labels_np = np.array(label_list)
    files_np  = np.array(fname_list)
    tokens_packed = []
    for e in emb_list:
        d = euclidean_distances(e.reshape(-1,1), centroids)
        tokens_packed.append(np.argmin(d, axis=1).astype(np.int16))
    tokens_arr = np.stack(tokens_packed, axis=0)  

    np.savez_compressed(
        EMB_KB_PATH,
        embeddings=emb_arr,   
        labels=labels_np,     
        filenames=files_np,  
        tokens=tokens_arr     
    )
    print(f"✅ Saved training embedding KB: {EMB_KB_PATH}")
    print(f"   Train samples in KB: {emb_arr.shape[0]}")

if __name__ == "__main__":
    main()