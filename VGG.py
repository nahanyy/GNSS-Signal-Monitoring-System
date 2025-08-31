# VGG
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

L1_FOLDER = r'E:/L1'   # L1-band file
L2_FOLDER = r'E:/L2'   # L2-band file
SPOOF_START = datetime.strptime('2023-12-21 12-32-10', '%Y-%m-%d %H-%M-%S')
SPOOF_END   = datetime.strptime('2023-12-21 12-49-10', '%Y-%m-%d %H-%M-%S')

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
VAL_SPLIT = 0.9     
RANDOM_STATE = 42
SAVE_BEST_PATH = '/vgg_dualband_best.pth'
SAVE_FEAT_PATH = '/vgg_dualband_feat.pth'
USE_CLASS_WEIGHTS = True  

class CN0DualBandDataset(Dataset):

    def __init__(self, l1_dir, l2_dir):
        self.l1_dir = l1_dir
        self.l2_dir = l2_dir
        self.files = []
        self.labels = []

        l1_files = sorted([f for f in os.listdir(l1_dir) if f.endswith('.xlsx')])
        for f in l1_files:
            l1_path = os.path.join(l1_dir, f)
            l2_path = os.path.join(l2_dir, f)
            if not os.path.exists(l2_path):
                continue  

            time_str = f.replace('.xlsx', '')
            try:
                dt = datetime.strptime(time_str, '%Y-%m-%d %H-%M-%S')
            except:
                continue

            label = 1 if (SPOOF_START <= dt <= SPOOF_END) else 0
            self.files.append(f)
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        l1 = pd.read_excel(os.path.join(self.l1_dir, fname), header=None).values.astype(np.float32)  
        l2 = pd.read_excel(os.path.join(self.l2_dir, fname), header=None).values.astype(np.float32)  # (32,30)

        x = np.stack([l1, l2], axis=0)  # (2, 32, 30)
        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class VGGSpoofClassifierDual(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  

            # Block 3
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

def stratified_split_indices(labels, test_size=0.9, random_state=42):

    y = np.array(labels)
    n = len(y)
    unique_classes = sorted(set(y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, val_idx = next(sss.split(np.zeros(n), y))

        if set(y[train_idx]) == set(unique_classes) and set(y[val_idx]) == set(unique_classes):
            return train_idx.tolist(), val_idx.tolist()
    except Exception:
        pass

    rng = np.random.default_rng(random_state)
    train_idx, val_idx = [], []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)

        n_train_cls = int(round((1 - test_size) * n_cls))

        if n_cls >= 2:
            n_train_cls = max(1, min(n_train_cls, n_cls - 1))
        else:
            n_train_cls = 1  

        train_idx.extend(cls_idx[:n_train_cls].tolist())
        val_idx.extend(cls_idx[n_train_cls:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy().tolist())
            labels_all.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec = recall_score(labels_all, preds_all, zero_division=0)
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    return acc, prec, rec, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = CN0DualBandDataset(L1_FOLDER, L2_FOLDER)
    idx_all = list(range(len(dataset)))
    labels_all = [dataset.labels[i] for i in idx_all]

    if len(idx_all) == 0:
        print("Dataset is empty")
        return

    train_idx, val_idx = stratified_split_indices(
        labels_all,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE
    )

    all_cnt = Counter(labels_all)
    trn_cnt = Counter([dataset.labels[i] for i in train_idx])
    val_cnt = Counter([dataset.labels[i] for i in val_idx])
    print('All  :', all_cnt)
    print('Train:', trn_cnt)
    print('Val  :', val_cnt)

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = VGGSpoofClassifierDual(in_channels=2, num_classes=2).to(device)

    if USE_CLASS_WEIGHTS:
        num_neg = sum(1 for i in train_idx if dataset.labels[i] == 0)
        num_pos = sum(1 for i in train_idx if dataset.labels[i] == 1)
        if num_neg == 0 or num_pos == 0:
            print("One class in the training set is 0.")
            criterion = nn.CrossEntropyLoss()
        else:
            weights = torch.tensor([1.0/num_neg, 1.0/num_pos], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Class weights (neg,pos): ({weights[0].item():.6f}, {weights[1].item():.6f})")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, corr, tot = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            corr += (pred == y).sum().item()
            tot += y.size(0)

        train_acc = corr / max(tot, 1)
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | TrainLoss {running_loss/max(tot,1):.4f} "
              f"| TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} "
              f"| P {val_prec:.4f} R {val_rec:.4f} F1 {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_BEST_PATH)
            print(f"  ✅ Saved best to {SAVE_BEST_PATH} (ValAcc={best_acc:.4f})")

    torch.save(model.state_dict(), SAVE_FEAT_PATH)
    print(f"  ✅ Also saved feature-extractor weights to {SAVE_FEAT_PATH}")

    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print(f"\nFinal on Val | Acc {acc:.4f} | P {prec:.4f} | R {rec:.4f} | F1 {f1:.4f}")

if __name__ == "__main__":
    main()