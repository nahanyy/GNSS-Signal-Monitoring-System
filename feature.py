import os
import json
import numpy as np
from glob import glob

# ===================== Feature Configuration =====================

# Always use all 6 features: [0, 1, 2, 3, 4, 5]
FEATURE_IDXS = [0, 1, 2, 3, 4, 5]
N_COLS = len(FEATURE_IDXS)
TAG = "all6"  # used as a suffix for output file names

# ===================== Data and Output Paths =====================

# Two normalized data folders
DATA_DIRS = ["./data"]

BASE_OUTPUT_DIR = "./output/case5/k7"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TAG)

# Paths for merged features and scene IDs
OUTPUT_FEATURE_NPY = os.path.join(OUTPUT_DIR, f"handcrafted_features_{TAG}.npy")
OUTPUT_SCENE_ID_NPY = os.path.join(OUTPUT_DIR, f"scene_ids_{TAG}.npy")


def extract_handcrafted_features(sample_matrix: np.ndarray,
                                 feature_indices: list) -> np.ndarray:
    """
    Input:
        sample_matrix: shape = (num_sats, 6), original 6 normalized features
        feature_indices: list of column indices to use, e.g. [0,1,2,3,4,5]

    Output:
        1D feature vector, length = len(feature_indices) * 3 + 1
        - For each selected column, compute mean, std, max across valid satellites
        - Additionally append the number of valid satellites
    """
    # sample_matrix: (num_sats, 6)
    # rows with all zeros are treated as invalid satellites for this time window
    valid_mask = ~(np.all(sample_matrix == 0.0, axis=1))
    valid_rows = sample_matrix[valid_mask]

    num_valid_sats = valid_rows.shape[0]

    if num_valid_sats == 0:
        feat_dim = len(feature_indices) * 3 + 1
        return np.zeros(feat_dim, dtype=np.float32)

    # select columns of interest => shape = (num_valid_sats, C)
    sub_rows = valid_rows[:, feature_indices]

    # mean/std/max for each physical feature column
    feat_mean = np.mean(sub_rows, axis=0)   # (C,)
    feat_std = np.std(sub_rows, axis=0)     # (C,)
    feat_max = np.max(sub_rows, axis=0)     # (C,)

    # concatenate: mean(C) + std(C) + max(C) + num_valid_sats(1)
    feature_vec = np.concatenate([
        feat_mean,
        feat_std,
        feat_max,
        np.array([num_valid_sats], dtype=np.float32)
    ])

    return feature_vec.astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_features = []
    all_scene_ids = []  # 0 for first folder, 1 for second folder, etc.

    print(f"[INFO] Using FEATURE_IDXS = {FEATURE_IDXS}")

    # 1. Iterate over data folders
    for scene_id, data_dir in enumerate(DATA_DIRS):
        json_files = sorted(glob(os.path.join(data_dir, "*.json")))
        if not json_files:
            print(f"[WARN] No json files found in directory: {data_dir}")
            continue

        print(f"[INFO] Found {len(json_files)} json samples in {data_dir}.")

        for idx, fpath in enumerate(json_files, start=1):
            with open(fpath, "r") as f:
                data = json.load(f)

            # data should be a 32x6 list
            arr = np.array(data, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 6:
                print(f"[WARN] File {os.path.basename(fpath)} has invalid shape: {arr.shape}, skipped.")
                continue

            feat_vec = extract_handcrafted_features(arr, FEATURE_IDXS)
            all_features.append(feat_vec)
            all_scene_ids.append(scene_id)

            if idx % 500 == 0:
                print(f"[INFO] [scene_id={scene_id}] processed {idx} / {len(json_files)} samples.")

    if not all_features:
        print("[ERROR] No features were successfully extracted.")
        return

    features = np.stack(all_features, axis=0)          # (N_total, D_feat)
    scene_ids = np.array(all_scene_ids, dtype=int)     # (N_total,)

    print(f"[INFO] Final feature matrix shape: {features.shape} (D_feat = {features.shape[1]})")
    print(f"[INFO] Scene ID array shape: {scene_ids.shape}, unique values: {np.unique(scene_ids)}")

    # 2. Save feature matrix and scene IDs
    np.save(OUTPUT_FEATURE_NPY, features)
    np.save(OUTPUT_SCENE_ID_NPY, scene_ids)
    print(f"[INFO] Hand-crafted features saved to: {OUTPUT_FEATURE_NPY}")
    print(f"[INFO] Scene IDs saved to: {OUTPUT_SCENE_ID_NPY}")


if __name__ == "__main__":
    main()
