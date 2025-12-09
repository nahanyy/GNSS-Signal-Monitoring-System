import os
import json
import numpy as np

# ====================== Path Configuration ======================
# Hand-crafted features (.npy)
HANDCRAFT_NPY = "./output/handcrafted_features.npy"

# AE cluster labels (.npy)
AE_CLUSTER_NPY = "./output/kmeans_labels_k7.npy"

# file_list.txt (optional, used for cluster purity check)
FILE_LIST_TXT = "./output/file_list.txt"

# Output directory
OUTPUT_DIR = "./output"
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "cluster_summaries_ae_on_handcrafted.json")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "cluster_summaries_for_llm.txt")
# ======================================================


# ====================== Physical Feature Configuration ======================
# By default we use all 6 physical quantities. You can adjust if needed.
FEATURE_NAMES = [
    "cn0",
    "mean_cn0",
    "var_cn0",
    "doppler",
    "range",
    "range_residual",
]

# Human-readable English descriptions for each physical quantity
FEATURE_DESC_EN = {
    "cn0":             "carrier-to-noise density ratio (C/N0)",
    "mean_cn0":        "smoothed mean of C/N0",
    "var_cn0":         "variance of C/N0",
    "doppler":         "Doppler shift",
    "range":           "pseudorange measurement",
    "range_residual":  "pseudorange residual",
}
# ======================================================


def build_labels_from_filelist(file_list_txt: str) -> np.ndarray:
    labels = []
    with open(file_list_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for fp in lines:
        lower_fp = fp.lower()
        if "0_aa" in lower_fp:
            labels.append(0)
        elif "1_aa" in lower_fp:
            labels.append(1)
        else:
            raise ValueError(f"Cannot infer label from path: {fp}")

    labels = np.array(labels, dtype=int)
    print(f"[INFO] Built labels from file list, total = {len(labels)}, unique = {np.unique(labels)}")
    return labels


def summarize_clusters(features: np.ndarray,
                       cluster_labels: np.ndarray):
    n_samples, feat_dim = features.shape

    if (feat_dim - 1) % 3 != 0:
        raise ValueError(
            f"[ERROR] Feature dimension {feat_dim} does not match 3*M + 1. "
            f"Please check the handcrafted feature extraction logic."
        )

    M = (feat_dim - 1) // 3
    if M != len(FEATURE_NAMES):
        raise ValueError(
            f"[ERROR] Inferred number of physical quantities M={M}, "
            f"but len(FEATURE_NAMES) = {len(FEATURE_NAMES)}. Please keep them consistent."
        )

    print(
        f"[INFO] Feature dimension = {feat_dim}, "
        f"containing {M} physical quantities (each with mean/std/max) plus valid-satellite count."
    )

    cluster_ids = np.unique(cluster_labels)
    summaries = {}

    for cid in cluster_ids:
        mask = (cluster_labels == cid)
        X_c = features[mask]     # (Nc, feat_dim)
        Nc = X_c.shape[0]

        valid_sat = X_c[:, -1]
        valid_sat_stats = {
            "mean": float(np.mean(valid_sat)),
            "std": float(np.std(valid_sat)),
            "min": float(np.min(valid_sat)),
            "max": float(np.max(valid_sat)),
        }

        # Per-feature statistics
        feat_summary = {}
        for i, name in enumerate(FEATURE_NAMES):
            # Indices for mean/std/max of this feature inside the feature vector
            idx_mean = i
            idx_std = M + i
            idx_max = 2 * M + i

            sample_means = X_c[:, idx_mean]
            sample_stds = X_c[:, idx_std]
            sample_maxs = X_c[:, idx_max]

            feat_summary[name] = {
                "sample_mean_mean": float(np.mean(sample_means)),  # mean of per-sample means
                "sample_mean_std":  float(np.std(sample_means)),   # variation of per-sample means
                "sample_std_mean":  float(np.mean(sample_stds)),   # mean of per-sample std
                "sample_max_mean":  float(np.mean(sample_maxs)),   # mean of per-sample max
            }

        summaries[f"cluster_{cid}"] = {
            "cluster_id": int(cid),
            "num_samples": int(Nc),
            "ratio": float(Nc / n_samples),
            "valid_sat": valid_sat_stats,
            "features": feat_summary,
        }

    return summaries


def make_text_for_llm(summaries: dict) -> str:
    """
    Convert numerical cluster summaries into natural-language text
    for LLM consumption. No ground-truth labels are used; the text
    only reflects statistics of the clusters themselves.
    """
    lines = []

    for key, info in summaries.items():
        cid = info["cluster_id"]
        Nc = info["num_samples"]
        ratio = info["ratio"]
        vs = info["valid_sat"]

        lines.append(f"=== Cluster {cid} ===")
        lines.append(
            f"This cluster contains about {Nc} samples, "
            f"accounting for {ratio * 100:.2f}% of all samples."
        )
        lines.append(
            "Valid satellite count: mean {:.2f}, std {:.2f}, range [{:.0f}, {:.0f}].".format(
                vs["mean"], vs["std"], vs["min"], vs["max"]
            )
        )

        # Per-feature description
        for name in FEATURE_NAMES:
            desc_en = FEATURE_DESC_EN.get(name, name)
            stat = info["features"][name]

            lines.append(
                "{}: the average of per-window means is {:.3f} (variation {:.3f}), "
                "the average of per-window standard deviations is {:.3f}, "
                "and the average of per-window maxima is {:.3f}.".format(
                    desc_en,
                    stat["sample_mean_mean"],
                    stat["sample_mean_std"],
                    stat["sample_std_mean"],
                    stat["sample_max_mean"],
                )
            )

        lines.append("")  # blank line between clusters

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feats = np.load(HANDCRAFT_NPY)   # (N, D)
    clus = np.load(AE_CLUSTER_NPY)   # (N,)
    N = feats.shape[0]

    print(f"[INFO] Handcrafted features shape = {feats.shape}")
    print(f"[INFO] AE cluster labels shape = {clus.shape}, unique = {np.unique(clus)}")

    if feats.shape[0] != clus.shape[0]:
        raise ValueError("Handcrafted features and AE cluster labels have different sample counts.")

    # (Optional) Build ground-truth labels from file_list and print cluster purity
    if os.path.exists(FILE_LIST_TXT):
        y = build_labels_from_filelist(FILE_LIST_TXT)
        if len(y) != N:
            raise ValueError("Number of labels built from file_list does not match feature sample count.")

        print("\n===== Cluster vs True Label (for sanity check only) =====")
        print("cluster_id\tN\tN_normal\tN_abnormal\tabnormal_ratio")
        n_clusters = int(clus.max()) + 1
        for cid in range(n_clusters):
            mask = (clus == cid)
            Nc = np.sum(mask)
            if Nc == 0:
                continue
            y_c = y[mask]
            n_abn = np.sum(y_c == 1)
            n_norm = np.sum(y_c == 0)
            ratio = n_abn / Nc
            print(f"{cid}\t{Nc}\t{n_norm}\t{n_abn}\t{ratio:.3f}")

        print("\n[INFO] Global abnormal ratio = {:.3f}".format(np.mean(y == 1)))

    summaries = summarize_clusters(feats, clus)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Cluster summaries (JSON) saved to: {SUMMARY_JSON}")

    text_for_llm = make_text_for_llm(summaries)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(text_for_llm)
    print(f"[INFO] Cluster text summaries saved to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
