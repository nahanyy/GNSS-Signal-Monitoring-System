import json
from pathlib import Path
import numpy as np
import joblib

KMEANS_LABEL_NPY = Path("outputs/test_clustering/kmeans_labels_k9.npy")
LLM_OUTPUT_TXT = Path("outputs/llm/llm_output.txt")

GMM_OUT_DIR = Path("outputs/gmm_out")
GMM_PKL = GMM_OUT_DIR / "gmm.pkl"
SCALER_PKL = GMM_OUT_DIR / "latent_scaler.pkl"
PCA_PKL = GMM_OUT_DIR / "pca.pkl"  

Z_TRAIN_NPY = GMM_OUT_DIR / "z_train.npy"
Z_TEST_NPY = GMM_OUT_DIR / "z_test.npy"
Z_TRAIN_PCA_NPY = GMM_OUT_DIR / "z_train_pca.npy" 
Z_TEST_PCA_NPY = GMM_OUT_DIR / "z_test_pca.npy"    

OUT_TXT = Path("outputs/fusion/fused_results.jsonl")

EPS = 1e-12
CALIB_Q_LOW = 0.9
CALIB_Q_HIGH = 0.995
BOUNDARY_BAND = 0.10
MIN_GATE = 0.30

FINAL_THRESHOLD = 0.50

def load_llm_cluster_risk(path: Path):
    """
    Parse an LLM output text that contains a JSON array.
    Accepts leading/trailing noise: extracts the first '[' ... last ']'.
    Returns:
      cluster2risk: cluster_id -> risk in [0, 1]
      cluster2label: optional cluster_id -> label
      cluster2reason: optional cluster_id -> reason
    """
    raw = path.read_text(encoding="utf-8").strip()
    l = raw.find("[")
    r = raw.rfind("]")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("Invalid LLM output: cannot find a JSON array '[...]'.")

    arr = json.loads(raw[l:r + 1])

    cluster2risk = {}
    cluster2label = {}
    cluster2reason = {}

    for item in arr:
        cid = int(item["cluster_id"])
        risk_pct = item.get("jamming_risk", item.get("risk", 0))
        risk01 = float(risk_pct) / 100.0
        risk01 = max(0.0, min(1.0, risk01))
        cluster2risk[cid] = risk01

        if "predicted_label" in item:
            cluster2label[cid] = str(item["predicted_label"])
        if "reason" in item:
            cluster2reason[cid] = str(item["reason"])

    return cluster2risk, cluster2label, cluster2reason


def entropy_from_resp(resp: np.ndarray) -> np.ndarray:
    return -np.sum(resp * np.log(resp + EPS), axis=1)


def calibrate_p_gmm_by_train_quantiles(
    nll_train: np.ndarray,
    nll_test: np.ndarray,
    q_low: float,
    q_high: float
) -> np.ndarray:
    t_low = float(np.quantile(nll_train, q_low))
    t_high = float(np.quantile(nll_train, q_high))
    t_high = max(t_high, t_low + 1e-9)
    p = (nll_test - t_low) / (t_high - t_low)
    return np.clip(p, 0.0, 1.0)


def load_features_with_optional_pca(z_train: np.ndarray, z_test: np.ndarray):
    """
    Priority:
      1) If z_train_pca.npy and z_test_pca.npy exist -> use them
      2) Else if pca.pkl exists -> transform z_train/z_test
      3) Else -> use raw z_train/z_test
    """
    if Z_TRAIN_PCA_NPY.exists() and Z_TEST_PCA_NPY.exists():
        return (
            np.load(Z_TRAIN_PCA_NPY).astype(np.float64),
            np.load(Z_TEST_PCA_NPY).astype(np.float64),
        )

    if PCA_PKL.exists():
        pca = joblib.load(PCA_PKL)
        return (
            pca.transform(z_train).astype(np.float64),
            pca.transform(z_test).astype(np.float64),
        )

    return z_train.astype(np.float64), z_test.astype(np.float64)


def require_exists(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main():
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    require_exists(KMEANS_LABEL_NPY, LLM_OUTPUT_TXT, GMM_PKL, SCALER_PKL, Z_TRAIN_NPY, Z_TEST_NPY)

    clus = np.load(KMEANS_LABEL_NPY).astype(int)
    n = int(clus.shape[0])
    k = int(clus.max()) + 1

    cluster2risk, cluster2label, cluster2reason = load_llm_cluster_risk(LLM_OUTPUT_TXT)
    missing = [cid for cid in range(k) if cid not in cluster2risk]
    if missing:
        raise ValueError(f"LLM output is missing risk entries for cluster_id: {missing}")

    gmm = joblib.load(GMM_PKL)
    scaler = joblib.load(SCALER_PKL)

    z_train = np.load(Z_TRAIN_NPY)
    z_test = np.load(Z_TEST_NPY)

    if int(z_test.shape[0]) != n:
        raise ValueError(
            f"z_test samples ({z_test.shape[0]}) != kmeans_labels samples ({n}). "
            "Ensure KMeans and latent extraction use the same test set and order."
        )

    z_train_feat, z_test_feat = load_features_with_optional_pca(z_train, z_test)

    z_train_std = scaler.transform(z_train_feat)
    z_test_std = scaler.transform(z_test_feat)

    resp_train = gmm.predict_proba(z_train_std)
    nll_train = -gmm.score_samples(z_train_std)

    resp_test = gmm.predict_proba(z_test_std)
    nll_test = -gmm.score_samples(z_test_std)
    h_test = entropy_from_resp(resp_test)

    risk_llm = np.array([cluster2risk[int(c)] for c in clus], dtype=np.float64)

    p_gmm = calibrate_p_gmm_by_train_quantiles(
        nll_train=nll_train,
        nll_test=nll_test,
        q_low=CALIB_Q_LOW,
        q_high=CALIB_Q_HIGH,
    )

    gate = 1.0 - p_gmm
    boundary_mask = (np.abs(risk_llm - FINAL_THRESHOLD) <= BOUNDARY_BAND)
    gate = np.where(boundary_mask, np.minimum(gate, MIN_GATE), gate)

    final_risk = gate * risk_llm + (1.0 - gate) * p_gmm
    final_label = np.where(final_risk >= FINAL_THRESHOLD, "abnormal", "normal")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for i in range(n):
            cid = int(clus[i])
            rec = {
                "sample_index": int(i),
                "cluster_id": cid,
                "llm_cluster_risk": float(risk_llm[i]),
                "gmm_nll": float(nll_test[i]),
                "gmm_entropy": float(h_test[i]),
                "p_gmm": float(p_gmm[i]),
                "gate_weight": float(gate[i]),
                "final_risk": float(final_risk[i]),
                "final_label": str(final_label[i]),
                "is_boundary": bool(boundary_mask[i]),
            }
            if cid in cluster2label:
                rec["llm_cluster_label"] = cluster2label[cid]
            if cid in cluster2reason:
                rec["llm_reason"] = cluster2reason[cid]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
