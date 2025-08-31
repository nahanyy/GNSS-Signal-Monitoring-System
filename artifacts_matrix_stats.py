import os
import numpy as np
import pandas as pd

L1_FOLDER = r"E:/L1"
L2_FOLDER = r"E:/L2"

OUT_DIR   = os.path.join(os.path.dirname(__file__), "artifacts_matrix_stats")
os.makedirs(OUT_DIR, exist_ok=True)

def is_visible(series: np.ndarray) -> bool:
    return np.any(series != 0)

def compute_visible_stats(matrix: np.ndarray):

    assert matrix.shape == (32, 30), f"{matrix.shape}"
    stats = {}
    for i in range(32):
        row = matrix[i].astype(np.float32)
        if not is_visible(row):
            continue
        stats[i+1] = {
            "mean": float(np.mean(row)),
            "var":  float(np.var(row)),
            "min":  float(np.min(row)),
            "max":  float(np.max(row)),
            "diffs": np.diff(row).astype(float).tolist()
        }
    return stats

def fmt_float(x, nd=2):
    return f"{x:.{nd}f}"

def format_block(stats: dict, band_label: str) -> str:
    """
    Format the results of compute_visible_stats into a Chinese-formatted text block:
    [1] Mean / Variance
    [2] Minimum / Maximum
    [3] Carrier-to-noise ratio variation (differences)
    """
    if not stats:
        return f"[{band_label}] 可见卫星：0 个（全部不可见或数据为 0）\n"

    sat_ids = sorted(stats.keys())
    lines = []
    lines.append(f"[{band_label}] 可见卫星（{len(sat_ids)} 个）：{', '.join(map(str, sat_ids))}\n")

    lines.append("【1】均值 / 方差：")
    for sid in sat_ids:
        s = stats[sid]
        lines.append(f"  卫星 {sid}: 均值 = {fmt_float(s['mean'])}, 方差 = {fmt_float(s['var'])}")

    lines.append("\n【2】最小值 / 最大值：")
    for sid in sat_ids:
        s = stats[sid]
        lines.append(f"  卫星 {sid}: 最小 = {fmt_float(s['min'], 2)}, 最大 = {fmt_float(s['max'], 2)}")

    lines.append("\n【3】载噪比变化率（差分）：")
    for sid in sat_ids:
        s = stats[sid]
        diffs_str = ", ".join(fmt_float(v, 1) for v in s["diffs"])
        lines.append(f"  卫星 {sid}: 差分 = [{diffs_str}]")

    return "\n".join(lines) + "\n"

def build_matrix_stats_text(L1: np.ndarray, L2: np.ndarray) -> str:

    l1_stats = compute_visible_stats(L1)
    l2_stats = compute_visible_stats(L2)

    text_parts = []
    text_parts.append("[Matrix Statistical Feature Description (Chinese)]\n")
    text_parts.append(format_block(l1_stats, "L1"))
    text_parts.append(format_block(l2_stats, "L2"))
    return "\n".join(text_parts)

def main():
    l1_files = sorted([f for f in os.listdir(L1_FOLDER) if f.lower().endswith(".xlsx")])
    total = 0
    paired = 0

    for fname in l1_files:
        l1_path = os.path.join(L1_FOLDER, fname)
        l2_path = os.path.join(L2_FOLDER, fname)
        total += 1
        if not os.path.exists(l2_path):
            continue

        L1 = pd.read_excel(l1_path, header=None).values.astype(np.float32)
        L2 = pd.read_excel(l2_path, header=None).values.astype(np.float32)
        if L1.shape != (32, 30) or L2.shape != (32, 30):
            continue

        desc_text = build_matrix_stats_text(L1, L2)

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUT_DIR, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(desc_text)

        paired += 1
        if paired % 20 == 0:
            print(f"Progress: {paired} out of {total} generated")

    print(f"✅ Completed: processed {total} files, successfully saved {paired}. Results saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
