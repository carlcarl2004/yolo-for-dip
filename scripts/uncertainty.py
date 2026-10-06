#!/usr/bin/env python
"""How much can we trust these numbers?

The leak-free test set is only 33 images / 41 boxes, so any single number is noisy.
This bootstraps over images (resample with replacement) to put a 95% interval on the
frame-level metrics, which are the ones the cart actually cares about.

    python scripts/uncertainty.py

Writes results/uncertainty.png and prints a table.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "evidence", "eval_all")
OUT = os.path.join(HERE, "results", "uncertainty.png")
RNG = np.random.default_rng(0)
N_BOOT = 5000

LABEL = {
    "frame_n_merged_640_c50_iou50": "yolov8n @640  conf 0.5",
    "frame_s_merged_640_c50_iou50": "yolov8s @640  conf 0.5",
    "frame_m_merged_640_c50_iou50": "yolov8m @640  conf 0.5",
    "frame_s_960m_960_c50_iou50": "yolov8s @960  conf 0.5",
    "frame_s_ftown_640_c10_iou50": "yolov8s 微调(自采)  conf 0.1",
    "frame_s_ftown_640_c20_iou50": "yolov8s 微调(自采)  conf 0.2",
    "frame_s_ftown_640_c30_iou50": "yolov8s 微调(自采)  conf 0.3",
    "frame_s_ftstreet_640_c50_iou50": "yolov8s 微调(TACO+自采) conf 0.5",
}


def boot(rows, key):
    """Bootstrap one metric over images."""
    tp = np.array([r["tp"] for r in rows], float)
    fp = np.array([r["pred"] for r in rows], float) - tp
    fn = np.array([r["gt"] for r in rows], float) - tp
    has_gt = np.array([r["gt"] for r in rows], float) > 0
    n = len(rows)
    out = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        t, p, f = tp[idx], fp[idx], fn[idx]
        m = has_gt[idx]
        if key == "hit":
            out.append((((t > 0) & m).sum() / m.sum()) if m.sum() else np.nan)
        elif key == "recall":
            den = t.sum() + f.sum()
            out.append(t.sum() / den if den else np.nan)
        elif key == "precision":
            den = t.sum() + p.sum()
            out.append(t.sum() / den if den else np.nan)
    out = np.array(out, float)
    return np.nanmean(out), np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


def point(rows, key):
    tp = sum(r["tp"] for r in rows)
    pred = sum(r["pred"] for r in rows)
    gt = sum(r["gt"] for r in rows)
    if key == "hit":
        m = [r for r in rows if r["gt"] > 0]
        return sum(1 for r in m if r["tp"] > 0) / len(m)
    if key == "recall":
        return tp / gt if gt else float("nan")
    return tp / pred if pred else float("nan")


def main():
    files = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8"))
             for f in glob.glob(os.path.join(SRC, "frame_*_iou50.json"))}
    names = [k for k in LABEL if k in files]
    names.sort(key=lambda k: point(files[k]["per_image"], "hit"))

    print("%-34s %-24s %-24s" % ("model", "帧命中率 (95% CI)", "框召回率 (95% CI)"))
    print("-" * 88)
    for k in names:
        rows = files[k]["per_image"]
        h, hl, hh = boot(rows, "hit")
        r, rl, rh = boot(rows, "recall")
        print("%-34s %.3f  [%.3f, %.3f]      %.3f  [%.3f, %.3f]"
              % (LABEL[k], point(rows, "hit"), hl, hh, point(rows, "recall"), rl, rh))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, key, title in ((axes[0], "hit", "帧命中率：有垃圾的帧里至少框对一个"),
                           (axes[1], "recall", "框召回率：所有垃圾里框对了多少")):
        vals, los, his, labs = [], [], [], []
        for k in names:
            rows = files[k]["per_image"]
            v = point(rows, key)
            _, lo, hi = boot(rows, key)
            vals.append(v)
            los.append(v - lo)
            his.append(hi - v)
            labs.append(LABEL[k])
        y = np.arange(len(vals))
        ax.barh(y, vals, color="#2f6fed", alpha=0.8)
        ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor="#12336e", capsize=4, lw=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=8.5)
        ax.set_xlim(0, 1.0)
        ax.axvline(0.9, color="#d04040", ls="--", lw=1.5)
        ax.text(0.905, len(vals) - 0.4, "目标 90%", color="#d04040", fontsize=9)
        ax.set_xlabel("比例")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("干净验证集只有 33 张图 / 41 个框 —— 误差棒是 95% 自助法区间", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()