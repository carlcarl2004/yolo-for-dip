#!/usr/bin/env python
"""Build the final comparison figure from results/eval_all/*.json."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "evidence", "eval_all")
OUT = os.path.join(HERE, "results", "final_compare.png")

rows = {}
for f in glob.glob(os.path.join(SRC, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    if not isinstance(d, dict) or "tag" not in d:
        continue
    tag = d["tag"]
    if "__" not in tag or tag.split("__")[1] != "clean_val" and tag.split("__")[1] != "leaky_val":
        pass
    model, ds = tag.split("__")
    rows[(model, ds)] = d

models = sorted({m for m, _ in rows})
order = ["yolov8n_640_base", "yolov8s_640_base", "yolov8m_640_base",
         "yolov8s_640_rot", "yolov8s_960m", "yolov8n_960m"]
models = [m for m in order if m in models] + [m for m in models if m not in order]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

ax = axes[0]
x = range(len(models))
w = 0.38
leak = [rows.get((m, "leaky_val"), {}).get("mAP50", 0) for m in models]
clean = [rows.get((m, "clean_val"), {}).get("mAP50", 0) for m in models]
b1 = ax.bar([i - w / 2 for i in x], leak, w, label="旧验证集（有泄漏，73张）", color="#c9d6ff")
b2 = ax.bar([i + w / 2 for i in x], clean, w, label="干净验证集（无泄漏，33张）", color="#2f6fed")
ax.bar_label(b1, fmt="%.3f", fontsize=8)
ax.bar_label(b2, fmt="%.3f", fontsize=8, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
ax.set_ylabel("mAP50")
ax.set_title("数据泄漏让成绩虚高约 50%")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(leak + clean) * 1.25)

ax = axes[1]
for m in models:
    d = rows.get((m, "clean_val"))
    if not d:
        continue
    cs = sorted(d["conf_sweep"], key=float)
    ax.plot([float(c) for c in cs], [d["conf_sweep"][c]["R"] for c in cs],
            marker="o", ms=3, label=m.replace("_", " ") + " recall")
ax.set_xlabel("置信度阈值 conf")
ax.set_ylabel("召回率 (干净验证集)")
ax.set_title("提高置信度阈值换来的只是大量漏检")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)

rows_txt = []
for m in models:
    l = rows.get((m, "leaky_val"), {})
    c = rows.get((m, "clean_val"), {})
    rows_txt.append("%-18s leaky mAP50=%.4f  clean mAP50=%.4f  clean@conf0.5 P=%s R=%s" % (
        m, l.get("mAP50", -1), c.get("mAP50", -1),
        c.get("conf_sweep", {}).get("0.50", {}).get("P"),
        c.get("conf_sweep", {}).get("0.50", {}).get("R")))
open(os.path.join(HERE, "results", "evidence", "final_table.txt"), "w", encoding="utf-8").write("\n".join(rows_txt) + "\n")
print("\n".join(rows_txt))