#!/usr/bin/env python
"""训练完成后：在你自己的真实场景图上评估 + 出可视化对比图"""
import os, glob, json, random
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

RUN   = os.environ.get("RUN_NAME", "s_merged")
W     = f"/root/autodl-tmp/trash_project/runs/{RUN}/weights/best.pt"
DATA  = "/root/autodl-tmp/datasets/trash_merged_bin/data.yaml"
VIMG  = "/root/autodl-tmp/datasets/trash_merged_bin/val/images"
VLAB  = "/root/autodl-tmp/datasets/trash_merged_bin/val/labels"
OUT   = f"/root/autodl-tmp/eval_{RUN}"
os.makedirs(OUT, exist_ok=True)

model = YOLO(W)
res = {}

# ---------- 1) 标准指标 ----------
m = model.val(data=DATA, split="val", imgsz=640, batch=16,
              conf=0.001, iou=0.6, plots=False, verbose=False,
              project=OUT, name="valmap")
mp, mr = float(m.box.mp), float(m.box.mr)
res["mAP50"]    = float(m.box.map50)
res["mAP50_95"] = float(m.box.map)
res["precision"]= mp
res["recall"]   = mr
res["F1"]       = 2*mp*mr/(mp+mr+1e-9)
print("== mAP ==", json.dumps({k: round(v,4) for k,v in res.items()}))

# ---------- 2) 不同置信度下的实际表现 ----------
res["at_conf"] = {}
for c in (0.10, 0.25, 0.40, 0.60):
    v = model.val(data=DATA, split="val", imgsz=640, batch=16, conf=c, iou=0.6,
                  plots=False, verbose=False, project=OUT, name=f"c{int(c*100)}")
    p, r = float(v.box.mp), float(v.box.mr)
    res["at_conf"][str(c)] = {"precision": round(p,4), "recall": round(r,4),
                              "F1": round(2*p*r/(p+r+1e-9),4)}
    print(f"== conf {c} ==", res["at_conf"][str(c)])

open(os.path.join(OUT, "metrics.json"), "w").write(json.dumps(res, indent=1))

# ---------- 3) 可视化：绿=你标的  橙=模型预测 ----------
def load_gt(p):
    f = os.path.join(VLAB, os.path.splitext(os.path.basename(p))[0] + ".txt")
    out = []
    if os.path.exists(f):
        for line in open(f):
            s = line.split()
            if len(s) >= 5:
                out.append(tuple(float(x) for x in s[1:5]))
    return out

files = sorted(glob.glob(os.path.join(VIMG, "*")))
preds = model.predict(files, imgsz=640, conf=0.25, iou=0.6, verbose=False)

CELL, COLS = 340, 4
rows = (len(files) + COLS - 1) // COLS
grid = Image.new("RGB", (CELL*COLS, CELL*rows), (20,20,20))
gd = ImageDraw.Draw(grid)
stats = []

for i, (p, r) in enumerate(zip(files, preds)):
    im = Image.open(p).convert("RGB"); Wd, Hd = im.size
    d = ImageDraw.Draw(im)
    for x, y, w, h in load_gt(p):
        d.rectangle([(x-w/2)*Wd, (y-h/2)*Hd, (x+w/2)*Wd, (y+h/2)*Hd],
                    outline=(0,255,90), width=max(3, Wd//180))
    n = 0
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        cf = float(b.conf[0]); n += 1
        col = (255,140,0) if cf >= 0.5 else (255,60,60)
        d.rectangle([x1,y1,x2,y2], outline=col, width=max(3, Wd//180))
        d.text((x1+3, max(0,y1-16)), f"{cf:.2f}", fill=col)
    stats.append({"file": os.path.basename(p), "gt": len(load_gt(p)), "pred": n})
    im.thumbnail((CELL, CELL))
    cx, cy = (i % COLS)*CELL, (i // COLS)*CELL
    grid.paste(im, (cx, cy))
    ImageDraw.Draw(grid).rectangle([cx,cy,cx+CELL-1,cy+CELL-1], outline=(60,60,60))

grid.save(os.path.join(OUT, "real_grid.jpg"), quality=88)
open(os.path.join(OUT, "per_image.json"), "w").write(json.dumps(stats, indent=1))

gt_t  = sum(s["gt"] for s in stats)
pd_t  = sum(s["pred"] for s in stats)
miss  = sum(1 for s in stats if s["gt"] > 0 and s["pred"] == 0)
print(f"共 {len(stats)} 张 | 你标了 {gt_t} 个框 | 模型检出 {pd_t} 个框 | 完全漏检的图 {miss} 张")
print("SAVED", os.path.join(OUT, "real_grid.jpg"))
