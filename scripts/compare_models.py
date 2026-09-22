#!/usr/bin/env python
"""三个模型横向对比：精度 / 速度 / 体积 + 并排可视化"""
import os, glob, json, time
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

DATA = "/root/autodl-tmp/datasets/trash_merged_bin/data.yaml"
VIMG = "/root/autodl-tmp/datasets/trash_merged_bin/val/images"
VLAB = "/root/autodl-tmp/datasets/trash_merged_bin/val/labels"
OUT  = "/root/autodl-tmp/compare"
os.makedirs(OUT, exist_ok=True)

RUNS = [("yolov8n (tiny)", "n_merged"),
        ("yolov8s (small)", "s_merged"),
        ("yolov8m (medium)", "m_merged")]

def load_gt(p):
    f = os.path.join(VLAB, os.path.splitext(os.path.basename(p))[0] + ".txt")
    out = []
    if os.path.exists(f):
        for line in open(f):
            s = line.split()
            if len(s) >= 5:
                x, y, w, h = (float(v) for v in s[1:5])
                out.append([x-w/2, y-h/2, x+w/2, y+h/2])   # xyxy 归一化
    return np.array(out) if out else np.zeros((0,4))

def iou(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:,None,0], b[None,:,0]); y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2]); y2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(x2-x1,0,None) * np.clip(y2-y1,0,None)
    ar = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]); br = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter / (ar[:,None] + br[None,:] - inter + 1e-9)

files = sorted(glob.glob(os.path.join(VIMG, "*")))
GT = {f: load_gt(f) for f in files}
n_gt = sum(len(v) for v in GT.values())

report, store = {}, {}
for label, run in RUNS:
    w = f"/root/autodl-tmp/trash_project/runs/{run}/weights/best.pt"
    if not os.path.exists(w):
        print(f"-- 跳过 {label}（{w} 不存在）"); continue
    model = YOLO(w)
    v = model.val(data=DATA, split="val", imgsz=640, batch=16, conf=0.001, iou=0.6,
                  plots=False, verbose=False, project=OUT, name=f"val_{run}")
    p, r = float(v.box.mp), float(v.box.mr)
    size_mb = os.path.getsize(w) / 1e6
    n_par = sum(x.numel() for x in model.model.parameters()) / 1e6

    # 推理速度
    model.predict(files[:20], imgsz=640, verbose=False)
    t0 = time.time(); model.predict(files, imgsz=640, conf=0.25, verbose=False)
    ms = (time.time()-t0)/len(files)*1000

    # 宽松口径：IoU 0.25 下算 P/R/F1（应对标注不精细）
    res = model.predict(files, imgsz=640, conf=0.25, iou=0.6, verbose=False)
    tp = fp = fn = 0
    preds_by_file = {}
    for f, rr in zip(files, res):
        pb = rr.boxes.xyxy.cpu().numpy().copy()
        img = np.array(Image.open(f).size)          # (w,h)
        pb = pb / np.array([img[0], img[1], img[0], img[1]])
        preds_by_file[f] = pb
        g = GT[f]; M = iou(pb, g)
        used_g, used_p = set(), set()
        if M.size:
            order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
            for i, j in order:
                if M[i,j] < 0.25: break
                if i in used_p or j in used_g: continue
                used_p.add(i); used_g.add(j)
        tp += len(used_p); fp += len(pb)-len(used_p); fn += len(g)-len(used_g)
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)

    report[label] = {
        "run": run, "mAP50": round(float(v.box.map50),4), "mAP50_95": round(float(v.box.map),4),
        "precision@.001": round(p,4), "recall@.001": round(r,4),
        "relaxed_P_iou25": round(prec,4), "relaxed_R_iou25": round(rec,4),
        "relaxed_F1_iou25": round(2*prec*rec/(prec+rec+1e-9),4),
        "ms_per_img": round(ms,1), "params_M": round(n_par,1), "size_MB": round(size_mb,1),
    }
    store[label] = preds_by_file
    print(f"{label}: mAP50={v.box.map50:.3f} mAP50-95={v.box.map:.3f} "
          f"宽松F1={2*prec*rec/(prec+rec+1e-9):.3f} {ms:.1f}ms {n_par:.1f}M {size_mb:.1f}MB")

open(os.path.join(OUT,"compare.json"),"w").write(json.dumps(report, ensure_ascii=False, indent=1))

# ---------- 可视化：挑 6 张真实图，逐模型并排 ----------
sel = [f for f in files if len(GT[f]) >= 1][:6]
CELL, COLS = 300, len(sel)
rows = len(store) + 1
grid = Image.new("RGB", (CELL*COLS, CELL*rows), (18,18,18))
gd = ImageDraw.Draw(grid)
for c, f in enumerate(sel):
    im = Image.open(f).convert("RGB"); Wd, Hd = im.size
    d = ImageDraw.Draw(im)
    for x1,y1,x2,y2 in GT[f]:
        d.rectangle([x1*Wd, y1*Hd, x2*Wd, y2*Hd], outline=(0,255,90), width=max(3, Wd//170))
    im.thumbnail((CELL,CELL)); grid.paste(im, (c*CELL, 0))
    gd.text((c*CELL+6, 6), "你标的 (绿)", fill=(0,255,90))
gd.rectangle([0,0,CELL*COLS-1,CELL-1], outline=(70,70,70))

for r, (label, _) in enumerate(store.items(), start=1):
    for c, f in enumerate(sel):
        im = Image.open(f).convert("RGB"); Wd, Hd = im.size
        d = ImageDraw.Draw(im)
        for x1,y1,x2,y2 in store[label][f]:
            d.rectangle([x1*Wd, y1*Hd, x2*Wd, y2*Hd], outline=(255,140,0), width=max(3, Wd//170))
        im.thumbnail((CELL,CELL)); grid.paste(im, (c*CELL, r*CELL))
        d2 = ImageDraw.Draw(grid)
        d2.rectangle([c*CELL, r*CELL, c*CELL+CELL-1, r*CELL+CELL-1], outline=(70,70,70))
        d2.text((c*CELL+6, r*CELL+6), label, fill=(255,200,60))

grid.save(os.path.join(OUT,"side_by_side.jpg"), quality=90)
print("SAVED", os.path.join(OUT,"side_by_side.jpg"))
print(json.dumps(report, ensure_ascii=False, indent=1))
