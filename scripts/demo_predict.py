#!/usr/bin/env python
"""Render a visual demo: ground truth in green, predictions in red, side by side.

Usage: python demo_predict.py <weights> <imgsz> <data.yaml> <conf> <n_show> <out.jpg>
Picks the val images the model finds hardest first, so the picture does not flatter it.
"""
import glob, os, sys
from PIL import Image, ImageDraw
from ultralytics import YOLO

W = "/root/autodl-tmp"


def load_gt(p, w, h):
    out = []
    for line in open(p):
        t = line.split()
        if len(t) >= 5:
            cx, cy, bw, bh = (float(x) for x in t[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def main():
    weights, imgsz, data, conf, n_show, out_path = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]),
        int(sys.argv[5]), sys.argv[6])
    root = os.path.dirname(data)
    imgs = sorted(glob.glob(os.path.join(root, "val", "images", "*")))
    lab_dir = os.path.join(root, "val", "labels")
    model = YOLO(weights)
    res = model.predict(imgs, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, stream=True)

    tiles, total_gt, total_hit = [], 0, 0
    for path, r in zip(imgs, res):
        stem = os.path.splitext(os.path.basename(path))[0]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        gt = load_gt(os.path.join(lab_dir, stem + ".txt"), w, h)
        preds = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        cfs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        total_gt += len(gt)
        total_hit += len(preds)
        d = ImageDraw.Draw(im)
        for b in gt:
            d.rectangle(b, outline=(0, 255, 0), width=3)
        for b, c in zip(preds, cfs):
            d.rectangle(b, outline=(255, 40, 40), width=3)
            d.text((b[0] + 4, b[1] + 4), "%.2f" % c, fill=(255, 40, 40))
        im = im.resize((400, 400))
        d2 = ImageDraw.Draw(im)
        d2.rectangle([0, 0, 399, 20], fill=(0, 0, 0))
        d2.text((4, 4), "%s  GT=%d  pred=%d" % (stem, len(gt), len(preds)), fill=(255, 255, 255))
        tiles.append(((len(gt) - len(preds)) ** 2, stem, im))

    tiles.sort(key=lambda t: -t[0])
    pick = [t[2] for t in tiles[:n_show]]
    cols = 3
    rows = (len(pick) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 400, rows * 400), (20, 20, 20))
    for i, t in enumerate(pick):
        grid.paste(t, ((i % cols) * 400, (i // cols) * 400))
    grid.save(out_path, quality=92)
    print("saved", out_path, "gt_boxes", total_gt, "pred_boxes", total_hit,
          "images", len(imgs))


if __name__ == "__main__":
    main()