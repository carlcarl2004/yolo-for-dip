#!/usr/bin/env python
"""Frame-level evaluation: the numbers that actually matter for a pick-up robot.

For every validation image we run one forward pass at a fixed confidence and a fixed
IoU, greedily match predictions to ground-truth boxes, then report:

  frame_hit_rate   fraction of trash frames where at least one box was found correctly
                   -> "did the robot see the trash in this frame?"
  box_precision    tp / (tp + fp)  -> "when it reports trash, how often is it right?"
  box_recall       tp / (tp + fn)  -> "how much of the trash does it catch?"
  false_alarm_rate fraction of empty frames on which it still reports something

Usage: python frame_eval.py <weights> <imgsz> <data.yaml> <conf>
Writes eval_all/frame_<tag>.json
"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from ultralytics import YOLO

W = "/root/autodl-tmp"
OUT = W + "/eval_all"
IOU = float(os.environ.get("IOU", "0.5"))


def load_gt(label_path):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) >= 5:
                boxes.append([float(x) for x in p[1:5]])
    return boxes


def xywhn2xyxy(b, w, h):
    cx, cy, bw, bh = b
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + \
         max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    weights, imgsz, data, conf = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    root = os.path.dirname(data)
    img_dir = os.path.join(root, "val", "images")
    lab_dir = os.path.join(root, "val", "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
    model = YOLO(weights)

    tp = fp = fn = 0
    frames_with_gt = 0
    frames_hit = 0
    empty_frames = 0
    false_alarm_frames = 0
    per_image = []

    results = model.predict(imgs, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, stream=True,
                            batch=8)
    for path, r in zip(imgs, results):
        stem = os.path.splitext(os.path.basename(path))[0]
        gt = load_gt(os.path.join(lab_dir, stem + ".txt"))
        with Image.open(path) as im:
            w, h = im.size
        g = [xywhn2xyxy(b, w, h) for b in gt]
        preds = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        preds = sorted(zip(preds, confs), key=lambda t: -t[1])
        preds = [p for p, _ in preds]

        used = [False] * len(g)
        n_tp = 0
        for pb in preds:
            best, bi = IOU, -1
            for gi, gb in enumerate(g):
                if used[gi]:
                    continue
                v = iou(pb, gb)
                if v >= best:
                    best, bi = v, gi
            if bi >= 0 and best >= IOU:
                used[bi] = True
                n_tp += 1
            else:
                fp += 1
        n_fn = used.count(False)
        tp += n_tp
        fn += n_fn

        if len(g) > 0:
            frames_with_gt += 1
            if n_tp > 0:
                frames_hit += 1
        else:
            empty_frames += 1
            if len(preds) > 0:
                false_alarm_frames += 1

        per_image.append({"image": os.path.basename(path), "gt": len(g),
                          "pred": len(preds), "tp": n_tp, "fn": n_fn})

    rep = {
        "weights": weights, "imgsz": imgsz, "data": data, "conf": conf, "iou_thr": IOU,
        "val_images": len(imgs),
        "frames_with_gt": frames_with_gt,
        "frames_empty": empty_frames,
        "frame_hit_rate": round(frames_hit / frames_with_gt, 4) if frames_with_gt else None,
        "frame_false_alarm_rate": round(false_alarm_frames / empty_frames, 4) if empty_frames else None,
        "box_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "box_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "per_image": per_image,
    }
    tag = "%s_%s_c%d_iou%d" % (os.path.basename(os.path.dirname(os.path.dirname(weights))), imgsz, int(conf * 100), int(IOU * 100))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "frame_" + tag + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_image"}, ensure_ascii=False))


if __name__ == "__main__":
    main()