#!/usr/bin/env python
"""Zero-shot / open-vocabulary baseline: can a pretrained model find trash with no training?

Uses YOLO-World (open-vocabulary YOLOv8): you give it class names as text, no fine-tuning.
If this beats our trained models on the leak-free split, the whole training problem changes.

    python zeroshot_eval.py yolov8s-worldv2.pt "trash,litter,garbage" --imgsz 640

Writes eval_all/zero_<tag>.json with the same frame-level metrics as frame_eval.py.
"""
import argparse
import glob
import json
import os

from PIL import Image

W = "/root/autodl-tmp"
OUT = W + "/eval_all"
IOU = 0.5


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
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
          + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def evaluate(weights, classes, data, imgsz, conf, tag, iou_thr=IOU):
    from ultralytics import YOLO

    root = os.path.dirname(data)
    imgs = sorted(glob.glob(os.path.join(root, "val", "images", "*")))
    lab_dir = os.path.join(root, "val", "labels")

    model = YOLO(weights)
    if classes:
        model.set_classes(classes)

    tp = fp = fn = 0
    frames_with_gt = frames_hit = 0
    empty_frames = alarms = 0
    per_image = []

    for path, r in zip(imgs, model.predict(imgs, imgsz=imgsz, conf=conf, iou=0.6,
                                            verbose=False, stream=True)):
        stem = os.path.splitext(os.path.basename(path))[0]
        gt = load_gt(os.path.join(lab_dir, stem + ".txt"))
        with Image.open(path) as im:
            w, h = im.size
        g = [xywhn2xyxy(b, w, h) for b in gt]
        preds = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        cfs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        preds = [p for p, _ in sorted(zip(preds, cfs), key=lambda t: -t[1])]

        used = [False] * len(g)
        n_tp = 0
        for pb in preds:
            best, bi = iou_thr, -1
            for gi, gb in enumerate(g):
                if used[gi]:
                    continue
                v = iou(pb, gb)
                if v >= best:
                    best, bi = v, gi
            if bi >= 0 and best >= iou_thr:
                used[bi] = True
                n_tp += 1
            else:
                fp += 1
        tp += n_tp
        fn += used.count(False)
        if len(g) > 0:
            frames_with_gt += 1
            frames_hit += 1 if n_tp > 0 else 0
        else:
            empty_frames += 1
            alarms += 1 if len(preds) > 0 else 0
        per_image.append({"image": os.path.basename(path), "gt": len(g),
                          "pred": len(preds), "tp": n_tp, "fn": used.count(False)})

    rep = {
        "weights": weights, "classes": classes, "imgsz": imgsz, "conf": conf, "iou_thr": iou_thr,
        "data": data, "val_images": len(imgs),
        "frames_with_gt": frames_with_gt, "frames_empty": empty_frames,
        "frame_hit_rate": round(frames_hit / frames_with_gt, 4) if frames_with_gt else None,
        "frame_false_alarm_rate": round(alarms / empty_frames, 4) if empty_frames else None,
        "box_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "box_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn, "per_image": per_image,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "zero_%s.json" % tag), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_image"}, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("classes", nargs="?", default="", help="comma separated prompts, empty = use COCO classes")
    ap.add_argument("--data", default=W + "/datasets/trash_clean/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    tag = a.tag or (os.path.basename(a.weights).replace(".pt", "") + "_" +
                    ("_".join(classes)[:40].replace(" ", "-") if classes else "coco"))
    evaluate(a.weights, classes, a.data, a.imgsz, a.conf, tag)


if __name__ == "__main__":
    main()