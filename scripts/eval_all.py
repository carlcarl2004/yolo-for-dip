#!/usr/bin/env python
"""Evaluate every trained trash detector on both the (leaky) original val split and the
leak-free val split, with a confidence sweep.

Writes one JSON per (model, dataset) pair into /root/autodl-tmp/eval_all/.
Re-running is cheap: pairs whose JSON already exists are skipped unless FORCE=1.
"""
import json, os
from ultralytics import YOLO

W = "/root/autodl-tmp"
OUT = W + "/eval_all"
os.makedirs(OUT, exist_ok=True)
CONFS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FORCE = os.environ.get("FORCE", "0") == "1"

MODELS = [
    ("yolov8n_640_base", W + "/trash_project/runs/n_merged/weights/best.pt", 640),
    ("yolov8s_640_base", W + "/trash_project/runs/s_merged/weights/best.pt", 640),
    ("yolov8m_640_base", W + "/trash_project/runs/m_merged/weights/best.pt", 640),
    ("yolov8s_640_rot", W + "/trash_project/runs/s_rot/weights/best.pt", 640),
    ("yolov8s_960m", W + "/trash_project/runs/s_960m/weights/best.pt", 960),
    ("yolov8s_ftstreet", W + "/trash_project/runs/s_ftstreet/weights/best.pt", 640),
    ("yolov8s_ftown", W + "/trash_project/runs/s_ftown/weights/best.pt", 640),
]
DATASETS = [
    ("leaky_val", W + "/datasets/trash_merged_bin/data.yaml"),
    ("street_val", W + "/datasets/trash_street/data.yaml"),
    ("clean_val", W + "/datasets/trash_clean/data.yaml"),
]


def evaluate(tag, weights, imgsz, data):
    model = YOLO(weights)
    res = {"tag": tag, "weights": weights, "imgsz": imgsz, "data": data, "conf_sweep": {}}
    b = model.val(data=data, split="val", imgsz=imgsz, batch=16, conf=0.001, iou=0.6,
                  plots=False, verbose=False, project=OUT, name="m_" + tag)
    res["mAP50"] = round(float(b.box.map50), 4)
    res["mAP50_95"] = round(float(b.box.map), 4)
    res["P_conf0.001"] = round(float(b.box.mp), 4)
    res["R_conf0.001"] = round(float(b.box.mr), 4)
    for c in CONFS:
        v = model.val(data=data, split="val", imgsz=imgsz, batch=16, conf=c, iou=0.6,
                      plots=False, verbose=False, project=OUT, name="c%d_%s" % (int(c * 100), tag))
        p, r = float(v.box.mp), float(v.box.mr)
        res["conf_sweep"]["%.2f" % c] = {
            "P": round(p, 4), "R": round(r, 4),
            "F1": round(2 * p * r / (p + r + 1e-9), 4)}
    return res


def main():
    summary = []
    for name, weights, imgsz in MODELS:
        if not os.path.exists(weights):
            print("SKIP missing", name, weights, flush=True)
            continue
        for ds, data in DATASETS:
            if not os.path.exists(data):
                print("SKIP dataset", ds, flush=True)
                continue
            tag = name + "__" + ds
            fn = os.path.join(OUT, tag + ".json")
            if os.path.exists(fn) and not FORCE:
                summary.append(json.load(open(fn)))
                print("cached", tag, flush=True)
                continue
            print("RUN", tag, flush=True)
            res = evaluate(tag, weights, imgsz, data)
            json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
            summary.append(res)
            print("DONE", tag, res["mAP50"], res["mAP50_95"], flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print("ALL DONE", len(summary), flush=True)


if __name__ == "__main__":
    main()