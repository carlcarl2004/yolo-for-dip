#!/usr/bin/env python
"""Detailed evaluation of trained trash detectors on the real-photo val split.

For every model it reports mAP50 / mAP50-95 and a confidence sweep, so we can
answer the operational question: "if the robot only acts on detections whose
confidence is >= X, how precise are those detections?"

Usage:  python eval_suite.py            # reads eval_suite/jobs.json
"""
import os, json
from ultralytics import YOLO

W = "/root/autodl-tmp"
DATA = W + "/datasets/trash_merged_bin/data.yaml"
OUT = W + "/eval_suite"
os.makedirs(OUT, exist_ok=True)
CONFS = (0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def evaluate(tag, weights, imgsz=640, tta=False):
    if not os.path.exists(weights):
        return {"tag": tag, "weights": weights, "missing": True}
    model = YOLO(weights)
    suffix = tag + ("_tta" if tta else "")
    res = {"tag": suffix, "weights": weights, "imgsz": imgsz, "tta": tta, "conf_sweep": {}}
    base = model.val(data=DATA, split="val", imgsz=imgsz, batch=16, conf=0.001,
                     iou=0.6, plots=False, verbose=False, augment=tta,
                     project=OUT, name="map_" + suffix)
    res["mAP50"] = round(float(base.box.map50), 4)
    res["mAP50_95"] = round(float(base.box.map), 4)
    res["P_conf0.001"] = round(float(base.box.mp), 4)
    res["R_conf0.001"] = round(float(base.box.mr), 4)
    for c in CONFS:
        v = model.val(data=DATA, split="val", imgsz=imgsz, batch=16, conf=c, iou=0.6,
                      plots=False, verbose=False, augment=tta,
                      project=OUT, name="c%d_%s" % (int(c * 100), suffix))
        p, r = float(v.box.mp), float(v.box.mr)
        res["conf_sweep"]["%.2f" % c] = {
            "P": round(p, 4), "R": round(r, 4),
            "F1": round(2 * p * r / (p + r + 1e-9), 4)}
    print("RESULT " + json.dumps(res, ensure_ascii=False))
    return res


def main():
    jobs = json.load(open(os.path.join(OUT, "jobs.json")))
    out = []
    for j in jobs:
        for tta in (False, True):
            out.append(evaluate(j["tag"], j["weights"], j.get("imgsz", 640), tta))
            json.dump(out, open(os.path.join(OUT, "eval_all.json"), "w"),
                      indent=1, ensure_ascii=False)
    print("ALL DONE")


if __name__ == "__main__":
    main()