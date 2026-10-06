#!/usr/bin/env python
"""Honest evaluation of a trash detector on held-out sets, with a confidence sweep.

  bigX_val  : trash_bigX/val (1387 imgs) - kept out of trash_bigX/train AND out of clean_val
  big_val   : trash_big/val  (1399 imgs) - the earlier cluster split
  clean_val : the 33 real captured photos that have no near-duplicate in any training set

Usage: python eval_honest.py <weights>[:label] ...
Writes eval_all/honest_<label>__<set>.json
"""
import argparse, json, os

W = "/root/autodl-tmp"
OUT = W + "/eval_all"
CONFS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SETS = [("bigX_val", W + "/datasets/trash_bigX/data.yaml"),
        ("big_val", W + "/datasets/trash_big/data.yaml"),
        ("clean_val", W + "/datasets/trash_clean/data.yaml")]


def n_images(data):
    import glob
    return len(glob.glob(os.path.join(os.path.dirname(data), "val", "images", "*")))


def evaluate(label, weights, imgsz, data, setname):
    from ultralytics import YOLO
    res = {"label": label, "set": setname, "weights": weights, "imgsz": imgsz,
           "data": data, "images": n_images(data)}
    model = YOLO(weights)
    b = model.val(data=data, split="val", imgsz=imgsz, batch=16, conf=0.001, iou=0.6,
                  plots=False, verbose=False, project=OUT, name="h_%s_%s" % (label, setname))
    res["mAP50"] = round(float(b.box.map50), 4)
    res["mAP50_95"] = round(float(b.box.map), 4)
    res["P_full"] = round(float(b.box.mp), 4)
    res["R_full"] = round(float(b.box.mr), 4)
    sweep = {}
    for c in CONFS:
        v = model.val(data=data, split="val", imgsz=imgsz, batch=16, conf=c, iou=0.6,
                      plots=False, verbose=False, project=OUT,
                      name="hc%d_%s_%s" % (int(c * 100), label, setname))
        p, r = float(v.box.mp), float(v.box.mr)
        sweep["%.2f" % c] = {"P": round(p, 4), "R": round(r, 4),
                             "F1": round(2 * p * r / (p + r + 1e-9), 4)}
    res["conf_sweep"] = sweep
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for spec in a.models:
        path, _, label = spec.partition(":")
        label = label or os.path.basename(path).replace(".pt", "")
        for sname, data in SETS:
            if not os.path.exists(data):
                continue
            res = evaluate(label, path, a.imgsz, data, sname)
            json.dump(res, open(os.path.join(OUT, "honest_%s__%s.json" % (label, sname)), "w"),
                      indent=1, ensure_ascii=False)
            print("%-10s %-10s imgs=%-5d mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f" %
                  (label, sname, res["images"], res["mAP50"], res["mAP50_95"],
                   res["P_full"], res["R_full"]), flush=True)


if __name__ == "__main__":
    main()