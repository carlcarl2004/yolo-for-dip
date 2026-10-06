#!/usr/bin/env python
"""Audit the merged trash dataset: duplicates, near-duplicates, leakage, label stats."""
import os, glob, json, hashlib
import numpy as np
from PIL import Image

ROOT = os.environ.get("DS", "/root/autodl-tmp/datasets/trash_merged_bin")
OUT = os.environ.get("AU", "/root/autodl-tmp/audit")
os.makedirs(OUT, exist_ok=True)
S = 16


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(p):
    im = Image.open(p).convert("L").resize((S, S), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).ravel()


def load(split):
    imgs = sorted(glob.glob(os.path.join(ROOT, split, "images", "*")))
    labs = [os.path.join(ROOT, split, "labels",
            os.path.splitext(os.path.basename(p))[0] + ".txt") for p in imgs]
    return imgs, labs


def label_stats(labs):
    nimg = nbox = empty = 0
    areas = []
    for lp in labs:
        rows = [r.split() for r in open(lp)] if os.path.exists(lp) else []
        rows = [r for r in rows if r]
        if not rows:
            empty += 1
            continue
        nimg += 1
        for r in rows:
            nbox += 1
            if len(r) >= 5:
                areas.append(float(r[3]) * float(r[4]))
    a = np.array(areas) if areas else np.array([0.0])
    pct = np.percentile(a, [0, 10, 25, 50, 75, 90, 100]) * 100
    return {
        "images_with_boxes": nimg,
        "images_empty_or_missing_label": empty,
        "boxes": nbox,
        "box_area_pct_min_p10_p25_p50_p75_p90_max": [round(float(x), 4) for x in pct],
        "boxes_lt_0.5pct": int((a < 0.005).sum()),
        "boxes_lt_1pct": int((a < 0.01).sum()),
    }


def main():
    rep = {}
    tr, trl = load("train")
    va, val = load("val")
    rep["counts"] = {"train": len(tr), "val": len(va)}
    rep["train_label_stats"] = label_stats(trl)
    rep["val_label_stats"] = label_stats(val)

    for name, imgs in (("train", tr), ("val", va)):
        h = {}
        for p in imgs:
            h.setdefault(md5(p), []).append(os.path.basename(p))
        d = {k: v for k, v in h.items() if len(v) > 1}
        rep[name + "_exact_dup_groups"] = len(d)
        rep[name + "_exact_dup_examples"] = list(d.values())[:10]

    A = np.stack([ahash(p) for p in tr])
    B = np.stack([ahash(p) for p in va])
    D = (A[None, :, :] != B[:, None, :]).sum(axis=2)
    thr = 12
    hits = []
    for i in range(len(va)):
        j = int(np.argmin(D[i]))
        if D[i, j] <= thr:
            hits.append({"val": os.path.basename(va[i]),
                         "train": os.path.basename(tr[j]),
                         "hamming": int(D[i, j])})
    rep["val_train_near_dup_threshold"] = thr
    rep["val_train_near_dup_count"] = len(hits)
    rep["val_train_near_dup_examples"] = hits[:20]

    Dv = (B[:, None, :] != B[None, :, :]).sum(axis=2)
    pairs = []
    for i in range(len(va)):
        for j in range(i + 1, len(va)):
            if Dv[i, j] <= thr:
                pairs.append({"a": os.path.basename(va[i]),
                              "b": os.path.basename(va[j]),
                              "hamming": int(Dv[i, j])})
    rep["val_internal_near_dup_count"] = len(pairs)
    rep["val_internal_near_dup_examples"] = pairs[:20]

    json.dump(rep, open(os.path.join(OUT, "audit.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()