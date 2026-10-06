#!/usr/bin/env python
"""Fold extra public litter datasets from /root/autodl-tmp/newdata into one single-class
YOLO 'train-only' set at datasets/trash_extra, then drop any image that near-duplicates
the held-out trash_big/val so the test set stays honest.

Handles: Roboflow COCO export (`_annotations.coco.json`), Roboflow YOLO export, CVAT XML.
"""
import glob, json, os, shutil, xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
from PIL import Image

W = "/root/autodl-tmp"
ND = W + "/newdata"
OUT = W + "/datasets/trash_extra"
VAL = W + "/datasets/trash_big/val/images"
S = 16
THR = 8
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def reset():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT + "/train/images")
    os.makedirs(OUT + "/train/labels")


def emit(src, name, lines):
    dst = os.path.join(OUT, "train", "images", name)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)
    with open(os.path.join(OUT, "train", "labels",
                           os.path.splitext(name)[0] + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def do_coco(d, tag):
    js = os.path.join(d, "_annotations.coco.json")
    if not os.path.exists(js):
        return 0, 0, {}
    coco = json.load(open(js))
    by_id = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    per = {}
    for ann in coco["annotations"]:
        per.setdefault(ann["image_id"], []).append(ann)
    n = nb = 0
    for iid, im in by_id.items():
        src = os.path.join(d, im["file_name"])
        if not os.path.exists(src):
            continue
        w, h = im["width"], im["height"]
        lines = []
        for a in per.get(iid, []):
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            lines.append("0 %.6f %.6f %.6f %.6f" % ((x + bw / 2) / w, (y + bh / 2) / h,
                                                    bw / w, bh / h))
            per.setdefault("_cls", {})
            per["_cls"][cats[a["category_id"]]] = per["_cls"].get(cats[a["category_id"]], 0) + 1
        emit(src, "%s_%s" % (tag, os.path.basename(src)), lines)
        n += 1
        nb += len(lines)
    print("  coco %-22s imgs=%-6d boxes=%-7d cats=%s" % (tag, n, nb, per.get("_cls")), flush=True)
    return n, nb, per.get("_cls") or {}


def do_yolo(root, tag):
    n = nb = 0
    for split in sorted(os.listdir(root)):
        img = os.path.join(root, split, "images")
        lab = os.path.join(root, split, "labels")
        if not os.path.isdir(img):
            continue
        for p in sorted(os.listdir(img)):
            if not os.path.splitext(p)[1].lower() in IMAGE_EXT:
                continue
            lines = []
            lp = os.path.join(lab, os.path.splitext(p)[0] + ".txt")
            if os.path.exists(lp):
                for line in open(lp):
                    q = line.split()
                    if len(q) == 5:
                        lines.append("0 " + " ".join(q[1:]))
            emit(os.path.join(img, p), "%s_%s_%s" % (tag, split, p), lines)
            n += 1
            nb += len(lines)
    print("  yolo %-22s imgs=%-6d boxes=%d" % (tag, n, nb), flush=True)
    return n, nb


def do_cvat(xml_path, img_dir, tag):
    root = ET.parse(xml_path).getroot()
    n = nb = 0
    for image in root.findall("image"):
        name = image.get("name")
        w, h = float(image.get("width")), float(image.get("height"))
        cand = [os.path.join(img_dir, name), os.path.join(img_dir, os.path.basename(name))]
        src = next((c for c in cand if os.path.exists(c)), None)
        if src is None:
            continue
        lines = []
        for box in image.findall("box"):
            if box.get("outside") == "1":
                continue
            x1, y1 = float(box.get("xtl")), float(box.get("ytl"))
            x2, y2 = float(box.get("xbr")), float(box.get("ybr"))
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append("0 %.6f %.6f %.6f %.6f" % ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh))
        emit(src, "%s_%s" % (tag, os.path.basename(src)), lines)
        n += 1
        nb += len(lines)
    print("  cvat %-22s imgs=%-6d boxes=%d" % (tag, n, nb), flush=True)
    return n, nb


def ahash(path):
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((S, S), Image.LANCZOS), dtype=np.float32)
        return (a > a.mean()).ravel().astype(np.int8)
    except Exception:
        return None


def deval():
    """Drop extra images that near-duplicate a held-out image."""
    vp = sorted(glob.glob(VAL + "/*"))
    ep = sorted(glob.glob(OUT + "/train/images/*"))
    with Pool(32) as pool:
        Hv = [h for h in pool.map(ahash, vp, chunksize=32) if h is not None]
        He = pool.map(ahash, ep, chunksize=32)
    V = np.stack(Hv).astype(np.int16)
    drop = []
    B = 512
    for i0 in range(0, len(ep), B):
        blk = He[i0:i0 + B]
        idx = [i0 + i for i, h in enumerate(blk) if h is None]
        hs = np.stack([h for h in blk if h is not None]).astype(np.int16)
        if not len(hs):
            continue
        d = (hs[:, None, :] != V[None, :, :]).sum(2).min(1)
        drop += [i0 + i for i in np.where(d <= THR)[0].tolist()]
    for i in sorted(drop, reverse=True):
        p = ep[i]
        os.remove(p)
        lp = os.path.join(OUT, "train", "labels",
                          os.path.splitext(os.path.basename(p))[0] + ".txt")
        if os.path.exists(lp):
            os.remove(lp)
    print("  dropped %d extra images that near-duplicate trash_big/val" % len(drop), flush=True)
    return len(drop)


def main():
    reset()
    os.makedirs(ND + "/unz", exist_ok=True)
    for z in sorted(glob.glob(ND + "/*.zip")):
        target = os.path.join(ND, "unz", os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(target):
            shutil.unpack_archive(z, target)
    n = nb = 0
    for d in sorted(glob.glob(ND + "/unz/*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "_annotations.coco.json")):
            a, b, _ = do_coco(d, os.path.basename(d))
            n += a
            nb += b
    for root2, dirs, files in os.walk(ND):
        if "data.yaml" in files:
            a, b = do_yolo(root2, os.path.basename(root2))
            n += a
            nb += b
    for xml in sorted(glob.glob(ND + "/**/annotations.xml", recursive=True)):
        img_dir = os.path.dirname(xml)
        sub = [os.path.join(img_dir, x) for x in os.listdir(img_dir)
               if os.path.isdir(os.path.join(img_dir, x))]
        if sub:
            img_dir = sub[0]
        a, b = do_cvat(xml, img_dir, os.path.basename(os.path.dirname(xml)))
        n += a
        nb += b
    dropped = deval()
    n = len(glob.glob(OUT + "/train/images/*"))
    nb = 0
    for fn in os.listdir(OUT + "/train/labels"):
        nb += sum(1 for _ in open(os.path.join(OUT, "train", "labels", fn)))
    with open(OUT + "/data.yaml", "w") as f:
        f.write("path: %s\ntrain: train/images\nval: train/images\nnames:\n  0: trash\n" % OUT)
    info = {"extra_images": n, "extra_boxes": nb, "dropped_as_val_dup": dropped}
    json.dump(info, open(W + "/night/extra_info.json", "w"), indent=1)
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()