#!/usr/bin/env python
"""Build datasets/trash_bigX: a hardlink copy of trash_big with every image that
near-duplicates a clean_val photo removed from BOTH splits.  The rest of the split is
byte-for-byte the same, so val stays comparable, and clean_val becomes a valid test set.
Source files are never deleted.
"""
import glob, json, os, shutil
from multiprocessing import Pool

import numpy as np
from PIL import Image

W = "/root/autodl-tmp"
SRC = W + "/datasets/trash_big"
DST = W + "/datasets/trash_bigX"
EXC = W + "/datasets/trash_clean/val/images"
S, THR = 16, 8


def ahash(p):
    try:
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((S, S), Image.LANCZOS), dtype=np.float32)
        return (a > a.mean()).ravel().astype(np.uint8)
    except Exception:
        return None


def main():
    ex = sorted(glob.glob(EXC + "/*"))
    with Pool(32) as pool:
        V = np.stack([h for h in pool.map(ahash, ex, chunksize=8) if h is not None]).astype(np.int16)
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    stats = {}
    for split in ("train", "val"):
        ip = sorted(glob.glob(os.path.join(SRC, split, "images", "*")))
        with Pool(32) as pool:
            H = pool.map(ahash, ip, chunksize=64)
        os.makedirs(os.path.join(DST, split, "images"))
        os.makedirs(os.path.join(DST, split, "labels"))
        kept = dropped = 0
        for i, p in enumerate(ip):
            h = H[i]
            if h is not None and int((h.astype(np.int16) != V).sum(1).min()) <= THR:
                dropped += 1
                continue
            name = os.path.basename(p)
            os.link(p, os.path.join(DST, split, "images", name))
            lp = os.path.join(SRC, split, "labels", os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lp):
                os.link(lp, os.path.join(DST, split, "labels", os.path.splitext(name)[0] + ".txt"))
            kept += 1
        stats[split] = {"kept": kept, "dropped": dropped}
        print(split, stats[split], flush=True)
    with open(DST + "/data.yaml", "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: trash\n" % DST)
    json.dump(stats, open(W + "/night/bigX_prune.json", "w"), indent=1)
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    main()