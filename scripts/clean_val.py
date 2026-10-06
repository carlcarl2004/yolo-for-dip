#!/usr/bin/env python
"""Build a leak-free validation split.

The merged dataset was assembled from Roboflow exports where the same source photo
can appear in train under one name and in val under another (identical `rf_` hash).
Here we detect those cases with a 16x16 average-hash and remove every val image that
has a near-identical twin in train. The result is written to datasets/trash_clean.
"""
import glob, json, os, shutil
import numpy as np
from PIL import Image

W = "/root/autodl-tmp"
SRC = W + "/datasets/trash_merged_bin"
DST = W + "/datasets/trash_clean"
OUT = W + "/night"
S = 16
THR = 10
os.makedirs(OUT, exist_ok=True)


def ahash(p):
    im = Image.open(p).convert("L").resize((S, S), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).ravel()


def hashes(paths, cache):
    if os.path.exists(cache):
        arr = np.load(cache)
        if arr.shape[0] == len(paths):
            print("cache hit", cache, flush=True)
            return arr
    out = []
    for i, p in enumerate(paths):
        out.append(ahash(p))
        if i % 500 == 0:
            print("hash", i, "/", len(paths), flush=True)
    arr = np.stack(out)
    np.save(cache, arr)
    return arr


def imgs(split):
    return sorted(glob.glob(os.path.join(SRC, split, "images", "*")))


def label_for(p):
    return os.path.join(SRC, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")


def relink(paths, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for p in paths:
        dst = os.path.join(out_dir, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)
        lab = label_for(p)
        if os.path.exists(lab):
            dl = os.path.join(out_dir, os.path.basename(lab))
            if not os.path.exists(dl):
                os.symlink(os.path.abspath(lab), dl)


def main():
    tr, va = imgs("train"), imgs("val")
    print("train", len(tr), "val", len(va), flush=True)
    A = hashes(tr, OUT + "/hash_train.npy")
    B = hashes(va, OUT + "/hash_val.npy")
    D = (A[None, :, :] != B[:, None, :]).sum(2)
    mind = D.min(1)
    argmin = D.argmin(1)
    leak_idx = [i for i in range(len(va)) if mind[i] <= THR]
    clean_idx = [i for i in range(len(va)) if mind[i] > THR]
    clean = [va[i] for i in clean_idx]

    rep = {
        "train_images": len(tr),
        "val_images": len(va),
        "val_near_dup_of_train": len(leak_idx),
        "val_clean": len(clean),
        "threshold_hamming_16x16": THR,
        "examples": [{"val": os.path.basename(va[i]),
                      "train": os.path.basename(tr[int(argmin[i])]),
                      "hamming": int(mind[i])} for i in leak_idx][:15],
    }
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, ensure_ascii=False), flush=True)
    print("examples:", json.dumps(rep["examples"], ensure_ascii=False)[:500], flush=True)

    relink(tr, os.path.join(DST, "train", "images"))
    relink(tr, os.path.join(DST, "train", "labels"))
    relink(clean, os.path.join(DST, "val", "images"))
    relink(clean, os.path.join(DST, "val", "labels"))
    with open(os.path.join(DST, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: trash\n" % DST)
    json.dump(rep, open(os.path.join(OUT, "clean_split.json"), "w"), indent=1, ensure_ascii=False)
    print("WROTE", DST, flush=True)


if __name__ == "__main__":
    main()