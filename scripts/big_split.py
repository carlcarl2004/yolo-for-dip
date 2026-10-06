#!/usr/bin/env python
"""Build a large, cluster-aware, leak-free train/val split over the whole litter corpus.

Every public dataset we merged has a broken val split (val images are pixel-identical
copies of train images), and our own 73-image val shared 40 frames with train.  So all
earlier numbers were measured on 33 images at best.

Fix: pool *every* image we have, group near-duplicates with a 16x16 average hash, then
split by GROUP (not by image) so no source photo can appear on both sides.  --exclude lets
us guarantee that a chosen test set (e.g. the 33-frame clean_val) never appears anywhere in
the output, so the old and new models stay comparable on it.
"""
import argparse, glob, json, os, shutil
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

W = "/root/autodl-tmp"
S = 16
THR = 8
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ahash(path):
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((S, S), Image.LANCZOS), dtype=np.float32)
        return (a > a.mean()).ravel().astype(np.uint8)
    except Exception:
        return None


def collect(roots):
    out = []
    for r in roots:
        for sub in ("train", "val", "valid", ""):
            d = os.path.join(r, sub, "images") if sub else os.path.join(r, "images")
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    if n.lower().endswith(IMAGE_EXT):
                        out.append(os.path.join(d, n))
    return sorted(set(out))


def label_for(p):
    parts = p.split(os.sep)
    i = parts.index("images")
    return os.path.join(os.sep.join(parts[:i]), "labels",
                        os.path.splitext(parts[-1])[0] + ".txt")


def neighbours(H, V, thr, B=512):
    """indices of rows in H that are within `thr` of some row in V"""
    out = []
    for i0 in range(0, len(H), B):
        hs = H[i0:i0 + B].astype(np.int16)
        d = (hs[:, None, :] != V[None, :, :]).sum(2).min(1)
        out += [i0 + i for i in np.where(d <= thr)[0].tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=W + "/datasets/trash_big")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-hamming-sep", type=int, default=THR)
    ap.add_argument("--exclude", default="", help="dir of images that must not appear anywhere")
    ap.add_argument("--extra", action="append", default=[], help="extra root dirs to pool")
    ap.add_argument("--val-link", default="", help="reuse an existing val dir instead of splitting")
    args = ap.parse_args()

    roots = [W + "/datasets/" + d for d in
             ("trash_bin", "taco_bin", "rf_bin", "trash3_bin",
              "trash_own", "trash_street", "trash_merged_bin")] + list(args.extra)
    paths = collect(roots)
    src = [os.path.relpath(p, W + "/datasets").split(os.sep)[0] for p in paths]
    print("pooled images:", len(paths), flush=True)
    print("by source:", Counter(src), flush=True)

    with Pool(32) as pool:
        H = pool.map(ahash, paths, chunksize=64)
    keep = [i for i, h in enumerate(H) if h is not None]
    paths = [paths[i] for i in keep]
    src = [src[i] for i in keep]
    H = np.stack([H[i] for i in keep]).astype(np.uint8)

    if args.exclude:
        ex = sorted(glob.glob(os.path.join(args.exclude, "*")))
        with Pool(32) as pool:
            HE = [h for h in pool.map(ahash, ex, chunksize=32) if h is not None]
        bad = set(neighbours(H, np.stack(HE).astype(np.int16), args.min_hamming_sep))
        keep = [i for i in range(len(paths)) if i not in bad]
        paths = [paths[i] for i in keep]
        src = [src[i] for i in keep]
        H = H[keep]
        print("excluded %d pooled images near-duplicating %s" % (len(bad), args.exclude),
              flush=True)

    # union-find over near-duplicate pairs
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    B = 512
    for i0 in range(0, len(paths), B):
        blk = H[i0:i0 + B].astype(np.int16)
        for j0 in range(i0, len(paths), B):
            blk2 = H[j0:j0 + B].astype(np.int16)
            d = (blk[:, None, :] != blk2[None, :, :]).sum(2)
            if i0 == j0:
                np.fill_diagonal(d, 999)
            ii, jj = np.where(d <= args.min_hamming_sep)
            for a, b in zip(ii.tolist(), jj.tolist()):
                union(i0 + a, j0 + b)
    groups = {}
    for i in range(len(paths)):
        groups.setdefault(find(i), []).append(i)
    gl = list(groups.values())
    print("groups:", len(gl), "largest:", max(len(g) for g in gl), flush=True)

    if args.val_link:
        val_g = set()
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(gl))
        val_imgs = int(round(args.val_frac * len(paths)))
        val_g, n = set(), 0
        for gi in order:
            if n >= val_imgs:
                break
            if len(gl[gi]) > 60:
                continue
            val_g.add(int(gi))
            n += len(gl[gi])
        print("val groups:", len(val_g), "val images:", n, flush=True)

    info = {"pooled": len(paths), "groups": len(gl), "hamming": args.min_hamming_sep,
            "val_by_source": {}, "train_by_source": {}}
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, split, "images"))
        os.makedirs(os.path.join(args.out, split, "labels"))
    if args.val_link:
        shutil.rmtree(os.path.join(args.out, "val"))
        os.symlink(args.val_link, os.path.join(args.out, "val"))
        info["val_link"] = args.val_link

    vc, tc = Counter(), Counter()
    nboxes = 0
    for gi, idxs in enumerate(gl):
        if args.val_link:
            split = "train"
        else:
            split = "val" if gi in val_g else "train"
        for i in idxs:
            p = paths[i]
            name = "%s__%s" % (src[i], os.path.basename(p))
            dst = os.path.join(args.out, split, "images", name)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.abspath(p), dst)
                except OSError:
                    shutil.copy(os.path.abspath(p), dst)
            lab = label_for(p)
            lines = []
            if os.path.exists(lab):
                for line in open(lab):
                    q = line.split()
                    if len(q) == 5:
                        lines.append(" ".join(q))
            nboxes += len(lines)
            with open(os.path.join(args.out, split, "labels",
                                   os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            (vc if split == "val" else tc)[src[i]] += 1
    info["val_by_source"] = dict(vc)
    info["train_by_source"] = dict(tc)
    info["boxes_written"] = nboxes
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: trash\n" % args.out)
    json.dump(info, open(W + "/night/%s_split.json" % os.path.basename(args.out), "w"),
              indent=1, ensure_ascii=False)
    print("train imgs:", sum(tc.values()), "val imgs:", sum(vc.values()), flush=True)
    print(json.dumps({k: v for k, v in info.items() if not k.endswith("_by_source")}),
          flush=True)


if __name__ == "__main__":
    main()