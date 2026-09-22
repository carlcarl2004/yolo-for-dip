#!/usr/bin/env python
"""把 TACO（COCO 格式）转成 YOLO 训练格式。

  python taco2yolo.py --mode bin    --out /root/autodl-tmp/datasets/taco_bin      # 所有垃圾合成 1 类 trash
  python taco2yolo.py --mode 9cls   --out /root/autodl-tmp/datasets/taco_9cls     # 映射到你现有的 9 类

图片用硬链接，不额外占磁盘。
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter

TACO = "/root/autodl-tmp/TACO"
NAMES9 = ["plastic", "paper", "tissue", "bottle", "cup", "can", "cardboard", "organic", "wood"]

# TACO 类别名 -> 你的 9 类序号；None = 丢弃
MAP9 = {
    "Other plastic bottle": 3, "Clear plastic bottle": 3, "Glass bottle": 3, "Glass jar": 3,
    "Plastic bottle cap": 0, "Metal bottle cap": 5, "Plastic lid": 0, "Metal lid": 5,
    "Food Can": 5, "Aerosol": 5, "Drink can": 5, "Pop tab": 5, "Scrap metal": 5,
    "Paper cup": 4, "Disposable plastic cup": 4, "Foam cup": 4, "Glass cup": 4, "Other plastic cup": 4,
    "Toilet tube": 6, "Other carton": 6, "Egg carton": 6, "Drink carton": 6,
    "Corrugated carton": 6, "Meal carton": 6, "Pizza box": 6,
    "Magazine paper": 1, "Wrapping paper": 1, "Normal paper": 1, "Paper bag": 1,
    "Plastified paper bag": 1, "Paper straw": 1,
    "Tissues": 2,
    "Food waste": 7,
    "Plastic film": 0, "Six pack rings": 0, "Garbage bag": 0, "Other plastic wrapper": 0,
    "Single-use carrier bag": 0, "Polypropylene bag": 0, "Crisp packet": 0,
    "Spread tub": 0, "Tupperware": 0, "Disposable food container": 0, "Foam food container": 0,
    "Other plastic container": 0, "Other plastic": 0, "Plastic glooves": 0,
    "Plastic utensils": 0, "Plastic straw": 0, "Styrofoam piece": 0, "Squeezable tube": 0,
    "Rope & strings": 0,
    "Aluminium foil": None, "Battery": None, "Aluminium blister pack": None, "Carded blister pack": None,
    "Broken glass": None, "Shoe": None, "Cigarette": None, "Unlabeled litter": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco", default=TACO)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["bin", "9cls"], default="bin")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    coco = json.load(open(f"{a.taco}/data/annotations.json"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    def to_class(taco_name):
        if a.mode == "bin":
            return 0  # 一切垃圾都是 trash
        return MAP9.get(taco_name)

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    imgs = []
    stats = Counter()
    dropped = Counter()
    for img in coco["images"]:
        src = os.path.join(a.taco, "data/images", img["file_name"])
        if not os.path.exists(src):
            continue
        W, H = img["width"], img["height"]
        lines = []
        for ann in anns_by_img.get(img["id"], []):
            cname = cat_name[ann["category_id"]]
            cid = to_class(cname)
            if cid is None:
                dropped[cname] += 1
                continue
            x, y, w, h = ann["bbox"]  # COCO: 左上角 + 宽高
            if w <= 1 or h <= 1:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(nw, 1), min(nh, 1)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[cid] += 1
        imgs.append((img["file_name"], src, lines))

    random.seed(a.seed)
    random.shuffle(imgs)
    n_val = int(len(imgs) * a.val_ratio)
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

    for split, items in splits.items():
        d_img = f"{a.out}/{split}/images"
        d_lab = f"{a.out}/{split}/labels"
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lab, exist_ok=True)
        for name, src, lines in items:
            flat = name.replace("/", "_")
            dst = f"{d_img}/{flat}"
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy(src, dst)
            with open(f"{d_lab}/{os.path.splitext(flat)[0]}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    names = ["trash"] if a.mode == "bin" else NAMES9
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {a.out}\ntrain: train/images\nval: val/images\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print(f"模式 {a.mode}: train={len(splits['train'])} val={len(splits['val'])} 图")
    print("每个类的框数:", {names[k]: v for k, v in sorted(stats.items())})
    if dropped:
        print("丢弃的类别:", dict(dropped))
    print("输出:", a.out)


if __name__ == "__main__":
    main()
