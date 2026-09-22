#!/usr/bin/env python
"""把外部 YOLO 数据集整理成本项目要用的两个版本（单类 trash / 项目 9 类）。

  python roboflow2yolo.py --src /root/autodl-tmp/datasets/roboflow_rf \
      --bin-out /root/autodl-tmp/datasets/rf_bin \
      --nine-out /root/autodl-tmp/datasets/rf_9cls

支持两种目录结构：
  1) 已分好 split：train/images、valid/images、test/images ...
  2) 扁平结构：根目录下直接是 images/ 和 labels/
类别按名字归一化后映射到项目 9 类，映射不到的类别（口罩、抹布等）丢弃。
"""
import argparse
import os
import random
import shutil
from collections import Counter

import yaml

NAMES9 = ["plastic", "paper", "tissue", "bottle", "cup", "can", "cardboard", "organic", "wood"]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAP9 = {
    "plastic": 0, "plasticbag": 0, "plasticwrapper": 0, "styrofoam": 0, "plasticfilm": 0,
    "paper": 1, "paperbag": 1,
    "tissue": 2, "tissues": 2,
    "bottle": 3, "plasticbottle": 3, "glassbottle": 3, "coloredglassbottles": 3, "glassjar": 3,
    "cup": 4, "plasticcup": 4, "plasticcups": 4, "papercup": 4,
    "can": 5, "cans": 5,
    "cardboard": 6, "paperboard": 6, "tetrapak": 6, "tetrapack": 6,
    "organic": 7, "peel": 7, "pileofleaves": 7,
    "wood": 8,
}


def normalise(name):
    return "".join(character for character in str(name).lower() if character.isalnum())


def read_names(source):
    with open(os.path.join(source, "data.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw = config["names"]
    if isinstance(raw, dict):
        return [raw[key] for key in sorted(raw, key=lambda key: int(key))]
    return list(raw)


def collect_dir(image_dir, label_dir):
    if not os.path.isdir(image_dir):
        return []
    found = []
    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith(IMAGE_EXT):
            continue
        label = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
        found.append((os.path.join(image_dir, name), label))
    return found


def collect(source):
    items = []
    used = []
    for split in ("train", "valid", "test", "val"):
        found = collect_dir(os.path.join(source, split, "images"), os.path.join(source, split, "labels"))
        if found:
            used.append(split)
            items += found
    if not items:
        items = collect_dir(os.path.join(source, "images"), os.path.join(source, "labels"))
        if items:
            used.append("(扁平 images/labels)")
    print("目录结构: %s" % (", ".join(used) if used else "未找到图片"))
    return items


def build(mode, out, names, splits):
    if os.path.isdir(out):
        if "datasets" not in out:
            raise SystemExit("拒绝删除 %s：输出目录必须在 datasets 下" % out)
        shutil.rmtree(out)
    counts = Counter()
    dropped = Counter()
    boxes = 0
    negatives = 0
    for split, group in splits.items():
        image_out = os.path.join(out, split, "images")
        label_out = os.path.join(out, split, "labels")
        os.makedirs(image_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)
        for image, label_path in group:
            lines = []
            if os.path.exists(label_path):
                with open(label_path, encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        old_id = int(float(parts[0]))
                        name = names[old_id] if 0 <= old_id < len(names) else ""
                        if mode == "bin":
                            new_id = 0
                        else:
                            new_id = MAP9.get(normalise(name))
                            if new_id is None:
                                dropped[name or ("id%d" % old_id)] += 1
                                continue
                        lines.append("%d %s" % (new_id, " ".join(parts[1:])))
                        counts[new_id] += 1
            if not lines:
                negatives += 1
            base = os.path.basename(image)
            target = os.path.join(image_out, base)
            if not os.path.exists(target):
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy(image, target)
            with open(os.path.join(label_out, os.path.splitext(base)[0] + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
            boxes += len(lines)
    labels = ["trash"] if mode == "bin" else NAMES9
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as handle:
        handle.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n" % out)
        for index, name in enumerate(labels):
            handle.write("  %d: %s\n" % (index, name))
    print("[%s] %s" % (mode, out))
    print("  train=%d val=%d 框=%d 无标注图=%d" % (len(splits["train"]), len(splits["val"]), boxes, negatives))
    print("  每类框数: %s" % {labels[key]: value for key, value in sorted(counts.items())})
    if dropped:
        print("  丢弃的类别: %s" % dict(dropped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--bin-out")
    parser.add_argument("--nine-out")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = read_names(args.src)
    print("原始类别(%d): %s" % (len(names), names))
    items = collect(args.src)
    print("图片总数: %d" % len(items))
    if not items:
        raise SystemExit("没有找到任何图片，请检查 --src")
    random.seed(args.seed)
    random.shuffle(items)
    count = int(len(items) * args.val_ratio)
    splits = {"val": items[:count], "train": items[count:]}
    if args.bin_out:
        build("bin", args.bin_out, names, splits)
    if args.nine_out:
        build("9cls", args.nine_out, names, splits)


if __name__ == "__main__":
    main()
