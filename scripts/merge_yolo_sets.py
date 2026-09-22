#!/usr/bin/env python
"""把多个 YOLO 数据集合并成一个：验证集用真实场景数据，其余图片并入训练集。

  python merge_yolo_sets.py --out /root/autodl-tmp/datasets/trash_merged_bin \
      --val-from /root/autodl-tmp/datasets/trash_bin \
      --train-from /root/autodl-tmp/datasets/trash_bin \
      --train-from /root/autodl-tmp/datasets/taco_bin \
      --train-from /root/autodl-tmp/datasets/rf_bin

图片用硬链接，不额外占磁盘；不同来源的同名文件自动加来源前缀。
"""
import argparse
import os
import shutil
from collections import Counter

import yaml

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def names_of(directory):
    with open(os.path.join(directory, "data.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw = config["names"]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda key: int(key))]
    return [str(item) for item in raw]


def place(source, split, out_split, prefix, stats):
    image_dir = os.path.join(source, split, "images")
    label_dir = os.path.join(source, split, "labels")
    if not os.path.isdir(image_dir):
        return 0
    added = 0
    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith(IMAGE_EXT):
            continue
        base = "%s_%s" % (prefix, name)
        target = os.path.join(out_split, "images", base)
        if os.path.exists(target):
            continue
        try:
            os.link(os.path.join(image_dir, name), target)
        except OSError:
            shutil.copy(os.path.join(image_dir, name), target)
        lines = []
        label_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 5:
                        lines.append(" ".join(parts))
                        stats[int(float(parts[0]))] += 1
        with open(os.path.join(out_split, "labels", os.path.splitext(base)[0] + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))
        added += 1
    print("  %s/%s <- %d 张" % (source, split, added))
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-from", required=True)
    parser.add_argument("--train-from", action="append", default=[])
    args = parser.parse_args()

    reference = names_of(args.val_from)
    sources = args.train_from or [args.val_from]
    for source in sources:
        if names_of(source) != reference:
            raise SystemExit("类别不一致: %s=%s ; %s=%s" % (source, names_of(source), args.val_from, reference))
    print("类别: %s" % reference)

    if os.path.isdir(args.out):
        if "datasets" not in args.out:
            raise SystemExit("拒绝删除 %s" % args.out)
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, "labels"), exist_ok=True)

    stats = Counter()
    print("验证集:")
    val_count = place(args.val_from, "val", os.path.join(args.out, "val"), "own", stats)
    print("训练集:")
    train_count = 0
    for source in sources:
        prefix = os.path.basename(source.rstrip("/")) or "src"
        # 每个来源的训练集和验证集都并入合并后的训练集，避免浪费已标注图片；
        # 合并后的验证集固定来自 --val-from，保证和之前的实验可比。
        # 作为验证集来源的那份数据，它的 val 不能再进训练集（否则验证集泄漏）
        splits_to_use = ("train",) if os.path.abspath(source) == os.path.abspath(args.val_from) else ("train", "val")
        for split in splits_to_use:
            train_count += place(source, split, os.path.join(args.out, "train"), prefix, stats)

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as handle:
        handle.write("path: %s\ntrain: train/images\nval: val/images\nnames:\n" % args.out)
        for index, name in enumerate(reference):
            handle.write("  %d: %s\n" % (index, name))
    print("合并完成: train=%d val=%d 框=%d" % (train_count, val_count, sum(stats.values())))
    print("每类框数: %s" % {reference[key]: value for key, value in sorted(stats.items())})


if __name__ == "__main__":
    main()
