#!/usr/bin/env python
"""检查 YOLO 数据集的完整性与标注合法性。

  python check_dataset.py --root /root/autodl-tmp/datasets/trash_merged_bin
"""
import argparse
import collections
import os

import yaml

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.root, "data.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw = config["names"]
    if isinstance(raw, dict):
        names = [str(raw[key]) for key in sorted(raw, key=lambda key: int(key))]
    else:
        names = [str(item) for item in raw]
    print("root=%s" % args.root)
    print("类别(%d): %s" % (len(names), names))

    problems = []
    for split in ("train", "val"):
        image_dir = os.path.join(args.root, split, "images")
        label_dir = os.path.join(args.root, split, "labels")
        if not os.path.isdir(image_dir):
            print("%s: 目录不存在，跳过" % split)
            continue
        images = {os.path.splitext(name)[0] for name in os.listdir(image_dir) if name.lower().endswith(IMAGE_EXT)}
        labels = {os.path.splitext(name)[0] for name in os.listdir(label_dir) if name.endswith(".txt")}
        stats = collections.Counter()
        negatives = 0
        boxes = 0
        for name in sorted(images & labels):
            with open(os.path.join(label_dir, name + ".txt"), encoding="utf-8") as handle:
                lines = [line.split() for line in handle if line.strip()]
            if not lines:
                negatives += 1
            for parts in lines:
                if len(parts) != 5:
                    problems.append("%s/%s: 字段数 %d" % (split, name, len(parts)))
                    continue
                index = int(float(parts[0]))
                if not 0 <= index < len(names):
                    problems.append("%s/%s: 类别 %d 越界" % (split, name, index))
                    continue
                try:
                    x, y, w, h = (float(value) for value in parts[1:])
                except ValueError:
                    problems.append("%s/%s: 坐标不是数字" % (split, name))
                    continue
                # 容差 1e-3：标注框正好贴图片边缘时会有浮点误差，不算真问题
                tolerance = 0.001
                if not (-tolerance <= x <= 1 + tolerance and -tolerance <= y <= 1 + tolerance
                        and 0 < w <= 1 + tolerance and 0 < h <= 1 + tolerance):
                    problems.append("%s/%s: 坐标越界 %s" % (split, name, parts[1:]))
                elif not (-tolerance <= x - w / 2 and x + w / 2 <= 1 + tolerance
                          and -tolerance <= y - h / 2 and y + h / 2 <= 1 + tolerance):
                    problems.append("%s/%s: 框超出图片 %s" % (split, name, parts[1:]))
                stats[index] += 1
                boxes += 1
        extra_images = images - labels
        extra_labels = labels - images
        print("%s: 图=%d 标注文件=%d 框=%d 无标注图=%d" % (split, len(images), len(labels), boxes, negatives))
        print("  每类框数: %s" % {names[key]: value for key, value in sorted(stats.items())})
        if extra_images:
            problems.append("%s: %d 张图没有标注文件，例如 %s" % (split, len(extra_images), sorted(extra_images)[:3]))
        if extra_labels:
            problems.append("%s: %d 个标注文件没有对应图片，例如 %s" % (split, len(extra_labels), sorted(extra_labels)[:3]))

    print("问题数: %d" % len(problems))
    for item in problems[:20]:
        print("  - %s" % item)


if __name__ == "__main__":
    main()
