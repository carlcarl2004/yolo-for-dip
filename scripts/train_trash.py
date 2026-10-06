#!/usr/bin/env python
"""Train YOLOv8 litter detector (AutoDL)."""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--data", default="/root/autodl-tmp/datasets/trash_merged_bin/data.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--degrees", type=float, default=5.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--project", default="/root/autodl-tmp/trash_project/runs")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.set_defaults(cache=True)
    ap.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=15)
    ap.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    name = a.name or ("trash_" + a.weights.replace(".pt", ""))
    model = YOLO(a.weights)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=a.project,
        name=name,
        device=a.device,
        workers=a.workers,
        cache=a.cache,
        patience=a.patience,
        seed=0,
        lr0=a.lr0,
        cos_lr=a.cos_lr,
        mixup=a.mixup,
        scale=a.scale,
        degrees=a.degrees,
        flipud=a.flipud,
        fliplr=a.fliplr,
        close_mosaic=a.close_mosaic,
        plots=True,
        exist_ok=True,
    )
    m = model.val()
    print(f"FINAL {name} mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} "
          f"precision={m.box.mp:.4f} recall={m.box.mr:.4f}")


if __name__ == "__main__":
    main()