#!/usr/bin/env python
"""Export a trained trash detector for the Raspberry Pi.

Run this on the training machine (the one with ultralytics installed), then copy the
resulting folder to the Pi.

    python scripts/export_pi.py weights/yolov8s_ftown_best.pt --imgsz 416 --format ncnn

Why 416: the cart camera sits ~0.5 m above the ground, so a 10 cm object is ~50 px wide
at 1 m and ~25 px at 2 m. 416 input still resolves anything inside the working range
(0.4-2.0 m for a 30-35 deg down-tilt) while costing only ~42% of 640's compute.
See docs/pi-deployment.md.
"""
import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="trained .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=416, choices=[320, 416, 512, 640])
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "openvino", "tflite"])
    ap.add_argument("--half", action="store_true", help="fp16 (Pi 5 supports it, Pi 4 mostly does not)")
    ap.add_argument("--int8", action="store_true", help="int8 quantisation (fastest, needs calibration data)")
    ap.add_argument("--data", default=None, help="dataset yaml, required for --int8 calibration")
    ap.add_argument("--out", default="pi_model", help="folder to collect the exported model into")
    a = ap.parse_args()

    if a.int8 and not a.data:
        ap.error("--int8 needs --data <dataset.yaml> for calibration")

    print("loading", a.weights)
    model = YOLO(a.weights)

    kwargs = {"format": a.format, "imgsz": a.imgsz, "simplify": True}
    if a.half:
        kwargs["half"] = True
    if a.int8:
        kwargs["int8"] = True
        kwargs["data"] = a.data

    path = model.export(**kwargs)
    print("exported ->", path)

    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, os.path.basename(str(path).rstrip("/")))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    if os.path.isdir(str(path)):
        shutil.copytree(str(path), dest)
    else:
        shutil.copy(str(path), dest)

    note = os.path.join(a.out, "README.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write(
            "model: %s\nformat: %s\nimgsz: %d\nhalf: %s\nint8: %s\n\n"
            "On the Pi:\n"
            "  python scripts/pi_infer.py %s --imgsz %d --camera\n\n"
            "Benchmark different resolutions before committing to one:\n"
            "  --imgsz 320 / 416 / 640\n"
            % (a.weights, a.format, a.imgsz, a.half, a.int8, dest, a.imgsz)
        )
    print("collected into", a.out)
    print("=" * 60)
    print("下一步：整份拷贝到树莓派，然后跑")
    print("  python scripts/pi_infer.py %s --imgsz %d --camera" % (dest, a.imgsz))
    print("记得分别测 320 / 416 / 640 的真实帧率再定。")


if __name__ == "__main__":
    main()