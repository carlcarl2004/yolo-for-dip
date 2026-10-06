#!/usr/bin/env python
"""Raspberry Pi inference skeleton for the litter cart.

This is a starting point, not a finished product -- it has not been run on real Pi
hardware. Adapt the camera part to your mounting and the GPIO part to your motor driver.

Install on the Pi:
    pip install ultralytics            # brings torch; heavy but simplest
    sudo apt install -y python3-picamera2

Run:
    python scripts/pi_infer.py pi_model/yolov8s_cart_ncnn_model --imgsz 416 --camera
    python scripts/pi_infer.py pi_model/yolov8s_cart_ncnn_model --imgsz 416 --bench

Design notes that come from the evaluation in this repo:
  * conf 0.4-0.5 is the sensible operating point; do not raise it to 0.8, you would
    lose almost every detection (measured: recall 29% -> 15% between 0.5 and 0.8).
  * Treat detections as candidate targets. The model does produce false positives on
    railings, road markings and hard shadows -- require 2 consecutive frames before
    acting on a detection.
  * Keep the input at 416 unless you measured that you need 640.
"""
import argparse
import time

import numpy as np


def load_model(path):
    from ultralytics import YOLO
    return YOLO(path, task="detect")


def bench(model, imgsz, conf, iters=50):
    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
    t0 = time.time()
    for _ in range(iters):
        model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
    dt = time.time() - t0
    print("imgsz=%d  %.2f FPS  (%.1f ms/frame, %d frames)" % (imgsz, iters / dt, 1000 * dt / iters, iters))
    return iters / dt


def run_camera(model, imgsz, conf, iou, show):
    """Grab frames from the Pi camera and print detections above `conf`."""
    from picamera2 import Picamera2

    cam = Picamera2()
    # keep the resolution close to the network input: less resizing, less latency
    cam.configure(cam.create_preview_configuration(main={"size": (imgsz, imgsz), "format": "RGB888"}))
    cam.start()
    time.sleep(2)                      # let auto-exposure settle
    print("camera up, %dx%d. Ctrl-C to stop." % (imgsz, imgsz))

    hits = 0
    total = 0
    t0 = time.time()
    try:
        while True:
            frame = cam.capture_array()
            res = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
            total += 1
            boxes = res.boxes
            if boxes is not None and len(boxes):
                hits += 1
                best = float(boxes.conf.max())
                x1, y1, x2, y2 = boxes.xyxy[int(boxes.conf.argmax())].tolist()
                # x centre tells the robot which way to steer
                cx = (x1 + x2) / 2 / imgsz
                side = "left" if cx < 0.4 else ("right" if cx > 0.6 else "ahead")
                print("TRASH conf=%.2f box=(%.0f,%.0f,%.0f,%.0f) -> steer %s"
                      % (best, x1, y1, x2, y2, side))
                # TODO: drive the motors here (GPIO), but see the "2 consecutive frames" note
            if total % 30 == 0:
                fps = total / (time.time() - t0)
                print("[%5.1f FPS] frames with a detection: %d/%d" % (fps, hits, total))
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="exported NCNN/ONNX model folder or .pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--camera", action="store_true", help="run on the live Pi camera")
    ap.add_argument("--bench", action="store_true", help="synthetic frame-rate benchmark only")
    ap.add_argument("--show", action="store_true", help="also save annotated frames (slow)")
    a = ap.parse_args()

    model = load_model(a.model)
    if a.bench or not a.camera:
        bench(model, a.imgsz, a.conf)
        if not a.camera:
            return
    run_camera(model, a.imgsz, a.conf, a.iou, a.show)


if __name__ == "__main__":
    main()