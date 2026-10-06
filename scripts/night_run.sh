#!/bin/bash
# Overnight driver for the trash-detector experiments.
#   bash night_run.sh 0   -> dataset audit (CPU only)
#   bash night_run.sh 1   -> yolov8s @640  + rotation/flip augmentation
#   bash night_run.sh 2   -> yolov8s @960  + rotation/flip augmentation
#   bash night_run.sh 3   -> yolov8n @640  + rotation/flip augmentation
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp
D=$W/datasets/trash_merged_bin/data.yaml
cd $W/trash_project || exit 1

mkdir -p /root/.config/Ultralytics
if [ ! -f /root/.config/Ultralytics/Arial.ttf ]; then
  cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /root/.config/Ultralytics/Arial.ttf
fi
if [ ! -f $W/trash_project/yolo11n.pt ]; then
  source /etc/network_turbo 2>/dev/null
  $PY -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || true
fi

case "$1" in
  0)
    $PY $W/trash_project/audit_data.py
    ;;
  1)
    $PY $W/trash_project/train_trash.py --weights yolov8s.pt --data $D \
      --imgsz 640 --epochs 150 --batch 32 --name s_rot \
      --degrees 180 --flipud 0.5 --mixup 0.1 --close-mosaic 20
    ;;
  2)
    $PY $W/trash_project/train_trash.py --weights yolov8s.pt --data $D \
      --imgsz 960 --epochs 150 --batch 16 --name s_rot960 \
      --degrees 180 --flipud 0.5 --mixup 0.1 --close-mosaic 20
    ;;
  3)
    $PY $W/trash_project/train_trash.py --weights yolov8n.pt --data $D \
      --imgsz 640 --epochs 150 --batch 32 --name n_rot \
      --degrees 180 --flipud 0.5 --mixup 0.1 --close-mosaic 20
    ;;
  *)
    echo "usage: bash night_run.sh {0|1|2|3}"
    exit 2
    ;;
esac
echo "STEP $1 EXIT=$?"