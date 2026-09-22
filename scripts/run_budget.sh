#!/bin/bash
# ============================================================
#  预算内一次性训练：全部跑完约 3.5 ~ 4.5 小时（4090D，¥2/时 -> ¥7~9）
#     bash run_budget.sh        全部依次跑
#     bash run_budget.sh 1      只跑第 1 步 (yolov8n + 合并集)
#     bash run_budget.sh 2      只跑第 2 步 (yolov8s + 合并集)
#     bash run_budget.sh 3      只跑第 3 步 (yolov8n + TACO)
#  日志: /root/autodl-tmp/run_budget.log
# ============================================================
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/trash_project || exit 1
export PYTHONUNBUFFERED=1
LOG=/root/autodl-tmp/run_budget.log

MERGED=/root/autodl-tmp/datasets/trash_merged_bin/data.yaml
TACO=/root/autodl-tmp/datasets/taco_bin/data.yaml

# --- 0) 先确认显卡已挂上，没卡就别烧钱 ---
if ! $PY -c "import torch,sys; n=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''; print('GPU:', n or 'NONE'); sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "!! 没有检测到显卡：请先把实例切到 4090D 再运行"
  exit 1
fi

run() {
  name=$1; w=$2; d=$3; ep=$4
  echo ""                                                                      | tee -a "$LOG"
  echo "########## $(date '+%m-%d %H:%M:%S')  START  $name  ${ep} 轮 ##########" | tee -a "$LOG"
  $PY train_trash.py --weights "$w" --data "$d" --epochs "$ep" \
      --batch 32 --workers 16 --name "$name" 2>&1 | tee -a "$LOG"
  echo "########## $(date '+%m-%d %H:%M:%S')  DONE   $name ##########"           | tee -a "$LOG"
}

P=${1:-all}
if [ "$P" = "all" ] || [ "$P" = "1" ]; then run n_merged yolov8n.pt "$MERGED" 150; fi
if [ "$P" = "all" ] || [ "$P" = "2" ]; then run s_merged yolov8s.pt "$MERGED" 150; fi
if [ "$P" = "all" ] || [ "$P" = "3" ]; then run n_taco   yolov8n.pt "$TACO"   100; fi

echo ""                                                                          | tee -a "$LOG"
echo "========== 全部完成 $(date '+%m-%d %H:%M:%S') =========="                   | tee -a "$LOG"
grep -h "^FINAL" "$LOG"
echo ""
echo ">>> 跑完了，记得关机停止计费 <<<"
