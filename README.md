# 捡垃圾小车 · 垃圾检测模型对比

面向**户外自主捡拾机器人**的单类垃圾检测（`trash`），对比 YOLOv8 的 n / s / m 三个规格，
目标是回答一个问题：**最小的 yolov8n 够不够用？**

## 结论（TL;DR）

**够用。选 `yolov8n`。**

三个模型在真实场景上的精度差异只有 0.04，落在统计噪声范围内；而体积差 8 倍，
边缘设备上的帧率差会进一步放大。在更贴合机器人实际需求的宽松口径下，**n 反而排第一**。

## 结果

评测集：**自采的 73 张真实户外照片**（97 个标注框），未参与任何训练。

| 指标 | yolov8n | yolov8s | yolov8m |
|---|---|---|---|
| mAP50 | 0.347 | 0.363 | **0.387** |
| mAP50-95 | 0.125 | 0.129 | **0.142** |
| Precision | 0.475 | 0.494 | **0.524** |
| Recall | 0.412 | 0.423 | **0.442** |
| 宽松 F1（IoU 0.25） | **0.537** | 0.495 | 0.522 |
| 检出框数 | 93 | 85 | **106** |
| 完全漏检的图 | 11 | 15 | **6** |
| 参数量 | 3.0 M | 11.1 M | 25.9 M |
| 权重体积 | 6.3 MB | 22.5 MB | 52.0 MB |
| 训练轮数（早停） | 113 | 95 | 104 |
| 训练耗时（4090D） | 19 min | 24 min | 47 min |

**为什么加"宽松 F1"**：标注框存在偏移，IoU 0.5 的严格口径会同时压低所有模型的分数。
对捡拾机器人而言，只需要知道垃圾在哪、不需要像素级贴合，因此额外统计了 IoU 0.25 下的
精确率/召回率。标注噪声对三个模型的影响一致，横向排名可信，绝对值偏低。

**对照历史成绩**：同一批真实照片上，早期的 8 类 `yolov8s` 只有 mAP50 = 0.150。
改为单类 + 扩充数据后达到 0.347~0.387，**提升约 2.3~2.6 倍**。
说明此任务的主要瓶颈是**数据规模与标注质量**，而不是模型大小。

## 数据集

训练集 `trash_merged_bin`：**4775 张训练 / 73 张验证，共 16343 个框，单类 `trash`**。

| 来源 | 图片数 | 占比 |
|---|---|---|
| Roboflow（公开） | 3358 | 70% |
| TACO（公开） | 1034 | 22% |
| 自采 | 383 | 8% |

- 验证集固定为**自采的 73 张真实照片**，与历史实验口径一致，保证可比
- 合并时带**防泄漏检查**：各来源的验证图不会混入训练集
- 图片使用**硬链接**合并，不额外占用磁盘

> ⚠️ 注意：Roboflow 部分多为摆拍/商品图（该数据集有较重的类别不平衡，如 Plastic Wrapper 1990 个框、PaperBag 仅 5 个），，和真实街景域差异较大。
> 该数据集适合让模型入门，**最终效果取决于自采真实数据的数量**。

## 复现

### 环境

```bash
pip install ultralytics          # 本项目使用 8.4.157
```

其余脚本只依赖 `numpy` / `Pillow`（`ultralytics` 会自动带上）。

### 目录约定

```
/root/autodl-tmp/
├── trash_project/            # 本仓库脚本所在位置
│   ├── yolov8n.pt            # 预训练权重
│   └── runs/                 # 训练输出
└── datasets/
    └── trash_merged_bin/
        ├── data.yaml
        ├── train/{images,labels}
        └── val/{images,labels}
```

### 训练

```bash
python scripts/train_trash.py \
  --weights yolov8n.pt \
  --data /root/autodl-tmp/datasets/trash_merged_bin/data.yaml \
  --epochs 150 --batch 32 --workers 16 --name n_merged
```

`train_trash.py` 的默认设置：`imgsz=640`、`patience=50`（验证集 50 轮不涨即早停）、
`cache=True`（图片缓存进内存）、`degrees=5`。三组实验均在第 95~113 轮触发早停。

> 首次运行前若服务器无法访问 GitHub，需手动准备两样东西，否则训练会卡在下载上：
> 1. 字体：把任意 TTF 复制为 `~/.config/Ultralytics/Arial.ttf`
> 2. AMP 检查权重：把 `yolo11n.pt` 放到工作目录

### 评估与可视化

```bash
# 单模型：标准指标 + 多置信度 + 真实场景对比图
RUN_NAME=n_merged python scripts/eval_real.py

# 三模型横向对比（精度/速度/体积 + 并排可视化）
python scripts/compare_models.py
```

### 数据集构建

```bash
python scripts/download_taco.py --out data/images --workers 32   # 下载 TACO
python scripts/taco2yolo.py                                      # 转 YOLO 格式
python scripts/roboflow2yolo.py                                  # 转 Roboflow 数据集
python scripts/merge_yolo_sets.py --val-from <自采集>            # 合并 + 防泄漏
python scripts/check_dataset.py                                  # 完整性校验
```

## 目录结构

```
scripts/            训练、评估、对比、数据集构建脚本
  roboflow/         Roboflow 数据集下载与格式整理
results/            评测指标、训练曲线、可视化图
  compare.json      三模型完整指标
  side_by_side.jpg  同一批真实照片的逐模型预测对照
  curves/           各模型训练曲线
weights/            训练好的权重（n 与 s；m 体积较大未收录）
```

## 数据来源

- [TACO](https://github.com/pedropro/TACO) — Trash Annotations in Context，公开数据集
- Roboflow Universe — [YoloV7 Trash](https://universe.roboflow.com/technological-institute-of-the-philippines/yolov7-trash-05-04-2023)，Technological Institute of the Philippines 公开数据集（CC BY 4.0）
- 自采数据 — 户外真实场景拍摄

## 相关文献

- Kulshreshtha et al. *OATCR: Outdoor Autonomous Trash-Collecting Robot Design Using YOLOv4-Tiny*, Electronics 2021 — 同类捡拾机器人，YOLOv4-tiny 以 95.2% mAP / 5.2 ms 击败 YOLOv4（97.1% / 32.8 ms）并被最终采用
- *SS-YOLOv8: A Lightweight Algorithm for Surface Litter Detection*, Applied Sciences 2024 — 2.3 MB 模型达到 79.9% mAP / 128 FPS
- *TrashDet*, 2025 — TACO 上 TinyML 系列（1.2M~30.5M 参数）mAP50 仅 11.4~19.5，说明该任务本身难度高
- *pLitterStreet*, 2024 — 13000 张车载街景，Faster R-CNN / RetinaNet / YOLOv3 / YOLOv5 均仅约 40% AP
- *A Performance Analysis of YOLO Models on Constrained Edge Devices*, Electronics 2025 — Jetson Orin NX 上 yolov8n 达 52 FPS，INT8 量化后 65 FPS；树莓派 5 无法满足实时
