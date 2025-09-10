# Training sleepy-pose (YOLOv11 Pose)

This folder contains scripts to auto-label, train, and evaluate a pose model tailored for sleepy detection.

## 1) Build dataset (semi-auto)

- Put your raw media (images/videos) into a folder, e.g. `data_raw/`.
- Run auto-label to create train/val images and labels (pose + class: binhthuong/ngugat/gucxuongban):

```powershell
cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools
python auto_label_pose.py --source ..\..\data_raw --out ..\datasets\sleepy_pose --model ..\yolo11n-pose.pt --imgsz 960 --frame-stride 5
```

Notes:
- Inspect and fix labels for hard cases. The heuristics are decent but not perfect.
- Ensure `yolo-sleepy-allinone-final/datasets/sleepy_pose/sleepy.yaml` exists (in repo).

## 2) Train pose model

```powershell
cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools
python train_pose.py --data ..\datasets\sleepy_pose\sleepy.yaml --model ..\yolo11n-pose.pt --epochs 120 --imgsz 640 --cos_lr
```

Outputs in `..\runs\pose-train\<exp>/weights/best.pt`.

Tips:
- If you have a GPU: add `--device cuda:0` and consider `--model ..\yolo11s-pose.pt`.
- If RAM is tight: keep `imgsz` at 512–640, and let batch auto-select.
- Watch mAP50/95, Precision/Recall; stop early if overfitting.

## 3) Evaluate

```powershell
python eval_pose.py --data ..\datasets\sleepy_pose\sleepy.yaml --weights ..\runs\pose-train\exp\weights\best.pt --imgsz 640
```

This prints validation metrics and saves PR/RC curves in the run folder.

## 4) Use the new model in the app

- Copy the trained `best.pt` to project root or reference it directly:

```powershell
cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final
python standalone_app.py --cam 0 --res 640x480 --mjpg --model runs\pose-train\exp\weights\best.pt
```

## 5) Data quality checklist

- Balanced classes: ~700 normal, ~600 ngugat, ~300–500 gucxuongban (phase 1 suggestion).
- Include hard cases: head down but sitting straight, occlusions, lighting changes.
- Validate a small subset manually; fix mislabeled keypoints/class.

## 6) Troubleshooting

- Empty dataset folders? Run auto-label, or add images first.
- Slow training on CPU? Reduce `imgsz` (e.g., 512), fewer epochs, or switch to GPU.
- Pose keypoints missing? Check that YAML has `kpt_shape: [17, 3]` and labels have 51 kpt values.
