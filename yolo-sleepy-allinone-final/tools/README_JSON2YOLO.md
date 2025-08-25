# JSON2YOLO (Ultralytics) — Convert COCO Keypoints → YOLO Pose
1) Cài đặt: `pip install ultralytics`
2) Chuẩn bị: COCO keypoint JSON + ảnh tương ứng.
3) Convert: dùng `yolo` cli
```
yolo convert model=yolo11n-pose.pt source=/path/to/coco_keypoints.json format=yolo
```
4) Đặt output vào `datasets/sleepy_pose/{images,labels}/{train,val}` theo sleepy.yaml.
