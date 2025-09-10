# Nhật ký phát triển (Dev Log)

Tài liệu ghi chép tiến độ và các quyết định kỹ thuật của dự án YOLO-Sleepy từ lúc bắt đầu đến hiện tại (cập nhật: 2025-09-09).

## Mục tiêu ban đầu
- Phát hiện buồn ngủ/ngủ gật thời gian thực từ camera.
- Hỗ trợ cả trường hợp “gục xuống bàn”.
- Giao diện overlay tiếng Việt, dễ đọc; có log, FPS, và thống kê thời lượng.

## Dòng thời gian và mốc chính

### Giai đoạn 1 — Khởi động & thử nghiệm web (Streamlit)
- Xây dựng bản thử nghiệm chạy mô hình YOLO Pose trên web.
- Vấn đề: giật/lag mạnh khi stream video → không phù hợp cho thời gian thực.

### Giai đoạn 2 — Chuyển sang ứng dụng Desktop (OpenCV)
- Viết ứng dụng OpenCV thuần (Python) để cải thiện độ trễ.
- Xử lý hiển thị Unicode tiếng Việt bằng Pillow (PIL) → hàm `draw_text_unicode`.
- Thêm đa backend camera (CAP_DSHOW, CAP_MSMF) và tùy chọn MJPG để cải thiện FPS.
- Thêm ước lượng FPS (EMA) để hiển thị ổn định.

### Giai đoạn 3 — Heuristics Pose cho “Ngủ gật” và “Gục xuống bàn”
- Trích xuất keypoint mũi + vai (YOLO-Pose) để tính:
  - Góc nghiêng so với trục dọc (nose–neck) → phát hiện cúi/ngả.
  - Độ rơi theo chiều dọc (nose dưới vai) theo tỉ lệ ảnh và bề rộng vai.
- Ngưỡng nhạy (có thể tinh chỉnh):
  - Gục xuống bàn: drop_h_ratio > 0.22 hoặc drop_sw_ratio > 0.65.
  - Ngủ gật: angle_v > 25° hoặc drop_h_ratio > 0.12 hoặc drop_sw_ratio > 0.35.
- Debounce bằng SLEEP_FRAMES (15 khung) để tránh nhấp nháy.

### Giai đoạn 4 — Trạng thái, log và thống kê
- Máy trạng thái per-person: `sleep_states`, `sleep_status`, thời điểm bắt đầu ngủ.
- Ghi log sự kiện “Ngủ gật”, “Thức dậy” vào panel.
- Tính và hiển thị “Ngủ gật lâu nhất”.
- Overlay tiếng Việt rõ ràng gần mũi từng người.

### Giai đoạn 5 — Dữ liệu & huấn luyện (YOLO-Pose)
- Tạo khung dữ liệu `datasets/sleepy_pose` (train/val images & labels).
- Cập nhật YAML:
  - `names = {0: binhthuong, 1: ngugat, 2: gucxuongban}`
  - `kpt_shape = [17,3]` (chuẩn COCO pose), `train/val` paths.
- Viết công cụ auto-label `tools/auto_label_pose.py`:
  - Chạy inference + heuristics để sinh nhãn YOLO-Pose.
  - Chia train/val tự động.
- Hướng dẫn huấn luyện Ultralytics. Lưu ý: lần đầu lỗi vì thư mục ảnh trống → đã hướng dẫn bổ sung dữ liệu/auto-label.

### Giai đoạn 6 — Chế độ ảnh tĩnh & đại tu file app
- Thêm tham số `--image` để kiểm thử trên 1 ảnh tĩnh.
- Dọn file `standalone_app.py` bị trùng đoạn code sau refactor; sửa lỗi FOURCC MJPG.
- Xác thực CLI `--help`, chạy image-mode, chạy webcam.

### Giai đoạn 7 — Tích hợp kỹ thuật “real-time drowsy driving” (mắt/ngáp)
- Tùy chọn bật pipeline phụ (qua `--enable-eyes`):
  - MediaPipe FaceMesh để cắt ROI mắt/miệng.
  - Hai mô hình YOLO phụ: mắt (open/close) và ngáp (yawn/no-yawn).
  - Bộ đếm: Blinks, Microsleeps (giây mắt nhắm), Yawns, Yawn Duration.
  - Cảnh báo khi `microsleeps ≥ 3s` hoặc `yawn_duration ≥ 7s` (tùy chỉnh được).
- Thêm tham số CLI: `--eye-weights`, `--yawn-weights`, `--secondary-interval`, `--microsleep-thresh`, `--yawn-thresh`.
- Panel thông tin phụ xếp bên phải, dưới Log.

### Giai đoạn 8 — Cải thiện UI Overlay
- Thu nhỏ & dời Log panel sang góc phải-trên, nền bán trong suốt, thích ứng kích thước.
- Xếp panel Eye/Yawn ngay bên dưới Log, tránh che khung chính.

### Giai đoạn 9 — Sửa chuyển trạng thái “Thức dậy”
- Thêm hysteresis hai ngưỡng:
  - Vào “Ngủ gật/Gục” khi `SLEEP_FRAMES ≥ 15` khung liên tiếp buồn ngủ.
  - Thoát về “Bình thường” khi `AWAKE_FRAMES ≥ 5` khung liên tiếp bình thường.
- Kết quả: overlay đổi trạng thái nhanh và ổn định khi ngồi dậy.

## Tệp & thư mục quan trọng
- `yolo-sleepy-allinone-final/standalone_app.py`: Ứng dụng desktop chính (webcam/ảnh tĩnh), overlay VN, heuristics pose, pipeline mắt/ngáp (tùy chọn), log & thống kê.
- `yolo-sleepy-allinone-final/datasets/sleepy_pose/sleepy.yaml`: Cấu hình dataset 3 lớp cho YOLO-Pose.
- `yolo-sleepy-allinone-final/tools/auto_label_pose.py`: Gán nhãn bán tự động từ ảnh/video.
- `real-time-drowsy-driving-detection/`: Tham chiếu mô hình mắt/ngáp và logic drowsiness phụ.

## Hiệu năng (tham chiếu nhanh)
- YOLO11n-Pose, imgsz ~960: thời gian suy luận ~85–120 ms/khung trên CPU → ~8–11 FPS (quan sát từ log và HUD).
- Có thể tăng FPS bằng: MJPG camera, giảm imgsz, dùng GPU/TensorRT/ONNX, hoặc đổi kiến trúc nhỏ hơn.

## So sánh mô hình YOLO (Pose) — số liệu đo trên máy hiện tại (CPU)
- Bối cảnh đo: Windows, CPU, 1 khung hình webcam (480×640), imgsz=640, Ultralytics YOLO v8.3.x.
- Script đo: `yolo-sleepy-allinone-final/tools/benchmark_pose_models.py` (đã thêm vào repo).

Kết quả (FPS cao hơn là tốt hơn):
- yolo11n-pose.pt: 6.02 FPS (15 vòng, tổng 2.49s)
- yolo11n.pt (detector thường, không pose): 4.26 FPS (15 vòng, tổng 3.52s)
- yolo11s-pose.pt: 2.93 FPS (15 vòng, tổng 5.11s)
- yolo11m-pose.pt: 1.28 FPS (15 vòng, tổng 11.71s)

Nhận xét nhanh:
- yolo11n-pose nhanh nhất trên CPU → phù hợp realtime hơn. yolo11s/m-pose chậm đáng kể trên CPU.
- Mô hình detector thường (yolo11n.pt) không xuất keypoints, nên không dùng trực tiếp cho heuristics pose của ứng dụng.
- FPS trong app sẽ thấp hơn đôi chút do overlay, tracking, và pipeline phụ (mắt/ngáp).

Khuyến nghị lựa chọn mô hình:
- CPU-only: dùng yolo11n-pose.pt để đạt FPS tốt; kết hợp heuristic (góc/độ rơi/độ rơi so với bbox) và hysteresis như hiện tại.
- Có GPU (CUDA): có thể nâng lên yolo11s-pose.pt để tăng độ chính xác pose, chấp nhận giảm FPS; tinh chỉnh imgsz để cân bằng.
- Trường hợp nhiều người cùng lúc: ưu tiên model nhanh (n-pose) + tracking (đã thêm) để giữ ổn định ID và overlay rõ ràng.

Tái lập benchmark (tùy chọn):
```powershell
cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools
python benchmark_pose_models.py --models yolo11n-pose.pt "yolo11n.pt" "yolo11s-pose.pt" "yolo11m-pose.pt" --iters 15 --imgsz 640
```

## Phụ: Kết quả huấn luyện mô hình phụ (mắt/ngáp) hiện có
- Eye (Open/Close) — epoch 10 (val): precision ~0.73, recall ~0.86, mAP50 ~0.78, mAP50-95 ~0.73.
- Yawn (Yawn/No-Yawn) — epoch 10 (val): precision ~0.77, recall ~0.73, mAP50 ~0.79, mAP50-95 ~0.59.

Gợi ý: tiếp tục thu thập và cân bằng dữ liệu, đặc biệt các ca khó (nhìn xuống, ánh sáng yếu, che khuất) để cải thiện độ tin cậy.

## Tổng quan lý thuyết YOLO (tóm tắt)
- YOLO (You Only Look Once) là họ mô hình phát hiện đối tượng “một bước” (single-stage):
  - Backbone trích xuất đặc trưng (CSP/Darknet, v11 cải thiện hiệu năng/độ chính xác).
  - Neck (FPN/PAN) hợp nhất đa tỉ lệ.
  - Head dự đoán trực tiếp trên lưới: hộp (x,y,w,h), độ tin cậy, lớp; với YOLO-Pose, head bổ sung keypoints (tọa độ + độ tin cậy từng điểm).
- Anchor-based/anchor-free: các phiên bản mới thiên về anchor-free, giảm độ phức tạp và cải thiện tốc độ trên CPU/GPU.
- Post-processing: NMS/NMS phân lớp để loại bỏ trùng lặp; với pose, còn lọc điểm theo độ tin cậy.
- Loss phổ biến: box (IoU/GIoU/CIoU), cls (BCE/CE), DFL (Distribution Focal Loss) cho hồi quy hộp mượt hơn; pose có thêm loss cho keypoints.
- Ưu/nhược:
  - Ưu: nhanh, triển khai gọn, phù hợp realtime; hệ sinh thái Ultralytics tiện dụng (train/export/infer CLI & Python).
  - Nhược: độ chính xác có thể kém hơn hai-bước trên bài toán đặc thù; kết quả pose phụ thuộc chất lượng dữ liệu và điều kiện ánh sáng.
- YOLO-Pose: học vị trí 17 điểm (chuẩn COCO) → cho phép suy luận tư thế, góc cúi/nghiêng, độ rơi đầu, v.v.

## Ứng dụng thực tế: phát hiện buồn ngủ/ngủ gật
- Tín hiệu thường dùng:
  - PERCLOS (tỉ lệ thời gian mắt nhắm), tần suất chớp mắt, thời lượng ngáp.
  - Tư thế đầu/cổ: cúi mặt, gật đầu, gục xuống mặt bàn.
  - Chuyển động nhỏ/vi chuyển động giảm (head pose ổn định bất thường).
- Pipeline đề xuất (đã triển khai trong app):
  1) Camera capture (OpenCV) + tối ưu backend (DSHOW/MSMF), MJPG để tăng FPS.
  2) YOLO-Pose → keypoints; tùy chọn YOLO phụ (eye/yawn) + MediaPipe FaceMesh để cắt ROI.
  3) Heuristics: góc mũi-so-vai, độ rơi theo ảnh/bề rộng vai, và tỉ lệ rơi theo chiều cao bbox (drop_bb_ratio) để bắt ca “cúi mặt nhưng ngồi thẳng”.
  4) Tracking (IoU) gán ID ổn định theo người; hysteresis vào/ra trạng thái để tránh nhấp nháy.
  5) Cảnh báo + overlay VN: log, FPS, thống kê thời lượng, dải nhãn màu rõ ràng trên mỗi người.
- Thách thức:
  - Ánh sáng yếu, che khuất (tay/khẩu trang), góc camera thấp/cao, nhiều người chồng lấn.
  - Sai số pose ở khoảng cách xa/độ phân giải thấp; yêu cầu tinh chỉnh imgsz/ngưỡng.
  - Quyền riêng tư: hạn chế lưu trữ video gốc, chỉ lưu sự kiện/ảnh đã làm mờ nếu cần.

## Bảng so sánh chi tiết mô hình
| Mô hình             | Nhiệm vụ | Pose keypoints | imgsz (đo) | FPS CPU (đo) | Điểm mạnh | Hạn chế | Khuyến nghị |
|---------------------|----------|----------------|------------|--------------|-----------|--------|------------|
| yolo11n-pose.pt     | Pose     | Có             | 640        | 6.02         | Nhanh nhất trên CPU; có keypoints cho heuristics | Độ chính xác pose thấp hơn bản lớn | Mặc định cho máy không GPU, nhiều người |
| yolo11s-pose.pt     | Pose     | Có             | 640        | 2.93         | Pose ổn hơn n-pose | Chậm hơn đáng kể trên CPU | Dùng nếu có GPU hoặc cần pose ổn định hơn |
| yolo11m-pose.pt     | Pose     | Có             | 640        | 1.28         | Tiềm năng chính xác cao hơn | Quá chậm trên CPU | Chỉ khi có GPU mạnh |
| yolo11n.pt          | Detect   | Không          | 640        | 4.26         | Nhanh, gọn | Không có keypoints (không dùng heuristics pose) | Không khuyến nghị cho bài toán này |

Ghi chú:
- Số FPS đo được bằng `tools/benchmark_pose_models.py` trên CPU máy hiện tại, imgsz=640, 15 vòng; FPS thực tế trong ứng dụng thấp hơn chút do overlay, tracking và pipeline mắt/ngáp.
- Có thể tăng FPS bằng cách giảm imgsz (ví dụ 512/480), bật MJPG, hoặc sử dụng GPU/ONNX/TensorRT.

## So sánh YOLOv3 vs YOLOv5 vs YOLOv11 cho nhận diện ngủ gật (dựa trên pose)

| Phiên bản | Năm | Pose gốc | Hệ sinh thái/Train | Triển khai | Phù hợp CPU | Liên quan bài toán (pose/keypoints) | Khi nên dùng |
|---|---|---|---|---|---|---|---|
| YOLOv3 | 2018 | Không (cần repo/phụ trợ) | Darknet cổ điển; training ít thuận tiện hơn PyTorch | Darknet/ONNX (chuyển đổi) | Trung bình/Chậm trên CPU hiện đại | Thiếu pose gốc → không trực tiếp tính góc/độ rơi đầu | Chỉ khi hệ thống legacy yêu cầu Darknet |
| YOLOv5 | 2020 | Có (v5-pose) | PyTorch/Ultralytics, dễ train/finetune, export tiện | ONNX/TensorRT/CoreML | Khá tốt trên CPU | Có keypoints; đủ dùng nếu phần cứng yếu và cần ổn định | Biên/nhúng yếu, hoặc cần tương thích lâu năm |
| YOLOv11 | 2024 | Có (v11-pose) | PyTorch/Ultralytics thế hệ mới, tối ưu hơn | ONNX/TensorRT/OpenVINO… | Tốt nhất (cân bằng tốc độ/độ chính xác) | Keypoints tốt hơn, ổn định; phù hợp drowsiness realtime | Lựa chọn mặc định hiện tại |

Kết luận nhanh cho ứng dụng ngủ gật:
- Ưu tiên YOLOv11-pose (n/s tuỳ phần cứng) vì: có keypoints ổn định, tốc độ/độ chính xác tốt, hệ công cụ Ultralytics mới nhất.
- Chỉ cân nhắc YOLOv5-pose khi cần export/triển khai trên phần cứng biên rất hạn chế hoặc phải giữ tương thích cũ.
- Tránh YOLOv3 cho bài toán pose trừ khi ràng buộc legacy, vì thiếu pose gốc và hệ sinh thái train/triển khai kém linh hoạt hơn.

## Nhật ký huấn luyện gần đây (2025-09-09)

Tóm tắt những gì đã train, train như thế nào, và kết quả hiện có cho bộ dữ liệu `sleepy_pose` (3 lớp: `binhthuong`, `ngugat`, `gucxuongban`).

- Dataset & cấu hình:
  - YAML: `yolo-sleepy-allinone-final/datasets/sleepy_pose/sleepy.yaml`
  - kpt_shape: `[17,3]`, names: `{0: binhthuong, 1: ngugat, 2: gucxuongban}`
  - ĐÃ thêm `flip_idx` cho COCO-17 để bật augment lật ngang đúng:
    - `flip_idx: [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]`

- Công cụ huấn luyện/đánh giá:
  - Train: `yolo-sleepy-allinone-final/tools/train_pose.py` (đã cứng hoá an toàn cho CPU: `workers=0` khi `device=cpu`, batch tự động được ấn định; `deterministic=True`).
  - Đánh giá: `yolo-sleepy-allinone-final/tools/eval_pose.py`.

- Các lần train đã thực hiện (thư mục kết quả):
  - Run A — `sleepy_pose_v11n2` (baseline đầu tiên, model: `yolo11n-pose.pt`):
    - Weights: `yolo-sleepy-allinone-final/runs/pose-train/sleepy_pose_v11n2/weights/best.pt`
    - Đánh giá (val nhỏ, sơ bộ): mAP50(B) ≈ 0.286; mAP50(P) ≈ 0.0087. Gợi ý: dữ liệu còn ít/thiếu cân bằng → cần mở rộng.
  - Run B — `sleepy_pose_v11n3` (tiếp theo, cùng kiến trúc cơ sở):
    - Weights: `yolo-sleepy-allinone-final/runs/pose-train/sleepy_pose_v11n3/weights/best.pt`
    - Đánh giá (val rất nhỏ — 1 ảnh/2 targets → số liệu phồng):
      - Precision(B)=1.0000, Recall(B)=0.9797, mAP50(B)=0.9950, mAP50-95(B)=0.8458
      - Precision(P)=1.0000, Recall(P)=0.9797, mAP50(P)=0.9950, mAP50-95(P)=0.6965
      - Tốc độ (CPU, ms/img): preprocess ≈ 1.13, inference ≈ 187.6, postprocess ≈ 2.02
      - Cảnh báo: `nt_per_image=[0,0,1]`, `nt_per_class=[0,0,2]` → tập val quá nhỏ, cần mở rộng để số liệu có ý nghĩa.

- Lệnh đánh giá đã dùng (tái lập):
  ```powershell
  # Đánh giá run v11n2
  python -X utf8 d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools\eval_pose.py \
    --data d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\datasets\sleepy_pose\sleepy.yaml \
    --weights d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\runs\pose-train\sleepy_pose_v11n2\weights\best.pt \
    --imgsz 640

  # Đánh giá run v11n3
  python -X utf8 d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools\eval_pose.py \
    --data d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\datasets\sleepy_pose\sleepy.yaml \
    --weights d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\runs\pose-train\sleepy_pose_v11n3\weights\best.pt \
    --imgsz 512
  ```

- Nhận xét & khuyến nghị:
  - Số liệu v11n3 rất cao do val bé tí → chưa phản ánh chất lượng thật; cần tăng kích thước và đa dạng tập val/test.
  - (ĐÃ LÀM) Bổ sung `flip_idx` vào YAML để bật flip augmentation đúng cho keypoints COCO-17.
  - Tiếp tục auto-label + rà soát thủ công ca khó (cúi mặt nhưng ngồi thẳng, che khuất một phần, chói/thiếu sáng).
  - Tích hợp tạm thời `best.pt` của v11n3 vào ứng dụng để test thực tế và hiệu chỉnh ngưỡng heuristics.

## Cập nhật cấu hình & ứng dụng (2025-09-09)

Các thay đổi chức năng vừa thực hiện để đồng bộ hóa cấu hình dữ liệu và ứng dụng:

- Dataset (sleepy_pose):
  - Thêm `flip_idx` vào `sleepy.yaml` (COCO-17) → bật augment lật ngang đúng cho keypoints khi train/eval.

- Ứng dụng (app):
  - Đổi model mặc định `--model` trong `yolo-sleepy-allinone-final/standalone_app.py` sang:
    - `runs/pose-train/sleepy_pose_v11n3/weights/best.pt`
    - Vẫn có thể ghi đè bằng tham số CLI; GUI cũng dùng mặc định này qua `args.model`.
  - Sửa lỗi thụt lề nhỏ trong `yolo-sleepy-allinone-final/gui_app.py` (khởi tạo model) để tránh lỗi cú pháp; không đổi hành vi.

- Kiểm thử nhanh (smoke test):
  - Chạy chế độ CLI với ảnh tĩnh thành công (cửa sổ hiển thị kết quả; thoát thủ công) → xác nhận import và đường dẫn weights mặc định hoạt động.


## Cách chạy
- Webcam cơ bản:
  ```powershell
  cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final
  python standalone_app.py --cam 0 --res 640x480 --mjpg
  ```
- Ảnh tĩnh:
  ```powershell
  python standalone_app.py --image path\to\image.jpg
  ```
- Bật mắt/ngáp (tùy chọn):
  ```powershell
  pip install mediapipe
  python standalone_app.py --cam 0 --res 640x480 --mjpg --enable-eyes \
    --eye-weights ..\real-time-drowsy-driving-detection\runs\detecteye\train\weights\best.pt \
    --yawn-weights ..\real-time-drowsy-driving-detection\runs\detectyawn\train\weights\best.pt
  ```

## Bài học & quyết định kỹ thuật
- Streamlit không phù hợp cho realtime video → chuyển OpenCV desktop.
- Unicode overlay ổn định cần Pillow, không dùng `cv2.putText` cho tiếng Việt dấu.
- Kết hợp lớp mô hình (nếu có) với heuristics pose + debounce giúp thực dụng hơn.
- Hysteresis (vào/ra) giúp trạng thái ổn định và phản hồi đúng kỳ vọng người dùng.
- Auto-label tăng tốc tạo dữ liệu, nhưng cần kiểm định thủ công mẫu khó để tránh nhiễu nhãn.

## Việc tiếp theo (đề xuất)
- Tracking (BYTE/OC-SORT) để gắn ID ổn định per-person (PERCLOS theo người).
- Huấn luyện lại YOLO-Pose với dữ liệu thật (ba lớp) để giảm phụ thuộc heuristics.
- Gộp tín hiệu: pose + mắt/ngáp → bộ phân loại trạng thái cuối cùng mạnh hơn.
- Tối ưu suy luận: FP16/GPU/TensorRT/ONNX, cân chỉnh `imgsz`, thay model nhẹ.
- Đóng gói: script khởi chạy, cấu hình .bat, hoặc gói app (PyInstaller) để dùng nhanh.

---
Nếu cần log chi tiết hơn theo ngày/commit, có thể bổ sung bảng mốc với ngày-giờ và thay đổi file cụ thể (CHANGELOG).

## Báo cáo tiến độ lần 1 (09/2025)

### 1) Tổng quan đề tài và tham khảo
- Mục tiêu: Phát hiện hành vi buồn ngủ/ngủ gật của sinh viên trong lớp học theo thời gian thực, đưa cảnh báo kịp thời.
- Ứng dụng: Camera lớp học/PC; overlay tiếng Việt; nhật ký sự kiện; hỗ trợ nhiều người cùng lúc.
- Tham khảo chính (đề nghị trích dẫn trong đề cương):
  - Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection” (YOLOv1, 2016)
  - Redmon & Farhadi, “YOLOv3: An Incremental Improvement” (2018)
  - Bochkovskiy et al., “YOLOv4: Optimal Speed and Accuracy of Object Detection” (2020)
  - Ultralytics YOLOv5 Docs (2020–)
  - Wang et al., “YOLOv7: Trainable bag-of-freebies sets new SOTA” (2022)
  - Ultralytics YOLOv8 Docs (2023–)
  - Ultralytics YOLOv11 Release (2024–)
  - Drowsiness: PERCLOS, head pose, blink/yawn literature (tổng hợp nhiều nguồn học thuật)
  - Gợi ý liên hệ GVHD: chốt phạm vi (pose-based vs. face-based), kế hoạch dữ liệu, chuẩn đánh giá.

### 2) Nghiên cứu lý thuyết (tóm tắt)
- Học máy & CNN: tích chập (conv), pooling, activation, kiến trúc trích xuất đặc trưng.
- Các thuật toán nhận dạng đối tượng phổ biến:
  - R-CNN, Fast/Faster R-CNN: hai-bước, độ chính xác cao, tốc độ chậm hơn.
  - SSD: một-bước, cân bằng tốc độ/độ chính xác.
  - YOLO: một-bước, tối ưu realtime; hệ Ultralytics tiện train/export/infer.
- YOLO-Pose: mở rộng head để dự đoán keypoints (COCO 17 điểm), phù hợp suy luận tư thế (góc, độ rơi).

### 3) Phân tích YOLO (đến YOLOv8) và lựa chọn
- So sánh v3/v5/v11 và bảng chi tiết đã thêm phía trên (xem các mục So sánh).
- Bổ sung nhận định v8: YOLOv8 (2023) là thế hệ Ultralytics trước YOLOv11, có nhánh v8-pose; v11 cải thiện tốc độ/độ chính xác hơn, ecosystem tương thích tương tự.
- Lựa chọn cho bài toán: YOLOv11-pose (n/s tùy phần cứng) do:
  - Có keypoints ổn định cho heuristics đầu-cổ; tốc độ tốt trên CPU (v11n-pose) và GPU (v11s/m-pose).
  - Hệ công cụ huấn luyện/triển khai mới nhất của Ultralytics.

### 4) Dữ liệu và gán nhãn
- Phạm vi: hình ảnh/video trong lớp học; đa góc độ, nhiều tư thế, ánh sáng khác nhau.
- Nhãn đề xuất:
  - Đối tượng: Person (pose keypoints 17 điểm)
  - Trạng thái: {binhthuong, ngugat, gucxuongban}
- Quy trình thu thập/đạo đức:
  - Xin phép, đảm bảo riêng tư; không chia sẻ dữ liệu nhạy cảm; làm mờ nếu cần.
  - Đa dạng hóa: chiều cao camera, khoảng cách, che khuất một phần, đeo khẩu trang.
- Công cụ: `tools/auto_label_pose.py` hỗ trợ gán nhãn bán tự động; kiểm định thủ công mẫu khó.
- Tổ chức thư mục: theo chuẩn Ultralytics (train/val split) — file `sleepy.yaml` đã sẵn trong repo.

### 5) Tiền xử lý & huấn luyện
- Tiền xử lý: cân bằng lớp, resize/augment (flip, brightness/contrast, blur nhẹ), kiểm tra/loại label lỗi.
- Huấn luyện YOLO-Pose (Ultralytics):
  - Tệp cấu hình: `yolo-sleepy-allinone-final/datasets/sleepy_pose/sleepy.yaml`
  - Tham số gợi ý: imgsz 640–960, batch theo RAM/GPU, epochs 50–150 tùy dữ liệu.
  - Theo dõi: mAP50/95, Precision, Recall; kiểm tra overfit.
- Kết quả hiện có: Chúng tôi đã benchmark mô hình pose và có số liệu mắt/ngáp; kết quả mAP pose sẽ cập nhật sau khi hoàn tất thu thập/label đủ dữ liệu lớp học.

### 6) Ứng dụng và cảnh báo
- Ứng dụng desktop (OpenCV) + GUI (PyQt5):
  - Kết nối camera/video; hiển thị kết quả, overlay VN; log sự kiện; FPS HUD.
  - Heuristics pose: góc mũi-so-vai, độ rơi theo ảnh/bề rộng vai, và tỉ lệ rơi theo chiều cao bbox (drop_bb_ratio) để phát hiện “cúi mặt nhưng ngồi thẳng”.
  - Tracking ID (IoU) per-person; hysteresis vào/ra trạng thái; nhãn màu nổi bật.
  - Tùy chọn pipeline mắt/ngáp (MediaPipe + YOLO phụ) → PERCLOS/yawn.

### 7) Đánh giá mô hình (kế hoạch)
- Độ đo chính: mAP50/95, Precision, Recall, FPS.
- Thiết lập đánh giá:
  - Val/Test set độc lập, đa cảnh/nhiễu; chạy inference → tính metrics.
  - So sánh v11n-pose vs v11s-pose (và/hoặc v5-pose nếu cần) theo mAP & FPS.
- Hiện trạng số liệu:
  - Pose: đang xây thêm dữ liệu để đánh giá; tạm thời ưu tiên v11n-pose do FPS cao (đã đo 6.02 FPS @640, CPU) và heuristics hỗ trợ tốt.
  - Eye/Yawn: đã có val precision/recall/mAP (mục “Phụ” phía trên), sẽ hợp nhất vào cảnh báo.

### 8) Tiến độ so với kế hoạch
- Nghiên cứu & đề cương: [x] Tổng quan YOLO/CNN; [x] so sánh v3/v5/v11; [ ] chốt đề cương với GVHD.
- Dữ liệu: [~] Thu thập; [~] gán nhãn (auto-label + kiểm định thủ công).
- Mô hình: [x] Benchmark; [ ] train pose full dữ liệu; [ ] đánh giá mAP/PR.
- Ứng dụng: [x] CLI + GUI; [x] tracking + hysteresis; [x] overlay VN; [x] video I/O; [~] tinh chỉnh UX.
- Báo cáo: [x] Cập nhật PROGRESS.md; [ ] Báo cáo PDF lần 1.

### 9) Việc tiếp theo
- Hoàn thiện đề cương với GVHD (mục tiêu, dữ liệu, phương pháp, đánh giá, rủi ro).
- Tăng dữ liệu ca khó (cúi mặt-chưa-gục, ngả nghiêng nhẹ, chói/thiếu sáng, che khuất).
- Huấn luyện pose với dữ liệu lớp học; báo cáo mAP50/95, PR/RC, FPS.
- Tối ưu triển khai: ONNX/TensorRT hoặc GPU nếu có; tinh chỉnh imgsz/ngưỡng.
- Chuẩn bị báo cáo PDF tiến độ lần 1 (tích hợp số liệu và hình ảnh minh họa).

---

## Bổ sung chi tiết các mục còn thiếu

### A) Đề cương (draft đề xuất để gửi GVHD)
- Vấn đề & động cơ: phát hiện buồn ngủ trong lớp để nâng cao hiệu suất học tập, cảnh báo sớm.
- Phương pháp: YOLO-Pose (v11n/s) + heuristics đầu-cổ + tracking + hysteresis; tùy chọn mắt/ngáp.
- Dữ liệu: thu thập trong lớp (đã xin phép), ba trạng thái (binhthuong/ngugat/gucxuongban), chuẩn COCO 17 điểm.
- Đánh giá: mAP50/95, Precision/Recall, FPS; thử nghiệm đa cảnh; báo cáo lỗi phổ biến.
- Sản phẩm: ứng dụng desktop (GUI), tài liệu hướng dẫn, mã nguồn, bộ dữ liệu xử lý ẩn danh.
- Kế hoạch & rủi ro: tiến độ theo tuần; rủi ro riêng tư/dữ liệu/thiết bị; phương án giảm thiểu (ẩn danh, mờ mặt, cài đặt quyền). 

### B) Kế hoạch dữ liệu & target số lượng
- Mục tiêu tối thiểu (đợt 1): ~1,500–2,000 ảnh gán keypoints, phân bố lớp tương đối cân bằng.
  - binhthuong: ~700
  - ngugat: ~600
  - gucxuongban: ~300–500
- Split: train 80% / val 10% / test 10% (theo cảnh để giảm rò rỉ). 
- Video → frame extraction: 2–3 fps cho cảnh ổn định để tránh trùng lặp quá nhiều.
- Quy định chú thích: mũi/2 vai/… theo COCO; trạng thái gán theo khung hình dựa vào tư thế.

### C) Tiền xử lý & Augmentation
- Resize theo imgsz (512–960), letterbox.
- Photometric: brightness/contrast/gamma; color jitter nhẹ.
- Geometric: flip ngang; rotate nhẹ (±10°) nếu phù hợp; blur nhẹ.
- Cutout/mosaic (thử sau) — theo dõi tác động đến keypoints.

### D) Cấu hình huấn luyện (gợi ý)
- Mô hình: `yolo11n-pose.pt` (CPU) hoặc `yolo11s-pose.pt` (GPU).
- imgsz: 640 (thử thêm 512/736/960 để cân bằng FPS/độ chính xác).
- Epochs: 80–150 (theo độ hội tụ).
- Early stopping & Cosine LR; batch theo RAM/GPU.
- Theo dõi: train/val loss, mAP50/95, PR/RC per class.

### E) Quy trình đánh giá
- Tập test độc lập theo cảnh; chạy infer → tính mAP50/95, Precision, Recall.
- Lưu confusion matrix, PR/RC curve; so sánh các biến thể (n-pose vs s-pose; imgsz khác nhau).
- Báo cáo FPS trung bình (CPU/GPU), độ trễ, mức sử dụng tài nguyên.

### F) Ứng dụng: kiểm checklist tính năng
- [x] Kết nối camera/video; [x] đọc ảnh tĩnh; [x] lưu video đã gán nhãn.
- [x] Overlay VN: nhãn trạng thái nổi bật, HUD FPS; [x] panel log; [x] panel mắt/ngáp (tùy chọn).
- [x] Tracking IoU + hysteresis; [x] person limit; [x] drop_bb_ratio cho ca cúi mặt.
- [x] GUI PyQt5: chọn nguồn, tham số; [ ] thêm control lưu cấu hình/nguồn mặc định; [ ] chụp ảnh nhanh (snapshot) từ GUI.

### G) Rủi ro & giảm thiểu
- Riêng tư & đạo đức: xin phép quay; ẩn danh; lưu tối thiểu; mã hóa/giới hạn truy cập.
- Môi trường ánh sáng/che khuất: tăng dữ liệu ca khó; tinh chỉnh ngưỡng; dùng eyes/yawn hỗ trợ.
- Hiệu năng CPU thấp: dùng `yolo11n-pose`, giảm imgsz, MJPG, hoặc GPU/ONNX/TensorRT.

### H) Lộ trình ngắn hạn (2–4 tuần)
- Tuần 1: hoàn thiện đề cương; hướng dẫn gán nhãn; bắt đầu thu thập dữ liệu đợt 1.
- Tuần 2: gán nhãn & QC; train baseline (v11n-pose @640); đánh giá bước đầu.
- Tuần 3: mở rộng dữ liệu ca khó; tinh chỉnh ngưỡng/augment; thử v11s-pose (GPU nếu có).
- Tuần 4: tổng hợp số liệu mAP/PR/RC/FPS; hoàn thiện báo cáo PDF + demo GUI.
