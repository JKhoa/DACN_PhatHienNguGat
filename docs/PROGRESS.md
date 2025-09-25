# Nhật ký phát triển hệ thốn### Giai đoạn 3 ### Giai đoạn 4 — Xây dựng hệ ### Giai đoạn 6 — Bổ sung ### Giai đoạn 8 — Cải thiện giao diện hiển thị
- Thu nhỏ và di chuyển bảng ghi chép sự kiện lên góc phải-trên với nền bán trong suốt, tự động thích ứng kích thước
- Sắp xếp bảng thông tin mắt/ngáp ngay bên dưới bảng ghi chép, tránh che khuất khung hình chính

### Giai đoạn 9 — Cải thiện chuyển đổi trạng thái "Thức dậy"
- Áp dụng kỹ thuật hysteresis với hai ngưỡng khác nhau:
  - Chuyển vào trạng thái "Ngủ gật/Gục xuống" khi có ≥ 15 khung hình liên tiếp phát hiện buồn ngủ
  - Trở về trạng thái "Bình thường" khi có ≥ 5 khung hình liên tiếp không phát hiện buồn ngủ
- **Kết quả đạt được**: Giao diện chuyển đổi trạng thái nhanh chóng và ổn định khi người dùng ngồi dậy lý ảnh tĩnh và tối ưu hóa mã nguồn
- Thêm tham số dòng lệnh `--image` để kiểm thử hệ thống trên một ảnh tĩnh
- Dọn dẹp file `standalone_app.py` khỏi các đoạn mã trùng lặp sau quá trình tái cấu trúc; sửa lỗi codec FOURCC MJPG
- Xác thực hoạt động của giao diện dòng lệnh `--help`, chế độ xử lý ảnh và chế độ webcam

### Giai đoạn 7 — Tích hợp công nghệ phát hiện buồn ngủ nâng cao (phân tích mắt và ngáp)
- Thêm tùy chọn kích hoạt pipeline xử lý bổ sung (thông qua `--enable-eyes`):
  - Sử dụng MediaPipe FaceMesh để cắt vùng quan tâm của mắt và miệng
  - Hai mô hình YOLO bổ sung: phân loại mắt (mở/nhắm) và phân loại ngáp (có ngáp/không ngáp)
  - Hệ thống đếm: số lần chớp mắt, thời gian nhắm mắt liên tục (microsleeps), số lần ngáp, thời lượng ngáp
  - Cảnh báo khi thời gian nhắm mắt ≥ 3 giây hoặc thời lượng ngáp ≥ 7 giây (có thể tùy chỉnh)
- Bổ sung các tham số dòng lệnh: `--eye-weights`, `--yawn-weights`, `--secondary-interval`, `--microsleep-thresh`, `--yawn-thresh`
- Bảng thông tin bổ sung được đặt ở bên phải màn hình, dưới phần ghi chép sự kiệni và ghi chép
- Tạo hệ thống trạng thái riêng cho từng người: `sleep_states` (trạng thái ngủ), `sleep_status` (tình trạng hiện tại), thời điểm bắt đầu ngủ
- Ghi lại các sự kiện "Ngủ gật" và "Thức dậy" vào bảng thông tin
- Tính toán và hiển thị thống kê "Thời gian ngủ gật lâu nhất"
- Tạo nhãn hiển thị bằng tiếng Việt rõ ràng, đặt gần vị trí mũi của từng người

### Giai đoạn 5 — Chuẩn bị dữ liệu và huấn luyện mô hình
- Tạo cấu trúc dữ liệu `datasets/sleepy_pose` (ảnh huấn luyện/kiểm tra và nhãn tương ứng)
- Cập nhật file cấu hình YAML:
  - `names = {0: binhthuong, 1: ngugat, 2: gucxuongban}` (tên các trạng thái)
  - `kpt_shape = [17,3]` (17 điểm đặc trưng theo chuẩn COCO), đường dẫn dữ liệu huấn luyện/kiểm tra
- Phát triển công cụ gán nhãn tự động `tools/auto_label_pose.py`:
  - Chạy suy luận mô hình và áp dụng thuật toán để tạo nhãn YOLO-Pose
  - Tự động chia dữ liệu thành tập huấn luyện và kiểm tra
- Tạo hướng dẫn huấn luyện sử dụng framework Ultralytics. Lưu ý: gặp lỗi lần đầu do thư mục ảnh trống → đã bổ sung hướng dẫn thu thập dữ liệu và gán nhãn tự độnguật toán phát hiện "Ngủ gật" và "Gục xuống bàn"
- Sử dụng các điểm đặc trưng của mũi và vai từ YOLO-Pose để tính toán:
  - Góc nghiêng đầu so với trục thẳng đứng (từ mũi đến cổ) → phát hiện việc cúi đầu hoặc ngả nghiêng
  - Mức độ đầu rơi xuống so với vai, tính theo tỷ lệ ảnh và khoảng cách giữa hai vai
- Thiết lập các ngưỡng nhạy cảm (có thể điều chỉnh):
  - Gục xuống bàn: tỷ lệ rơi theo chiều cao > 0.22 hoặc tỷ lệ rơi theo vai > 0.65
  - Ngủ gật nhẹ: góc nghiêng > 25° hoặc tỷ lệ rơi theo chiều cao > 0.12 hoặc tỷ lệ rơi theo vai > 0.35
- Áp dụng kỹ thuật khử nhiễu (15 khung hình liên tiếp) để tránh hiệu ứng nhấp nháy không mong muốniện ngủ gật

Tài liệu này ghi lại toàn bộ quá trình phát triển dự án phát hiện ngủ gật sử dụng công nghệ YOLO từ lúc bắt đầu đến hiện tại (cập nhật: tháng 9/2025).

## Mục tiêu của dự án
- Phát hiện tình trạng buồn ngủ và ngủ gật của con người trong thời gian thực qua camera
- Hỗ trợ nhận diện cả hai trường hợp: ngủ gật nhẹ và "gục xuống bàn" 
- Tạo giao diện hiển thị bằng tiếng Việt dễ đọc, có ghi chép sự kiện, hiển thị tốc độ xử lý và thống kê thời gian ngủý phát triển (Dev Log)

Tài liệu ghi chép tiến độ và các quyết định kỹ thuật của dự án YOLO-Sleepy từ lúc bắt đầu đến hiện tại (cập nhật: 2025-09-09).

## Mục tiêu ban đầu
- Phát hiện buồn ngủ/ngủ gật thời gian thực từ camera.
- Hỗ trợ cả trường hợp “gục xuống bàn”.
- Giao diện overlay tiếng Việt, dễ đọc; có log, FPS, và thống kê thời lượng.

## Các giai đoạn phát triển chính

### Giai đoạn 1 — Bắt đầu với ứng dụng web
- Xây dựng phiên bản thử nghiệm đầu tiên sử dụng mô hình YOLO để phát hiện tư thế trên nền tảng web (Streamlit)
- **Vấn đề gặp phải**: Video bị giật và chậm trễ nghiêm trọng → không đáp ứng được yêu cầu xử lý thời gian thực

### Giai đoạn 2 — Chuyển sang ứng dụng máy tính để bàn
- Phát triển ứng dụng sử dụng OpenCV thuần túy bằng Python để giảm độ trễ
- Giải quyết vấn đề hiển thị tiếng Việt có dấu bằng thư viện Pillow (PIL) → tạo hàm `draw_text_unicode`
- Bổ sung nhiều phương thức kết nối camera (CAP_DSHOW, CAP_MSMF) và định dạng MJPG để tăng tốc độ khung hình
- Thêm tính năng ước lượng tốc độ khung hình (EMA) để hiển thị ổn định

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

## Các tệp và thư mục chính
- `yolo-sleepy-allinone-final/standalone_app.py`: Ứng dụng chính cho máy tính để bàn (hỗ trợ webcam/ảnh tĩnh), hiển thị tiếng Việt, thuật toán phát hiện tư thế, pipeline phân tích mắt/ngáp (tùy chọn), ghi chép và thống kê
- `yolo-sleepy-allinone-final/datasets/sleepy_pose/sleepy.yaml`: File cấu hình bộ dữ liệu 3 trạng thái cho YOLO-Pose
- `yolo-sleepy-allinone-final/tools/auto_label_pose.py`: Công cụ gán nhãn bán tự động từ ảnh/video
- `real-time-drowsy-driving-detection/`: Thư mục tham khảo chứa các mô hình phân tích mắt/ngáp và logic phát hiện buồn ngủ bổ sung

## Hiệu suất hệ thống (tham khảo nhanh)
- YOLO11n-Pose với kích thước ảnh ~960: thời gian xử lý ~85–120 ms cho mỗi khung hình trên CPU → đạt ~8–11 FPS (theo quan sát từ log và giao diện)
- Có thể tăng tốc độ khung hình bằng cách: sử dụng định dạng MJPG cho camera, giảm kích thước ảnh đầu vào, sử dụng GPU/TensorRT/ONNX, hoặc chuyển sang mô hình nhỏ hơn

## So sánh các mô hình YOLO (Phát hiện tư thế) — Số liệu đo trên hệ thống hiện tại (CPU)
- **Điều kiện đo**: Windows, chỉ CPU, 1 khung hình từ webcam (480×640 pixel), kích thước ảnh đầu vào=640, Ultralytics YOLO phiên bản 8.3.x
- **Công cụ đo**: Script `yolo-sleepy-allinone-final/tools/benchmark_pose_models.py` (đã có trong kho mã nguồn)

**Kết quả đo đạc** (FPS cao hơn = hiệu suất tốt hơn):
- yolo11n-pose.pt: 6.02 FPS (15 lần chạy, tổng thời gian 2.49 giây)
- yolo11n.pt (mô hình phát hiện thông thường, không có tư thế): 4.26 FPS (15 lần chạy, tổng thời gian 3.52 giây)
- yolo11s-pose.pt: 2.93 FPS (15 lần chạy, tổng thời gian 5.11 giây)
- yolo11m-pose.pt: 1.28 FPS (15 lần chạy, tổng thời gian 11.71 giây)

**Nhận xét tổng quan:**
- yolo11n-pose có tốc độ nhanh nhất trên CPU → phù hợp cho ứng dụng thời gian thực. Các mô hình yolo11s/m-pose chạy chậm đáng kể trên CPU
- Mô hình phát hiện thông thường (yolo11n.pt) không tạo ra các điểm đặc trưng tư thế, nên không sử dụng được cho thuật toán phân tích tư thế của ứng dụng
- Tốc độ FPS trong ứng dụng thực tế sẽ thấp hơn một chút do có thêm giao diện hiển thị, theo dõi đối tượng, và pipeline phân tích mắt/ngáp

**Khuyến nghị lựa chọn mô hình:**
- **Chỉ có CPU**: sử dụng yolo11n-pose.pt để đạt FPS tốt; kết hợp với thuật toán phân tích (góc nghiêng/mức độ rơi đầu/tỷ lệ rơi so với khung) và kỹ thuật hysteresis như hiện tại
- **Có GPU (CUDA)**: có thể nâng cấp lên yolo11s-pose.pt để tăng độ chính xác phát hiện tư thế, chấp nhận giảm FPS; điều chỉnh kích thước ảnh để cân bằng
- **Trường hợp nhiều người cùng lúc**: ưu tiên mô hình nhanh (n-pose) kết hợp với theo dõi đối tượng (đã được bổ sung) để giữ ổn định ID và hiển thị rõ ràng

**Cách tái lập benchmark** (nếu cần):
```powershell
cd d:\Study\DoAnChuyenNganh\Yolo-v11-testing\yolo-sleepy-allinone-final\tools
python benchmark_pose_models.py --models yolo11n-pose.pt "yolo11n.pt" "yolo11s-pose.pt" "yolo11m-pose.pt" --iters 15 --imgsz 640
```

## Phụ lục: Kết quả huấn luyện các mô hình bổ sung (phân tích mắt/ngáp)
- **Mô hình phân loại mắt** (Mở/Nhắm) — epoch 10 (kiểm tra): độ chính xác ~0.73, độ nhạy ~0.86, mAP50 ~0.78, mAP50-95 ~0.73
- **Mô hình phân loại ngáp** (Có ngáp/Không ngáp) — epoch 10 (kiểm tra): độ chính xác ~0.77, độ nhạy ~0.73, mAP50 ~0.79, mAP50-95 ~0.59

**Gợi ý cải thiện**: tiếp tục thu thập và cân bằng dữ liệu, đặc biệt tập trung vào các trường hợp khó (nhìn xuống, ánh sáng yếu, bị che khuất) để nâng cao độ tin cậy của hệ thống

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

## So sánh YOLOv3 vs YOLOv5 vs YOLOv8 vs YOLOv11 cho nhận diện ngủ gật (dựa trên pose)

| Phiên bản | Năm | Pose gốc | Hệ sinh thái/Train | Triển khai | Phù hợp CPU | Liên quan bài toán (pose/keypoints) | Khi nên dùng |
|---|---|---|---|---|---|---|---|
| YOLOv3 | 2018 | Không (cần repo/phụ trợ) | Darknet cổ điển; training ít thuận tiện hơn PyTorch | Darknet/ONNX (chuyển đổi) | Trung bình/Chậm trên CPU hiện đại | Thiếu pose gốc → không trực tiếp tính góc/độ rơi đầu | Chỉ khi hệ thống legacy yêu cầu Darknet |
| YOLOv5 | 2020 | Có (v5-pose) | PyTorch/Ultralytics, dễ train/finetune, export tiện | ONNX/TensorRT/CoreML | Tốt trên CPU cũ | Có keypoints; đủ dùng nếu phần cứng yếu và cần ổn định | Biên/nhúng yếu, hoặc cần tương thích lâu năm |
| YOLOv8 | 2023 | Có (v8-pose) | PyTorch/Ultralytics thế hệ mới, cải tiến architecture | ONNX/TensorRT/OpenVINO… | Tốt trên CPU/GPU | Keypoints cải thiện, tốc độ ổn định | Cân bằng giữa hiệu suất và tương thích |
| YOLOv11 | 2024 | Có (v11-pose) | PyTorch/Ultralytics thế hệ mới nhất, tối ưu hơn | ONNX/TensorRT/OpenVINO… | Tốt nhất (cân bằng tốc độ/độ chính xác) | Keypoints tốt nhất, ổn định; phù hợp drowsiness realtime | Lựa chọn mặc định hiện tại |

Kết luận nhanh cho ứng dụng ngủ gật:
- **Ưu tiên cao nhất**: YOLOv11-pose (n/s tuỳ phần cứng) vì có keypoints ổn định nhất, tốc độ/độ chính xác tốt, hệ công cụ Ultralytics mới nhất.
- **Lựa chọn thứ hai**: YOLOv8-pose khi cần cân bằng hiệu suất và tương thích, hoặc khi YOLOv11 chưa ổn định trên hệ thống cụ thể.
- **Phần cứng yếu**: YOLOv5-pose khi cần export/triển khai trên phần cứng biên rất hạn chế hoặc phải giữ tương thích cũ.
- **Tránh**: YOLOv3 cho bài toán pose trừ khi ràng buộc legacy, vì thiếu pose gốc và hệ sinh thái train/triển khai kém linh hoạt hơn.

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

## Cập nhật 2025-09-24 — Chọn mô hình (YOLOv11/YOLOv8/YOLOv5/Custom) trong GUI

- Bổ sung selector mô hình trong `yolo-sleepy-allinone-final/gui_app.py` (tab Settings):
  - Preset: YOLOv11n-pose (mặc định), YOLOv11s-pose, YOLOv8n-pose, YOLOv5n-pose, và Custom…
  - Đường dẫn mặc định: tự dò `yolo11n-pose.pt`/`yolo11s-pose.pt`/`yolov5n-pose.pt` ở thư mục gốc dự án; v8/v5 dùng alias `yolov8n-pose.pt`/`yolov5n-pose.pt` (Ultralytics sẽ tự tải nếu thiếu).
  - Nút Browse… để chọn `.pt` bất kỳ (bao gồm các phiên bản YOLO khác).
- Tải nóng mô hình theo lựa chọn, không cần khởi động lại ứng dụng; hiển thị tên mô hình đang dùng ở status bar và ghi log khi đổi.
- Gợi ý so sánh nhanh:
  - Chất lượng: YOLOv11 Pose ≥ YOLOv8 Pose ≥ YOLOv5 Pose (tuỳ dữ liệu/weights).
  - Tốc độ: YOLOv5n-pose thường nhẹ nhất trên CPU cũ; YOLOv8n-pose cân bằng; YOLOv11n-pose tối ưu nhất.
  - Khuyến nghị: ưu tiên YOLOv11n/s-pose cho hiệu suất tốt nhất; dùng YOLOv5n-pose khi phần cứng rất yếu hoặc cần tương thích cũ.

## Cập nhật 2025-09-24 — Huấn luyện mô hình với dữ liệu mới (sleepy_pose_new_data)

### Thông tin huấn luyện:
- **Dữ liệu**: 25 ảnh từ thư mục `data_raw` được tự động gán nhãn bởi `auto_label_pose.py`
- **Tổng số nhãn**: 64 annotations được tạo ra từ 25 ảnh đầu vào
- **Phân chia dữ liệu**: 
  - Train: 22 ảnh (88%)
  - Validation: 3 ảnh (12%)
- **Mô hình base**: YOLOv11n-pose.pt (2.87M parameters, 196 layers)
- **Cấu hình training**:
  - Epochs: 50
  - Batch size: 4
  - Image size: 640x640
  - Optimizer: AdamW (lr=0.001429, momentum=0.9)
  - Device: CPU
  - Patience: 20 (early stopping)

### Kết quả huấn luyện:
- **Thời gian**: ~395 giây (6.6 phút) cho 50 epochs
- **Tốc độ**: ~7.9 giây/epoch trung bình
- **Chuyển giao học习 (Transfer Learning)**: 535/541 items từ pretrained weights

#### Metrics cuối cùng (Epoch 50):
- **Box Detection**:
  - Precision: 98.70%
  - Recall: 100%
  - mAP50: 99.50%
  - mAP50-95: 92.93%
- **Pose Estimation**:
  - Precision: 98.70%
  - Recall: 100%
  - mAP50: 44.25%
  - mAP50-95: 44.25%
- **Loss Values**:
  - Box Loss: 0.584
  - Pose Loss: 3.003
  - Keypoint Object Loss: 0.329
  - Classification Loss: 1.092
  - DFL Loss: 1.089

#### Xu hướng cải thiện:
- **Box mAP50**: Tăng từ 47.4% (epoch 1) → 99.5% (epoch 50)
- **Pose mAP50**: Dao động và ổn định ở ~44% từ epoch 15 trở đi
- **Training Loss**: Giảm dần và ổn định, không có dấu hiệu overfitting

### Tệp kết quả:
- **Weights**: `runs/pose-train/sleepy_pose_new_data/weights/best.pt`
- **Kết quả đầy đủ**: `runs/pose-train/sleepy_pose_new_data/results.csv`
- **Biểu đồ**: Training curves, confusion matrix, PR curves đã được tạo tự động

### Nhận xét:
- Mô hình đạt hiệu suất box detection rất cao (99.5% mAP50)
- Pose estimation đạt mức trung bình (44.25% mAP50), phù hợp cho ứng dụng thời gian thực
- Training ổn định, không có overfitting
- Tốc độ training nhanh nhờ pretrained weights và dataset nhỏ gọn
- Model weights đã sẵn sàng cho việc tích hợp vào ứng dụng

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

---

## 📊 Cập Nhật Mới Nhất (Phase 5 - Automated Collection Tools)

### ✅ Hoàn thành Phase 5: Công Cụ Thu Thập Tự Động
**Ngày**: 2025-01-08

#### Công cụ được tạo:
1. **`download_images.py`**: Script tải ảnh từ URLs miễn phí
   - Hỗ trợ Pexels, Unsplash, Pixabay URLs
   - Retry logic và error handling  
   - Hướng dẫn chi tiết lấy URL trực tiếp
   - Validation kích thước file

2. **`collect_data.py` nâng cấp**: Công cụ tổng hợp hoàn chỉnh
   - `--download`: Tích hợp download_images.py
   - `--full-pipeline`: Quy trình hoàn chỉnh tự động
   - Thống kê chi tiết theo nguồn ảnh
   - Hỗ trợ video frame extraction

3. **Tài liệu hướng dẫn**:
   - `README_DATA_COLLECTION.md`: Hướng dẫn sử dụng đầy đủ
   - `EXAMPLE_URLS.md`: Ví dụ cụ thể và templates
   - Integration với DATA_COLLECTION_GUIDE.md hiện có

#### Workflow được tối ưu:
```bash
# Quy trình thu thập hoàn chỉnh
python collect_data.py --full-pipeline

# Hoặc từng bước:
python download_images.py          # Tải ảnh từ URLs
python collect_data.py --copy      # Sao chép về data_raw  
python collect_data.py --auto-label # Tạo labels tự động
python collect_data.py --stats     # Xem thống kê
```

#### Tiến độ dataset:
- **Hiện tại**: 27 ảnh (19 gốc + 8 khác)
- **Mục tiêu**: 300-400 ảnh
- **Cần thêm**: ~273-373 ảnh

#### Next Steps:
1. Cấu hình URLs thực tế vào download_images.py
2. Thu thập batch đầu tiên 50-100 ảnh
3. Kiểm tra chất lượng auto-labeling
4. Retrain model với dataset mở rộng
