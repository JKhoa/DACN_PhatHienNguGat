# Tóm tắt quá trình phát triển (tháng 9/2025)

## 1) Tổng quan dự án

- *## 8) Các bước tiếp theo

- **Thêm cài đặt GUI cho**:
  - Ngưỡng vi giấc ngủ (microsleep threshold)
  - Ngưỡng ngáp (yawn threshold)  
  - Khoảng thời gian thứ cấp cho pipeline mắt/ngáp
- **Lưu trữ cài đặt** (JSON/YAML) và mở rộng kiểm tra trên các thiết lập camera/trọng số khác nhau

---

**Tóm tắt**: Ứng dụng GUI đã được ổn định hóa, và pipeline phân tích khuôn mặt/mắt/ngáp toàn diện của CLI (với bộ đếm và lớp phủ) đã được tích hợp vào `gui_app.py`. Các kiểm tra tĩnh đều pass và việc khởi chạy nền thành công; điều khiển trải nghiệm người dùng và lưu trữ là những cải tiến trọng tâm tiếp theo.hính của phiên làm việc:**
  - Tiếp tục cải thiện giao diện đồ họa của người dùng (GUI)
  - Chuyển các tính năng phân tích khuôn mặt/mắt/ngáp từ giao diện dòng lệnh vào giao diện chính
  - Viết lại tài liệu tiến độ theo cấu trúc phù hợp với Word
  - Cung cấp bản tóm tắt chi tiết, định dạng cải tiến tập trung vào các hoạt động gần đây
- **Quy trình làm việc:**
  - Sửa lỗi khởi tạo GUI và vấn đề thụt lề → tích hợp FaceMesh + logic mắt/ngáp → xác thực → cập nhật tài liệu → tóm tắtion Summary (2025-09-10)

## 1) Overview

- Primary objectives:
  - Continue iterating on the GUI.
  - Port the CLI’s richer face/eye/yawn pipeline into the main GUI.
  - Rewrite progress documentation in a Word-friendly structure.
  - Provide a detailed, enhanced-format summary focusing on recent operations.
- Session flow:
  - Fix GUI init/indentation issues → integrate FaceMesh + eye/yawn logic → validate → update docs → summarize.

## 2) Nền tảng kỹ thuật

- **Thị giác máy tính**: OpenCV (thu nhận và hiển thị video), Ultralytics YOLO (phát hiện tư thế + đối tượng), MediaPipe FaceMesh (phân tích khuôn mặt)
- **Giao diện người dùng**: PyQt5 (cửa sổ chính, thanh công cụ, thanh trạng thái, bố cục chia đôi, định dạng giao diện)
- **Theo dõi và trạng thái**: Bộ theo dõi đơn giản dựa trên IoU; hysteresis theo ID (SLEEP_FRAMES/AWAKE_FRAMES); nhật ký sự kiện
- **Hiển thị**: Vẽ Unicode thông qua Pillow (ImageFont/ImageDraw); bảng thông tin bên cạnh
- **Hiệu suất**: Camera MJPG, ước lượng FPS EMA; thực thi định kỳ pipeline mắt/ngáp để tiết kiệm tính toán

## 3) Trạng thái mã nguồn

- **File chính**: `yolo-sleepy-allinone-final/gui_app.py`
  - Sửa lỗi thụt lề và loại bỏ các chú thích kiểu dữ liệu inline có vấn đề trong `__init__`
  - Thêm các hàm hỗ trợ: `draw_panel` (vẽ bảng thông tin), `_safe_crop` (cắt ảnh an toàn), `_predict_eye` (dự đoán trạng thái mắt), `_predict_yawn` (dự đoán ngáp)
  - Thêm khởi tạo lazy cho FaceMesh và các mô hình YOLO phân tích mắt/ngáp; thực thi định kỳ sử dụng `secondary_interval`
  - Bộ đếm và trạng thái: `blinks` (số lần chớp mắt), `microsleeps` (vi giấc ngủ), `yawns` (số lần ngáp), `yawn_duration` (thời lượng ngáp); biến boolean cho trạng thái mắt trái/phải nhắm và đang ngáp
  - Mở rộng `process_frame_once` để thực hiện phân loại tư thế, theo dõi/hysteresis, phát hiện mắt/ngáp tùy chọn, và vẽ bảng thông tin bên cạnh với cảnh báo
  - Giữ lại hỗ trợ ghi video (`_ensure_writer`, `_write_frame`, `_release_writer`)
  - **Mới**: Bộ chọn mô hình trong tab Cài đặt để chuyển đổi giữa YOLOv11n/s-pose, YOLOv8n-pose, hoặc file `.pt` tùy chỉnh; tải lại động không cần khởi động lại
- **File tham khảo**: `yolo-sleepy-allinone-final/standalone_app.py`
  - Được sử dụng làm bản thiết kế cho logic phân tích khuôn mặt/mắt/ngáp toàn diện và các ngưỡng

## 4) Các vấn đề gặp phải và giải pháp

- **Vấn đề về thụt lề**: Các khối mã bị thụt lề sai và các dòng không thụt lề đúng trong `__init__` gây ra lỗi "self is not defined" và "Unexpected indentation"
  - **Giải pháp**: Thụt lề lại chính xác trong class; đảm bảo tất cả phép gán nằm trong `__init__`
- **Chú thích kiểu dữ liệu inline**: Gây ra lỗi phân tích cú pháp trong môi trường này
  - **Giải pháp**: Loại bỏ các chú thích inline trong phép gán; giữ lại các phép gán đơn giản
- **Tích hợp phức tạp**: Việc tích hợp thêm nhiều mã nguồn tăng nguy cơ lỗi thụt lề
  - **Giải pháp**: Lặp lại các bản vá với kiểm tra tĩnh cho đến khi sạch

## 5) Theo dõi tiến độ

- **Đã hoàn thành**:
  - Ổn định hóa khởi tạo GUI; không còn lỗi cú pháp được báo cáo bởi kiểm tra tĩnh
  - Tích hợp FaceMesh + phát hiện mắt/ngáp vào GUI với thực thi định kỳ và bộ đếm
  - Thêm bảng thông tin trên màn hình và lớp phủ cảnh báo
  - Viết lại tài liệu tiến độ theo định dạng thân thiện với Word
- **Đang chờ/tùy chọn**:
  - Điều khiển GUI cho ngưỡng vi giấc ngủ/ngáp và khoảng thời gian thứ cấp
  - Lưu trữ cấu hình (nguồn, độ phân giải, ngưỡng) và cải thiện chụp ảnh nhanh

## 6) Hành vi hiện tại

- GUI chạy với hysteresis dựa trên bộ theo dõi cho trạng thái buồn ngủ
- FaceMesh tùy chọn + phát hiện mắt/ngáp thực thi định kỳ để cân bằng hiệu suất
- Bảng thông tin bên cạnh hiển thị bộ đếm và trạng thái; cảnh báo xuất hiện cho các trạng thái nguy hiểm

## 7) Các hoạt động gần đây (lệnh/kết quả)

- Các bản vá lặp lại cho `gui_app.py` để sửa thụt lề, tích hợp các hàm hỗ trợ/bảng điều khiển, và thêm lựa chọn đa mô hình + tải lại nóng
- Kiểm tra lỗi tĩnh:
  - **Ban đầu**: lỗi chú thích kiểu dữ liệu, "self is not defined", và các vấn đề thụt lề
  - **Cuối cùng**: "No errors found" (Không tìm thấy lỗi) sau khi sửa
- Kiểm tra sơ bộ: Khởi chạy GUI ở chế độ nền (không có lỗi runtime quan trọng nào hiển thị)

## 8) Next steps

- Add GUI settings for:
  - Microsleep threshold
  - Yawn threshold
  - Secondary interval for eye/yawn pipeline
- Persist settings (JSON/YAML) and broaden testing across camera/weights setups.

---

Summary: The GUI app was stabilized, and the CLI’s comprehensive face/eye/yawn pipeline (with counters and overlays) was integrated into `gui_app.py`. Static checks pass and a background launch succeeded; UX controls and persistence are the next focused improvements.
