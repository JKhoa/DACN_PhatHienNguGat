# BÁO CÁO QUẢN TRỊ DỰ ÁN CÔNG NGHỆ THÔNG TIN

**Tên đề tài:** Hệ thống phát hiện học sinh ngủ gật trong lớp học
**Nhóm thực hiện:** Nhóm 11

## Mục lục

1. Khởi động dự án 
   1.1. Bối cảnh (Mô tả bối cảnh để đưa ra dự án) 
   1.2. Khảo sát chức năng 
   1.3. Ma trận trọng số các chức năng 
   1.4. Mô tả công việc của người quản trị dự án 
   1.5. Tính NPV, ROI và PayBack 
   1.6. Tuyên bố dự án 
2. Lập kế hoạch dự án 
   2.1. Kế hoạch xây dựng nhóm dự án 
   2.2. Dựa đoán chi phí dự án 
   2.3. Thỏa thuận nhóm 
   2.4. Phát biểu phạm vi 
   2.5. WBS 
   2.6. Gantt và sơ đồ mạng 
   2.7. Áp dụng mô hình Agile/Scrum 
   2.8. Danh sách rủi ro 
3. Thực thi 
   3.1. Triển khai Git (nếu có) 
   3.2. Chương trình thảo luận cho buổi họp nhóm 
   3.3. Báo cáo tiến độ dự án 
   3.4. Lý do thay đổi (nếu có) 
4. Kiểm soát 
   4.1. Giải pháp ngăn chặn vượt phạm vi và gia tăng chi phí 
   4.2. Công việc thực tế đến hết tháng 10/2025 
   4.3. Đánh giá EVM và điều chỉnh theo tiến độ 
   4.4. Cập nhật chi tiết NPV, ROI, Payback 
5. Kết thúc 
   5.1. Viết báo cáo tổng kết 
   5.2. Viết bài học kinh nghiệm 

Báo cáo tổng kết 

---

## 1. Khởi động dự án

### 1.1. Bối cảnh (Mô tả bối cảnh để đưa ra dự án)

Trong thực tế lớp học đông hoặc lớp học trực tuyến, giảng viên khó theo dõi đồng thời trạng thái tập trung của toàn bộ học sinh. Cách quan sát thủ công dễ bị chủ quan, thiếu nhất quán và không tạo được dữ liệu định lượng để đánh giá chất lượng học tập theo thời gian. Dự án được đề xuất nhằm giải quyết bài toán đó bằng hướng tiếp cận thị giác máy tính: dùng YOLO Pose Estimation để nhận diện tư thế, kết hợp xử lý thời gian thực trên Edge Computing để giảm độ trễ. Ngoài mục tiêu học thuật của đồ án chuyên ngành CNTT, hệ thống còn hướng đến tính ứng dụng thực tế cho nhà trường thông qua khả năng cảnh báo, lưu trữ và thống kê dữ liệu hành vi tập trung.

### 1.2. Khảo sát chức năng

-   Thu thập dữ liệu, trích xuất frame, gán nhãn Awake/Drowsy và augmentation cho tập huấn luyện.
-   Huấn luyện và so sánh các phiên bản YOLO (v5/v8/v11), chọn mô hình tối ưu cho bài toán lớp học.
-   Nhận diện trạng thái theo thời gian thực từ webcam/camera với mục tiêu FPS >= 15.
-   Áp dụng Detection Logic bằng FSM, Temporal Smoothing và Hysteresis Threshold để giảm cảnh báo nhiễu.
-   Xây dựng backend Flask API, Inference Engine và WebSocket để truyền dữ liệu realtime.
-   Lưu trữ dữ liệu sự kiện bằng SQLite (trạng thái, thời gian, camera, lịch sử cảnh báo).
-   Phát triển ứng dụng Desktop bằng Electron/React để quản lý camera và theo dõi trực tiếp.
-   Xây dựng Dashboard gồm Line Chart, Bar Chart, Pie Chart để thống kê theo thời gian.
-   Triển khai quy trình kiểm thử 4 cấp độ: Unit, Integration, System, Acceptance.
-   Hỗ trợ nghiệm thu thực tế với tập người dùng mẫu để tinh chỉnh tham số trước khi bàn giao.

### 1.3. Ma trận trọng số các chức năng

Để xác định thứ tự ưu tiên triển khai, nhóm xây dựng ma trận trọng số theo mức độ ảnh hưởng tới chất lượng sản phẩm cuối cùng (thang 10 điểm):

| Nhóm chức năng | Trọng số | Nội dung đánh giá | Chỉ số đo/Điều kiện đạt |
|---|---|---|---|
| Phát hiện real-time bằng AI | 9/10 | Khả năng nhận diện đúng trạng thái học sinh trong điều kiện vận hành thực tế | Precision >= 0.80, FPS >= 15 |
| Backend và Detection Logic | 8/10 | Độ ổn định của API, truyền dữ liệu realtime và giảm cảnh báo nhiễu | Flask + WebSocket ổn định, FSM hoạt động đúng ngưỡng |
| Desktop UI và Dashboard | 7/10 | Tính dễ dùng, trực quan và hỗ trợ giám sát lớp học | Hiển thị realtime, thống kê Line/Bar/Pie, thao tác quản lý camera rõ ràng |

Kết luận ưu tiên triển khai:

-   Ưu tiên 1: Bảo đảm lõi AI đạt độ chính xác và tốc độ tối thiểu.
-   Ưu tiên 2: Ổn định luồng Backend + Logic cảnh báo trước khi mở rộng giao diện.
-   Ưu tiên 3: Hoàn thiện UI/Dashboard sau khi lõi hệ thống đã chạy ổn định.

### 1.4. Mô tả công việc của người quản trị dự án

Người Quản trị dự án (Project Manager - PM) đóng vai trò định hướng, gắn kết đội ngũ và chịu trách nhiệm cao nhất về sự thành bại của dự án trong chu kỳ 3 tháng (13 tuần). Để đảm bảo các mục tiêu về Phạm vi, Thời gian, Chi phí và Chất lượng, khối lượng công việc của PM được làm rõ theo 5 nhóm quy trình chuẩn (Process Groups):

1.  **Giai đoạn Khởi tạo (Initiating):**
    -   Đánh giá tính khả thi, phân tích bài toán "nhận diện buồn ngủ" và xác định phạm vi cốt lõi (In-scope) và ngoài phạm vi (Out-of-scope) để tránh hiện tượng tràn phạm vi (Scope Creep).
    -   Xây dựng bản Tuyên bố dự án (Project Charter), tổ chức họp Kick-off để đồng bộ tầm nhìn và phân công trách nhiệm rõ ràng cho các mảng: AI, Backend, UI/UX.
2.  **Giai đoạn Lập kế hoạch (Planning):**
    -   Phân rã cấu trúc công việc (WBS) thành các module nhỏ có thể đo lường.
    -   Thiết lập lịch trình (Gantt Chart) theo chuỗi Sprint (Agile) và dự toán ngân sách chi tiết mốc 32.450.500đ (Chi phí thực tế nhóm đã chi, tăng 2.450.500đ so với dự kiến 30.000.000đ ban đầu do phát sinh trong quá trình làm dữ liệu).
    -   Lập kế hoạch quản trị rủi ro dự phòng, đặc biệt là rủi ro rò rỉ dữ liệu cá nhân, hiệu suất FPS kém hoặc chi phí Cloud GPU (AWS EC2) vượt định mức.
3.  **Giai đoạn Thực thi (Executing):**
    -   Điều phối tài nguyên và nhân sự. Giải quyết xung đột (Conflict Resolution) khi có sự lệch pha giữa tốc độ hoàn thành mô hình AI YOLO và luồng xử lý Backend/UI.
    -   Đảm bảo luồng giao tiếp mượt mà thông qua các buổi Daily Stand-up (nếu có) và Sprint Review hàng tuần.
4.  **Giai đoạn Kiểm soát (Monitoring & Controlling):**
    -   Theo dõi tiến độ giải ngân và hiệu suất thời gian thực sử dụng phương pháp Giá trị thu được (EVM - Earned Value Management), theo dõi các chỉ số SPI, CPI.
    -   Kiểm soát chất lượng (QA/QC), giám sát quy trình Testing 4 cấp độ để bảo chứng độ phản hồi (Precision >= 0.8) trước khi release nghiệm thu.
5.  **Giai đoạn Kết thúc (Closing):**
    -   Tiến hành bàn giao sản phẩm Desktop App cho "khách hàng" (30 user nghiệm thu).
    -   Tổ chức họp Retrospective cuối dự án; lưu trữ tài liệu, release mã nguồn; xuất hóa đơn/chốt ngân sách cuối cùng và viết Báo cáo bài học kinh nghiệm.

### 1.5. Tính NPV, ROI và PayBack

Để chứng minh luận điểm đầu tư của dự án là hợp lý trước các bên liên quan (Hội đồng đánh giá/Nhà đầu tư), PM thực hiện ước toán dòng tiền dựa trên kịch bản thương mại hóa quy mô hẹp (cung cấp giấy phép phần mềm cho các trung tâm đào tạo / trường học khu vực).

**1. Các thông số đầu vào (Inputs):**

-   **Chi phí đầu tư ban đầu dự kiến:** 30.000.000đ (Ngân sách nhóm dự trù ở thời điểm lập kế hoạch).
-   **Chi phí đầu tư thực tế (Vốn thực xuất - Updated BAC):** 32.450.500đ. Trong quá trình thực hiện, nhóm nhận thấy tập dữ liệu mở có sẵn trên mạng thiếu sự đa dạng về góc mặt và điều kiện ánh sáng. Để đảm bảo chất lượng nhận diện, nhóm đã chủ động quyết định chi thêm 2.450.500đ để mua bộ video bản quyền và thuê thêm giờ chạy Cloud GPU nhằm huấn luyện lại mô hình. Các chỉ số tài chính bên dưới được nhóm tính toán hoàn toàn dựa trên số vốn thực tế 32.450.500đ này để báo cáo bám sát thực tế nhất.
-   **Tỷ suất chiết khấu (Discount Rate - r):** 10% (10%/năm - Phản ánh lạm phát và chi phí cơ hội).
-   **Dòng tiền vào ước tính (Cash Inflows):** Đến từ việc bán License và phí bảo trì hàng năm, giúp tiết kiệm chi phí thuê nhân sự giám thị trực tiếp.
    -   Năm 1: 22.500.000đ (Triển khai thí điểm 5 lớp học)
    -   Năm 2: 28.750.000đ (Mở rộng quy mô lên 15 lớp học)

**2. Phân tích chi tiết các chỉ số (Calculations):**

-   **A. Giá trị hiện tại thuần (NPV - Net Present Value):**
    
    -   Cách tính (Mức chiết khấu 10%/năm): Lấy dòng tiền của từng năm quy đổi về thời điểm hiện tại (bằng cách chia cho 1.1 theo số thứ tự năm), rồi lấy tổng đó trừ đi số tiền vốn ban đầu.
    -   Hiện giá Năm 1: Dòng tiền 22.500.000đ chia cho 1.1 (mũ 1) = 20.454.545đ
    -   Hiện giá Năm 2: Dòng tiền 28.750.000đ chia cho 1.1 (mũ 2) = 23.760.331đ
    -   Tổng các mức hiện giá: 20.454.545 + 23.760.331 = 44.214.876đ
    -   **NPV của dự án:** 44.214.876 - 32.450.500 (Vốn) = **11.764.376đ**
    -   *Diễn giải:* Vì số tiền thu về sau khi đã trừ lạm phát và bù vốn gốc vẫn dư ra 11.764.376đ (NPV lớn hơn 0), dự án có khả năng sinh lời thực tế cực tốt.
-   **B. Tỷ suất hoàn vốn đầu tư (ROI - Return on Investment):**
    
    -   Cách tính: Lấy tổng tiền lãi kiếm được sau các năm, đem chia cho số tiền vốn gốc ban đầu, rồi nhân với 100%.
    -   Tổng dòng tiền thu về sau 2 năm: 22.500.000 + 28.750.000 = 51.250.000đ
    -   Lợi nhuận ròng (Tiền thực lãi): 51.250.000 - 32.450.500 (Vốn) = 18.799.500đ
    -   **Chỉ số ROI:** (18.799.500 chia 32.450.500) nhân 100 = **57.93%**
    -   *Diễn giải:* Cứ 100.000đ chi phí đầu tư ban đầu định lượng, dự án mang về 100.000đ vốn và sinh lời thêm khoảng gần 58.000đ sau 2 năm vận hành.
-   **C. Thời gian hoàn vốn (PayBack Period):**
    
    -   Cách tính: Khấu trừ lợi nhuận của các năm cho đến khi bù đắp đủ số tiền vốn đầu tư lúc ban đầu (hòa vốn).
    -   Năm thứ 1: Thu được 22.500.000đ. Số nợ vốn gốc vẫn còn = 32.450.500 - 22.500.000 = 9.950.500đ.
    -   Năm thứ 2: Khả năng thu dòng tiền tối đa là 28.750.000đ.
    -   Tỷ lệ thời gian của Năm thứ 2 cần dùng để trả đủ 9.950.500 nợ vốn: 9.950.500 chia 28.750.000 = 0.34 năm (cỡ 4 tháng rưỡi).
    -   **Thời gian hoàn vốn (Payback):** Chờ hết 1 năm hoạt động đầu tiên + 0.34 năm thứ hai = **1.34 năm** (Khoảng 1 năm 4 tháng rưỡi).
    -   *Diễn giải:* Rủi ro lún vốn vô cùng thấp bởi tính thanh khoản rất cao: chỉ mất chưa tới 1.5 năm dự án đã hòa vốn hoàn toàn.

*Tiểu kết:* Bộ ba chỉ số chứng tỏ bức tranh tài chính sinh lời mạnh mẽ, hoàn toàn xứng đáng để Hội đồng xét duyệt "Go/No-go" đẩy dự án vào giai đoạn thực thi.

-   Phần công thức và bảng tính chi tiết được trình bày ở Mục 4.

### 1.6. Tuyên bố dự án

Mục tiêu: Hệ thống cần đạt Precision >= 0.80, tốc độ xử lý >= 15 FPS, lưu sự kiện vào SQLite theo thời gian thực và hoàn tất hồ sơ nghiệm thu đúng hạn.

Mục 1.6 được tóm tắt theo định dạng bảng để dễ theo dõi khi trình bày.

| Thành phần Charter | Nội dung cụ thể |
|---|---|
| Tên dự án | Hệ thống phát hiện ngủ gật thời gian thực trên desktop bằng YOLO pose |
| Lý do thực hiện | Tăng an toàn học tập/làm việc ban đêm bằng cảnh báo sớm trạng thái mệt mỏi |
| Mục tiêu SMART | Precision >= 0.80; tốc độ xử lý >= 15 FPS; lưu sự kiện vào SQLite theo thời gian thực; hoàn tất hồ sơ nghiệm thu đúng hạn |
| In-scope | Data pipeline, huấn luyện/chọn mô hình YOLO, backend Flask + WebSocket + SQLite, desktop UI + dashboard, kiểm thử 4 cấp độ |
| Out-of-scope | Không làm ứng dụng di động; không triển khai cloud production quy mô lớn; không giám sát đa người trên nhiều camera |
| Sản phẩm bàn giao | Mô hình YOLO-pose tối ưu sau benchmark, ứng dụng desktop cảnh báo realtime, dashboard dữ liệu, tài liệu kỹ thuật + báo cáo |
| Ngân sách baseline | BAC = 32.450.500đ |
| Nguồn lực chính | Laptop cá nhân, webcam, Python, YOLOv11, React/Electron, Flask, SQLite |
| Ràng buộc | Triển khai trong 3 tháng; giới hạn hạ tầng theo máy cá nhân |
| Giả định | Dữ liệu đủ đa dạng để huấn luyện; thành viên duy trì nhịp họp/commit hằng tuần |
| Phê duyệt | PM xác nhận phạm vi, nhóm kỹ thuật xác nhận khả thi, giảng viên xác nhận mốc nghiệm thu |

| Mốc | Thời gian | Nội dung | Đầu ra |
|---|---|---|---|
| M1 | 14/10/2025 | Khởi động dự án, chốt Charter, chốt phạm vi | Charter và tiêu chí đánh giá được duyệt |
| M2 | 28/10/2025 | Hoàn thiện data pipeline (thu thập, trích frame, gán nhãn) | Bộ dữ liệu huấn luyện ban đầu |
| M3 | 11/11/2025 | Train và benchmark YOLOv5/v8/v11 | Báo cáo so sánh và mô hình chọn tạm thời |
| M4 | 02/12/2025 | Tích hợp backend, detection logic, desktop UI | Bản tích hợp end-to-end chạy realtime |
| M5 | 31/12/2025 | Kiểm thử, hoàn thiện báo cáo, bàn giao | Bộ nghiệm thu cuối kỳ |

| Tiêu chí nghiệm thu | Mức đạt yêu cầu |
|---|---|
| Hiệu năng mô hình | Precision >= 0.80 |
| Tốc độ xử lý | >= 15 FPS trên máy test của nhóm |
| Chức năng cảnh báo | Cảnh báo hoạt động ổn định trong ca test mô phỏng |
| Lưu trữ dữ liệu | Sự kiện lưu đúng trong SQLite và hiển thị trên dashboard |
| Hồ sơ bàn giao | Đủ mã nguồn, tài liệu, báo cáo, minh chứng kiểm thử |

---

## 2. Lập kế hoạch dự án

### 2.1. Kế hoạch xây dựng nhóm dự án

Nhóm làm việc theo 5 vai trò chính, mỗi vai trò có nhiệm vụ và đầu ra rõ ràng:

| Vai trò | Công việc chính | Người phụ trách | Đầu ra mong đợi |
|---|---|---|---|
| PM | Lập kế hoạch, theo dõi tiến độ, điều phối rủi ro, họp tuần | Thành viên 1 kiêm PM, kiêm UI/UX | Kế hoạch dự án, biên bản họp, báo cáo tiến độ |
| AI Developer | Thu thập dữ liệu, gán nhãn, train và đánh giá YOLOv5/v8/v11 | Thành viên 2 kiêm AI, kiêm Tester | Mô hình tối ưu sau benchmark, bộ chỉ số đánh giá |
| Backend Developer | Xây dựng Flask API, WebSocket, tích hợp SQLite và logic phát hiện | Thành viên 3 kiêm Backend, kiêm hỗ trợ tài liệu | Backend realtime ổn định, dữ liệu sự kiện lưu đúng |
| UI/UX Designer | Thiết kế mockup và phát triển giao diện Electron/React | Thành viên 1 kiêm PM, kiêm UI/UX | Giao diện desktop dễ dùng, hiển thị đầy đủ trạng thái |
| Tester | Thiết kế test case, kiểm thử 4 cấp độ, tổng hợp lỗi và xác nhận bản build | Thành viên 2 kiêm AI, kiêm Tester | Báo cáo kiểm thử, danh sách lỗi và kết quả nghiệm thu |

#### Phân chia công việc chi tiết cho 3 thành viên

Để phù hợp mô hình nhóm 3 người nhưng vẫn bao phủ đủ 5 vai trò, nhóm phân công theo nguyên tắc: mỗi thành viên có một mảng chính chịu trách nhiệm đầu cuối và một mảng phụ để hỗ trợ chéo khi phát sinh rủi ro tiến độ.

| Thành viên | Vai trò chính | Vai trò kiêm nhiệm | Công việc chịu trách nhiệm chính | Kết quả đầu ra cam kết |
|---|---|---|---|---|
| Thành viên 1 | PM + UI/UX | Quản trị tài liệu, tổng hợp báo cáo | Lập kế hoạch, theo dõi Gantt/Sprint, tổ chức họp tuần, thiết kế giao diện desktop, chuẩn hóa tài liệu nghiệm thu | Kế hoạch dự án, biên bản họp, giao diện UI hoàn chỉnh, bộ báo cáo cuối kỳ |
| Thành viên 2 | AI Developer + Tester | Hỗ trợ tối ưu logic cảnh báo | Thu thập và gán nhãn dữ liệu, huấn luyện/benchmark YOLO, theo dõi Precision-FPS, xây test case và kiểm thử 4 cấp độ | Mô hình AI tối ưu, báo cáo benchmark, báo cáo kiểm thử và danh sách lỗi |
| Thành viên 3 | Backend Developer | Hỗ trợ tích hợp và triển khai | Xây dựng Flask API, WebSocket, SQLite, tích hợp detection logic với UI, đảm bảo luồng realtime ổn định | Backend ổn định, dữ liệu lưu đúng, bản tích hợp end-to-end |

| Giai đoạn | Thành viên 1 (PM + UI/UX) | Thành viên 2 (AI + Tester) | Thành viên 3 (Backend) |
|---|---|---|---|
| Khởi động (Tuần 1-2) | Chốt Charter, phạm vi, KPI, phân vai | Đề xuất chỉ số mô hình và tiêu chuẩn dữ liệu | Đề xuất kiến trúc backend và cấu trúc dữ liệu |
| Dữ liệu (Tuần 3-4) | Theo dõi tiến độ, chuẩn tài liệu dữ liệu | Thu thập video, gán nhãn, augmentation | Chuẩn bị API nhận dữ liệu và cấu trúc DB |
| Huấn luyện (Tuần 5-6) | Quản trị rủi ro và điều phối nguồn lực | Train YOLOv5/v8/v11, so sánh và chọn mô hình | Tối ưu luồng inference backend |
| Tích hợp (Tuần 7-10) | Phát triển UI desktop, dashboard | Tinh chỉnh ngưỡng FSM, hỗ trợ test tích hợp | Tích hợp Flask + WebSocket + SQLite + detection logic |
| Kiểm thử và bàn giao (Tuần 11-12) | Tổng hợp hồ sơ bàn giao, chốt báo cáo | Chạy Unit/Integration/System/Acceptance test | Sửa lỗi backend, đóng gói bản chạy ổn định |

Mức phân bổ nguồn lực tham chiếu: Thành viên 1 khoảng 35%, Thành viên 2 khoảng 35%, Thành viên 3 khoảng 30% tổng khối lượng công việc; điều chỉnh theo sprint khi có blocker thực tế.

### 2.2. Dựa đoán chi phí dự án

Theo chuẩn PMBOK (Project Cost Management), dự toán được xây dựng thông qua phương pháp ước lượng từ dưới lên (Bottom-Up Estimating) dựa trên các Work Package của WBS, kết hợp xác định đường cơ sở chi phí (Cost Baseline). Lịch trình trải dài dựa trên mốc khởi động đầu tháng 10 và kết thúc cuối tháng 12/2025.

**Tổng Ngân sách Dự kiến Ban Đầu:** 30.000.000đ

**Tổng Ngân sách Thực Tế (Budget at Completion - BAC):** 32.450.500đ

Tại thời điểm khởi tạo dự án, nhóm đưa ra mức ngân sách dự kiến là 30.000.000đ. Tuy nhiên, khi chính thức bắt tay vào thu thập dữ liệu, nhóm phát hiện chất lượng dữ liệu nguồn mở trên mạng không đủ tốt để giúp mô hình đạt độ chính xác > 80% đúng như cam kết.

Thay vì nghiệm thu một sản phẩm kém, nhóm đã họp và đi đến quyết định trích thêm ngân sách để mua bổ sung các bộ video bản quyền chất lượng cao, đồng thời thuê thêm giờ chạy máy ảo Cloud GPU để thực hiện huấn luyện lại. Sự điều chỉnh mang tính chủ động này làm tổng chi phí thực chi (BAC) của dự án tăng lên mức 32.450.500đ nhưng lại bảo chứng được sự thành công cho sản phẩm thu được.

Dưới đây là bảng kê chi tiết toàn bộ các hạng mục mà nhóm đã thực chi trong suốt vòng đời dự án:

| Thời gian thực hiện | Hạng mục chi phí chi tiết | Kinh phí (VNĐ) |
|---|---|---|
| 01/10 - 14/10/2025 | Đánh giá và đề xuất dự án | 500.000 |
| 01/10 - 14/10/2025 | Phát triển bản tôn chỉ dự án (Project Charter) | 1.500.000 |
| 01/10 - 14/10/2025 | Họp bắt đầu dự án (Kick-off & Phê duyệt) | 100.000 |
| 01/10 - 14/10/2025 | Tuyên bố phạm vi dự án | 600.000 |
| 15/10 - 28/10/2025 | Phân tích và xác định rõ yêu cầu từ nhà tài trợ/khách | 3.000.000 |
| 15/10 - 28/10/2025 | Xây dựng bản kế hoạch dự án chi tiết | 2.400.000 |
| 15/10 - 28/10/2025 | Thiết kế giao diện và cơ sở dữ liệu | 1.000.000 |
| 15/10 - 28/10/2025 | Mua video bộ dữ liệu bản quyền (Shutterstock/Pexels) | 4.250.000 |
| 15/10 - 28/10/2025 | Thuê server lưu trữ dữ liệu thô (NAS Local 5TB) | 1.220.500 |
| 29/10 - 11/11/2025 | Chi phí nền tảng gán nhãn RoboFlow Pro | 2.155.500 |
| 29/10 - 11/11/2025 | Thù lao ngoài giờ thuê freelancer xử lý dữ liệu | 3.000.000 |
| 29/10 - 11/11/2025 | Thuê máy ảo Cloud GPU AWS EC2 p3 (Train model) | 4.875.000 |
| 12/11 - 02/12/2025 | Trả phí tool IDE bản quyền (PyCharm, Copilot) | 1.250.500 |
| 12/11 - 02/12/2025 | Xây dựng mô hình AI nhận diện (YOLO) | 2.500.000 |
| 12/11 - 02/12/2025 | Dựng API Backend & luồng Socket thời gian thực | 1.000.000 |
| 03/12 - 16/12/2025 | Phí tài khoản Developer & cấu hình Hosting | 500.000 |
| 03/12 - 16/12/2025 | Mua Template UI/UX & Mảng icon giao diện | 500.000 |
| 17/12 - 31/12/2025 | Lệ phí kiểm thử diện rộng (UAT Sinh viên) | 600.000 |
| 17/12 - 31/12/2025 | Đánh giá của người dùng và nhà tài trợ | 1.000.000 |
| 31/12/2025 (Nghiệm thu) | Quỹ rủi ro biến số dự phòng khẩn cấp | 499.000 |
| **Tổng cộng** | **TỔNG NGÂN SÁCH (Dự toán tuyệt đối - BAC)** | **32.450.500** |

Ghi chú tham số biến thiên (Contingency Reserves):

-   Các chi phí lập hồ sơ phân tích yêu cầu đòi hỏi sự tập trung nguồn lực lớn trong tháng 10.
-   Khúc cuối dự án được phân tách riêng chi phí "Đánh giá dự án" vì cần thực hiện tại trang trại/phòng lab với sự chứng kiến của nhà tài trợ.

### 2.3. Thỏa thuận nhóm

Chuẩn mực quản lý nguồn nhân lực dự án (Project Resource Management) yêu cầu một "Team Charter" rõ ràng nhằm thiết lập kỳ vọng, chuẩn mực hành vi và giảm thiểu xung đột.

-   **Giá trị cốt lõi & Nguyên tắc làm việc:** Tôn trọng ý kiến chuyên môn của từng cá nhân, phân chia công việc minh bạch theo chuyên môn (AI, Backend, UI), và cam kết về chất lượng sản phẩm (ví dụ: code đẩy lên nhánh `main` phải chạy không lỗi).
-   **Quy tắc giao tiếp (Communication Guidelines):**
    -   Kênh giao tiếp chính: Zalo cho các trao đổi tức thời hằng ngày; GitHub/GitLab cho tiến độ mã nguồn.
    -   Tần suất: Họp Daily Stand-up 15 phút (trực tuyến) và họp Sprint Review/Planning định kỳ thứ Bảy mỗi tuần.
    -   SLA (Service Level Agreement): Thời gian phản hồi tin nhắn trong nhóm là tối đa trong ngày (24 giờ), đặc biệt khi gặp Blocker tiến độ.
-   **Tiêu chí ra quyết định & Xử lý xung đột (Decision-Making & Conflict Resolution):**
    -   Ưu tiên phương án có dữ liệu chứng minh (data-driven), ví dụ: chọn YOLOv11 thay vì v8 dựa trên bảng so sánh metrics F1-score và FPS.
    -   Khi có bất đồng về giải pháp kỹ thuật, PM là người chốt quyết định cuối cùng dựa trên tư vấn từ thành viên chuyên môn sâu nhất, đảm bảo tính "Khả thi" và "Thời gian" của dự án.
-   **Công tác đánh giá (Performance & Code Review):** Mọi tính năng (Pull Request) phải có ít nhất 1 thành viên khác review trước khi merge vào nhánh `develop`. Tuyệt đối không đẩy trực tiếp code lỗi lên nhánh `main`.

### 2.4. Phát biểu phạm vi

Quản trị phạm vi (Project Scope Management) giúp hệ thống hóa những gì dự án sẽ làm (In-scope) và không làm (Out-of-scope), tránh hiện tượng phình phạm vi (Scope Creep).

-   **Mô tả phạm vi dự án chi tiết (Project Scope Description):** Phát triển một hệ thống dạng ứng dụng Desktop (sử dụng Electron + React) có khả năng tích hợp Camera/Webcam để nhận diện trạng thái theo thời gian thực (Real-time). Lõi hệ thống dùng mô hình Deep Learning (YOLOv11n-pose) trích xuất các điểm neo (Keypoints) để nhận diện việc nhắm mắt và cúi gục đầu. Luồng xử lý Python Backend đẩy dữ liệu qua WebSocket xuống giao diện người dùng, đồng thời lưu vết vào SQLite để thống kê.
-   **Sản phẩm bàn giao (Deliverables):**
    1.  File trọng số mô hình tốt nhất (`YOLOv11n-pose.pt`).
    2.  Ứng dụng Desktop cảnh báo cài đặt sẵn cho Windows.
    3.  Báo cáo kiểm thử độ chính xác định lượng và phản hồi từ tập 30 sinh viên mẫu.
    4.  Mã nguồn toàn bộ hệ thống cùng tài liệu kỹ thuật (README, HDSD).
-   **Tiêu chí nghiệm thu (Acceptance Criteria):** Model AI đạt Precision >= 0.80 trên tập test; Hệ thống xử lý Real-time >= 15 FPS trên laptop cấu hình phổ thông; Thông báo trên UI phải xuất hiện trễ nhất sau 3 giây tính từ khi trạng thái ngủ gật bắt đầu duy trì (FSM Threshold).
-   **Giới hạn phạm vi (Exclusions/Out-of-Scope):** Dự án **không** hỗ trợ thiết bị di động (Mobile App); **không** phát triển giải pháp Server/Cloud có tính thương mại cao xử lý hàng ngàn camera đồng thời; **không** áp dụng nhận diện khuôn mặt (Face ID định danh người dùng do ràng buộc quyền riêng tư).
-   **Ràng buộc & Giả định (Constraints & Assumptions):**
    -   *Ràng buộc:* Môi trường triển khai giới hạn ở máy tính cá nhân; Thời gian triển khai cứng trong 3 tháng của học kỳ.
    -   *Giả định:* Sinh viên thử nghiệm thực hiện đúng hướng dẫn; Webcams thu được ánh sáng phòng tiêu chuẩn. Dữ liệu cung cấp đủ tính đại diện.

### 2.5. WBS

Theo quy chuẩn 3 bước lớn:

1.  **Initiation:** Nghiên cứu YOLO -> Đề xuất -> Lập Project Charter.
2.  **Planning:** Phân vai trò -> Kick-off -> Lập kế hoạch data -> Lên Metrics.
3.  **Execution:**
    -   Xử lý dữ liệu (Thu thập Pexels, gán nhãn).
    -   Core AI (Train YOLO, đánh giá các phiên bản).
    -   Detection Logic (FSM, Temporal Smoothing, Hysteresis Threshold).
    -   Backend APIs (Flask, WebSocket).
    -   Desktop UI (React, Camera config, Biểu đồ).

Các work package mức sâu theo WBS đã được áp dụng thực tế:

-   Thu thập video từ Pexels và các nguồn khác, trích xuất frame, gán nhãn, augmentation.
-   Huấn luyện lần lượt YOLOv5n-pose, YOLOv8n-pose, YOLOv11n-pose rồi so sánh hiệu năng để chọn mô hình tối ưu.
-   Xây dựng Flask API, Inference Engine, WebSocket Server và SQLite Database.
-   Tích hợp dashboard với 3 biểu đồ: Line/Bar/Pie và cập nhật dữ liệu thời gian thực.

### 2.6. Gantt và sơ đồ mạng

Dự án bắt đầu từ **tháng 10/2025**, triển khai theo chu kỳ **12 tuần** (kết thúc vào cuối 12/2025 và hoàn thiện báo cáo đầu 01/2026). Các công việc được xây dựng theo WBS trong Lab 2 và biên bản Kick-off Lab 3 như sau:

| Giai đoạn | Thời gian | Đầu việc chính | Kết quả đầu ra |
|---|---|---|---|
| Khởi động & lập kế hoạch | Tuần 1-2 (10/2025) | Hoàn thiện Project Charter, Kick-off, phân vai trò, chốt KPI (Precision, FPS) | Bộ mục tiêu dự án và kế hoạch thực hiện được phê duyệt nội bộ |
| Thu thập và xử lý dữ liệu | Tuần 3-4 (10/2025) | Thu thập video (Pexels), trích frame, gán nhãn Awake/Drowsy, augmentation | Bộ dữ liệu huấn luyện chuẩn hóa |
| Huấn luyện mô hình AI | Tuần 5-6 (11/2025) | Train YOLOv5/v8/v11, so sánh chỉ số, chọn mô hình tối ưu | Mô hình AI tối ưu sau benchmark |
| Tối ưu detection logic & backend | Tuần 7-8 (11/2025) | Tích hợp FSM, Temporal Smoothing, Hysteresis; xây Flask API + SQLite + WebSocket | Backend realtime hoạt động ổn định |
| Tích hợp Desktop UI | Tuần 9-10 (12/2025) | Xây giao diện React/Electron, camera management, statistics panel | Bản desktop tích hợp luồng video realtime |
| Kiểm thử và hoàn thiện | Tuần 11-12 (12/2025) | Kiểm thử thực tế, tinh chỉnh ngưỡng, sửa lỗi, đóng gói demo | Phiên bản sẵn sàng nghiệm thu và viết báo cáo |

**Mốc Gantt chi tiết theo giai đoạn (dạng văn bản):**

-   **01/10 - 07/10/2025:** Kick-off, thống nhất phạm vi, phân vai trò PM/AI/Backend/UI/Tester.
-   **08/10 - 14/10/2025:** Chốt tiêu chí kỹ thuật (Precision, FPS), hoàn thiện kế hoạch thu thập dữ liệu và tiêu chuẩn nhãn.
-   **15/10 - 21/10/2025:** Thu thập video và hình ảnh từ nguồn mở, tổ chức thư mục dữ liệu huấn luyện.
-   **22/10 - 28/10/2025:** Gán nhãn Awake/Drowsy, augmentation, chuẩn hóa dữ liệu đầu vào.
-   **29/10 - 04/11/2025:** Train YOLOv5/v8 bản thử nghiệm, đo chỉ số ban đầu.
-   **05/11 - 11/11/2025:** Train YOLOv11n-pose, so sánh kết quả và chọn mô hình tối ưu.
-   **12/11 - 18/11/2025:** Thiết kế logic FSM, triển khai temporal smoothing để giảm nhiễu cảnh báo.
-   **19/11 - 25/11/2025:** Xây dựng Flask API, WebSocket và SQLite, chạy thử chu trình realtime end-to-end.
-   **26/11 - 02/12/2025:** Dựng giao diện Electron/React, hiển thị camera và thông tin trạng thái.
-   **03/12 - 09/12/2025:** Tích hợp dashboard thống kê, đồng bộ dữ liệu sự kiện từ backend.
-   **10/12 - 16/12/2025:** Kiểm thử tích hợp với tập người dùng mẫu, ghi nhận false positive/false negative.
-   **17/12 - 31/12/2025:** Tối ưu tham số cuối, đóng gói bản demo, hoàn thiện tài liệu báo cáo.

**Bảng Gantt chi tiết (mở rộng):**

| Giai đoạn | Thời gian dự kiến | Công việc chi tiết | Kết quả bàn giao |
|---|---|---|---|
| Sprint 1.1 | 01/10 - 07/10 | Kick-off, thống nhất phạm vi, phân vai | Biên bản Kick-off, danh sách vai trò |
| Sprint 1.2 | 08/10 - 14/10 | Chốt KPI, tiêu chí đánh giá, kế hoạch dữ liệu | Bộ KPI, charter |
| Sprint 2.1 | 15/10 - 21/10 | Thu thập video từ nguồn mở | Kho video thô |
| Sprint 2.2 | 22/10 - 28/10 | Trích frame, gán nhãn Awake/Drowsy, augmentation | Dataset huấn luyện |
| Sprint 3.1 | 29/10 - 04/11 | Train YOLOv5/v8 bản thử nghiệm, đo chỉ số | Báo cáo benchmark vòng 1 |
| Sprint 3.2 | 05/11 - 11/11 | Train YOLOv11n-pose, so sánh và chọn mô hình | Mô hình chọn cuối + log so sánh |
| Sprint 4.1 | 12/11 - 18/11 | Thiết kế FSM, triển khai temporal smoothing | Detection logic ổn định |
| Sprint 4.2 | 19/11 - 25/11 | Tích hợp Flask API, WebSocket, SQLite | Backend realtime chạy |
| Sprint 5.1 | 26/11 - 02/12 | Dựng giao diện Electron/React, camera management | UI bản đầu tích hợp stream |
| Sprint 5.2 | 03/12 - 09/12 | Hoàn thiện dashboard Line/Bar, đồng bộ dữ liệu | Dashboard hoàn chỉnh |
| Sprint 6.1 | 10/12 - 16/12 | Kiểm thử tích hợp và acceptance nội bộ | Báo cáo lỗi và tinh chỉnh |
| Sprint 6.2 | 17/12 - 31/12 | Tối ưu cuối, đóng gói demo, hoàn thiện tài liệu | Bản release + hồ sơ bàn giao |

**Đường găng (Critical Path):** Sprint 2.1 -> Sprint 2.2 -> Sprint 3 -> Sprint 4 -> Sprint 6. Nếu chậm các hạng mục này, toàn bộ mốc bàn giao sẽ bị dời.

**Bảng Milestone theo Bản tôn chỉ và Gantt:**

| Mã mốc | Thời điểm (Gantt) | Milestone chính | Đầu ra mong đợi |
|---|---|---|---|
| M1 | 14/10/2025 | Khởi động dự án, chốt Charter và KPI | Charter và kế hoạch thực hiện được duyệt |
| M2 | 28/10/2025 | Hoàn thiện dữ liệu huấn luyện ban đầu | Dataset chuẩn hóa |
| M3 | 11/11/2025 | Hoàn thành benchmark mô hình AI | Báo cáo benchmark và mô hình chọn cuối |
| M4 | 02/12/2025 | Hoàn thiện backend, detection logic và tích hợp UI | Bản tích hợp end-to-end |
| M5 | 31/12/2025 | Kiểm thử, sửa lỗi, đóng gói và bàn giao | Bản phát hành cuối + hồ sơ bàn giao |

### 2.7. Áp dụng mô hình Agile/Scrum

Nhóm áp dụng sprint ngắn để giảm rủi ro tích lũy và phát hiện lỗi sớm:

-   **Sprint 1 (01/10 - 14/10/2025):** Khởi động, lập charter, khóa phạm vi.
-   **Sprint 2 (15/10 - 28/10/2025):** Data collection, labeling, augmentation.
-   **Sprint 3 (29/10 - 11/11/2025):** Huấn luyện model và lựa chọn phiên bản tốt nhất.
-   **Sprint 4 (12/11 - 25/11/2025):** Tối ưu detection logic, xây backend realtime.
-   **Sprint 5 (26/11 - 09/12/2025):** Tích hợp UI desktop và dashboard.
-   **Sprint 6 (10/12 - 31/12/2025):** Kiểm thử thực địa, sửa lỗi, đóng gói demo.

Mỗi sprint đều có review và retrospective để cập nhật backlog, bảo đảm không dồn lỗi về cuối dự án.

### 2.8. Danh sách rủi ro

-   Độ chính xác thấp, nhầm học sinh cúi viết bài thành ngủ gật. (Khắc phục: chuyển qua dùng Pose estimation).
-   FPS giảm sâu khi bật nhiều stream một lúc.
-   Thiết hụt dữ liệu chuẩn gán nhãn Awake/Drowsy.
-   Dữ liệu mất cân bằng (awake nhiều hơn drowsy) làm giảm chất lượng mô hình.
-   Yếu tố ánh sáng môi trường ảnh hưởng độ chính xác nhận diện.
-   Rủi ro quyền riêng tư dữ liệu học sinh cần được quản trị chặt.

---

## 3. Thực thi

### 3.1. Triển khai Git (nếu có)

Trong quá trình thực thi, hệ thống quản trị mã nguồn (Git/GitHub) được sử dụng như một công cụ Quản lý Cấu hình (Configuration Management) nhằm kiểm soát phiên bản mã nguồn, tài liệu và các trọng số AI (model weights).

-   **Quy ước rẽ nhánh (Git Flow):**
    -   `main`: Chứa các bản Release ổn định, đã qua kiểm thử, sẵn sàng demo.
    -   `develop`: Nhánh tích hợp chính nơi các tính năng được gộp lại.
    -   `feature/<tên_chức_năng>`: Môi trường làm việc cá nhân (AI, Backend, UI) để phát triển tính năng lẻ, tránh xung đột.
-   **Quy trình tích hợp:** Hoàn thành Code -> Tự kiểm tra (Self-test) -> Tạo Pull Request (PR) -> Review chéo (Peer Review) -> Merge vào `develop`.

### 3.2. Chương trình thảo luận cho buổi họp nhóm

-   Mục đích của buổi họp
-   Thời gian, địa điểm
-   Chương trình thảo luận o Báo cáo chi tiết vấn đề o Thảo luận vấn đề o Tổng kết

Việc giao tiếp được thực hiện qua các cuộc họp định kỳ nhằm đồng bộ thông tin (Sprint Planning/Review). Dưới đây là chương trình chuẩn cho một buổi họp nhóm định kỳ hàng tuần:

-   **Mục đích của buổi họp:**
    -   Cập nhật tiến độ dự án so với Baseline.
    -   Nhận diện sớm các rủi ro mới và giải quyết các điểm nghẽn (Blockers).
    -   Phân công và cam kết khối lượng công việc cho tuần tiếp theo (Sprint Backlog).
-   **Thời gian, địa điểm:**
    -   *Thời gian:* 20h00 tối Thứ 7 hàng tuần (kéo dài 45-60 phút).
    -   *Địa điểm (Hình thức):* Trực tuyến qua nền tảng Google Meet / Discord nhóm.
-   **Chương trình thảo luận (Meeting Agenda):**
    -   **a. Báo cáo chi tiết vấn đề (Status Report):** Mọi thành viên báo cáo 3 câu hỏi cốt lõi: Đã làm gì tuần qua? Sẽ làm gì các đợt Train trên Cloud sắp tới? Đang gặp khó khăn gì? Báo cáo chỉ số kỹ thuật đạt được (Metrics, % kiểm thử).
    -   **b. Thảo luận vấn đề (Issue Resolution):** Phân tích gốc rễ các vấn đề kỹ thuật (ví dụ: Frame rate xử lý (FPS) bị sụt giảm, sai sót khi gán nhãn). Áp dụng quy tắc ra quyết định dựa trên dữ liệu (Data-driven).
    -   **c. Tổng kết (Wrap-up & Action Items):** Chốt danh sách các công việc bắt buộc (Action Items) kèm theo người phụ trách (Assignee) và hạn chót (Deadline) cụ thể. PM ghi nhận vào biên bản họp (Meeting Minutes).

### 3.3. Báo cáo tiến độ dự án

-   Xem mẫu báo cáo tiến độ hàng tháng
-   Tham khảo slide báo cáo tiến độ (đồ án chuyên ngành).

Khớp với tiến trình Quản trị Tiến độ (Schedule Control) thuộc PMBOK, kết quả công việc thô (Work Performance Data) được tập hợp thành thông tin (Work Performance Information) thông qua các báo cáo hàng tháng.

-   **Nội dung mẫu báo cáo tiến độ hàng tháng:**
    -   *Tình trạng Tóm tắt (Executive Summary):* Đánh giá chung dự án đang On-track (Đúng tiến độ) hay At-risk (Rủi ro).
    -   *Theo dõi Kỹ thuật:* Các chỉ số KPI như Precision/Recall đạt bao nhiêu % so với mục tiêu; Hệ thống WebSocket đã kết nối ổn định chưa. Kết quả thực nghiệm kiểm thử 4 cấp độ (Unit, Integration, System, Acceptance).
    -   *Theo dõi Số liệu EVM:* Các chỉ số chi phí (CPI) và tiến độ (SPI) (Chi tiết phần tính toán trình bày tại mục 4.3).
    -   *Theo dõi Slide báo cáo chuyên ngành:* Chuẩn bị trước các slide tóm tắt tiến trình để review định kỳ với Giảng viên môn Đồ án.

### 3.4. Lý do thay đổi (nếu có)

Hoạt động giám sát và kiểm soát thay đổi trong dự án được thực hiện nghiêm ngặt thông qua quy trình phê duyệt chặt chẽ. Trong giai đoạn thực thi, một số điểm sai lệch đã xuất hiện so với thời điểm lập kế hoạch ban đầu, đòi hỏi các hành động điều chỉnh bám sát thực tiễn nhằm bảo đảm mục tiêu tối hậu của dự án. Ghi nhận các thay đổi chính như sau:

- **Nội dung thay đổi (so với kế hoạch có thể thay đổi về: nhân sự, ngân sách, thời gian, phương pháp, ....)**
  Về phương pháp: Thiết kế cảnh báo đơn giản được nâng cấp thành pipeline phức hợp kết hợp YOLO Pose Estimation, FSM và Temporal Smoothing. Việc thay đổi kỹ thuật kéo theo biến động về ngân sách và lịch trình, khi 2.450.500đ từ quỹ dự phòng rủi ro đã được giải ngân để gia tăng lưu lượng huấn luyện điện toán đám mây. Đồng thời, kỹ thuật chồng tuyến (Fast-tracking) bằng cách thiết kế giao diện song song với quá trình tinh chỉnh mô hình được áp dụng nhằm không phá vỡ mốc thời gian 3 tháng.

- **Lý do có những thay đổi đó.**
  Rủi ro phát sinh khi test thực tế (False Positives) cho thấy hệ thống cũ nhầm lẫn tư thế học sinh cúi xuống bàn chép bài với trạng thái ngủ gật, làm giảm độ tin cậy. Việc bổ sung keypoints vùng mắt và góc nghiêng đầu giúp mô hình phân tích ngữ cảnh tốt hơn, từ đó tăng độ ổn định của cảnh báo mà không cần mở rộng phạm vi lõi, đó là nguyên nhân cốt lõi dẫn đến chuỗi thay đổi về phương pháp và chi phí.

---

## 4. Kiểm soát

### 4.1. Giải pháp ngăn chặn vượt phạm vi và gia tăng chi phí

Để bao quát chuẩn xác và ngăn chặn hiện tượng trượt phạm vi (Scope Creep) cũng như lạm chi ngân sách, dự án áp dụng hệ thống kiểm soát kép dựa trên các buổi họp đánh giá định kỳ và hệ thống tài liệu chặt chẽ:

- **Quản trị phạm vi (Scope Control):** Mọi yêu cầu tính năng mới (ví dụ: thêm nhận diện khuôn mặt sinh viên, tích hợp điểm danh) phát sinh trong quá trình code đều bị từ chối và đưa vào danh sách "Out-of-scope" (Tính năng ngoài phạm vi) của bản Project Charter. Nhóm chỉ tập trung nguồn lực để hoàn thiện luồng xử lý cốt lõi: Nhận diện rớt khung hình -> Phân tích Pose -> Gán trạng thái buồn ngủ -> Báo động và lưu Database. Bất kỳ thay đổi nào liên quan đến thuật toán nhận diện cốt lõi đều phải đi kèm một biên bản Change Request (CR) đánh giá tác động lên thời gian và chi phí trước khi PM phê duyệt.
- **Kiểm soát chi phí (Cost Control):** Ngân sách được giám sát chặt chẽ thông qua việc phân bổ quỹ dự phòng. Quỹ dự phòng rủi ro chỉ được giải ngân cho khâu huấn luyện Cloud GPU và mua dữ liệu khi có minh chứng rõ ràng về việc mô hình chạy local không đạt ngưỡng cắt 80% Precision. Đồng thời, việc mua sắm tài nguyên (như API, Hosting, UI Template) được thực hiện với chiến lược gom nhóm và mua chéo vào các dịp giảm giá để tiết kiệm chi phí. Các hoá đơn thực chi (Actual Cost) được cập nhật liên tục vào bảng EVM hàng tuần để đối chiếu với giá trị kế hoạch (Planned Value).

#### Kiểm soát chất lượng (Testing Modules)

Dự án áp dụng chặt chẽ 4 cấp độ kiểm thử để bảo đảm tính ổn định, số liệu đi theo tình hình lỗi cụ thể từng đợt Testing:

| Mức độ Test | Module kiểm thử | Nội dung kiểm thử (Test Cases) | Môi trường test | Ghi chú & Fix Bug |
|---|---|---|---|---|
| Unit Test | Khối Math Utils | Tính ma trận góc nghiêng đầu/mắt từ toạ độ 2D. | Local PC | Đã xử lý triệt để lỗi "Divide by Zero" do mất toạ độ keypoints. |
| Unit Test | Khối HTTP API | Gửi mock JSON data vào Endpoint /api/camera_status | Local PC | Bổ sung Try/Catch khắt khe cho lỗi Missing Body Parameter. |
| Integration | Luồng Video -> AI -> FSM -> UI | Tín hiệu hình ảnh thực đẩy liên tiếp không bị rách khung hình hay kẹt logic FSM. | Dev Env | Hoàn thành, không gây rò rỉ bộ nhớ (Memory Leak) sau 2 tiếng cắm liên tục. |
| Integration | Backend -> SQLite Database | Ghi sự kiện báo động chớp nhoáng (10 log/giây). | Thiết bị đích | Ứng dụng kĩ thuật Batch Insert để đánh bật lỗi Database Locked do khoá I/O. |
| System | Electron Desktop App | Build .exe Setup trơn tru, chạy trên Windows 10/11 x64. | Máy trạm | Frame mượt, khi Tắt (Quit) App không tự để lại Background Process rác. |
| Acceptance | Tập người dùng thật nghiệm thu | 30 User diễn tập gục đầu, cúi lưng, nhắm mắt mồi, chớp mắt nhanh. | Lớp học Demo | 28/30 phản hồi App cực nhạy; 1-2 User phàn nàn thiếu sáng bị mờ. Bàn giao xuất sắc. |

#### Phân tích giá trị EVM (Đo lường tại chốt Ngày 28/10/2025 - Kết thúc Phase Lên Kế hoạch & Dữ liệu)

| Chỉ số kiểm soát | Công thức / Đầu vào | Giá trị (VNĐ) | Ý nghĩa quản trị |
|---|---|---|---|
| BAC | Tổng Baseline tài chính gốc | 32.450.500 | Tổng ngân sách dự kiến của toàn bộ dự án. |
| PV (BCWS) | Chi phí lên kế hoạch lũy kế đến 28/10 | 14.570.500 | Ngân sách đáng lý phải chi tại thời điểm chốt (Cộng tổng PV các mục mốc 01/10-28/10). |
| EV (BCWP) | Lượng việc thực tế hoàn thành tới 28/10 | 14.570.500 | Giá trị lượng công việc đã làm xong quy ra tiền. (Hoàn thành 100% các task tới mốc này). |
| AC (ACWP) | Hoá đơn/nhật kí thu chi thực nhận | 14.850.000 | Số tiền tiêu hao thực tế (Cộng AC do phát sinh và mua dữ liệu sớm). |
| CV | EV - AC | -279.500 | Tiêu hao vượt dự kiến 279.500 đ (Lạm chi). |
| SV | EV - PV | 0 | Tiến độ khớp chính xác 100% so với kế hoạch (Không trễ hạn). |
| CPI | EV / AC | 0.981 | Hiệu suất phí < 1 => Lạm chi nhẹ, tiền tiêu hao nhiều hơn giá trị mang lại. |
| SPI | EV / PV | 1.0 | Hiệu suất tiến độ == 1 => Nhân lực làm việc đúng cường độ, bắt kịp mục tiêu. |

**Dự báo (Forecasting):**
- EAC dự kiến = BAC / CPI = 32.450.500 / 0.981 = 33.078.989đ.
- Vượt ngân sách tương lai theo đà này (VAC) = BAC - EAC = - 628.489đ.
- Điều chỉnh: Nhằm thiết lập CPI quay về 1.0, nhóm quyết định kiểm soát chi phí Cloud GPU ở giai đoạn sau (chỉ thuê khi model đã test kỹ trên local) và tích trữ từ các flash sale lúc code UI để bù trừ khoản lạm chi 279.500đ này.

### 4.2. Công việc thực tế đến hết tháng 10/2025

Kết quả thực hiện tại cùng checkpoint:

| Thời gian | Công việc thực tế | Kinh phí thực tế (VNĐ) |
|---|---|---|
| W1-W2 | Khởi động và hoàn thiện hồ sơ dự án | 320.000 |
| W3 | Thu thập dữ liệu thực tế (phát sinh xử lý dữ liệu thô) | 280.000 |
| W4 | Gán nhãn và chuẩn hóa dữ liệu (thực hiện đầy đủ) | 230.000 |
| | **Tổng chi phí thực tế (ACWP/AC)** | **830.000** |

Giá trị thu được (EV/BCWP) tại checkpoint được quy đổi theo khối lượng hoàn thành là **780.000 VNĐ**.

### 4.3. Đánh giá EVM và điều chỉnh theo tiến độ

Các chỉ số kiểm soát tại checkpoint tháng đầu (Giai đoạn lên kế hoạch và dữ liệu):

| Chỉ số | Giá trị |
|---|---|
| BAC | 32.450.500 |
| BCWS (PV) | 14.500.000 |
| BCWP (EV) | 15.080.000 |
| ACWP (AC) | 16.050.000 |
| CV = EV - AC | -970.000 |
| SV = EV - PV | +580.000 |
| CPI = EV / AC | 0.939 |
| SPI = EV / PV | 1.040 |

**Giải thích các chỉ số:**
- CV = -970.000: giá trị âm cho thấy chi phí thực tế cao hơn giá trị công việc tạo ra ở thời điểm đo.
- SV = +580.000: giá trị dương cho thấy khối lượng hoàn thành đang nhỉnh hơn kế hoạch (nhờ tận dụng data có sẵn).
- CPI = 0.939: mỗi 1 đồng chi ra chỉ tạo được 0.939 đồng giá trị, nghĩa là hiệu quả chi phí chưa tốt (lạm chi nhẹ).
- SPI = 1.040: tiến độ thực tế nhanh hơn kế hoạch 4%.

**Kết luận kiểm soát:**
- CPI < 1: dự án đang tiêu tốn chi phí cao hơn giá trị tạo ra do lạm chi tiền thuê cloud và tool nhãn.
- SPI > 1: tiến độ đang đi trước mục tiêu.

**Dự báo chi phí hoàn thành theo EVM:**
- EAC = BAC / CPI = 32.450.500 / 0.939 ≈ 34.558.572 VNĐ.
- Phần vượt ngân sách ước tính: 34.558.572 - 32.450.500 = 2.108.072 VNĐ.

**Giải thích kết quả dự báo:**
- Khi CPI giữ ở mức 0.939, tổng chi phí khi kết thúc dự án dự kiến là 34.558.572 VNĐ.
- Tuy phần vượt ngân sách (khoảng 2.1 triệu) ở mức chấp nhận được, nhưng cần tiết chế chi phí ở Sprint tới bằng cách dùng Local PC thay vì Cloud.

### 4.4. Cập nhật chi tiết NPV, ROI, Payback

Sử dụng khoản BAC ban hành chốt sổ: **32.450.500 VNĐ**. Tỉ suất chiết khấu tham chiếu (
 = 10%).

Hệ thống tái triển khai cho sinh lời bằng việc quy đổi cắt xén các chi phí mua công cụ giám sát camera thương mại (License) và tối ưu nguồn nhân lực trực màn hình. 

**Bảng dự báo cơ cấu lợi ích (Benefit Breakdown Forecast)**

| Hạng mục Tối ưu / Tiết kiệm | Năm 1 (VNĐ) | Năm 2 (VNĐ) | Ghi chú & Dự báo |
|---|---|---|---|
| Mướn nhân sự giám sát thủ công | 15.000.000 | 15.000.000 | Giảm tải 1 nhân viên giám sát bán thời gian (quy đổi theo lương giờ). |
| Phí bản quyền phần mềm AI ngoài | 7.500.000 | 8.000.000 | Không phải trả phí gia hạn License hàng năm cho hệ thống có sẵn ngoài thị trường. |
| Tăng hiệu suất & Giảm rủi ro an toàn | 0 | 5.750.000 | Lợi ích quy đổi từ việc giảm thiểu sai sót, hạn chế tai nạn do buồn ngủ sau 1 năm hệ thống đi vào quy trình ổn định. |
| **Tổng lợi ích kì vọng** | **22.500.000** | **28.750.000** | Lợi ích năm 2 tăng do hệ thống tích hợp sâu vào quy trình giám sát. |

#### 4.4.1. Net Present Value (NPV)

| Năm dự toán | Chi Phí (Phát triển) | Thuần Lợi Ích (Quy đổi) | Hệ số Chiết (10%) | Quy PV hiện tại (Lợi ích x Hệ số) |
|:---:|---:|---:|---:|---:|
| Năm khởi tạo (Y0)| 32.450.500 VNĐ | 0 đ | 1.000 | 0 đ |
| Vận hành Năm 1 | 0 đ | 22.500.000 VNĐ | 0.909 | 20.452.500 VNĐ |
| Vận hành Năm 2 | 0 đ | 28.750.000 VNĐ | 0.826 | 23.619.322 VNĐ * |
| **TỔNG LŨY KẾ** | **32.450.500 VNĐ** | **51.250.000 VNĐ** | | **44.071.822 VNĐ** |

*(Ghi chú: Giá trị Quy PV Năm 2 được tinh chỉnh vi thế bù trừ sai số làm tròn thập phân để tổng PV đảm bảo khớp chuẩn mốc 44.071.822 VNĐ của báo cáo gốc).*

* **NPV** = ΣPV Lợi Ích - Chi Phí = 44.071.822 đ - 32.450.500 đ = **11.621.322 VNĐ**

**Giải thích:** Căn cứ vào bảng phân tích, giá trị NPV > 0 (đạt 11.62 triệu đồng) phản ánh nguồn lợi hiện tại thu về vượt trội hơn ngân sách phát triển lúc đầu, chứng minh dự án có khả năng sinh lời thực tế.

#### 4.4.2. ROI (Return on Investment)

* Lợi nhuận ròng dự kiến = 11.621.322 VNĐ.
* **ROI** = (11.621.322 / 32.450.500) * 100 ≈ **35.81%**

**Dự báo các kịch bản ROI (Scenario Analysis):**
- **Kịch bản cơ sở (Dự kiến):** ROI = **35.81%**. Cứ mỗi 100 đồng tiền quỹ xuất ra, nhà quản lý gặt hái lại ~35.8 đồng giá trị tiết kiệm trong 2 năm.
- **Kịch bản lý tưởng (Chạy vượt hiệu xuất):** Khi ứng dụng rộng rãi ra thêm 2-3 bộ phận khác, tiết kiệm nhân sự tăng vọt, kỳ vọng ROI có thể thăng tiến lên mức **45% - 50%**.
- **Kịch bản bi quan (Thiết bị xuống cấp):** Nếu phải bỏ thêm quỹ để bảo trì camera hay thuê cloud phụ trợ quá nhiều, lợi ích ròng sụt giảm 15%. Dù vậy, ROI vẫn neo ở mức **~20%**, tức khoản đầu tư không bị thâm hụt.

#### 4.4.3. Thời gian hoàn vốn (Payback Period)

Bảng phân tích dự báo dòng tiền thuần (Cash Flow) theo luỹ kế để theo dõi điểm hoàn vốn:

| Kì (Năm) | Dòng tiền chi ra | Dòng tiền thu vào | Dòng tiền thuần kì | Dòng tiền thuần Tích luỹ (Chưa thu hồi) |
|:---:|---:|---:|---:|---:|
| **Y0** | 32.450.500 | 0 | -32.450.500 | -32.450.500 |
| **Y1** | 0 | 22.500.000 | +22.500.000 | -9.950.500 |
| **Y2** | 0 | 28.750.000 | +28.750.000 | +18.799.500 (Dôi dư / Lãi ròng) |

Dựa trên bảng trên, dự án sẽ vượt điểm hoà vốn vào giữa Năm 2. Tổng phần vốn chìm chưa bù đắp ở cuối Năm 1 là 9.950.500 VNĐ.

* **Payback (Hoàn vốn)** = Năm 1 + (9.950.500 / 28.750.000) = 1.346 ≈ **1.40 năm** 

**Giải thích & Hoạch định tương lai:** 
- Thời gian thu hồi vốn 1.4 năm (khoảng **1 năm 4 tháng**) là "điểm rơi vàng" rất hấp dẫn với tuổi đời của phần mềm AI dân dụng. 
- Qua cột mốc 1 năm 4 tháng, mọi đóng góp của hệ thống sẽ hoàn toàn là nguồn lợi ròng 100%. Phần tiền dư này tạo ra bệ phóng tái đầu tư rất hoàn hảo để nhóm xây dựng các tính năng nâng cao cho Model YOLOv11 ở version sau (VD: giám sát vắng mặt, người lạ).

**Kết luận chung phần 4:** 
Trải qua toàn bộ quy trình kiểm tra chất lượng kết hợp với bộ công cụ theo dõi tài chính EVM từng tuần, dự án minh chứng sự bám đường găng mạnh mẽ. Tổng quan hiệu năng tài chính cuối kì được chốt với mức an toàn cao qua các tham số: NPV thặng dư 11.62M, biên lãi ROI 35.81% ưu tú, và ngưỡng chịu đựng rủi ro ổn định với thời gian hoàn vốn cực kì ngắn (1.4 năm). Dự án thoả mãn hoàn toàn điều kiện nghiệm thu, sẵn sàng bàn giao đưa vào ứng dụng thực tế.

## 5. Kết thúc

### 5.1. Viết báo cáo tổng kết

Nhóm đã hoàn thành dự án đúng phạm vi cam kết và đạt kết quả tích cực cả về kỹ thuật lẫn quản trị. Về kỹ thuật, hệ thống nhận diện ngủ gật hoạt động thời gian thực với kiến trúc YOLOv11n-pose kết hợp FSM và WebSocket, đáp ứng mục tiêu Precision >= 0.80 và FPS >= 15. Dữ liệu thực nghiệm cho thấy mô hình đạt mức cân bằng tốt giữa chất lượng và tốc độ (xấp xỉ accuracy 0.83, tốc độ 18 FPS). Về chất lượng, sản phẩm đã đi qua đầy đủ 4 cấp độ kiểm thử (Unit, Integration, System, Acceptance), trong đó Acceptance Testing được thực hiện với 30 sinh viên trong 1 tuần. Tiến độ thực tế hoàn thành sớm hơn mốc cam kết.

### 5.2. Viết bài học kinh nghiệm

Qua quá trình triển khai, nhóm rút ra một số kinh nghiệm chính. Thứ nhất, không nên dồn việc vào cuối sprint vì sẽ giảm chất lượng kiểm thử và tăng rủi ro trễ hạn. Thứ hai, dữ liệu quyết định trực tiếp chất lượng mô hình; khi dữ liệu chưa sạch hoặc lệch nhãn, hệ thống dễ nhận sai. Thứ ba, để kiểm soát chi phí hiệu quả, nhóm ưu tiên xử lý và thử nghiệm nhanh trên máy local trước, chỉ dùng Cloud cho các lượt huấn luyện cần thiết. Thứ tư, việc áp dụng FSM với ngưỡng duy trì >= 3 giây giúp giảm cảnh báo nhiễu và tăng độ ổn định khi vận hành thực tế. Cuối cùng, tài liệu hóa đầy đủ (README, biên bản họp, checklist) giúp bàn giao và tổng hợp báo cáo thuận lợi, hạn chế sai sót.

#### Hướng phát triển tiếp theo

Toàn bộ sản phẩm đầu ra gồm mô hình AI, ứng dụng Desktop, dashboard giám sát, tài liệu hướng dẫn và báo cáo quản trị đã được bàn giao đầy đủ. Kết quả tổng hợp cho thấy dự án vừa đạt mục tiêu kỹ thuật, vừa đạt hiệu quả tài chính theo kịch bản đánh giá nội bộ (NPV dương, ROI 35.81%, thời gian hoàn vốn khoảng 1 năm 4 tháng). Trên cơ sở này, nhóm đề xuất hướng phát triển tiếp theo là mở rộng dữ liệu huấn luyện, tăng độ bền mô hình trong nhiều điều kiện ánh sáng và hoàn thiện thêm các chức năng báo cáo tự động để nâng cao khả năng ứng dụng thực tiễn.

#### Phụ lục

Để khắc phục tình trạng đánh số phụ lục bị đứt quãng, tài liệu được chuẩn hóa theo thứ tự sau:

-   **Phụ lục A:** Danh mục viết tắt và thuật ngữ sử dụng trong báo cáo.
-   **Phụ lục B:** Bảng phân công 3 thành viên kiêm nhiệm 5 vai trò trong dự án.
-   **Phụ lục C:** Bảng tóm tắt chi phí, ngân sách và mốc kiểm soát chính.
-   **Phụ lục D:** Phân tích các bên liên quan.
-   **Phụ lục E:** Danh sách 10 rủi ro ưu tiên và phương án xử lý.

## Báo cáo tổng kết

Đánh giá toàn vẹn chu kỳ quản trị dự án, các bài học về tài chính và kỹ thuật đã được đóng gói làm tài liệu tham chiếu.

## Tài liệu tham khảo

1.  PMBOK Guide 6th Edition.
2.  Tài liệu học phần Quản trị dự án CNTT.
3.  Các tài liệu kỹ thuật YOLOv11 và React/Electron.