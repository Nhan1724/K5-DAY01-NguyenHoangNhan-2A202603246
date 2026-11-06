# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** T4-GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

# Báo cáo kết quả Thực hành Ngày 1: Phân tích Dữ liệu Nhãn từ YOLO11

## 1. Phân loại ảnh – prediction cấp ảnh
- **Thông tin record hạng 1 (mẫu traffic):** Lớp dự đoán có độ tin cậy cao nhất là `cab` (ID: 468, score: 0.510915, thuộc danh mục ImageNet-1K).
- **Mô tả toàn ảnh:** Nhãn phân loại ảnh gán một nhãn duy nhất đại diện cho nội dung nổi bật nhất của bức ảnh, không chỉ ra vị trí cụ thể của từng vật thể.
- **Định nghĩa class list:** Danh sách lớp được định nghĩa bởi nhóm xây dựng bộ dữ liệu gốc (ở đây là ImageNet-1K).
- **Ý nghĩa việc giữ ID, tên lớp và taxonomy:** Giúp tránh nhầm lẫn do trùng tên, dễ dàng tra cứu, hỗ trợ đọc hiểu trực quan và duy trì tính tương thích hệ thống giữa các phiên bản mô hình khác nhau.
- **Guideline khi có nhiều chủ thể:** Cần quy định rõ quy tắc ưu tiên chọn đối tượng trung tâm, đối tượng có diện tích lớn nhất, hoặc quy chuẩn chọn nhãn bao quát.
- **Model score không phải ground truth:** Vì nó chỉ là xác suất phán đoán dựa trên trọng số đã học của mô hình và vẫn có thể sai sót, trong khi ground truth là nhãn chuẩn được con người kiểm chứng chính xác tuyệt đối.

## 2. Phát hiện vật thể – lớp và box cho từng object
- **Thông tin record (mẫu kitchen):** Ví dụ đối tượng `person` có score 0.812513, tọa độ bbox_xyxy `[183.12, 114.78, 478.43, 421.1]` (đơn vị pixel).
- **Diễn giải vị trí box bằng lời:** Hộp giới hạn bao quanh đối tượng người từ tọa độ x=183.12, y=114.78 (góc trên bên trái) đến x=478.43, y=421.1 (góc dưới bên phải), với chiều rộng 295.31 pixel và chiều cao 306.32 pixel.
- **So sánh số lượng prediction ở hai ngưỡng:** Ở ngưỡng 0.35 phát hiện 11 vật thể, ở ngưỡng 0.60 chỉ còn lại 6 vật thể có độ tin cậy cao.
- **Sự thay đổi đối với độ bao phủ và công việc reviewer:** Hạ thấp ngưỡng giúp tăng độ bao phủ (Recall) nhưng tăng khối lượng công việc cho reviewer vì phải lọc bỏ nhiều nhiễu. Nâng cao ngưỡng giúp giảm nhiễu nhưng làm tăng nguy cơ bỏ sót vật thể thực tế, yêu cầu reviewer phải vẽ bù thủ công nhiều hơn.
- **Đề xuất quy tắc box chặt:** Hộp giới hạn phải ôm khít tối đa biên ngoài cùng của vật thể, không để thừa khoảng trống quá 2 pixel ở mỗi cạnh và không cắt lẹm vào cấu trúc vật thể.
- **Vấn đề che khuất/cắt mép:** Cần quy định rõ tỷ lệ diện tích bị che tối đa được phép gán nhãn, cách ước lượng phần khuất và quy trình leo thang khi gặp các vật thể mờ hoặc bị cắt mép không rõ danh tính.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance
- **Thông tin record (mẫu kitchen):** Một cá thể cái bát `bowl` (ID instance: `kitchen-002`) đạt score 0.887612 với đa giác gồm 52 điểm neo tọa độ bám sát viền.
- **Đặc trưng bổ sung của Polygon:** Cung cấp thông tin chi tiết về ranh giới hình học thực tế và hình dạng chính xác của vật thể tới cấp độ pixel thay vì chỉ bao khung hình chữ nhật thô.
- **Ý nghĩa của instance_id:** Giúp phân biệt và định danh duy nhất cho từng cá thể độc lập trong cùng một bức ảnh (phân đoạn thực thể), khác với ID danh mục lớp chung.
- **Quy tắc biên mask:** Đường đa giác phải đi sát ranh giới phân tách màu sắc của vật thể, số lượng điểm neo tối đa không quá 120 điểm để đảm bảo tối ưu hóa dữ liệu lưu trữ.
- **Vấn đề vùng mờ/tiếp xúc/che khuất:** Quy định rõ phương án bo viền khi đối tượng bị bóng mờ chuyển động, ranh giới chia cắt khi hai vật thể chồng lên nhau và quy trình báo cáo khi không thể phân định ranh giới.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| :--- | :--- | :--- | :--- | :--- |
| **Phân loại ảnh** | Tên nhãn / class_id duy nhất cho toàn ảnh. | Ảnh có quá nhiều chủ thể gây nhiễu, nhãn không đồng nhất. | Chọn nhãn theo quy tắc ưu tiên trong guideline. | Kiểm tra nhãn được chọn có đúng quy tắc ưu tiên không. |
| **Phát hiện vật thể** | Bbox dạng `[x_min, y_min, x_max, y_max]` kèm nhãn. | Bỏ sót các đồ vật nhỏ hoặc bị khuất, vẽ box quá rộng. | Vẽ hộp giới hạn ôm khít toàn bộ phần vật thể. | Kiểm tra độ phủ và độ khít của các hộp giới hạn. |
| **Phân đoạn đối tượng** | Tập hợp các tọa độ điểm `[x, y]` của đa giác và `instance_id`. | Đường biên răng cưa, nhầm lẫn ranh giới tiếp xúc chồng lấn. | Điểm đa giác bám sát viền, phân biệt rõ các cá thể riêng. | Đánh giá độ khớp của mặt nạ và kiểm tra ranh giới tiếp xúc. |

## 5. An toàn dữ liệu
- **Quy tắc bảo vệ dữ liệu:** Tuyệt đối không sao chép, chia sẻ dữ liệu hình ảnh nội bộ lên các kênh công cộng hoặc công cụ AI tự do ngoài hệ thống mà chưa có sự đồng ý của quản lý dự án.
- **Nếu thấy ảnh không đúng phạm vi hoặc nhạy cảm:** Tôi sẽ dừng công việc trên tệp đó, dán nhãn cảnh báo (flag) và lập tức báo cáo cho Trưởng nhóm Gán nhãn hoặc Quản lý Dự án (PM) để được xử lý kịp thời.

## 6. Danh sách bằng chứng
- `classification_predictions.json`
- `detection_predictions.json`
- `segmentation_predictions.json`
- `IMAGE_ATTRIBUTION.md`
- `visuals/classification_top5.png`
- `visuals/detection_predictions.png`
- `visuals/segmentation_prediction.png`
- Ô validation cuối notebook báo PASS.
- Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
