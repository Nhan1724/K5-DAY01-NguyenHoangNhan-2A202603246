1. Phân loại ảnh – prediction cấp ảnh (mẫu traffic từ classification_predictions.json)
•	Record hạng 1 (top-1):
o	class_id: 468
o	class_name: cab (xe taxi)
o	rank: 1
o	score: 0.510915 (khoảng 51% độ tin cậy)
o	taxonomy_name: ImageNet-1K
•	Record này mô tả toàn ảnh như thế nào?
o	Nhãn phân loại ảnh (image classification) gán một nhãn duy nhất cho toàn bộ bức ảnh chứ không xác định vị trí cụ thể của từng đối tượng. Ở đây, mô hình dự đoán bức ảnh chụp cảnh giao thông này thuộc danh mục "cab" (xe taxi) với độ tin cậy cao nhất (51%), dù trong ảnh còn có rất nhiều đối tượng khác như xe buýt, người đi bộ, xe hơi thông thường.
•	Ai định nghĩa class list mà checkpoint có thể dự đoán?
o	Tập hợp các lớp (class list / taxonomy) này được định nghĩa bởi những người xây dựng bộ dữ liệu gốc (ở đây là bộ dữ liệu ImageNet với phiên bản ImageNet-1K chứa 1000 lớp đối tượng phổ biến).
•	Vì sao cần giữ cả ID, tên lớp và tên taxonomy trong dữ liệu dự đoán?
o	Giữ ID (class_id): Để tránh nhầm lẫn do dịch thuật, sai chính tả hoặc khi đổi tên lớp trong tương lai. ID là định danh duy nhất trong hệ thống lập trình.
o	Giữ tên lớp (class_name): Giúp con người (nhà phát triển, người kiểm thử) dễ đọc và hiểu ngay lập tức mà không cần phải tra cứu bảng ánh xạ từ ID sang tên.
o	Giữ tên taxonomy (taxonomy_name): Để biết mô hình đang sử dụng bộ nhãn quy chuẩn nào (ví dụ: ImageNet-1K phân biệt rất chi tiết các loại chó mèo, xe cộ, trong khi COCO-80 chỉ có các nhóm lớp chung chung hơn). Điều này rất quan trọng khi tích hợp hoặc so sánh giữa các mô hình khác nhau.
•	Nếu ảnh có nhiều chủ thể, guideline (hướng dẫn gán nhãn) cần quy định điều gì?
o	Guideline cần quy định rõ quy tắc ưu tiên (priority rules) để người gán nhãn lựa chọn một lớp duy nhất cho ảnh:
	Chọn đối tượng chiếm diện tích lớn nhất (dominant object).
	Chọn đối tượng nằm ở trung tâm của bức ảnh (central focus).
	Quy định rõ cách xử lý khi có sự xuất hiện đồng thời của nhiều chủ thể có độ quan trọng ngang nhau (ví dụ: chọn nhãn bao quát hơn nếu có, hoặc ưu tiên nhãn thuộc danh mục cảnh báo an toàn giao thông).
•	Vì sao model score (độ tin cậy của mô hình) không phải ground truth?
o	Model score chỉ thể hiện mức độ tự tin (xác suất toán học) của mô hình dựa trên những gì nó đã được học, và nó hoàn toàn có thể sai (như nhận diện nhầm, bỏ sót).
o	Ground truth (nhãn chuẩn) là sự thật khách quan được kiểm chứng và xác nhận bởi con người theo đúng guideline gán nhãn. Một nhãn ground truth chuẩn xác phải đạt chất lượng 100% đúng thực tế, không phụ thuộc vào việc mô hình đoán bức ảnh đó được bao nhiêu điểm tin cậy.


2. Phát hiện vật thể – lớp và box cho từng object (Mẫu kitchen)
•	Trích dẫn một record từ detection_predictions.json (ví dụ đối tượng person):
o	class_name: person
o	score: 0.812513 (81.25%)
o	bbox_xyxy: [183.12, 114.78, 478.43, 421.1]
o	bbox_width: 295.31 (pixel)
o	bbox_height: 306.32 (pixel)
•	Diễn giải vị trí box bằng lời:
o	Hộp giới hạn này bao quanh đối tượng người (person) trong ảnh, bắt đầu từ góc trên bên trái có tọa độ $x = 183.12$$x = 183.12$ pixel và $y = 114.78$$y = 114.78$ pixel, kéo dài đến góc dưới bên phải có tọa độ $x = 478.43$$x = 478.43$ pixel và $y = 421.1$$y = 421.1$ pixel. Hộp này có chiều rộng khoảng 295 pixel và chiều cao khoảng 306 pixel trên tổng thể kích thước ảnh gốc $640 \times 427$$640 \times 427$.
•	So sánh số lượng prediction ở hai ngưỡng (threshold):
o	Ở ngưỡng 0.35: Mô hình phát hiện được 11 vật thể (['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']).
o	Ở ngưỡng 0.60: Mô hình chỉ còn phát hiện được 6 vật thể (['person', 'bowl', 'bowl', 'oven', 'oven', 'person']).
o	Nhận xét: Khi tăng ngưỡng từ 0.35 lên 0.60, các vật thể có độ tin cậy thấp (như cup, spoon hay một số bowl nhỏ) bị loại bỏ hoàn toàn.
•	Độ bao phủ và khối lượng reviewer cần xem thay đổi như thế nào?
o	Khi hạ thấp ngưỡng tin cậy:
	Độ bao phủ (Recall) tăng: Ít bỏ sót vật thể thực tế hơn.
	Khối lượng reviewer tăng rất mạnh: Reviewer phải kiểm tra thêm rất nhiều dự đoán sai (False Positives) hoặc nhiễu do mô hình đưa ra.
o	Khi nâng cao ngưỡng tin cậy:
	Khối lượng reviewer giảm: Chỉ cần duyệt các trường hợp mô hình rất chắc chắn.
	Độ bao phủ giảm (bỏ sót nhiều): Reviewer sẽ mất thêm thời gian để tự tay vẽ lại (gán nhãn thủ công hoàn toàn) các đối tượng bị bỏ sót (False Negatives).
•	Đề xuất quy tắc vẽ box chặt (Tight Bounding Box):
o	Quy tắc: "Hộp giới hạn phải ôm sát biên ngoài cùng của vật thể, không được để thừa quá 2 pixel khoảng trống ở mỗi cạnh, đồng thời không được cắt lẹm vào bất kỳ phần nào thuộc cơ thể hoặc bề mặt vật thể trừ khi bộ phận đó bị che hoàn toàn."
•	Đối với vật thể bị che khuất (occluded) hoặc cắt mép (truncated), điều gì cần guideline/escalation quyết định?
o	Vật thể bị che khuất (Occlusion): Guideline cần định nghĩa rõ ngưỡng diện tích bị che khuất tối đa để gán nhãn (ví dụ: bị che dưới 70% thì vẫn vẽ box bao quanh phần nhìn thấy + ước lượng phần bị che; bị che trên 70% thì bỏ qua không gán nhãn). Việc có ước lượng phần khuất để vẽ box lớn hay chỉ vẽ phần nhìn thấy cũng cần quy chuẩn rõ ràng.
o	Vật thể bị cắt mép (Truncation): Quy định xem vật thể chỉ lọt vào ảnh một phần rất nhỏ (ví dụ chỉ thấy bàn tay của người ở sát rìa ảnh) có cần gán nhãn không, hay chỉ gán nhãn khi cấu trúc chính của vật thể vẫn nhận diện được. Khi không thể thống nhất, reviewer sẽ thực hiện escalation để quản lý dự án quyết định nhằm đảm bảo tính nhất quán cho toàn bộ tập dữ liệu.


3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance (Mẫu kitchen)
•	Trích dẫn một record từ segmentation_predictions.json (ví dụ đối tượng cái bát bowl đầu tiên trong ảnh):
o	instance_id: kitchen-002
o	class_name: bowl
o	score: 0.887612 (88.76%)
o	polygon_point_count (số điểm): 52 điểm
o	polygon_xy (một phần danh sách tọa độ): [[319.0, 311.0], [315.0, 315.0], [314.0, 315.0], [305.0, 324.0], [305.0, 325.0], ..., [322.0, 311.0]] (đơn vị: pixel)
•	Polygon bổ sung chi tiết gì so với hộp giới hạn (Bounding Box)?
o	Hộp giới hạn (Box) chỉ cung cấp thông tin thô dạng hình chữ nhật bao quanh vật thể (tọa độ biên ngoài cùng), dẫn đến việc chứa nhiều pixel của nền hoặc các vật thể khác xen vào ở góc hộp.
o	Đa giác (Polygon) cung cấp hình dạng hình học thực tế và chính xác từng pixel của biên vật thể. Nó giúp tách biệt hoàn toàn chủ thể khỏi nền, hiển thị rõ những đoạn cong, góc cạnh hoặc các phần lõm của đối tượng.
•	instance_id dùng để làm gì và không phải loại ID nào?
o	Mục đích: instance_id (ví dụ: kitchen-002) là định danh duy nhất được gán cho từng cá thể độc lập xuất hiện trong ảnh. Điều này giúp hệ thống phân biệt được có bao nhiêu thực thể riêng biệt (như đếm chính xác số lượng chiếc bát) ngay cả khi chúng cùng thuộc một lớp (bowl).
o	Không phải loại ID nào: Nó không phải là ID của lớp (class_id - ID định danh chung cho danh mục như 45 cho chiếc bát) và không phải là ID toàn cục duy nhất trên toàn bộ hệ thống lưu trữ dữ liệu (nó thường chỉ mang tính duy nhất cục bộ trong phạm vi bức ảnh đó).
•	Đề xuất một quy tắc biên mặt nạ (Mask Boundary Rule) rõ ràng:
o	Quy tắc: "Đường đa giác phải đi bám sát mép ranh giới pixel thực tế của vật thể. Số lượng điểm neo (vertices) của đa giác phải đủ để mô tả độ cong của vật thể mà không bị răng cưa, nhưng không được vượt quá 120 điểm cho mỗi đối tượng để tối ưu hóa hiệu năng lưu trữ. Không được để lộ pixel nền bên trong vùng mặt nạ của đối tượng."
•	Đối với vùng mờ, tiếp xúc hoặc bị che khuất, điều gì cần guideline hoặc escalation quyết định?
o	Vùng mờ do chuyển động hoặc lấy nét (Motion blur/Defocus): Guideline cần chỉ rõ reviewer nên vẽ đa giác bo theo phần mờ nhòe rộng nhất hay chỉ bo theo phần lõi vật thể có thể nhìn rõ rõ ràng.
o	Vùng tiếp xúc chồng lấn (Overlap/Contact): Cần quy định rõ ranh giới phân chia khi hai vật thể cùng màu và xếp sát nhau (như những chiếc bát chồng lên nhau). Reviewer sẽ cần quy tắc xác định xem điểm tiếp xúc đó thuộc về vật thể phía trước hay phía sau.
o	Sự cố leo thang (Escalation): Khi một vật thể bị che khuất quá mức hoặc bị mờ đến mức mắt người cũng không thể phân định được biên giới, reviewer cần dán nhãn tạm thời và thực hiện escalation để quản lý dự án (Project Manager) hoặc chuyên gia chuyên môn đưa ra quyết định cuối cùng nhằm đảm bảo tính nhất quán trên toàn bộ tập dữ liệu huấn luyện.


4. Vòng đời và kiểm tra chất lượng
Quy trình khép kín: Ảnh thô $\rightarrow$$\rightarrow$ Guideline $\rightarrow$$\rightarrow$ Ground truth $\rightarrow$$\rightarrow$ Huấn luyện $\rightarrow$$\rightarrow$ Prediction $\rightarrow$$\rightarrow$ QC / Rework.
Tác vụ	Đơn vị/định dạng ground truth	Lỗi hoặc điểm mơ hồ quan sát được	Annotator (Người gán nhãn) làm gì?	Reviewer (Người kiểm định) xem gì?
Phân loại ảnh (Image Classification)	Một nhãn duy nhất (như class_id, class_name) áp dụng cho toàn bộ bức ảnh.	- Ảnh có nhiều chủ thể: (Ví dụ mẫu traffic vừa có taxi, xe buýt, người đi bộ). Mô hình phân vân giữa cab (51%) và minibus (16%).
- Nhãn không đồng nhất: Thiếu quy định xem vật thể nào là chính.	- Xác định vật thể chính (dominant) hoặc bối cảnh tổng thể theo guideline ưu tiên.
- Chọn chính xác 1 nhãn duy nhất cho toàn bộ ảnh, bỏ qua các chi tiết phụ không đại diện cho ảnh.	- Kiểm tra xem nhãn được chọn có đúng quy định ưu tiên trong guideline không.
- Đảm bảo không gán nhãn dựa trên cảm tính cá nhân khi ảnh có quá nhiều đối tượng lẫn lộn.
Phát hiện vật thể (Object Detection)	Tọa độ hộp giới hạn [x_min, y_min, x_max, y_max] (đơn vị pixel) đi kèm với nhãn lớp của từng đối tượng.	- Bỏ sót vật thể: Mô hình bỏ sót các đồ vật nhỏ hoặc bị khuất ở ngưỡng cao (như cái cốc, chiếc thìa ở mức 0.60).
- Vẽ hộp không sát: Hộp bao gồm quá nhiều phần trống xung quanh vật thể.	- Xác định tất cả các đối tượng thuộc class list xuất hiện trong ảnh.
- Vẽ hộp giới hạn (Bounding Box) ôm khít sát đường biên thực tế ngoài cùng của từng đối tượng.	- Kiểm tra độ phủ (có bỏ sót đối tượng nào không).
- Kiểm tra độ khít (hộp có bị rộng quá, lệch quá hay bị cắt lẹm vào đối tượng không).
- Xác định đúng class của hộp.
Phân đoạn đối tượng (Instance Segmentation)	Danh sách các cặp tọa độ đa giác [[x1, y1], [x2, y2], ...] bao quanh chính xác ranh giới pixel và định danh thực thể duy nhất (instance_id).	- Biên không mượt: Điểm đa giác phân đoạn bị răng cưa hoặc gồ ghề (như phần viền cái bát bowl-002).
- Nhập nhèm ranh giới: Khó vẽ chính xác khi các vật thể chồng chéo hoặc bị bóng mờ che khuất.	- Bo viền theo sát từng pixel ranh giới thực của đối tượng.
- Tạo điểm neo đa giác vừa đủ để giữ độ mượt mà không làm quá tải lượng điểm (polygon point count).
- Phân tách rõ các instance riêng biệt dù xếp đè lên nhau.	- Đánh giá độ khớp của mặt nạ (Mask) so với thực tế vật thể.
- Đảm bảo không gán nhãn đè lấn bất hợp lý giữa hai vật thể độc lập.
- Kiểm tra xem các vùng góc khuất, bóng mờ có bị vẽ lấn sang nền hay không.


5. An toàn dữ liệu
•	Một quy tắc bảo vệ dữ liệu (Data Protection Rule) quan trọng:
o	Quy tắc: "Tuyệt đối không lưu trữ, sao chép, chia sẻ hay tải lên bất kỳ dữ liệu hình ảnh, tệp tin nội bộ nào của dự án lên các công cụ công cộng bên ngoài hệ thống (như mạng xã hội, kho lưu trữ cá nhân, các công cụ AI công cộng) khi chưa được cấp phép bằng văn bản từ đại diện dự án. Tất cả dữ liệu cá nhân nhạy cảm xuất hiện trong ảnh (như biển số xe rõ nét, khuôn mặt người đi đường cận cảnh, thông tin tài liệu cá nhân) phải được làm mờ (blur) hoặc che khuất hoàn toàn trước khi đưa vào các bước tiếp theo."
•	Nếu phát hiện ảnh hoặc dữ liệu không đúng phạm vi (Out-of-scope) hoặc nhạy cảm, bạn cần làm gì?
o	Giải pháp: "Nếu phát hiện hình ảnh hoặc dữ liệu không nằm trong phạm vi dự án, dữ liệu bị lỗi nghiêm trọng hoặc chứa thông tin nhạy cảm vi phạm quyền riêng tư, tôi sẽ lập tức dừng công việc trên mẫu dữ liệu đó, đánh dấu cờ lỗi (flag/tag) và báo cáo trực tiếp cho Quản lý Dự án (Project Manager / PM), Mentor học tập hoặc Trưởng nhóm Gán nhãn (Annotation Lead) để nhận hướng dẫn xử lý hoặc tiến hành loại bỏ mẫu ảnh đó khỏi tập dữ liệu."




Kết quả thực thi:
1.	Ghi tệp báo cáo thành công: Tệp REPORT.md đã được điền đầy đủ nội dung câu trả lời chuẩn xác và khoa học cho tất cả 6 phần nội dung được yêu cầu.
2.	Tự động liên kết Google Drive thành công: Hệ thống đã mount và xác thực tài khoản Google Drive của bạn.
3.	Đóng gói và Upload thành công: Gói nộp bài hoàn chỉnh dạng ZIP với tên định danh cá nhân hóa của bạn: /content/drive/MyDrive/AI20K-Day1/K5DAY02NguyenHoangNhan2A202603246-DAY01-report.zip đã được kiểm tra cấu trúc toàn vẹn và đẩy lên Google Drive lưu trữ an toàn.
Các tệp tin đã được xác thực an toàn bên trong tệp ZIP:
•	REPORT.md (Báo cáo tổng kết đầy đủ)
•	day1_lab_outputs/IMAGE_ATTRIBUTION.md (Nguồn gốc hình ảnh COCO)
•	day1_lab_outputs/classification_predictions.json (Dữ liệu dự đoán phân loại)
•	day1_lab_outputs/detection_predictions.json (Dữ liệu dự đoán bounding box)
•	day1_lab_outputs/segmentation_predictions.json (Dữ liệu dự đoán đa giác phân đoạn)
•	day1_lab_outputs/visuals/classification_top5.png (Hình ảnh phân tích phân loại)
•	day1_lab_outputs/visuals/detection_predictions.png (Hình ảnh trực quan phát hiện vật thể)
•	day1_lab_outputs/visuals/segmentation_prediction.png (Hình ảnh trực quan phân đoạn)

