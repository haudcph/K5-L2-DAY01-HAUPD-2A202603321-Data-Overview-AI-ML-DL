# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/nov/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K 
    - class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K 


2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    - Record này mô tả toàn bộ bức ảnh, không phải một vật thể riêng lẻ. Mô hình phân loại ảnh đưa ra nhãn ở cấp ảnh; record không xác định vị trí hay tạo bounding box cho từng vật thể.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    - Danh sách tên lớp đi kèm taxonomy/checkpoint đã huấn luyện; model không tự nghĩ ra tên lớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    - Để xác định lớp chính xác: `class_id` là mã lớp, `class_name` là tên dễ đọc, còn `taxonomy_name` cho biết lớp thuộc bộ nhãn nào. Tên lớp có thể trùng hoặc khác cách gọi giữa các taxonomy.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
	- Guideline cần nêu tiêu chí chọn lớp chính (ví dụ vật thể nổi bật nhất hoặc chủ đề trung tâm) và cách xử lý khi không thể xác định rõ; trường hợp mơ hồ thì hỏi reviewer/Lab Coach, không tự đoán.

6. Vì sao không được lấy `score` của model làm ground truth?
    - `score` chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không xác nhận dự đoán đó đúng. Ground truth phải được gán theo guideline và kiểm tra bởi người có trách nhiệm.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
    - `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? 
    - Box có góc trên-trái tại (385.33, 69.24), góc dưới-phải tại (498.92, 348.92), rộng 113.58 pixel, cao 279.68 pixel.
    
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    - 0.20 có 17 prediction, 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    - Reviewer cần kiểm tra thêm các hộp mới được giữ lại, loại false positive và hộp trùng, đồng thời xác nhận lớp và vị trí hộp; các trường hợp khó cần được escalte để làm rõ.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    - Vẽ box nhỏ nhất bao trọn phần nhìn thấy của vật thể; không gộp vật thể khác vào box. Với phần bị che khuất hoặc nằm ngoài khung hình, chỉ đánh dấu phần nhìn thấy và áp dụng nhất quán theo guideline.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Kiểm tra đối tượng `person` trong ảnh `kitchen` (box từ x=385.33 đến 498.92, y=69.24 đến 348.92) để xác định phần nào bị che khuất hoặc cắt mép. Nếu ranh giới phần nhìn thấy không rõ, đưa trường hợp này lên hỏi Lab Coach để thống nhất cách gán; không tự suy đoán phần bị khuất.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    - instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[53.0,344.0],[52.0 345.0],[50.0,345.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - Polygon bám theo đường biên của vật thể nên thể hiện hình dạng và phần lồi/lõm chi tiết hơn. Box chỉ cho biết khung chữ nhật bao quanh, có thể chứa cả vùng nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - `instance_id` phân biệt từng đối tượng được gán nhãn riêng trong kết quả, kể cả khi nhiều đối tượng cùng lớp. Nó không cho biết tên lớp như class ID và cũng không theo dõi cùng một đối tượng qua nhiều khung hình như tracking ID.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    - Vẽ mask sát theo đường biên phần vật thể nhìn thấy, không lấy vùng nền; tại chỗ bị che, chỉ đánh dấu phần nhìn thấy và áp dụng cùng một quy tắc cho mọi ảnh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Kiểm tra đường biên của chiếc bowl, nhất là đoạn sát vật thể hoặc nền xung quanh. Nếu ảnh không cho thấy rõ đường biên ở đoạn đó, tôi sẽ escalation lên Lab Coach để thống nhất cách xử lý thay vì tự đoán phần bị che.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho mỗi ảnh, theo taxonomy và guideline. | Ảnh `traffic` có top-1 `cab` (class_id 468, score 0.510915); ảnh có thể có nhiều vật/chủ đề nên lớp đại diện có thể mơ hồ. | Chọn một lớp theo tiêu chí trong guideline; không dùng score làm ground truth, ca mơ hồ thì hỏi. | Kiểm tra lớp có phù hợp ảnh và taxonomy, guideline được áp dụng nhất quán. |
| Phát hiện vật thể | Một nhãn lớp và một bbox `[x1,y1,x2,y2]` cho mỗi vật thể. | Ảnh `kitchen` có 17 prediction ở ngưỡng 0.20 nhưng 6 ở 0.60; cần xem các hộp điểm thấp, hộp trùng và vật bị che/cắt mép. | Gán lớp và vẽ box quanh phần nhìn thấy theo quy tắc thống nhất; không tự đoán phần khuất. | Kiểm tra bỏ sót/false positive, lớp và độ khít của box; rà hộp trùng và trường hợp khó. |
| Instance segmentation | Một nhãn lớp và một mask/polygon cho từng đối tượng riêng. | JSON có instance `kitchen-002` là `bowl`, polygon 67 điểm; biên mờ, vật chạm nhau hoặc bị che có thể khiến ranh giới khó xác định. | Vẽ mask theo biên phần nhìn thấy, tách từng instance và theo guideline; trường hợp không rõ thì hỏi. | Kiểm tra mask có bám biên, không lấy nền, instance/lớp đúng và cách xử lý vùng che khuất nhất quán. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu được phép cho bài lab; không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào báo cáo/output, và không chia sẻ dữ liệu ngoài phạm vi được phê duyệt.
2. Tôi sẽ dừng xử lý, không tải lên hoặc chia sẻ dữ liệu đó, rồi báo cho Lab Coach hoặc giảng viên phụ trách.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
