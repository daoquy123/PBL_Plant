# So sánh chi tiết mô hình `VGG16+CBAM` và `ResNet50`

Tài liệu này tổng hợp khác biệt kiến trúc, quy trình huấn luyện và kết quả đánh giá giữa hai mô hình CNN dùng cho bài toán phân loại 5 lớp:
`la_khoe`, `la_vang`, `la_sau`, `sau`, `co`.

---

## 1. Mục tiêu so sánh

- Đánh giá việc **đổi backbone** từ `VGG16` sang `ResNet50` có cải thiện hiệu năng hay không.
- Giữ cùng khung thực nghiệm để công bằng: cùng dữ liệu, cùng cách split, cùng pipeline train/eval.
- Kiểm tra trade-off nghiệp vụ:
  - ưu tiên **không bỏ sót lá sâu** (recall `la_sau` cao),
  - hay ưu tiên **tổng thể ổn định, ít báo giả** (accuracy/macro-F1/precision cao).

---

## 2. Khác nhau về kiến trúc

## 2.1. `VGG16+CBAM`

Pipeline:

`Input -> VGG16 (freeze) -> CBAM (channel + spatial) -> GAP -> Dense(256)+Dropout -> Softmax(5)`

Đặc trưng chính:

- Dùng attention `CBAM` để tăng khả năng tập trung vào vùng/cụm đặc trưng quan trọng.
- Backbone VGG16 ít cơ chế hỗ trợ gradient hơn ResNet khi mạng sâu.
- Mạnh về hướng giải thích (visual attention/Grad-CAM).

## 2.2. `ResNet50`

Pipeline:

`Input -> ResNet50 (freeze/fine-tune) -> GAP -> Dense(256)+Dropout -> Softmax(5)`

Đặc trưng chính:

- Có `Residual Connection`: học hàm dư `F(x)` và cộng tắt `x`, giúp train mạng sâu ổn định hơn.
- Thường cho đặc trưng tổng quát tốt hơn trên nhiều bài toán transfer learning.
- Không có attention tường minh như CBAM (trừ khi bật biến thể `ResNet50+CBAM`).

## 2.3. `ResNet50+CBAM` (biến thể mở rộng)

Pipeline:

`Input -> ResNet50 -> CBAM -> GAP -> Dense -> Softmax`

Ý nghĩa:

- Kết hợp ưu điểm gradient flow của ResNet với khả năng focus vùng quan trọng của CBAM.
- Hợp cho vòng thử nghiệm tiếp theo nếu cần tăng độ thuyết phục trong báo cáo.

---

## 3. Thiết lập thực nghiệm đã dùng

- Dữ liệu đánh giá test: **314 ảnh**.
- Các chỉ số chính: `accuracy`, `macro_f1`, `precision_la_sau`, `recall_la_sau`.
- Huấn luyện theo hướng cost-sensitive để phản ánh ưu tiên nghiệp vụ lớp `la_sau`.
- Với script evaluate mới, có tune ngưỡng xác suất cho `la_sau` trên tập `val` rồi áp dụng sang `test`:
  - `VGG16+CBAM`: threshold `la_sau = 0.43`
  - `ResNet50`: threshold `la_sau = 0.47`

---

## 4. Kết quả định lượng (test)

Nguồn số liệu:

- `reports/vgg16_cbam_metrics_summary.json`
- `reports/resnet50_metrics_summary.json`

| Metric | VGG16+CBAM | ResNet50 | Delta (ResNet - VGG) |
|---|---:|---:|---:|
| Accuracy | 0.8025 | 0.8439 | +0.0414 |
| Macro F1 | 0.8343 | 0.8609 | +0.0266 |
| Recall `la_sau` | 0.9109 | 0.7723 | -0.1386 |
| Precision `la_sau` | 0.6815 | 0.8478 | +0.1663 |

Nhận xét nhanh:

- `ResNet50` vượt trội ở `accuracy`, `macro_f1`, `precision_la_sau`.
- `VGG16+CBAM` vượt trội ở `recall_la_sau`.
- Hai mô hình thể hiện trade-off rõ ràng giữa **an toàn cảnh báo** và **độ chính xác tổng thể**.

---

## 5. So sánh theo từng lớp (F1 trên test)

`VGG16+CBAM`:

- `la_khoe`: 0.6744
- `la_vang`: 0.7407
- `la_sau`: 0.7797
- `sau`: 0.9767
- `co`: 1.0000

`ResNet50`:

- `la_khoe`: 0.7926
- `la_vang`: 0.7037
- `la_sau`: 0.8083
- `sau`: 1.0000
- `co`: 1.0000

Diễn giải:

- `ResNet50` cải thiện rõ ở `la_khoe` và `la_sau`.
- `VGG16+CBAM` tốt hơn nhẹ ở `la_vang`.
- Hai lớp `sau`, `co` đều rất tốt ở cả hai mô hình.

---

## 6. Ý nghĩa nghiệp vụ

Nếu ưu tiên **không bỏ sót lá sâu**:

- Chọn `VGG16+CBAM` vì recall `la_sau` cao hơn.
- Chấp nhận nhiều cảnh báo dương tính giả hơn (`precision_la_sau` thấp hơn).

Nếu ưu tiên **vận hành ổn định, ít báo giả**:

- Chọn `ResNet50` vì accuracy/macro-F1 cao hơn và precision `la_sau` cao hơn nhiều.
- Chấp nhận giảm một phần khả năng bắt hết mẫu `la_sau`.

Kết luận thực dụng:

- Với nhu cầu cân bằng tổng thể cho báo cáo kỹ thuật, `ResNet50` là baseline mạnh.
- Với nhu cầu cảnh báo sớm thiên về an toàn, `VGG16+CBAM` vẫn có lợi thế.

---

## 7. Hạn chế và hướng cải thiện

- Dữ liệu lớp `la_vang` còn ít hơn các lớp chính nên dao động chỉ số cao hơn.
- Một phần ảnh `la_vang` và `la_sau` có biểu hiện trực quan chồng lấp, gây khó tách rạch ròi trong bài toán single-label.
- Hướng cải thiện:
  - thử `ResNet50+CBAM`,
  - cân nhắc multi-label nếu nhãn thực địa cho phép `la_vang` và `la_sau` cùng đúng,
  - chạy nhiều seed và báo cáo mean/std để tăng độ tin cậy.

---

## 8. Danh sách hình dùng cho slide

- Kiến trúc ResNet50: `reports/figures/08_resnet50_architecture.png`
- Kiến trúc ResNet50+CBAM: `reports/figures/09_resnet50_cbam_architecture.png`
- Minh họa residual block: `reports/figures/10_resnet50_residual_block.png`
- So sánh chỉ số hai mô hình: `reports/figures/11_vgg_vs_resnet_metrics.png`
- Bộ chart đánh giá riêng từng mô hình:
  - `reports/figures/vgg16_cbam_04_loss_accuracy.png` ... `vgg16_cbam_07_gradcam_samples.png`
  - `reports/figures/resnet50_04_loss_accuracy.png` ... `resnet50_07_gradcam_samples.png`
