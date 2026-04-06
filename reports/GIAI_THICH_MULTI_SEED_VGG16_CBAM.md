# Giải thích multi-seed cho VGG16+CBAM

## 1) Mục tiêu chạy nhiều seed

Trong huấn luyện Deep Learning, kết quả của **một lần chạy** có thể dao động do yếu tố ngẫu nhiên.
Vì vậy nhóm chạy nhiều `random_state/seed` để:

- đánh giá độ ổn định của mô hình,
- tránh kết luận dựa trên một run may mắn hoặc run xấu,
- báo cáo theo dạng **TB (std)** thay vì một con số đơn lẻ.

Trong thí nghiệm này, mô hình chạy với seed: `118, 119, 120, 121, 122, 123, 124, 125`.

---

## 2) Seed làm thay đổi những gì?

Khi đổi seed, dữ liệu và code giữ nguyên, nhưng các thành phần ngẫu nhiên thay đổi:

- thứ tự shuffle dữ liệu trong train,
- cách chia train/val khi fallback stratified split,
- thứ tự batch theo epoch,
- biến đổi augmentation ngẫu nhiên (flip/brightness/contrast),
- quỹ đạo tối ưu khi fine-tune (do RNG của Python/NumPy/TensorFlow).

=> Mỗi seed tạo ra một **quỹ đạo học khác nhau**, nên metric có thể dao động.

---

## 3) Kết quả tổng hợp 8 seed

Nguồn: `reports/MULTI_SEED_TL_SUMMARY.md`

- Accuracy TB(std): `0.7516 (0.1080)`
- Macro-F1 TB(std): `0.7684 (0.1467)`
- Recall `la_sau` TB(std): `0.8800 (0.0267)`
- Precision `la_sau` TB(std): `0.6358 (0.0958)`
- TL score TB(std): `0.7455 (0.0959)`

Với công thức:

`TL score = 0.65 * f1_la_sau + 0.35 * macro_f1`

---

## 4) Vì sao seed 125 bị outlier?

Seed `125` có kết quả thấp bất thường:

- Accuracy: `0.4904`
- Macro-F1: `0.4083`
- F1 theo lớp:
  - `la_khoe`: `0.0196`
  - `la_vang`: `0.0000`
  - `la_sau`: `0.5706`
  - `sau`: `0.6349`
  - `co`: `0.8163`

Diễn giải:

- Mô hình bị lệch mạnh theo hướng dự đoán sang `la_sau` (recall `la_sau` cao nhưng precision thấp),
- hai lớp lá còn lại (`la_khoe`, `la_vang`) bị sụt nặng,
- đây là một **run xấu (outlier)** do quỹ đạo tối ưu theo seed, không phải lỗi code bị crash.

---

## 5) Kết luận dùng trong báo cáo

- Mô hình VGG16+CBAM có xu hướng giữ được recall `la_sau` khá ổn định qua nhiều seed.
- Tuy nhiên chỉ số tổng thể còn dao động, cho thấy độ nhạy theo seed vẫn đáng kể.
- Vì vậy kết luận cuối nên dựa trên **multi-seed TB(std)**, không dựa trên 1 lần train.

Khuyến nghị trình bày với giảng viên:

1. Báo kết quả TB(std) của 8 seed (chính thức).
2. Nêu rõ seed 125 là outlier và giải thích cơ chế lệch dự đoán.
3. Nếu cần thêm độ tin cậy, báo thêm median/IQR hoặc chạy thêm seed.

---

## 6) Tệp liên quan

- Tổng hợp chính: `reports/MULTI_SEED_TL_SUMMARY.md`
- Dữ liệu chi tiết: `reports/multi_seed_tl_summary.json`
- Kết quả từng seed:
  - `reports/vgg16_cbam_seed118_metrics_summary.json`
  - ...
  - `reports/vgg16_cbam_seed125_metrics_summary.json`
