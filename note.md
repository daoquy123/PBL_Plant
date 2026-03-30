# Ghi chú thay đổi (30-03-2026)

## 1) Thay đổi chia dữ liệu: thêm tập `test` với tỷ lệ 6/2/2

File chỉnh sửa: `prepare_dataset.py`

- Cập nhật pipeline từ 2 tập (`train/val`) thành 3 tập:
  - `train`: 60%
  - `val`: 20%
  - `test`: 20%
- Tạo thư mục đầu ra cho cả 3 split:
  - `dataset/train/<class>/`
  - `dataset/val/<class>/`
  - `dataset/test/<class>/`
- Bổ sung guard để hạn chế `test` bị rỗng:
  - Với lớp có từ 3 ảnh trở lên, cố giữ tối thiểu 1 ảnh cho `test`.
- Thêm log in ra số lượng mẫu theo từng class sau khi chia:
  - `train=...`, `val=...`, `test=...`

## 2) Thay đổi huấn luyện để bám yêu cầu nghiệp vụ

File chỉnh sửa: `app/ml/train_vgg16_cbam.py`

- Tăng số epoch:
  - `EPOCHS`: `30 -> 60`
- Thêm callback giảm learning rate khi validation loss chững:
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)`
- Đổi tiêu chí lưu trọng số tốt nhất:
  - từ `val_accuracy`
  - sang `val_recall_la_sau` (ưu tiên không bỏ sót lá sâu)
- Thêm metric theo dõi lớp `la_sau`:
  - `recall_la_sau`
  - `precision_la_sau`
- Thêm `cost-sensitive loss` để phạt nhầm lẫn bất đối xứng:
  - Phạt mạnh hơn lỗi `la_sau -> la_khoe` (x3.0)
  - Phạt nhẹ hơn lỗi `la_khoe -> la_sau` (x1.3)

## 3) Ý nghĩa thay đổi

- Tách rõ `train/val/test` giúp đánh giá mô hình khách quan hơn trên dữ liệu chưa thấy.
- Huấn luyện theo hướng ưu tiên an toàn: giảm nguy cơ dự đoán `la_sau` thành `la_khoe`.
- Tăng epoch + giảm learning rate theo plateau giúp mô hình có thêm cơ hội hội tụ ổn định.

## 4) Lệnh chạy lại đề xuất

```bash
python split_dataset_existing.py --dataset-root D:/Downloads/PBL5/dataset --source-splits train --dry-run
python split_dataset_existing.py --dataset-root D:/Downloads/PBL5/dataset --source-splits train
python app/ml/train_vgg16_cbam.py
```

## 5) Lưu ý

- Cần đánh giá lại bằng confusion matrix/F1/recall sau khi train mới để xác nhận:
  - recall lớp `la_sau` đã tăng chưa
  - số nhầm `la_sau -> la_khoe` đã giảm chưa

## 6) Bổ sung ngày 30-03-2026 (tách lại từ dataset hiện có)

- Thêm script mới: `split_dataset_existing.py`
- Mục đích: tách lại trực tiếp từ `dataset/train` hiện có sang 3 tập `train/val/test = 6/2/2` theo từng lớp.
- Đã chạy script và nhận được phân bố:
  - `train`: `la_khoe=299`, `la_vang=85`, `la_sau=299`, `sau=126`, `co=120`
  - `val`: `la_khoe=99`, `la_vang=28`, `la_sau=99`, `sau=42`, `co=40`
  - `test`: `la_khoe=101`, `la_vang=30`, `la_sau=101`, `sau=42`, `co=40`
- Notebook `notebooks/pbl5_bao_cao_charts.ipynb` đã chỉnh để:
  - Đánh giá Confusion Matrix/F1 trên `dataset/test`
  - Grad-CAM lấy mẫu từ `dataset/test`

## 7) Ghi chú cách chia đều kiểu stratified theo từng lớp

Ý tưởng giống `stratify=y` trong `train_test_split`: không trộn toàn bộ dataset rồi cắt ngẫu nhiên,
mà **chia riêng từng lớp** để giữ tỷ lệ lớp ổn định giữa `train/val/test`.

Với mỗi lớp `c`, gọi:
- `n_c`: tổng số ảnh lớp `c`
- `r_train = 0.6`, `r_val = 0.2`, `r_test = 0.2`

Ta tính:
- `n_train_c = floor(n_c * r_train)`
- `n_val_c = floor(n_c * r_val)`
- `n_test_c = n_c - n_train_c - n_val_c`

Sau đó ghép tất cả lớp lại:
- `train = union(train_c)`
- `val = union(val_c)`
- `test = union(test_c)`

Pseudo-code:

```python
for class_name in CLASS_NAMES:
    files = list_images_of_class(class_name)
    shuffle(files, seed=123)

    n_total = len(files)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    save_to_split("train", class_name, train_files)
    save_to_split("val", class_name, val_files)
    save_to_split("test", class_name, test_files)
```

Lợi ích:
- Giữ phân bố lớp gần với dữ liệu gốc ở cả 3 tập.
- Tránh trường hợp lớp hiếm bị thiếu trong val/test.
- Kết quả đánh giá (đặc biệt confusion matrix/F1) ổn định và đáng tin hơn.

## 8) Cảnh báo quan trọng khi chia lại dataset

- Không chạy lặp lại lệnh split nhiều lần trên cùng dữ liệu nếu không cần.
- Script `split_dataset_existing.py` đã bổ sung `--source-splits`:
  - Sau khi reset về dataset gốc: dùng `--source-splits train` để tránh trộn dữ liệu cũ từ `val/test`.
- Luôn chạy `--dry-run` trước để kiểm tra số lượng, sau đó mới chạy thật.

## 9) Sửa lỗi evaluate test bị thiếu lớp `co`

- Vấn đề gặp phải:
  - Trong `dataset/test/co` ảnh là `.tif` (40 ảnh).
  - `tf.keras.utils.image_dataset_from_directory` có thể bỏ qua/không decode `.tif` -> báo `support=0` cho lớp `co`.
- Cách sửa:
  - Cell evaluate trong `notebooks/pbl5_bao_cao_charts.ipynb` đã chuyển sang decode bằng PIL (`load_image_rgb_from_path`) để đọc được `.tif`.
  - Confusion matrix/F1 trên test sau sửa đã phản ánh đủ 5 lớp.

## 10) Cập nhật tối ưu train (2-stage fine-tune)

File chỉnh sửa: `app/ml/train_vgg16_cbam.py`

- Chuyển sang huấn luyện 2 giai đoạn:
  - Stage 1: train head khi VGG16 frozen (`EPOCHS_STAGE1 = 30`).
  - Stage 2: fine-tune `block5` của VGG16 với LR thấp (`EPOCHS_STAGE2 = 20`, `Adam(2e-5)`).
- Thêm augmentation nhẹ cho train (`flip`, `brightness`, `contrast`).
- Giữ tiêu chí checkpoint theo `val_recall_la_sau`.

Kết quả run gần nhất (theo log):
- Best `val_recall_la_sau`: `0.86869` (epoch 34, đã lưu vào `vgg16_cbam_best.weights.h5`).
- Ở stage 2, train accuracy tăng nhanh hơn val accuracy (dấu hiệu fine-tune mạnh), nhưng checkpoint vẫn lấy epoch tốt nhất theo mục tiêu nghiệp vụ.
