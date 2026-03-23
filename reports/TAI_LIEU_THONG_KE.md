# TÀI LIỆU THỐNG KÊ - DỰ ÁN PBL5
## Hệ thống phân loại bệnh lá cây sử dụng Deep Learning

---

## 1. IOT (Internet of Things)
*Phần này cần được bổ sung dựa trên phần cứng thực tế của dự án*

Hệ thống IoT bao gồm:
- Camera thu thập ảnh lá cây
- Module xử lý trung tâm (Raspberry Pi/Arduino)
- Kết nối với server xử lý AI
- Hiển thị kết quả và cảnh báo

---

## 2. Nguyên liệu và Phương pháp nghiên cứu

### 2.1. Thu thập và tiền xử lý dữ liệu

Nghiên cứu này xây dựng một tập dữ liệu phân loại bệnh lá cây, với 5 classes (nhãn):

**Cấu trúc dataset:**
```
dataset/
├── train/          # Dữ liệu huấn luyện (đếm lại: 1,551 ảnh)
│   ├── la_khoe/    # 499 ảnh
│   ├── la_vang/    # 143 ảnh (bao gồm nhiều file .HEIC)
│   ├── la_sau/     # 499 ảnh
│   ├── sau/        # 210 ảnh
│   └── co/         # 200 ảnh
├── val/            # Hiện tại rỗng (0 ảnh)
│   ├── co/
│   ├── la_khoe/
│   ├── la_sau/
│   ├── la_sau_va_vang/  # thư mục thừa, không thuộc CLASS_NAMES
│   └── la_vang/
└── yolo/           # thư mục phụ, không dùng trong train_vgg16_cbam.py
```

**Thống kê dữ liệu:**
- Tổng số ảnh training: `1,551` ảnh
- Tổng số ảnh validation: `0` ảnh (thư mục có nhưng chưa có file ảnh)
- Tổng cộng hiện có thể dùng: `1,551` ảnh

**Phân bố dữ liệu theo lớp:**
1. `la_khoe` (lá khỏe): `499` ảnh (~32.17%)
2. `la_vang` (lá vàng): `143` ảnh (~9.22%)
3. `la_sau` (lá sâu bệnh): `499` ảnh (~32.17%)
4. `sau` (sâu): `210` ảnh (~13.54%)
5. `co` (cỏ/nền): `200` ảnh (~12.89%)

#### 2.1.1. Nguồn dữ liệu và Thành phần tập dữ liệu

Tập dữ liệu được thu thập từ:
- **Dữ liệu thực tế tự thu thập**: Chụp ảnh trực tiếp lá cây trong môi trường thực
- **Cân bằng dữ liệu khi huấn luyện (trong code)**: không dùng augmentation online; thay vào đó sử dụng chia train/val theo từng lớp (stratified split) và truyền `class_weight` vào quá trình `model.fit()`

*Lưu ý sau khi đếm lại: class `la_vang` đã tăng lên 143 ảnh (bao gồm 105 file `.HEIC`), nhưng vẫn thấp hơn `la_khoe`/`la_sau` (499 ảnh/lớp).*

#### 2.1.2. Tiền xử lý và Tăng cường dữ liệu

Các kỹ thuật được áp dụng:

**Preprocessing chuẩn:**
- Đọc ảnh bằng PIL, chuyển sang chế độ `RGB`
- Resize về `IMG_SIZE = (224, 224)` (dùng `Image.BILINEAR`)
- Ép kiểu ảnh thành `float32`
- Không chuẩn hóa thủ công về `[0, 1]`; sau đó model áp dụng `tf.keras.applications.vgg16.preprocess_input()` trước khi đưa vào backbone VGG16

**Thống kê định dạng file (train):**
- `.jpg`: `606`
- `.jpeg`: `640`
- `.tif`: `200`
- `.heic`: `105`

*Lưu ý quan trọng để train lại:* hiện tại dữ liệu có `.HEIC`, cần đảm bảo pipeline đọc được HEIC (hoặc chuyển HEIC sang JPG/PNG trước khi train) để tránh bỏ sót ảnh.

**Tăng cường dữ liệu:**
- Pipeline huấn luyện hiện tại không dùng augmentation online (không flip/rotate/zoom/brightness/contrast)
- “Tăng” cho lớp hiếm được thực hiện bằng:
  - `stratified split` để đảm bảo lớp hiếm xuất hiện trong validation
  - `class_weight` để tăng trọng số đóng góp của lớp hiếm vào loss

**Stratified splitting:**
- `VAL_SPLIT = 0.2`
- Nếu `dataset/val` không có ảnh, code sẽ tự tách validation theo từng lớp từ `dataset/train`
- Với mỗi lớp: `n_val = ceil(n_total * VAL_SPLIT)` rồi clamp để `val` không rỗng

Nếu tách stratified trực tiếp từ `train` hiện tại (1,551 ảnh, `VAL_SPLIT=0.2`) thì ước tính:
- `train ≈ 1240` ảnh
- `val ≈ 311` ảnh

Chi tiết từng lớp (ước tính theo `ceil(0.2 * n_c)`):
- `la_khoe`: val 100, train 399
- `la_vang`: val 29, train 114
- `la_sau`: val 100, train 399
- `sau`: val 42, train 168
- `co`: val 40, train 160

### 2.2. Lựa chọn và kiến trúc mô hình

Nghiên cứu này lựa chọn kiến trúc **VGG16 + CBAM (Convolutional Block Attention Module)** để phân loại bệnh lá cây.

#### Kiến trúc tổng thể:

```
Input (224×224×3) 
    → VGG16 Backbone (pretrained ImageNet, freeze)
    → CBAM Module (Channel + Spatial Attention)
    → Global Average Pooling
    → Dense(256) + Dropout
    → Softmax(5 classes)
```

#### Chi tiết các thành phần:

**1. VGG16 Backbone:**
- Pretrained trên ImageNet
- Freeze các layer để giữ nguyên feature extraction đã học
- Extract features từ các convolutional blocks

**2. CBAM (Convolutional Block Attention Module):**

CBAM bao gồm 2 module attention tuần tự:

**a) Channel Attention:**
```
Features (H×W×C)
    → AvgPool(spatial) và MaxPool(spatial) 
    → Shared MLP (reduction ratio r=8)
    → Sigmoid activation
    → Element-wise multiply với Features
    → Output: Channel-refined features
```

*Mục đích: Học "cái gì quan trọng" - channel nào chứa thông tin quan trọng*

**b) Spatial Attention:**
```
Channel-refined features
    → AvgPool(channel-wise) và MaxPool(channel-wise)
    → Concatenate
    → Conv2D 7×7
    → Sigmoid activation  
    → Element-wise multiply
    → Output: Spatially-refined features
```

*Mục đích: Học "ở đâu quan trọng" - vùng nào trong ảnh cần chú ý*

**3. Classification Head:**
- Global Average Pooling: Giảm spatial dimensions
- Dropout(0.4) + Dense(256, activation="relu") + Dropout(0.4): học patterns phức tạp và regularization
- Dense(5) + Softmax: Output probabilities cho 5 classes

#### Ưu điểm của kiến trúc VGG16 + CBAM:

1. **Transfer Learning hiệu quả**: VGG16 pretrained cung cấp features chất lượng cao
2. **Attention mechanism**: CBAM giúp model tập trung vào vùng bệnh quan trọng
3. **Lightweight**: CBAM thêm parameters rất ít (~0.1% so với backbone)
4. **Interpretability**: Có thể visualize attention maps để hiểu model học gì

### 2.3. Huấn luyện và tinh chỉnh siêu tham số

#### Cấu hình training:

**Hardware:**
- GPU: NVIDIA (cụ thể tùy môi trường - Google Colab/Local)
- Framework: TensorFlow/Keras

**Hyperparameters:**
```python
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
OPTIMIZER = Adam(learning_rate=1e-4)
```

**Training strategy:**
- **Loss function**: `sparse_categorical_crossentropy`
- **Metrics** (trong quá trình train): `accuracy` (precision/recall/F1 được tính riêng ở notebook khi suy luận trên tập `val`)
- **Early Stopping**: monitor `val_loss`, `patience=5`, `restore_best_weights=True`
- **Model Checkpoint**: monitor `val_accuracy`, `save_best_only=True`, `save_weights_only=True`

**Class Imbalance Handling:**
Do sự mất cân bằng dữ liệu (đặc biệt `la_vang` thấp hơn nhiều so với `la_khoe`/`la_sau`), huấn luyện dùng `class_weight`.

Trong `train_vgg16_cbam.py`, trọng số được tính theo:
`class_weight[c] = total / (K * n_c)`

Trong đó:
- `K = 5` (số lớp)
- `n_c` là số ảnh của class `c` trong `dataset/train`
- `total` là tổng số ảnh của tất cả các lớp trong `dataset/train`

Với số lượng đã đếm lại (train = 1,551 ảnh), nếu dùng đầy đủ ảnh để tính thì:
- `la_khoe`: `0.621`
- `la_vang`: `2.169`
- `la_sau`: `0.621`
- `sau`: `1.477`
- `co`: `1.551`

*Lưu ý:* các giá trị `0.58 / 7.611 / ...` trong lần train cũ được tạo ra từ tập ảnh ít hơn (khi HEIC bị bỏ sót).

### 2.4. Evaluation và Deployment

**Đánh giá mô hình:**
- Confusion Matrix: Xem chi tiết classification errors
- Classification Report: Precision, Recall, F1 cho từng class
- Grad-CAM visualization: Hiển thị vùng model chú ý

**Export model:**
- Lưu trọng số best model tại `app/checkpoints/vgg16_cbam_best.weights.h5`
- Script training cũng thử lưu model đầy đủ dạng `.keras` (nếu không gặp lỗi khi có `Lambda layer`)
- Triển khai hiện tại qua API `FastAPI` (`/api/predict`) để suy luận bằng `LeafHealthPredictor`

---

## 3. Kết quả

### 3.1. Kết quả huấn luyện

*Kết quả dưới đây là từ lần train trước đó (dựa trên `training_history.json` và notebook `pbl5_bao_cao_charts.ipynb`). Sau khi bổ sung HEIC và train lại, các chỉ số này cần cập nhật lại.*

**File kết quả**: `app/checkpoints/training_history.json`

**Training curves** (Loss & Accuracy):
- Vẽ đồ thị training/validation loss theo epochs
- Vẽ đồ thị training/validation accuracy theo epochs

**Observations:**
- Huấn luyện chạy `EPOCHS = 30` (số điểm trong `training_history.json` = 30)
- `val_accuracy` tốt nhất: `0.7973` ở epoch `29` (1-based)
- `val_loss` tốt nhất (thấp nhất): `0.4777` ở epoch `30` (1-based)

### 3.2. Kết quả trên tập validation

**Confusion Matrix:**
```
Confusion matrix được tạo và lưu tại `reports/figures/05_confusion_matrix.png` (ma trận chuẩn hoá theo hàng + số lượng trong từng ô).
```

**Classification Report:**
```
              precision    recall  f1-score   support

    la_khoe       0.684     0.780     0.729       100
    la_vang       0.222     0.250     0.235         8
     la_sau       0.812     0.690     0.746       100
        sau       1.000     1.000     1.000        42
         co       1.000     1.000     1.000        40

   accuracy                           0.797       290
  macro avg       0.744     0.744     0.742      290
weighted avg     0.805     0.797     0.798      290
```

**F1-Score per class:**
- Biểu đồ cột F1 theo từng lớp được lưu tại `reports/figures/06_f1_per_class.png`
- `la_vang` là class có F1 thấp nhất do support ít trong tập `val` (chỉ 8 mẫu).

### 3.3. Grad-CAM Visualization

**Ví dụ Grad-CAM cho mỗi class:**
- Hiển thị ảnh gốc
- Hiển thị heatmap attention (vùng model chú ý)
- Overlay heatmap lên ảnh gốc
- Notebook lưu mẫu Grad-CAM tại `reports/figures/07_gradcam_samples.png`

**Nhận xét:**
- Model có focus vào đúng vùng bệnh không?
- Có attention spurious (nhầm lẫn) không?

### 3.4. Phân tích kết quả

### 3.5. Logic 2-stage hợp lý (VGG16+CBAM -> YOLO)

VGG16 + CBAM **không phải** mô hình detection theo nghĩa YOLO. Vai trò phù hợp:

- Trích xuất đặc trưng và phân tích ngữ cảnh tổng quát của ảnh.
- Làm nổi bật vùng giàu thông tin qua attention (CBAM).
- Đưa ra đánh giá/phân loại sơ bộ để phát hiện ảnh có dấu hiệu bất thường.

YOLO là bước detection chi tiết:

- Định vị đối tượng/vùng cụ thể bằng bounding boxes.
- Chỉ rõ vị trí: lá vàng do hư hại, vùng sâu hại/lỗ sâu, con sâu, cỏ.
- Trực quan hoá kết quả để giải thích và hỗ trợ quyết định xử lý.

Pipeline đề xuất cho báo cáo:

1. **Bước 1 - VGG16+CBAM (sơ bộ):** đánh giá tổng thể + attention map.
2. **Bước 2 - YOLO (chi tiết):** detection + vị trí cụ thể để diễn giải.

Nói ngắn: **VGG16+CBAM = nhận thức ngữ cảnh**, **YOLO = định vị chi tiết**.

**Điểm mạnh:**
- Model phân loại tốt các class với nhiều dữ liệu (`la_khoe`, `la_sau`)
- CBAM attention giúp tăng interpretability

**Điểm yếu:**
- Class `la_vang` là lớp hiếm (support 8 trong `val`) nên precision/recall thấp.

**Hướng cải thiện:**
1. **Tăng dữ liệu đặc biệt cho `la_vang`**: thu thập thêm từ nhiều điều kiện ánh sáng/góc chụp
2. **Fine-tune VGG16**: unfreeze một số block cuối (hiện tại backbone đang freeze)
3. **Thử augmentation (nếu cần)**: bổ sung flip/rotate/brightness/contrast để tăng đa dạng hình ảnh (pipeline train hiện tại chưa có augmentation online)

---

## 4. Kết luận

### Tóm tắt đóng góp:

1. **Dataset**: Tập dữ liệu 5 classes sau khi đếm lại có thể dùng để train là 1,551 ảnh trong `dataset/train` (kèm HEIC)
2. **Model**: Triển khai VGG16 + CBAM với Transfer Learning
3. **Attention mechanism**: Áp dụng CBAM để tăng interpretability
4. **Evaluation**: Đánh giá toàn diện với Confusion Matrix, Classification Report, Grad-CAM

### Tiềm năng ứng dụng:

- **Nông nghiệp thông minh**: Tự động phát hiện bệnh lá cây trong real-time
- **IoT integration**: Kết hợp camera + edge computing để cảnh báo sớm
- **Web/Chatbot**: Deploy qua API `FastAPI` và giao diện Gradio (chatbot chẩn đoán)

### Các bước tiếp theo:

1. **Collect thêm data**: Đặc biệt cho class `la_vang`
2. **Hyperparameter tuning**: Tìm best learning rate, batch size, augmentation params
3. **Deploy IoT system**: Tích hợp với hardware thực tế
4. **Field test**: Test model trong môi trường thực tế với nhiều điều kiện ánh sáng, góc chụp khác nhau

---

## Phụ lục

### A. Cấu trúc thư mục project

```
PBL5/
├── app/
│   ├── ml/
│   │   ├── model_vgg16_cbam.py      # Định nghĩa model
│   │   ├── train_vgg16_cbam.py      # Script training
│   │   └── reporting.py             # Evaluation utilities
│   ├── checkpoints/
│   │   ├── training_history.json    # History training
│   │   └── vgg16_cbam_best.weights.h5 # Best weights
├── dataset/
│   ├── train/                       # Training data
│   └── val/                         # Validation data
├── notebooks/
│   └── pbl5_bao_cao_charts.ipynb    # Notebook visualization
└── reports/
    ├── figures/                     # Saved charts
    └── TAI_LIEU_THONG_KE.md        # Tài liệu này
```

### B. Chạy training

```bash
# Activate venv
source .venv/bin/activate  # hoặc .venv\Scripts\activate (Windows)

# Chạy full pipeline một lần:
# prepare data -> train VGG16+CBAM -> eval VGG -> train YOLO (nếu có data.yaml)
python run_full_pipeline.py

# Nếu chưa có dataset detection YOLO:
python run_full_pipeline.py --skip-yolo-train

# Visualization (notebook)
jupyter notebook notebooks/pbl5_bao_cao_charts.ipynb
```

### C. Dependencies chính

```
tensorflow>=2.12.0
keras
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

---

**Ngày tạo tài liệu**: [Điền ngày]
**Người thực hiện**: [Điền tên]
**Version**: 1.0
