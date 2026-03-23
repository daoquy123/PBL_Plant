# LeafCare AI – Phân tích lá cải bằng VGG16 + CBAM

Hệ thống demo AI giúp phân tích **tình trạng lá cải** (lá khỏe, lá vàng, lá sâu, lá sâu và vàng) từ ảnh, sử dụng **VGG16 + CBAM** theo đúng tài liệu đề tài của bạn, kèm UI web đơn giản kiểu chat.

## Cấu trúc thư mục

- `requirements.txt` – danh sách thư viện Python cần cài.
- `app/`
  - `main.py` – FastAPI backend, cung cấp:
    - `GET /` – giao diện web.
    - `POST /api/predict` – API nhận ảnh và trả kết quả phân tích.
  - `ml/model_vgg16_cbam.py` – định nghĩa mô hình VGG16 + CBAM.
  - `ml/predictor.py` – wrapper nạp model và dự đoán.
  - `templates/index.html` – UI web kiểu chat (giống ChatGPT đơn giản).
  - `static/style.css` – CSS giao diện.
  - `static/script.js` – JS xử lý upload ảnh và hiển thị kết quả.

> Gợi ý: bạn có thể tạo thư mục `models/` và đặt file trọng số huấn luyện, ví dụ `models/leaf_vgg16_cbam_best.h5`.

## Cài đặt

1. Tạo môi trường ảo (khuyến khích):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

2. Cài thư viện:

```bash
pip install -r requirements.txt
```

## Chạy server

Trong thư mục gốc (`d:\Downloads\PBL5`):

```bash
uvicorn app.main:app --reload
```

Sau đó mở trình duyệt và truy cập: `http://localhost:8000`.

Bạn sẽ thấy giao diện:

- Bên trái: thông tin hệ thống, gợi ý câu hỏi.
- Bên phải: vùng chat; bạn chọn **ảnh lá cải**, nhấn **Gửi** để xem kết quả.

## Huấn luyện và nạp model

- Mặc định, code sẽ cố gắng nạp trọng số từ `models/leaf_vgg16_cbam_best.h5` nếu tồn tại.
- Bạn có thể tái sử dụng kiến trúc trong `app/ml/model_vgg16_cbam.py` để viết script train riêng, sau đó lưu:

```python
model.save_weights("models/leaf_vgg16_cbam_best.h5")
```

Khi file này xuất hiện, server sẽ dùng trọng số đã huấn luyện để dự đoán.

