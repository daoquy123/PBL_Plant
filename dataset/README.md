# Cấu trúc dữ liệu (5 lớp)

Thư mục **`train/`** và **`val/`** mỗi thư mục chứa **đúng 5 folder** (tên khớp code):

| Thư mục   | Ý nghĩa |
|-----------|---------|
| `la_khoe` | Lá cải khỏe mạnh |
| `la_vang` | Lá vàng / suy dinh dưỡng |
| `la_sau`  | Lá có vết sâu bệnh (ăn lá, thủng) |
| `sau`     | Ảnh có **con sâu / côn trùng** thực thể |
| `co`      | Cỏ, nền đất — **mẫu nhiễu** khi chụp vườn |

**Lưu ý:** Nếu bạn còn thư mục cũ (`la_sau_va_vang`), hãy gộp ảnh vào `la_sau` hoặc tách tay theo tiêu chí nhóm.

Sau khi sắp xong, chạy huấn luyện từ thư mục gốc project:

```bash
python app/ml/train_vgg16_cbam.py
```

Biểu đồ báo cáo: mở `notebooks/pbl5_bao_cao_charts.ipynb`.
