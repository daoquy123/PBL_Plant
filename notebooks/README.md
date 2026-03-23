# Notebook báo cáo

1. Cài thêm (nếu chưa): `pip install scikit-learn matplotlib seaborn jupyter`
2. Tạo đủ thư mục `dataset/train/{la_khoe,la_vang,la_sau,sau,co}` và `val` tương tự (xem `dataset/README.md`).
3. Train: `python app/ml/train_vgg16_cbam.py` → sinh `app/checkpoints/training_history.json` và `vgg16_cbam_best.weights.h5`.
4. Mở `pbl5_bao_cao_charts.ipynb`, chạy **Run All** (hoặc từng ô).
5. Ảnh slide nằm trong `reports/figures/`.

**Lưu ý:** Nếu mở notebook khi thư mục làm việc là `notebooks/`, ô đầu vẫn tự tìm `REPO` chứa `app/ml`.
