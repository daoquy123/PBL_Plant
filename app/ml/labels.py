"""
Nhãn và mô tả tiếng Việt — dùng chung cho model, predictor, chatbot, notebook.
Thứ tự phải khớp CLASS_NAMES trong model_vgg16_cbam.py
"""

# Đồng bộ với model_vgg16_cbam.CLASS_NAMES
CLASS_LABELS_VI = {
    "la_khoe": "Lá cải khỏe mạnh",
    "la_vang": "Lá cải vàng / suy dinh dưỡng",
    "la_sau": "Lá cải có dấu hiệu sâu bệnh (vết ăn, thủng lá)",
    "sau": "Con sâu / côn trùng gây hại (thực thể)",
    "co": "Cỏ / nền không phải lá cải (mẫu nhiễu)",
}

EXPLANATIONS_VI = {
    "la_khoe": "Lá khỏe: màu xanh ổn định, không có vết sâu bệnh rõ rệt.",
    "la_vang": "Lá vàng: có thể thiếu dinh dưỡng, úng/khô, hoặc bệnh lý làm đổi màu.",
    "la_sau": "Lá bị sâu: vết cắn, lỗ thủng, mép lá không đều do sâu ăn.",
    "sau": "Phát hiện con sâu hoặc sâu non trong khung hình — ưu tiên cảnh báo can thiệp.",
    "co": "Vùng cỏ hoặc nền đất — hệ thống phân biệt để tránh nhầm với lá cải trong ảnh vườn.",
}
