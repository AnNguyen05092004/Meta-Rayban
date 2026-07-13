"""
Lưu/nạp ngưỡng quen/lạ (novelty gate) đã HIỆU CHỈNH theo (embedder, modality).

Vì `confidence = cos(query, prototype)`, phân bố điểm số phụ thuộc TỪNG model embedding:
- face  (ArcFace/InsightFace buffalo_l ~ facenet): cos người-khác-nhau ~0.3–0.5
- object(OpenCLIP ViT-B-32): cos đồ-khác-loại cũng khá cao
=> ngưỡng 0.35 mặc định (suy từ synthetic) SAI cho embedding thật. Ngưỡng đúng phải
   calibrate từ ROC thật (xem experiments/calibration.py + scripts/calibrate_threshold.py)
   rồi lưu ở đây, khoá theo (embedder_name, modality).

Module này CỐ Ý thuần-runtime (chỉ JSON + khoá), KHÔNG import experiments/metrics để giữ
lớp lõi `cpm` độc lập. Phần TÍNH ngưỡng nằm ở experiments/calibration.py.
"""

from __future__ import annotations

import json
import os

# Meta-Rayban/cpm/thresholds.py -> Meta-Rayban/
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_THRESHOLDS_PATH = os.path.join(_PKG_DIR, "configs", "thresholds.json")


# Tạo "khoá" tra ngưỡng dạng "tên_embedder:modality" (vd "real:face").
# Vào: tên model embedding + loại đối tượng. Ra: chuỗi khoá để tra/ghi trong file JSON.
def threshold_key(embedder_name: str, modality: str) -> str:
    """Khoá tra ngưỡng. Ngưỡng gắn với CẢ model lẫn modality (đổi model -> calibrate lại)."""
    return f"{embedder_name}:{modality}"


# Đọc bảng ngưỡng đã hiệu chuẩn từ file JSON. Không có file thì trả về bảng rỗng {}.
def load_thresholds(path: str = DEFAULT_THRESHOLDS_PATH) -> dict:
    """Đọc bảng ngưỡng {'<embedder>:<modality>': threshold}. Thiếu file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Ghi/cập nhật bảng ngưỡng vào file JSON. Đọc bảng cũ trước rồi trộn khoá mới vào,
# nên KHÔNG làm mất các ngưỡng đã lưu trước đó.
def save_thresholds(mapping: dict, path: str = DEFAULT_THRESHOLDS_PATH) -> None:
    """Ghi/cập nhật bảng ngưỡng (giữ nguyên các khoá cũ đã có)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # tạo thư mục chứa file nếu chưa có
    current = load_thresholds(path)  # đọc bảng ngưỡng hiện tại
    current.update(mapping)  # chèn/ghi đè bằng các khoá mới
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2, sort_keys=True)  # ghi lại file (giữ được tiếng Việt)


# Trả ngưỡng quen-lạ đã lưu cho (embedder, modality); không có thì dùng giá trị mặc định.
# recall_threshold = ngưỡng quen-lạ: độ giống ≥ ngưỡng -> coi là QUEN, nhỏ hơn -> "chưa biết".
def resolve_threshold(mapping: dict, embedder_name: str, modality: str, default: float) -> float:
    """Ngưỡng cho (embedder, modality); không có -> default (thường là 0.35 synthetic)."""
    entry = mapping.get(threshold_key(embedder_name, modality))  # tra theo khoá "embedder:modality"
    if entry is None:
        return float(default)  # chưa hiệu chuẩn cho cặp này -> dùng ngưỡng mặc định
    # cho phép entry là số, hoặc dict {'threshold': ..., 'policy': ...} (giàu thông tin hơn)
    if isinstance(entry, dict):
        return float(entry.get("threshold", default))
    return float(entry)
