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


def threshold_key(embedder_name: str, modality: str) -> str:
    """Khoá tra ngưỡng. Ngưỡng gắn với CẢ model lẫn modality (đổi model -> calibrate lại)."""
    return f"{embedder_name}:{modality}"


def load_thresholds(path: str = DEFAULT_THRESHOLDS_PATH) -> dict:
    """Đọc bảng ngưỡng {'<embedder>:<modality>': threshold}. Thiếu file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_thresholds(mapping: dict, path: str = DEFAULT_THRESHOLDS_PATH) -> None:
    """Ghi/cập nhật bảng ngưỡng (giữ nguyên các khoá cũ đã có)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    current = load_thresholds(path)
    current.update(mapping)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_threshold(mapping: dict, embedder_name: str, modality: str, default: float) -> float:
    """Ngưỡng cho (embedder, modality); không có -> default (thường là 0.35 synthetic)."""
    entry = mapping.get(threshold_key(embedder_name, modality))
    if entry is None:
        return float(default)
    # cho phép entry là số, hoặc dict {'threshold': ..., 'policy': ...} (giàu thông tin hơn)
    if isinstance(entry, dict):
        return float(entry.get("threshold", default))
    return float(entry)
