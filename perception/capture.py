"""
Nguồn ảnh: webcam hoặc file. Lazy import OpenCV để không bắt buộc khi chỉ chạy CPM.
"""

from __future__ import annotations

import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    """Đọc ảnh file -> numpy BGR (dùng cho InsightFace)."""
    import cv2  # lazy

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def load_image_rgb(path: str):
    """Đọc ảnh file -> PIL.Image RGB (dùng cho CLIP)."""
    from PIL import Image  # lazy

    return Image.open(path).convert("RGB")


def grab_webcam_bgr(cam_index: int = 0) -> np.ndarray:
    """Chụp 1 khung hình từ webcam -> numpy BGR."""
    import cv2  # lazy

    cap = cv2.VideoCapture(cam_index)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Không lấy được khung hình từ webcam #{cam_index}")
        return frame
    finally:
        cap.release()


def bgr_to_rgb_pil(frame_bgr: np.ndarray):
    """Chuyển khung BGR (cv2) -> PIL RGB (cho CLIP)."""
    import cv2  # lazy
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
