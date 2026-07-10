"""
Optional real providers for OCR and VLM skills.

All factories are lazy: importing this module does not require heavy packages or
API keys. Missing optional dependencies raise RuntimeError with actionable
messages so the orchestrator can fall back to stub skills.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np


def _as_rgb_array(frame) -> np.ndarray:
    if frame is None:
        raise ValueError("Chua co anh/khung hinh de xu ly.")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Dinh dang anh khong hop le: shape={arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def easyocr_fn(langs: tuple[str, ...] = ("vi", "en"), gpu: bool = False) -> Callable:
    """Return callable(frame_rgb) -> text using EasyOCR."""
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai easyocr. Cai bang: pip install easyocr") from exc

    reader = easyocr.Reader(list(langs), gpu=gpu)

    def run(frame_rgb) -> str:
        arr = _as_rgb_array(frame_rgb)
        results = reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(str(x).strip() for x in results if str(x).strip())

    return run


def gemini_vlm(model: str = "gemini-1.5-flash", api_key: str | None = None) -> Callable:
    """Return callable(frame_rgb, prompt) -> text using Google Gemini."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Thieu GEMINI_API_KEY trong bien moi truong.")
    try:
        import google.generativeai as genai
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai google-generativeai/Pillow. Cai bang: pip install google-generativeai pillow") from exc

    genai.configure(api_key=key)
    m = genai.GenerativeModel(model)

    def run(frame_rgb, prompt: str) -> str:
        arr = _as_rgb_array(frame_rgb)
        image = Image.fromarray(arr)
        response = m.generate_content([prompt, image])
        return getattr(response, "text", "") or ""

    return run


def moondream_vlm() -> Callable:
    """Placeholder for a local/offline VLM provider."""
    raise RuntimeError(
        "Moondream provider chua duoc cai dat trong repo nay. "
        "Co the them sau bang transformers/moondream hoac dung Gemini cho MVP."
    )


__all__ = ["easyocr_fn", "gemini_vlm", "moondream_vlm"]
