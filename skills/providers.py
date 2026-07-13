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


def _encode_image_data_url(frame, fmt: str = "PNG") -> str:
    """RGB array -> data URL base64 (PNG/JPEG) cho OpenAI vision."""
    import base64
    import io

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai Pillow. Cai bang: pip install pillow") from exc

    arr = _as_rgb_array(frame)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _openai_client(api_key: str | None = None, client=None):
    """Tao (hoac nhan) OpenAI client. `client` cho phep tiem gia lap khi test."""
    if client is not None:
        return client
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thieu OPENAI_API_KEY trong bien moi truong.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai openai SDK. Cai bang: pip install openai pillow") from exc
    return OpenAI(api_key=key)


def _openai_vision(client, model: str, prompt: str, frame, max_tokens: int) -> str:
    """Goi 1 luot chat.completions co kem anh, tra ve text."""
    data_url = _encode_image_data_url(frame)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def openai_vlm(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 400,
    client=None,
) -> Callable:
    """VLM (scene/VQA) dung OpenAI GPT-4o. Tra callable(frame_rgb, prompt) -> text."""
    cli = _openai_client(api_key, client)

    def run(frame_rgb, prompt: str) -> str:
        return _openai_vision(cli, model, prompt, frame_rgb, max_tokens)

    return run


_OPENAI_OCR_PROMPT = (
    "Trich xuat CHINH XAC toan bo van ban xuat hien trong anh, giu nguyen thu tu doc va dau "
    "tieng Viet. Chi tra ve phan chu, khong mo ta, khong giai thich. Neu khong co chu, tra ve chuoi rong."
)


def openai_ocr(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 800,
    client=None,
) -> Callable:
    """OCR dung OpenAI GPT-4o vision (khong can cai EasyOCR). Tra callable(frame_rgb) -> text."""
    cli = _openai_client(api_key, client)

    def run(frame_rgb) -> str:
        return _openai_vision(cli, model, _OPENAI_OCR_PROMPT, frame_rgb, max_tokens)

    return run


def moondream_vlm() -> Callable:
    """Placeholder for a local/offline VLM provider."""
    raise RuntimeError(
        "Moondream provider chua duoc cai dat trong repo nay. "
        "Co the them sau bang transformers/moondream hoac dung Gemini/OpenAI cho MVP."
    )


__all__ = [
    "easyocr_fn",
    "gemini_vlm",
    "openai_vlm",
    "openai_ocr",
    "moondream_vlm",
]
