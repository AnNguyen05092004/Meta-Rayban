"""
Thử OCR/VLM THẬT trên một ảnh (kiểm chứng key OpenAI/Gemini nhanh, không cần webcam/UI).

Cách dùng:
  # 1) đặt key (PowerShell):  $env:OPENAI_API_KEY = "sk-..."
  # 2) cài SDK:               pip install openai pillow
  # 3) chạy:
  python -m scripts.try_vlm --image data/mau_bien.jpg --query "Trước mặt có gì?"
  python -m scripts.try_vlm --image data/to_giay.jpg  --query "Đọc chữ giúp tôi"

Provider tự chọn: OPENAI_API_KEY -> Gemini -> Stub (xem log [skills] in ra stderr).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.orchestrator import VisionAssistant


def load_image_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Chưa cài Pillow. Cài bằng: pip install pillow") from exc
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description="Thử OCR/VLM thật trên một ảnh.")
    p.add_argument("--image", required=True, help="đường dẫn ảnh (jpg/png)")
    p.add_argument("--query", default="Trước mặt có gì?", help="câu hỏi tiếng Việt (mô tả cảnh / đọc chữ)")
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    args = p.parse_args()

    frame = load_image_rgb(args.image)
    print(f"[try_vlm] Ảnh {args.image} -> shape {frame.shape}", file=sys.stderr)

    assistant = VisionAssistant(embedder_kind=args.embedder, skills_mode="real")
    print(f"[Người dùng] {args.query}")
    print(f"[Trợ lý]     {assistant.handle(args.query, frame=frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
