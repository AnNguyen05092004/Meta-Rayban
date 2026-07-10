"""
Gradio UI for the CPM visual assistant demo.

Run:
  python -m app.demo_gradio
  python -m app.demo_gradio --embedder real
  python -m app.demo_gradio --embedder real --server-port 7861
"""

from __future__ import annotations

import argparse

import numpy as np

from app.cli import Assistant
from perception.embed import SyntheticEmbedder


def _normalize_image_rgb(image_rgb) -> np.ndarray:
    """Convert Gradio image input to contiguous uint8 RGB."""
    if image_rgb is None:
        raise ValueError("Chưa có ảnh. Hãy chụp webcam hoặc tải ảnh lên trước.")

    arr = np.asarray(image_rgb)
    if arr.size == 0:
        raise ValueError("Ảnh rỗng. Hãy chụp hoặc tải lại ảnh khác.")
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Định dạng ảnh không hợp lệ: shape={arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _friendly_error(exc: Exception, modality: str) -> str:
    msg = str(exc)
    lower = msg.lower()
    if "không phát hiện" in lower or "khong phat hien" in lower:
        return "Không phát hiện khuôn mặt rõ. Hãy nhìn thẳng camera, đủ sáng, chỉ một mặt chính."
    if "chưa có ảnh" in lower or "ảnh rỗng" in lower or "định dạng ảnh" in lower:
        return msg
    if modality == "face":
        return f"Lỗi xử lý khuôn mặt: {msg}"
    return f"Lỗi xử lý đồ vật: {msg}"


def _to_embedding(assistant: Assistant, image_rgb, modality: str) -> np.ndarray:
    image_rgb = _normalize_image_rgb(image_rgb)
    emb = assistant.embedder

    if isinstance(emb, SyntheticEmbedder):
        return emb.embed_face(image_rgb) if modality == "face" else emb.embed_object(image_rgb)

    if modality == "face":
        import cv2

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return emb.embed_face(bgr)

    from PIL import Image

    return emb.embed_object(Image.fromarray(image_rgb))


def build_demo(embedder_kind: str = "synthetic"):
    import gradio as gr

    assistant = Assistant(embedder_kind=embedder_kind)

    def _stats() -> str:
        return "Bộ nhớ - " + assistant.stats()

    def teach(image, modality, label):
        if not label or not label.strip():
            return "Nhập nhãn trước khi Ghi nhớ.", _stats()
        try:
            key = _to_embedding(assistant, image, modality)
            assistant.cpm[modality].write(key, label.strip())
            return f"Đã ghi nhớ ({modality}): {label.strip()}", _stats()
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    def ask(image, modality):
        try:
            key = _to_embedding(assistant, image, modality)
            res = assistant.cpm[modality].recall(key)[0]
            if not res["known"]:
                return f"Chưa biết ({modality}) - độ tương đồng {res['confidence']:.2f}", _stats()
            return (
                f"Đây là **{res['label']}** "
                f"(tin cậy {res['confidence']:.2f}, tầng {res['tier']})",
                _stats(),
            )
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    def fix(image, modality, label):
        if not label or not label.strip():
            return "Nhập nhãn đúng trước khi Sửa.", _stats()
        try:
            key = _to_embedding(assistant, image, modality)
            assistant.cpm[modality].correct(key, label.strip())
            return f"Đã sửa ({modality}): đây là {label.strip()}", _stats()
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    with gr.Blocks(title="Trợ lý CPM - nhận diện và ghi nhớ") as demo:
        gr.Markdown(
            "# Trợ lý thị giác - nhận diện và ghi nhớ (CPM / Nested Learning)\n"
            "Dạy hệ thống người quen và đồ cá nhân, hỏi lại, sửa sai - học liên tục, không quên."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(sources=["webcam", "upload"], type="numpy", label="Ảnh webcam hoặc upload")
                modality = gr.Radio(["face", "object"], value="face", label="Loại đối tượng")
                label = gr.Textbox(label="Nhãn", placeholder="VD: Lan / ví của tôi")
                with gr.Row():
                    btn_teach = gr.Button("Ghi nhớ", variant="primary")
                    btn_ask = gr.Button("Hỏi", variant="secondary")
                    btn_fix = gr.Button("Sửa")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Kết quả")
                stats = gr.Markdown(_stats())

        btn_teach.click(teach, [image, modality, label], [out, stats])
        btn_ask.click(ask, [image, modality], [out, stats])
        btn_fix.click(fix, [image, modality, label], [out, stats])

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    p.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    p.add_argument("--server-port", type=int, default=None, help="use another port if 7860 is busy")
    args = p.parse_args()
    build_demo(args.embedder).launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
