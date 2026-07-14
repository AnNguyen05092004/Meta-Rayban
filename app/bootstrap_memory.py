"""Nạp có chủ đích dataset thư mục vào CPM local cho demo."""

from __future__ import annotations

from pathlib import Path

from app.cli import Assistant
from perception.embed import SyntheticEmbedder


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def bootstrap_directory(assistant: Assistant, data_dir: str | Path, modality: str) -> dict[str, int]:
    """Đọc ``data_dir/<nhãn>/*`` và ghi embedding thật vào CPM.

    Chỉ dùng lúc khởi tạo bộ nhớ local. Việc bootstrap lại cùng dataset sẽ cộng
    thêm mẫu vào prototype, do đó CLI chặn nó khi memory đã có nhãn.
    """
    if isinstance(assistant.embedder, SyntheticEmbedder):
        raise RuntimeError("Bootstrap dataset cần --embedder real; synthetic embedding không đại diện ảnh thật.")
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")
    imported: dict[str, int] = {}
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        files = sorted(path for path in label_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            continue
        count = 0
        for image_path in files:
            if modality == "face":
                import cv2

                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                embedding = assistant.embedder.embed_face(image)
            else:
                from PIL import Image

                with Image.open(image_path) as source:
                    embedding = assistant.embedder.embed_object(source.convert("RGB"))
            assistant.cpm[modality].write(embedding, label_dir.name)
            count += 1
        if count:
            imported[label_dir.name] = count
    if not imported:
        raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong {root}")
    assistant.persist_memory()
    return imported
