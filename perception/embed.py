"""
Perception embedding — biến ảnh thành embedding cho CPM.

- RealEmbedder: dùng model thật (chạy trên M4):
    * face  -> InsightFace (ArcFace), 512-d normed embedding
    * object-> OpenCLIP (ViT-B-32), 512-d
- SyntheticEmbedder: embedding giả TẤT ĐỊNH theo 'danh tính' truyền vào -> để test wiring
  toàn bộ pipeline mà KHÔNG cần tải model / camera. Mặc định dùng cái này để chạy được ngay.

Hai không gian (face vs object) khác nhau -> CPM giữ bộ nhớ riêng theo modality.
"""

from __future__ import annotations

import hashlib

import numpy as np

FACE_DIM = 512
OBJECT_DIM = 512


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + eps)


def _hash_int(x) -> int:
    if isinstance(x, str):
        b = x.encode()
    elif isinstance(x, (bytes, bytearray)):
        b = bytes(x)
    elif isinstance(x, np.ndarray):
        b = x.tobytes()
    else:
        b = repr(x).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")


# ---------------------------------------------------------------- synthetic
class SyntheticEmbedder:
    """Embedding giả tất định. `image` có thể là chuỗi 'danh tính' để mô phỏng cùng người/đồ."""

    name = "synthetic"  # khoá ngưỡng (embedder, modality) — xem cpm/thresholds.py

    def __init__(self, face_dim: int = FACE_DIM, object_dim: int = OBJECT_DIM):
        self.face_dim = face_dim
        self.object_dim = object_dim

    def _embed(self, image, dim: int, namespace: str, jitter_cos: float = 0.85) -> np.ndarray:
        base_seed = _hash_int(f"{namespace}:{image}")
        rng = np.random.default_rng(base_seed)
        base = _unit(rng.standard_normal(dim))
        return base  # tất định: cùng 'danh tính' -> cùng vector (đủ cho wiring)

    def embed_face(self, image) -> np.ndarray:
        return self._embed(image, self.face_dim, "face")

    def embed_object(self, image) -> np.ndarray:
        return self._embed(image, self.object_dim, "object")


# --------------------------------------------------------------------- real
class RealEmbedder:
    """Model thật — lazy import để không bắt buộc cài nặng khi chỉ chạy CPM/synthetic."""

    # Khoá ngưỡng: "real" ngụ ý face=ArcFace(buffalo_l), object=OpenCLIP(ViT-B-32).
    # ĐỔI model (vd sang facenet) => phải calibrate lại; cân nhắc đổi name cho khỏi đụng khoá.
    name = "real"

    def __init__(self, device: str | None = None):
        self._face_app = None
        self._clip = None
        self._clip_preprocess = None
        self._device = device or self._auto_device()

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ---- face (InsightFace / ArcFace) ----
    def _ensure_face(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis  # lazy

            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0 if self._device != "cpu" else -1, det_size=(640, 640))
            self._face_app = app

    def embed_face(self, image_bgr: np.ndarray) -> np.ndarray:
        """image_bgr: ảnh numpy BGR (từ cv2). Trả embedding khuôn mặt lớn nhất."""
        self._ensure_face()
        faces = self._face_app.get(image_bgr)
        if not faces:
            raise ValueError("Không phát hiện khuôn mặt nào trong ảnh.")
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return _unit(faces[0].normed_embedding)

    # ---- object (OpenCLIP) ----
    def _ensure_clip(self):
        if self._clip is None:
            import open_clip  # lazy
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval().to(self._device)
            self._clip = model
            self._clip_preprocess = preprocess
            self._torch = torch

    def embed_object(self, image_rgb) -> np.ndarray:
        """image_rgb: PIL.Image RGB. Trả embedding ảnh CLIP."""
        self._ensure_clip()
        torch = self._torch
        x = self._clip_preprocess(image_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._clip.encode_image(x)
        return _unit(feat.squeeze(0).float().cpu().numpy())


# ------------------------------------------------------------------ factory
def get_embedder(kind: str = "auto"):
    """
    kind:
      - "synthetic": luôn dùng embedding giả (chạy được ngay, không cần model).
      - "real": bắt buộc model thật (M4).
      - "auto": thử real, lỗi thì fallback synthetic (kèm cảnh báo).
    """
    if kind == "synthetic":
        return SyntheticEmbedder()
    if kind == "real":
        return RealEmbedder()
    # auto
    try:
        emb = RealEmbedder()
        # không load model ngay (lazy); trả về, lỗi sẽ hiện khi gọi embed_*
        return emb
    except Exception as e:  # pragma: no cover
        print(f"[perception] Không dùng được RealEmbedder ({e}); fallback SyntheticEmbedder.")
        return SyntheticEmbedder()
