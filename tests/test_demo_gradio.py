from __future__ import annotations

import numpy as np
import pytest

from app.cli import Assistant
from app.demo_gradio import _normalize_image_rgb, _to_embedding, build_demo


def test_normalize_image_rgb_handles_grayscale_and_rgba():
    gray = np.zeros((16, 16), dtype=np.uint8)
    rgba = np.zeros((16, 16, 4), dtype=np.float32)
    rgba[..., 3] = 255

    assert _normalize_image_rgb(gray).shape == (16, 16, 3)
    assert _normalize_image_rgb(rgba).shape == (16, 16, 3)
    assert _normalize_image_rgb(rgba).dtype == np.uint8


def test_normalize_image_rgb_rejects_empty_image():
    with pytest.raises(ValueError):
        _normalize_image_rgb(None)


def test_to_embedding_synthetic_face_and_object_headless():
    assistant = Assistant("synthetic")
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    face = _to_embedding(assistant, img, "face")
    obj = _to_embedding(assistant, img, "object")

    assert face.shape == (512,)
    assert obj.shape == (512,)
    assert np.isclose(np.linalg.norm(face), 1.0)
    assert np.isclose(np.linalg.norm(obj), 1.0)


def test_build_demo_headless_does_not_require_camera():
    demo = build_demo("synthetic")
    try:
        assert demo is not None
        assert len(demo.blocks) > 0
    finally:
        demo.close()
