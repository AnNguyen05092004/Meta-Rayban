from __future__ import annotations

from app.orchestrator import VisionAssistant
from skills import RealOCR, RealScene, StubOCR, StubScene


def test_real_ocr_callable_is_used():
    ocr = RealOCR(ocr_fn=lambda frame: "Xin chao")
    assert "Xin chao" in ocr.run(frame="fake").text


def test_real_scene_callable_is_used():
    scene = RealScene(vlm=lambda frame, prompt: f"Mo ta: {prompt}")
    assert "Mo ta" in scene.run(frame="fake", query="co gi").text


def test_orchestrator_can_use_injected_real_skills():
    a = VisionAssistant(embedder_kind="synthetic")
    a.ocr = RealOCR(ocr_fn=lambda frame: "ABC 123")
    a.scene = RealScene(vlm=lambda frame, prompt: "Mot canh trong nha")
    a._query_skills = [a.obstacle, a.ocr, a.scene]

    assert "ABC 123" in a.handle("Đọc chữ giúp tôi", frame="fake")
    assert "Mot canh trong nha" in a.handle("Trước mặt có gì?", frame="fake")


def test_real_mode_falls_back_without_optional_dependencies(monkeypatch):
    import app.orchestrator as orch

    monkeypatch.setattr(orch.VisionAssistant, "_make_skills", staticmethod(lambda mode: (StubScene(), StubOCR())))
    a = orch.VisionAssistant(embedder_kind="synthetic", skills_mode="real")

    assert isinstance(a.scene, StubScene)
    assert isinstance(a.ocr, StubOCR)
