from __future__ import annotations

import numpy as np

from app.orchestrator import VisionAssistant
from skills import RealOCR, RealScene, StubOCR, StubScene
from skills.providers import openai_ocr, openai_vlm


class _FakeOpenAI:
    """Client OpenAI giả: ghi lại request, trả về text cố định. Không chạm mạng."""

    def __init__(self, reply: str = "Mot canh trong nha"):
        self.reply = reply
        self.captured = {}
        outer = self

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kw):
                outer.captured = kw
                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _fake_frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_openai_vlm_sends_text_and_image_then_parses_reply():
    client = _FakeOpenAI(reply="Mot canh trong nha")
    vlm = openai_vlm(client=client)
    out = vlm(_fake_frame(), "Mo ta di")
    assert out == "Mot canh trong nha"

    content = client.captured["messages"][0]["content"]
    assert any(p.get("type") == "text" and p["text"] == "Mo ta di" for p in content)
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_ocr_uses_ocr_prompt_and_returns_text():
    client = _FakeOpenAI(reply="CUA HANG 7:00-21:00")
    ocr = openai_ocr(client=client)
    out = ocr(_fake_frame())
    assert out == "CUA HANG 7:00-21:00"

    content = client.captured["messages"][0]["content"]
    prompt = next(p["text"] for p in content if p.get("type") == "text")
    assert "Trich xuat" in prompt  # dùng prompt OCR chuyên biệt, không phải mô tả cảnh


def test_openai_ocr_wired_into_realocr():
    client = _FakeOpenAI(reply="ABC 123")
    ocr = RealOCR(ocr_fn=openai_ocr(client=client))
    assert "ABC 123" in ocr.run(frame=_fake_frame()).text


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
