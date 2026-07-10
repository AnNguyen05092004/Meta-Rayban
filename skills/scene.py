"""
Skill: Mô tả cảnh & hỏi-đáp thị giác (VQA) — dùng VLM.

- StubScene: trả mô tả mẫu (chạy ngay, không cần model/mạng).
- RealScene : gọi một VLM (cloud hoặc on-device). VLM được tiêm vào dạng callable
  `vlm(frame_rgb, query) -> str` để KHÔNG khoá cứng nhà cung cấp. Xem docs/HUONG_DAN_M4.md
  để gắn Gemini/GPT-4o/Claude hoặc VLM on-device (Moondream/PaliGemma).
"""

from __future__ import annotations

from .base import PRIORITY_NORMAL, Skill, SkillResult

_KEYWORDS = ("trước mặt", "có gì", "mô tả", "xung quanh", "nhìn thấy", "cảnh", "quang cảnh")


class StubScene(Skill):
    name = "scene"
    keywords = _KEYWORDS

    def run(self, frame=None, query: str = "") -> SkillResult:
        return SkillResult(
            "Phía trước là một không gian trong nhà: có bàn, ghế và một ô cửa sổ bên phải. "
            "(mô tả mẫu — gắn VLM thật để có kết quả chính xác)",
            self.name,
            PRIORITY_NORMAL,
        )


class RealScene(Skill):
    name = "scene"
    keywords = _KEYWORDS

    def __init__(self, vlm=None):
        self.vlm = vlm  # callable(frame_rgb, query) -> str

    def run(self, frame=None, query: str = "") -> SkillResult:
        if self.vlm is None:
            raise RuntimeError("Chưa cấu hình VLM cho RealScene (xem docs/HUONG_DAN_M4.md).")
        prompt = query or "Mô tả ngắn gọn khung cảnh trước mặt cho người khiếm thị, bằng tiếng Việt."
        return SkillResult(self.vlm(frame, prompt), self.name)
