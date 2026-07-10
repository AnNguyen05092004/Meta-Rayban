"""
Skill: Đọc chữ (OCR) — biển báo, tài liệu, tiền, hạn sử dụng.

- StubOCR: trả chuỗi mẫu.
- RealOCR : VietOCR/PaddleOCR (lazy) hoặc VLM. Tiêm `ocr_fn(frame_rgb) -> str`.
"""

from __future__ import annotations

from .base import Skill, SkillResult

_KEYWORDS = ("đọc", "chữ", "biển", "tài liệu", "giấy", "hạn sử dụng", "tờ", "menu", "bảng")


class StubOCR(Skill):
    name = "ocr"
    keywords = _KEYWORDS

    def run(self, frame=None, query: str = "") -> SkillResult:
        return SkillResult(
            'Chữ đọc được: "CỬA HÀNG TẠP HOÁ — GIỜ MỞ CỬA 7:00–21:00". '
            "(kết quả mẫu — gắn OCR thật để đọc chính xác)",
            self.name,
        )


class RealOCR(Skill):
    name = "ocr"
    keywords = _KEYWORDS

    def __init__(self, ocr_fn=None):
        self.ocr_fn = ocr_fn  # callable(frame_rgb) -> str

    def run(self, frame=None, query: str = "") -> SkillResult:
        if self.ocr_fn is None:
            raise RuntimeError("Chưa cấu hình OCR cho RealOCR (VietOCR/PaddleOCR/VLM).")
        text = self.ocr_fn(frame).strip()
        if not text:
            return SkillResult("Không đọc được chữ nào trong khung hình.", self.name)
        return SkillResult(f"Chữ đọc được: {text}", self.name)
