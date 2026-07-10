"""Các kỹ năng nhận thức của trợ lý thị giác."""

from .base import PRIORITY_NORMAL, PRIORITY_SAFETY, Skill, SkillResult
from .obstacle import RealObstacle, StubObstacle
from .ocr import RealOCR, StubOCR
from .providers import easyocr_fn, gemini_vlm, moondream_vlm
from .scene import RealScene, StubScene

__all__ = [
    "Skill",
    "SkillResult",
    "PRIORITY_SAFETY",
    "PRIORITY_NORMAL",
    "StubScene",
    "RealScene",
    "StubOCR",
    "RealOCR",
    "StubObstacle",
    "RealObstacle",
    "easyocr_fn",
    "gemini_vlm",
    "moondream_vlm",
]
