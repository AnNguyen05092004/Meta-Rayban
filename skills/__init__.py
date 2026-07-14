"""Các kỹ năng nhận thức của trợ lý thị giác."""

from .base import PRIORITY_NORMAL, PRIORITY_SAFETY, Skill, SkillResult
from .obstacle import RealObstacle, SafetyMonitor, StubObstacle
from .ocr import RealOCR, StubOCR
from .providers import easyocr_fn, faster_whisper_stt, gemini_vlm, moondream_vlm, speak_text
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
    "SafetyMonitor",
    "easyocr_fn",
    "faster_whisper_stt",
    "gemini_vlm",
    "speak_text",
    "moondream_vlm",
]
