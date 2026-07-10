"""
Giao diện chung cho các "kỹ năng" (skill) nhận thức của trợ lý thị giác.

Mỗi skill:
- khai báo `keywords` (tiếng Việt) để orchestrator định tuyến intent,
- có bản `Stub*` chạy ngay (không cần model) và `Real*` (lazy import, chạy trên M4),
- trả về `SkillResult` với `priority` (SAFETY = cao nhất -> chen ngang).
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIORITY_SAFETY = 100   # cảnh báo an toàn (vật cản) -> chen ngang mọi thứ
PRIORITY_NORMAL = 10


@dataclass
class SkillResult:
    text: str                       # câu trả lời tiếng Việt cho người dùng
    skill: str
    priority: int = PRIORITY_NORMAL
    data: dict = field(default_factory=dict)


class Skill:
    name: str = "base"
    keywords: tuple[str, ...] = ()

    def can_handle(self, intent_text: str) -> bool:
        t = intent_text.lower()
        return any(k in t for k in self.keywords)

    def run(self, frame=None, query: str = "") -> SkillResult:
        raise NotImplementedError
