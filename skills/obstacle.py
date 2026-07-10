"""
Skill: Phát hiện vật cản & hỗ trợ di chuyển — AN TOÀN, chạy real-time & cục bộ.

Có `check(frame)` để orchestrator GỌI CHỦ ĐỘNG (safety monitor): nếu có vật cản trong ngưỡng
nguy hiểm -> trả priority=SAFETY để chen ngang mọi câu trả lời khác.

- StubObstacle: điều khiển bằng `set_distance()` để dựng kịch bản demo.
- RealObstacle : YOLO (phát hiện) + Depth Anything (khoảng cách), lazy import, chạy on-device.
"""

from __future__ import annotations

from .base import PRIORITY_NORMAL, PRIORITY_SAFETY, Skill, SkillResult

_KEYWORDS = ("vật cản", "chướng ngại", "cản đường", "đi được", "có cản", "tránh", "an toàn không")


def _describe(distance: float | None, danger_m: float) -> dict:
    if distance is None:
        return {"danger": False, "distance": None, "text": "Đường phía trước thông thoáng."}
    if distance <= danger_m:
        return {"danger": True, "distance": distance,
                "text": f"Cảnh báo: có vật cản phía trước khoảng {distance:.1f} mét, hãy dừng lại."}
    return {"danger": False, "distance": distance,
            "text": f"Có vật cách khoảng {distance:.1f} mét, vẫn đi được nhưng chú ý."}


class StubObstacle(Skill):
    name = "obstacle"
    keywords = _KEYWORDS

    def __init__(self, danger_m: float = 1.5):
        self.danger_m = danger_m
        self._distance: float | None = None

    def set_distance(self, distance: float | None):
        """Điều khiển kịch bản demo: đặt khoảng cách vật cản gần nhất (None = trống)."""
        self._distance = distance

    def check(self, frame=None) -> dict:
        return _describe(self._distance, self.danger_m)

    def run(self, frame=None, query: str = "") -> SkillResult:
        r = self.check(frame)
        return SkillResult(r["text"], self.name,
                           PRIORITY_SAFETY if r["danger"] else PRIORITY_NORMAL, data=r)


class RealObstacle(Skill):
    name = "obstacle"
    keywords = _KEYWORDS

    def __init__(self, danger_m: float = 1.5, detector=None, depth=None):
        self.danger_m = danger_m
        self._detector = detector   # callable(frame)-> list boxes
        self._depth = depth         # callable(frame)-> depth map

    def _lazy(self):
        if self._detector is None:
            from ultralytics import YOLO  # lazy
            model = YOLO("yolo11n.pt")
            self._detector = lambda f: model(f, verbose=False)
        # depth: gắn Depth Anything khi cần (xem docs)

    def check(self, frame=None) -> dict:
        self._lazy()
        # TODO(M4): chạy detector + depth -> khoảng cách vật gần nhất trong vùng đi.
        # Ở đây để khung; hiện thực chi tiết trên M4 (cần model + hiệu chỉnh khoảng cách).
        raise NotImplementedError("RealObstacle.check cần hiện thực YOLO+Depth trên M4.")

    def run(self, frame=None, query: str = "") -> SkillResult:
        r = self.check(frame)
        return SkillResult(r["text"], self.name,
                           PRIORITY_SAFETY if r["danger"] else PRIORITY_NORMAL, data=r)
