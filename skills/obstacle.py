"""Phát hiện vật cản cho trợ lý trợ thị.

``RealObstacle`` dùng YOLO để tìm vật trong hành lang phía trước và Depth
Anything V2 Small để ước lượng *độ sâu tương đối*. Camera đơn không đủ tin cậy
để suy ra mét trong mọi cảnh, nên API chỉ trả ``rất gần/gần/chưa gần``. Mốc mét
chỉ được thêm sau bước hiệu chuẩn Sprint 2 trên thiết bị thật.

Mọi dependency nặng được lazy-load. ``detector`` và ``depth`` có thể được tiêm
vào, vừa giúp unit test không tải model, vừa cho phép thay model khi deploy.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .base import PRIORITY_NORMAL, PRIORITY_SAFETY, Skill, SkillResult

_KEYWORDS = ("vật cản", "chướng ngại", "cản đường", "đi được", "có cản", "tránh", "an toàn không")


def _describe(distance: float | None, danger_m: float) -> dict:
    """Thông điệp của stub, giữ tương thích với kịch bản demo cũ."""
    if distance is None:
        return {"danger": False, "distance": None, "text": "Đường phía trước thông thoáng."}
    if distance <= danger_m:
        return {
            "danger": True,
            "distance": distance,
            "text": f"Cảnh báo: có vật cản phía trước khoảng {distance:.1f} mét, hãy dừng lại.",
        }
    return {
        "danger": False,
        "distance": distance,
        "text": f"Có vật cách khoảng {distance:.1f} mét, vẫn đi được nhưng chú ý.",
    }


class StubObstacle(Skill):
    name = "obstacle"
    keywords = _KEYWORDS

    def __init__(self, danger_m: float = 1.5):
        self.danger_m = danger_m
        self._distance: float | None = None

    def set_distance(self, distance: float | None):
        """Điều khiển kịch bản demo: ``None`` nghĩa là không có vật cản."""
        self._distance = distance

    def check(self, frame=None) -> dict:
        return _describe(self._distance, self.danger_m)

    def run(self, frame=None, query: str = "") -> SkillResult:
        result = self.check(frame)
        return SkillResult(
            result["text"], self.name, PRIORITY_SAFETY if result["danger"] else PRIORITY_NORMAL, data=result
        )


class RealObstacle(Skill):
    """YOLO + relative depth cho cảnh báo vật cản ở vùng đi phía trước.

    ``depth_nearer_is_larger`` đúng với output relative-depth mặc định của
    Depth Anything. Nếu đổi model có chiều ngược lại, cấu hình cờ này thay vì
    âm thầm đảo kết quả.
    """

    name = "obstacle"
    keywords = _KEYWORDS

    def __init__(
        self,
        *,
        detector: Callable[[np.ndarray], Any] | None = None,
        depth: Callable[[np.ndarray], np.ndarray] | None = None,
        detector_model: str = ".model_cache/yolo11n.pt",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        corridor_width: float = 0.60,
        near_score: float = 0.56,
        very_near_score: float = 0.76,
        depth_nearer_is_larger: bool = True,
    ):
        if not 0 < corridor_width <= 1:
            raise ValueError("corridor_width phải nằm trong (0, 1].")
        if not 0 <= near_score < very_near_score <= 1:
            raise ValueError("Ngưỡng danger phải thỏa 0 <= near < very_near <= 1.")
        self._detector = detector
        self._depth = depth
        self.detector_model = detector_model
        self.depth_model = depth_model
        self.corridor_width = corridor_width
        self.near_score = near_score
        self.very_near_score = very_near_score
        self.depth_nearer_is_larger = depth_nearer_is_larger

    def _lazy_detector(self) -> None:
        if self._detector is not None:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Chưa cài ultralytics. Chạy: pip install -r requirements.txt") from exc
        model_path = Path(self.detector_model)
        if model_path.parent != Path("."):
            model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_path))

        def run(frame_bgr: np.ndarray):
            return model(frame_bgr, verbose=False)

        self._detector = run

    def _lazy_depth(self) -> None:
        if self._depth is not None:
            return
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Chưa cài transformers cho Depth Anything. Chạy: pip install -r requirements.txt") from exc

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            processor = AutoImageProcessor.from_pretrained(self.depth_model)
            model = AutoModelForDepthEstimation.from_pretrained(self.depth_model).to(device).eval()
        except Exception as exc:  # pragma: no cover - model download/runtime dependent
            raise RuntimeError(
                "Không nạp được Depth Anything V2 Small. Kiểm tra mạng/dung lượng rồi thử lại."
            ) from exc

        def run(frame_bgr: np.ndarray) -> np.ndarray:
            import cv2

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                predicted = model(**inputs).predicted_depth
            resized = functional.interpolate(
                predicted.unsqueeze(1), size=frame_bgr.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
            return resized.detach().float().cpu().numpy()

        self._depth = run

    @staticmethod
    def _frame(frame) -> np.ndarray | None:
        if frame is None:
            return None
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Khung hình vật cản không hợp lệ.")
        image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _as_detections(raw: Any) -> list[dict[str, Any]]:
        """Chuẩn hóa list dict cho test và kết quả Ultralytics cho runtime."""
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)) and (not raw or isinstance(raw[0], dict)):
            return [dict(item) for item in raw]
        results = raw if isinstance(raw, (list, tuple)) else [raw]
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            names = getattr(result, "names", {})
            for coords, conf, cls_id in zip(xyxy, confidence, classes, strict=True):
                label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
                detections.append({"bbox": coords.tolist(), "confidence": float(conf), "label": str(label)})
        return detections

    def _candidates(self, detections: list[dict[str, Any]], depth_map: np.ndarray | None, shape: tuple[int, int]):
        height, width = shape
        corridor_left = (1 - self.corridor_width) * width / 2
        corridor_right = width - corridor_left
        if depth_map is not None:
            depth_map = np.asarray(depth_map, dtype=np.float64)
            if depth_map.shape != (height, width) or not np.isfinite(depth_map).all():
                raise ValueError("Bản đồ độ sâu không cùng kích thước hoặc chứa giá trị lỗi.")
            lo, hi = np.percentile(depth_map, (10, 90))
            depth_range = max(float(hi - lo), 1e-6)
        else:
            lo = depth_range = 0.0

        candidates = []
        for item in detections:
            bbox = item.get("bbox") or item.get("xyxy")
            if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
            if x2 <= x1 or y2 <= y1 or x2 <= corridor_left or x1 >= corridor_right:
                continue
            area = ((x2 - x1) * (y2 - y1)) / (width * height)
            # Cắt 20% viền bbox để tránh nền làm loãng giá trị depth của vật.
            margin_x, margin_y = max(1, int((x2 - x1) * 0.2)), max(1, int((y2 - y1) * 0.2))
            ix1, ix2 = int(x1) + margin_x, int(x2) - margin_x
            iy1, iy2 = int(y1) + margin_y, int(y2) - margin_y
            depth_score = None
            if depth_map is not None and ix2 > ix1 and iy2 > iy1:
                value = float(np.median(depth_map[iy1:iy2, ix1:ix2]))
                depth_score = float(np.clip((value - lo) / depth_range, 0.0, 1.0))
                if not self.depth_nearer_is_larger:
                    depth_score = 1.0 - depth_score
            size_score = float(np.clip(area / 0.22, 0.0, 1.0))
            proximity = size_score if depth_score is None else 0.78 * depth_score + 0.22 * size_score
            candidates.append(
                {
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "label": str(item.get("label", "vật thể")),
                    "confidence": float(item.get("confidence", 0.0)),
                    "area_ratio": area,
                    "depth_score": depth_score,
                    "proximity_score": proximity,
                }
            )
        return candidates

    def check(self, frame=None) -> dict:
        image = self._frame(frame)
        if image is None:
            return {
                "danger": False,
                "distance": None,
                "proximity": "unknown",
                "text": "Chưa có khung hình để kiểm tra vật cản.",
                "method": "unavailable",
            }

        started = time.perf_counter()
        self._lazy_detector()
        self._lazy_depth()
        assert self._detector is not None and self._depth is not None
        detections = self._as_detections(self._detector(image))
        depth_map = self._depth(image)
        candidates = self._candidates(detections, depth_map, image.shape[:2])
        latency_ms = (time.perf_counter() - started) * 1000
        base = {"distance": None, "method": "yolo11n+depth-anything-v2-relative", "latency_ms": latency_ms}
        if not candidates:
            return {
                **base,
                "danger": False,
                "proximity": "clear",
                "text": "Không phát hiện vật cản rõ trong vùng phía trước.",
                "detections": [],
            }

        selected = max(candidates, key=lambda item: item["proximity_score"])
        score = selected["proximity_score"]
        if score >= self.very_near_score:
            proximity, danger = "very_near", True
            text = "Cảnh báo: có vật cản rất gần phía trước, hãy dừng lại."
        elif score >= self.near_score:
            proximity, danger = "near", True
            text = "Cảnh báo: có vật cản gần phía trước, hãy đi chậm và chú ý."
        else:
            proximity, danger = "far", False
            text = "Phát hiện vật thể phía trước nhưng chưa ở mức cảnh báo."
        return {
            **base,
            "danger": danger,
            "proximity": proximity,
            "text": text,
            "selected": selected,
            "detections": candidates,
        }

    def run(self, frame=None, query: str = "") -> SkillResult:
        result = self.check(frame)
        return SkillResult(
            result["text"], self.name, PRIORITY_SAFETY if result["danger"] else PRIORITY_NORMAL, data=result
        )


class SafetyMonitor:
    """Chạy ``obstacle.check`` nền trên frame mới nhất, không mở camera thứ hai."""

    def __init__(
        self,
        obstacle: RealObstacle | StubObstacle,
        frame_source: Callable[[], Any],
        *,
        interval: float = 0.75,
        alert_cooldown: float = 3.0,
        on_alert: Callable[[dict], None] | None = None,
    ):
        if interval <= 0 or alert_cooldown < 0:
            raise ValueError("interval phải > 0 và alert_cooldown phải >= 0.")
        self.obstacle = obstacle
        self.frame_source = frame_source
        self.interval = interval
        self.alert_cooldown = alert_cooldown
        self.on_alert = on_alert
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._last_error: Exception | None = None
        self._checks = 0
        self._last_alert_at = float("-inf")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def checks(self) -> int:
        with self._lock:
            return self._checks

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def latest(self) -> dict | None:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def start(self) -> "SafetyMonitor":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="SafetyMonitor", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                assessment = self.obstacle.check(self.frame_source())
                now = time.monotonic()
                should_alert = assessment.get("danger", False) and now - self._last_alert_at >= self.alert_cooldown
                with self._lock:
                    self._latest = assessment
                    self._last_error = None
                    self._checks += 1
                    if should_alert:
                        self._last_alert_at = now
                if should_alert and self.on_alert is not None:
                    self.on_alert(assessment)
            except Exception as exc:  # pragma: no cover - device/model runtime path
                with self._lock:
                    self._last_error = exc
            self._stop.wait(self.interval)

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
