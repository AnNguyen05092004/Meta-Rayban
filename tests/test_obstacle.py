from __future__ import annotations

import time

import numpy as np

from app.orchestrator import VisionAssistant
from skills.obstacle import RealObstacle, SafetyMonitor


def _frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_real_obstacle_selects_close_central_object_from_relative_depth():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[25:75, 35:65] = 10.0
    obstacle = RealObstacle(
        detector=lambda image: [{"bbox": [30, 20, 70, 80], "label": "ghế", "confidence": 0.9}],
        depth=lambda image: depth,
    )

    result = obstacle.check(_frame())

    assert result["danger"]
    assert result["proximity"] == "very_near"
    assert result["selected"]["label"] == "ghế"
    assert result["distance"] is None
    assert "mét" not in result["text"]
    assert "ghế" not in result["text"]  # Không đọc nhãn YOLO có thể sai cho người dùng.


def test_real_obstacle_ignores_object_outside_forward_corridor():
    obstacle = RealObstacle(
        detector=lambda image: [{"bbox": [0, 20, 15, 90], "label": "bàn", "confidence": 0.9}],
        depth=lambda image: np.ones((100, 100), dtype=np.float32),
    )

    result = obstacle.check(_frame())

    assert not result["danger"]
    assert result["proximity"] == "clear"


def test_real_obstacle_reports_far_when_relative_depth_and_size_are_low():
    depth = np.full((100, 100), 10.0, dtype=np.float32)
    depth[45:55, 45:55] = 0.0
    obstacle = RealObstacle(
        detector=lambda image: [{"bbox": [43, 43, 57, 57], "label": "cốc", "confidence": 0.8}],
        depth=lambda image: depth,
    )

    result = obstacle.check(_frame())

    assert not result["danger"]
    assert result["proximity"] == "far"


def test_safety_monitor_publishes_assessment_and_rate_limits_alerts():
    alerts = []
    obstacle = RealObstacle(
        detector=lambda image: [{"bbox": [25, 20, 75, 85], "label": "người", "confidence": 0.9}],
        depth=lambda image: np.pad(np.full((50, 50), 9.0), ((25, 25), (25, 25))),
    )
    monitor = SafetyMonitor(obstacle, _frame, interval=0.01, alert_cooldown=10, on_alert=alerts.append)
    try:
        monitor.start()
        time.sleep(0.06)
        latest = monitor.latest()
        assert latest is not None and latest["danger"]
        assert monitor.checks >= 2
        assert len(alerts) == 1
    finally:
        monitor.stop()


def test_orchestrator_uses_injected_real_obstacle_for_safety_override():
    obstacle = RealObstacle(
        detector=lambda image: [{"bbox": [25, 20, 75, 85], "label": "người", "confidence": 0.9}],
        depth=lambda image: np.pad(np.full((50, 50), 9.0), ((25, 25), (25, 25))),
    )
    assistant = VisionAssistant(embedder_kind="synthetic", obstacle=obstacle)

    result = assistant.handle("Trước mặt có gì?", frame=_frame())

    assert result.startswith("Cảnh báo")
