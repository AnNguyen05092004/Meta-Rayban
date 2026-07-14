"""Nghiệm thu camera stream + cảnh báo vật cản thật của Sprint 2.

Ví dụ:
  HF_HUB_DISABLE_XET=1 python -m scripts.safety_monitor --cam 0 --show --seconds 60

Lần đầu chạy sẽ nạp checkpoint YOLO11n và Depth Anything V2 Small. Kết quả là
``gần/rất gần`` tương đối, không phải đo khoảng cách mét.
"""

from __future__ import annotations

import argparse
import time

from perception.capture import CameraStream
from skills import RealObstacle, SafetyMonitor


def _draw_overlay(frame, assessment):
    import cv2

    view = frame.copy()
    selected = assessment.get("selected") if assessment else None
    if selected:
        x1, y1, x2, y2 = selected["bbox"]
        color = (0, 0, 255) if assessment["danger"] else (0, 180, 255)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            view,
            f"{selected['label']} {assessment['proximity']} {selected['proximity_score']:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    if assessment:
        color = (0, 0, 255) if assessment["danger"] else (0, 220, 0)
        cv2.putText(view, assessment["text"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        cv2.putText(
            view,
            f"infer {assessment.get('latency_ms', 0):.0f}ms | camera fps {assessment.get('camera_fps', 0):.1f}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description="Theo dõi vật cản YOLO11n + Depth Anything V2 từ webcam.")
    parser.add_argument("--cam", type=int, default=0, help="chỉ số webcam")
    parser.add_argument("--seconds", type=float, default=60, help="thời gian chạy; 0 = tới khi Ctrl-C")
    parser.add_argument("--interval", type=float, default=0.75, help="chu kỳ infer vật cản, giây")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--show", action="store_true", help="hiện preview OpenCV; nhấn Esc để dừng")
    args = parser.parse_args()

    latest_alert = {"text": None}

    def on_alert(assessment):
        latest_alert["text"] = assessment["text"]
        print(f"[CẢNH BÁO] {assessment['text']} | infer={assessment.get('latency_ms', 0):.0f}ms")

    stream = CameraStream(args.cam, width=args.width, height=args.height, fps=args.fps)
    monitor = SafetyMonitor(RealObstacle(), stream.latest, interval=args.interval, on_alert=on_alert)
    cv2 = None
    if args.show:
        import cv2 as _cv2

        cv2 = _cv2

    stream.start()
    monitor.start()
    print("Safety monitor đang chạy. Đây là depth tương đối, không đọc mét. Nhấn Ctrl-C hoặc Esc để dừng.")
    deadline = None if args.seconds == 0 else time.monotonic() + args.seconds
    reported_checks = -1
    try:
        while deadline is None or time.monotonic() < deadline:
            assessment = monitor.latest()
            if assessment is not None:
                assessment = dict(assessment)
                assessment["camera_fps"] = stream.measured_fps
                if monitor.checks != reported_checks:
                    print(
                        f"[safety] {assessment['proximity']}: {assessment['text']} "
                        f"| infer={assessment.get('latency_ms', 0):.0f}ms"
                    )
                    reported_checks = monitor.checks
            if cv2 is not None:
                frame = stream.latest()
                if frame is not None:
                    cv2.imshow("Meta-Rayban safety monitor", _draw_overlay(frame, assessment))
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
            time.sleep(0.05)
    finally:
        monitor.stop()
        stream.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    if monitor.last_error is not None:
        print(f"Safety monitor dừng với lỗi: {monitor.last_error}")
        return 1
    print(f"Đã dừng. Đã kiểm tra {monitor.checks} lượt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
