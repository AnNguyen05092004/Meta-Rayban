"""Smoke test luồng camera liên tục.

Chạy:
  python -m scripts.camera_stream --seconds 10
  python -m scripts.camera_stream --seconds 30 --show

Mục tiêu: giữ webcam mở, đọc frame liên tục, in FPS/shape để kiểm chứng nền cho
voice loop và safety monitor real-time.
"""

from __future__ import annotations

import argparse
import time

from perception.capture import CameraStream


def main() -> int:
    p = argparse.ArgumentParser(description="Kiểm tra webcam stream liên tục.")
    p.add_argument("--cam", type=int, default=0, help="chỉ số webcam")
    p.add_argument("--seconds", type=float, default=10.0, help="thời gian chạy")
    p.add_argument("--width", type=int, default=None, help="chiều rộng mong muốn")
    p.add_argument("--height", type=int, default=None, help="chiều cao mong muốn")
    p.add_argument("--fps", type=float, default=30.0, help="FPS mục tiêu")
    p.add_argument("--show", action="store_true", help="hiện cửa sổ preview OpenCV")
    args = p.parse_args()

    cv2 = None
    if args.show:
        import cv2 as _cv2

        cv2 = _cv2

    stream = CameraStream(args.cam, width=args.width, height=args.height, fps=args.fps)
    stream.start()
    print(f"Camera #{args.cam} đang chạy. Nhấn Ctrl+C để dừng.")

    deadline = time.monotonic() + args.seconds
    last_report = 0.0
    try:
        while time.monotonic() < deadline:
            frame = stream.read(timeout=1.0)
            now = time.monotonic()
            if now - last_report >= 1.0:
                print(
                    f"frames={stream.frame_count} "
                    f"fps~{stream.measured_fps:.1f} "
                    f"shape={frame.shape}"
                )
                last_report = now

            if cv2 is not None:
                view = frame.copy()
                cv2.putText(
                    view,
                    f"frames={stream.frame_count} fps~{stream.measured_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("camera stream", view)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
    finally:
        stream.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    print("Đã dừng camera stream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
