"""
Nguồn ảnh: webcam hoặc file. Lazy import OpenCV để không bắt buộc khi chỉ chạy CPM.

`grab_webcam_bgr()` giữ lại đường chụp 1 khung hình cũ. Với các luồng cần chạy
liên tục (voice loop, safety monitor, vật cản real-time), dùng `CameraStream` để
giữ camera mở và luôn có khung hình mới nhất.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    """Đọc ảnh file -> numpy BGR (dùng cho InsightFace)."""
    import cv2  # lazy

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def load_image_rgb(path: str):
    """Đọc ảnh file -> PIL.Image RGB (dùng cho CLIP)."""
    from PIL import Image  # lazy

    return Image.open(path).convert("RGB")


def grab_webcam_bgr(cam_index: int = 0) -> np.ndarray:
    """Chụp 1 khung hình từ webcam -> numpy BGR."""
    import cv2  # lazy

    cap = cv2.VideoCapture(cam_index)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Không lấy được khung hình từ webcam #{cam_index}")
        return frame
    finally:
        cap.release()


class CameraStream:
    """Luồng camera liên tục, thread-safe, trả khung BGR mới nhất.

    Lớp này giải quyết vấn đề mở camera/chụp 1 frame/release lặp lại. Camera được
    mở một lần trong `start()`, một thread nền đọc frame liên tục, còn phần còn lại
    của app gọi `latest()` hoặc `read()` để lấy snapshot hiện tại.
    """

    def __init__(
        self,
        cam_index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        capture_factory: Callable[[int], object] | None = None,
    ):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_factory = capture_factory

        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: np.ndarray | None = None
        self._frame_count = 0
        self._started_at: float | None = None
        self._last_error: Exception | None = None

    def start(self, *, wait_first_frame: bool = True, timeout: float = 3.0) -> "CameraStream":
        """Mở camera và bắt đầu đọc nền.

        `wait_first_frame=True` giúp phát hiện sớm lỗi quyền camera/không có camera.
        """
        if self.running:
            return self

        import cv2  # lazy

        factory = self.capture_factory or cv2.VideoCapture
        cap = factory(self.cam_index)
        if hasattr(cap, "isOpened") and not cap.isOpened():
            raise RuntimeError(
                f"Không mở được webcam #{self.cam_index}. "
                "Kiểm tra quyền camera của Terminal/VS Code hoặc thử --cam 1."
            )

        if self.width is not None and hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None and hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None and hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._cap = cap
        self._stop.clear()
        self._latest = None
        self._frame_count = 0
        self._last_error = None
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._reader_loop, name="CameraStream", daemon=True)
        self._thread.start()

        if wait_first_frame:
            self.read(timeout=timeout)
        return self

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        with self._cond:
            return self._frame_count

    @property
    def measured_fps(self) -> float:
        with self._cond:
            if self._started_at is None:
                return 0.0
            elapsed = max(time.monotonic() - self._started_at, 1e-6)
            return self._frame_count / elapsed

    @property
    def last_error(self) -> Exception | None:
        with self._cond:
            return self._last_error

    def _reader_loop(self) -> None:
        delay = 1.0 / self.fps if self.fps and self.fps > 0 else 0.0
        assert self._cap is not None
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Không đọc được khung hình từ webcam #{self.cam_index}")
                frame = np.ascontiguousarray(frame)
                with self._cond:
                    self._latest = frame
                    self._frame_count += 1
                    self._last_error = None
                    self._cond.notify_all()
            except Exception as exc:  # pragma: no cover - phụ thuộc lỗi thiết bị thật
                with self._cond:
                    self._last_error = exc
                    self._cond.notify_all()
                time.sleep(0.05)

            if delay:
                time.sleep(delay)

    def read(self, *, timeout: float | None = None, copy: bool = True) -> np.ndarray:
        """Chờ tới khi có ít nhất một frame rồi trả về frame mới nhất."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._latest is None:
                if self._last_error is not None and not self.running:
                    raise self._last_error
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    err = f"Hết thời gian chờ frame từ webcam #{self.cam_index}"
                    if self._last_error is not None:
                        err += f": {self._last_error}"
                    raise TimeoutError(err)
                self._cond.wait(remaining)
            return self._latest.copy() if copy else self._latest

    def latest(self, *, copy: bool = True) -> np.ndarray | None:
        """Trả frame mới nhất ngay lập tức; chưa có frame thì trả None."""
        with self._cond:
            if self._latest is None:
                return None
            return self._latest.copy() if copy else self._latest

    def frames(self, *, interval: float = 0.0, copy: bool = True) -> Iterator[np.ndarray]:
        """Iterator tiện dụng cho vòng lặp xử lý frame liên tục."""
        while self.running:
            yield self.read(timeout=1.0, copy=copy)
            if interval > 0:
                time.sleep(interval)

    def stop(self, *, timeout: float = 2.0) -> None:
        """Dừng thread nền và release camera."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        if self._cap is not None and hasattr(self._cap, "release"):
            self._cap.release()
        self._cap = None
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def iter_webcam_bgr(
    cam_index: int = 0,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    interval: float = 0.0,
) -> Iterator[np.ndarray]:
    """Generator frame BGR liên tục, tự mở/đóng camera."""
    with CameraStream(cam_index, width=width, height=height, fps=fps) as stream:
        yield from stream.frames(interval=interval)


def bgr_to_rgb_pil(frame_bgr: np.ndarray):
    """Chuyển khung BGR (cv2) -> PIL RGB (cho CLIP)."""
    import cv2  # lazy
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
