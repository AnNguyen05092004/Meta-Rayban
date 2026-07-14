from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from perception.capture import CameraStream


class _FakeCapture:
    def __init__(self, cam_index: int, *, fail_open: bool = False, fail_read: bool = False):
        self.cam_index = cam_index
        self.fail_open = fail_open
        self.fail_read = fail_read
        self.released = False
        self.props = {}
        self._counter = itertools.count()

    def isOpened(self):
        return not self.fail_open

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self.fail_read:
            time.sleep(0.01)
            return False, None
        n = next(self._counter)
        frame = np.full((4, 5, 3), n % 255, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


def test_camera_stream_reads_latest_frames_and_releases():
    captures: list[_FakeCapture] = []

    def factory(cam_index):
        cap = _FakeCapture(cam_index)
        captures.append(cap)
        return cap

    stream = CameraStream(2, fps=60, capture_factory=factory)
    try:
        stream.start(timeout=1.0)
        first = stream.read(timeout=1.0)
        time.sleep(0.05)
        latest = stream.latest()

        assert first.shape == (4, 5, 3)
        assert latest is not None
        assert stream.frame_count >= 2
        assert stream.measured_fps > 0
        assert captures[0].cam_index == 2
    finally:
        stream.stop()

    assert captures[0].released
    assert not stream.running


def test_camera_stream_open_error_is_clear():
    stream = CameraStream(9, capture_factory=lambda cam: _FakeCapture(cam, fail_open=True))

    with pytest.raises(RuntimeError, match="Không mở được webcam #9"):
        stream.start()


def test_camera_stream_times_out_when_no_frame_can_be_read():
    stream = CameraStream(0, capture_factory=lambda cam: _FakeCapture(cam, fail_read=True))
    try:
        with pytest.raises(TimeoutError, match="Hết thời gian chờ frame"):
            stream.start(timeout=0.05)
    finally:
        stream.stop()
