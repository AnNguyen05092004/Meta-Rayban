"""Thiết bị âm thanh cho vòng lặp giọng nói.

Các dependency được import muộn để phần CPM/test vẫn chạy khi máy chưa cài
sounddevice hoặc faster-whisper. Audio đầu vào luôn được chuẩn hóa thành WAV
mono 16 kHz, phù hợp với các model Whisper.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque


DEFAULT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VADRecording:
    path: Path | None
    duration_seconds: float
    speech_detected: bool
    stop_reason: str


def record_until_silence_wav(
    output_path: str | Path,
    *,
    device: int | str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    start_timeout: float = 8.0,
    max_seconds: float = 12.0,
    silence_seconds: float = 1.0,
    pre_roll_seconds: float = 0.30,
) -> VADRecording:
    """Thu đến khi nghe tiếng nói rồi tự dừng sau một khoảng im lặng.

    WebRTC VAD chỉ nhận PCM mono 16-bit, 8/16/32/48 kHz và frame 10/20/30 ms.
    Hàm này cố định 16 kHz, 30 ms để Whisper nhận WAV sạch, tránh ghi cứng 5 giây.
    """
    if sample_rate not in {8_000, 16_000, 32_000, 48_000} or frame_ms not in {10, 20, 30}:
        raise ValueError("WebRTC VAD cần sample rate 8/16/32/48 kHz và frame 10/20/30 ms.")
    if not 0 < silence_seconds < max_seconds or start_timeout <= 0 or max_seconds <= 0:
        raise ValueError("Tham số VAD không hợp lệ.")
    try:
        import sounddevice as sd
        import soundfile as sf
        import webrtcvad
    except Exception as exc:  # pragma: no cover - optional/device dependency
        raise RuntimeError("Chưa cài WebRTC VAD/audio. Chạy: pip install -r requirements.txt") from exc

    blocksize = sample_rate * frame_ms // 1000
    silence_frames = max(1, round(silence_seconds * 1000 / frame_ms))
    preroll = deque(maxlen=max(1, round(pre_roll_seconds * 1000 / frame_ms)))
    vad = webrtcvad.Vad(aggressiveness)
    collected, voiced_started, quiet = [], False, 0
    frames_limit = round(max_seconds * 1000 / frame_ms)
    timeout_limit = round(start_timeout * 1000 / frame_ms)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=blocksize, device=device) as stream:
            for index in range(frames_limit):
                frame, _overflowed = stream.read(blocksize)
                pcm = frame[:, 0].copy()
                is_speech = vad.is_speech(pcm.tobytes(), sample_rate)
                if not voiced_started:
                    preroll.append(pcm)
                    if is_speech:
                        voiced_started = True
                        collected.extend(preroll)
                    elif index >= timeout_limit:
                        return VADRecording(None, 0.0, False, "no_speech")
                    continue
                collected.append(pcm)
                quiet = 0 if is_speech else quiet + 1
                if quiet >= silence_frames:
                    break
            else:
                return _save_vad(path, collected, sample_rate, "max_duration", sf)
    except Exception as exc:  # pragma: no cover - real hardware
        raise RuntimeError("Không thu được microphone. Kiểm tra quyền Microphone cho Terminal/VS Code.") from exc
    return _save_vad(path, collected, sample_rate, "silence", sf)


def _save_vad(path: Path, frames, sample_rate: int, stop_reason: str, soundfile_module) -> VADRecording:
    import numpy as np

    if not frames:
        return VADRecording(None, 0.0, False, "no_speech")
    audio = np.concatenate(frames).reshape(-1, 1)
    soundfile_module.write(path, audio, sample_rate, subtype="PCM_16")
    return VADRecording(path, len(audio) / sample_rate, True, stop_reason)


def record_wav(
    output_path: str | Path,
    seconds: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int | str | None = None,
) -> Path:
    """Thu microphone trong ``seconds`` giây và lưu WAV mono 16 kHz.

    Quyền microphone do macOS quản lý. Lỗi quyền được đổi thành hướng dẫn có
    thể hành động thay vì để callback âm thanh làm sập ứng dụng.
    """
    if seconds <= 0:
        raise ValueError("Thời lượng thu âm phải lớn hơn 0 giây.")
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Chưa cài audio dependencies. Chạy: pip install -r requirements.txt"
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:  # pragma: no cover - needs real hardware
        raise RuntimeError(
            "Không thu được microphone. Hãy cấp quyền Microphone cho Terminal/VS Code "
            "trong System Settings > Privacy & Security > Microphone, rồi chạy lại."
        ) from exc

    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def list_input_devices() -> list[dict[str, object]]:
    """Trả danh sách microphone theo định dạng dễ in ra CLI."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chưa cài sounddevice. Chạy: pip install -r requirements.txt") from exc

    devices = []
    for index, item in enumerate(sd.query_devices()):
        if item["max_input_channels"] > 0:
            devices.append({"index": index, "name": item["name"], "channels": item["max_input_channels"]})
    return devices
