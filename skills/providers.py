"""
Optional real providers for OCR and VLM skills.

All factories are lazy: importing this module does not require heavy packages or
API keys. Missing optional dependencies raise RuntimeError with actionable
messages so the orchestrator can fall back to stub skills.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable

import numpy as np


def _as_rgb_array(frame) -> np.ndarray:
    if frame is None:
        raise ValueError("Chua co anh/khung hinh de xu ly.")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Dinh dang anh khong hop le: shape={arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def easyocr_fn(langs: tuple[str, ...] = ("vi", "en"), gpu: bool = False) -> Callable:
    """Return callable(frame_rgb) -> text using EasyOCR."""
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai easyocr. Cai bang: pip install easyocr") from exc

    reader = easyocr.Reader(list(langs), gpu=gpu)

    def run(frame_rgb) -> str:
        arr = _as_rgb_array(frame_rgb)
        results = reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(str(x).strip() for x in results if str(x).strip())

    return run


def gemini_vlm(model: str = "gemini-1.5-flash", api_key: str | None = None) -> Callable:
    """Return callable(frame_rgb, prompt) -> text using Google Gemini."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Thieu GEMINI_API_KEY trong bien moi truong.")
    try:
        import google.generativeai as genai
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai google-generativeai/Pillow. Cai bang: pip install google-generativeai pillow") from exc

    genai.configure(api_key=key)
    m = genai.GenerativeModel(model)

    def run(frame_rgb, prompt: str) -> str:
        arr = _as_rgb_array(frame_rgb)
        image = Image.fromarray(arr)
        response = m.generate_content([prompt, image])
        return getattr(response, "text", "") or ""

    return run


def _encode_image_data_url(frame, fmt: str = "PNG") -> str:
    """RGB array -> data URL base64 (PNG/JPEG) cho OpenAI vision."""
    import base64
    import io

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai Pillow. Cai bang: pip install pillow") from exc

    arr = _as_rgb_array(frame)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _openai_client(api_key: str | None = None, client=None):
    """Tao (hoac nhan) OpenAI client. `client` cho phep tiem gia lap khi test."""
    if client is not None:
        return client
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thieu OPENAI_API_KEY trong bien moi truong.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chua cai openai SDK. Cai bang: pip install openai pillow") from exc
    return OpenAI(api_key=key)


def _openai_vision(client, model: str, prompt: str, frame, max_tokens: int) -> str:
    """Goi 1 luot chat.completions co kem anh, tra ve text."""
    data_url = _encode_image_data_url(frame)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def openai_vlm(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 400,
    client=None,
) -> Callable:
    """VLM (scene/VQA) dung OpenAI GPT-4o. Tra callable(frame_rgb, prompt) -> text."""
    cli = _openai_client(api_key, client)

    def run(frame_rgb, prompt: str) -> str:
        return _openai_vision(cli, model, prompt, frame_rgb, max_tokens)

    return run


_OPENAI_OCR_PROMPT = (
    "Trich xuat CHINH XAC toan bo van ban xuat hien trong anh, giu nguyen thu tu doc va dau "
    "tieng Viet. Chi tra ve phan chu, khong mo ta, khong giai thich. Neu khong co chu, tra ve chuoi rong."
)


def openai_ocr(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 800,
    client=None,
) -> Callable:
    """OCR dung OpenAI GPT-4o vision (khong can cai EasyOCR). Tra callable(frame_rgb) -> text."""
    cli = _openai_client(api_key, client)

    def run(frame_rgb) -> str:
        return _openai_vision(cli, model, _OPENAI_OCR_PROMPT, frame_rgb, max_tokens)

    return run


def faster_whisper_stt(
    model: str = "small",
    *,
    device: str = "auto",
    compute_type: str | None = None,
) -> Callable[[str], str]:
    """Trả callable ``wav_path -> văn bản tiếng Việt`` dùng faster-whisper.

    faster-whisper/CTranslate2 hiện chạy ổn định trên CPU của Apple Silicon;
    không ép MPS vì backend này không dùng PyTorch MPS. Model chỉ được tải lần
    đầu khi factory được gọi, tuyệt đối không lúc import module.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Chưa cài faster-whisper. Chạy: pip install -r requirements.txt"
        ) from exc

    resolved_device = "cpu" if device == "auto" else device
    resolved_compute_type = compute_type or ("int8" if resolved_device == "cpu" else "float16")
    try:
        whisper = WhisperModel(model, device=resolved_device, compute_type=resolved_compute_type)
    except Exception as exc:  # pragma: no cover - model download/runtime dependent
        raise RuntimeError(
            f"Không nạp được Whisper model '{model}'. Kiểm tra mạng/dung lượng rồi thử lại."
        ) from exc

    def run(audio_path: str) -> str:
        try:
            segments, _info = whisper.transcribe(
                audio_path,
                language="vi",
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            return "".join(segment.text for segment in segments).strip()
        except Exception as exc:  # pragma: no cover - needs model/audio
            raise RuntimeError("Không thể chuyển giọng nói thành văn bản. Hãy thử ghi âm lại.") from exc

    return run


def _speak_macos(text: str, voice: str | None = None) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("TTS macOS chỉ dùng được trên macOS. Chọn --tts piper hoặc --tts gtts.")
    # Linh là voice vi_VN có sẵn trên M4 đã kiểm tra cho môi trường project.
    # Người dùng vẫn có thể truyền --tts-voice để chọn giọng hệ thống khác.
    cmd = ["say", "-v", voice or "Linh"]
    cmd.append(text)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("Không tìm thấy lệnh say của macOS.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("macOS không đọc được câu trả lời. Thử chọn giọng khác bằng --tts-voice.") from exc


def synthesize_macos_wav(text: str, *, voice: str | None = None, output_dir: str = ".local_audio") -> str:
    """Tạo WAV cho browser autoplay; không phát loa trên server."""
    from pathlib import Path
    import time

    if platform.system() != "Darwin":
        raise RuntimeError("Xuất WAV hiện dùng macOS say; browser cần chạy trên máy Mac host.")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.wav"):
        if time.time() - old.stat().st_mtime > 24 * 3600:
            old.unlink(missing_ok=True)
    output = folder / f"reply_{time.time_ns()}.wav"
    try:
        subprocess.run(
            ["say", "-v", voice or "Linh", "-o", str(output), "--file-format=WAVE", "--data-format=LEI16@22050", text],
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("macOS không tạo được WAV cho browser.") from exc
    return str(output)


def _speak_piper(text: str, model_path: str | None) -> None:
    if not model_path:
        raise RuntimeError("TTS Piper cần --piper-model /đường/dẫn/vi_VN-*.onnx.")
    binary = shutil.which("piper")
    if not binary:
        raise RuntimeError("Chưa cài Piper. Chạy: pip install -r requirements.txt")
    player = shutil.which("afplay") if platform.system() == "Darwin" else None
    if not player:
        raise RuntimeError("Chưa có trình phát WAV cho Piper trên hệ điều hành này.")
    with tempfile.TemporaryDirectory(prefix="meta_rayban_tts_") as tmp:
        output = os.path.join(tmp, "reply.wav")
        try:
            subprocess.run(
                [binary, "--model", model_path, "--output_file", output],
                input=text,
                text=True,
                check=True,
            )
            subprocess.run([player, output], check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("Piper không tạo/phát được audio. Kiểm tra file model và thử lại.") from exc


def _speak_gtts(text: str) -> None:
    try:
        from gtts import gTTS
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chưa cài gTTS tùy chọn. Chạy: pip install gTTS") from exc
    player = shutil.which("afplay") if platform.system() == "Darwin" else None
    if not player:
        raise RuntimeError("gTTS hiện cần macOS afplay để phát file MP3.")
    with tempfile.TemporaryDirectory(prefix="meta_rayban_tts_") as tmp:
        output = os.path.join(tmp, "reply.mp3")
        try:
            gTTS(text=text, lang="vi").save(output)
            subprocess.run([player, output], check=True)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError("gTTS không tạo/phát được audio. Kiểm tra mạng rồi thử lại.") from exc


def speak_text(
    text: str,
    *,
    backend: str = "auto",
    voice: str | None = None,
    piper_model: str | None = None,
) -> None:
    """Đọc câu trả lời, ưu tiên native macOS để demo chạy ngay.

    ``auto`` chọn macOS ``say`` trên M4; Piper là lựa chọn offline chất lượng
    cao khi nhóm đã tải model tiếng Việt; gTTS là fallback online.
    """
    if not text or not text.strip():
        return
    selected = "macos" if backend == "auto" and platform.system() == "Darwin" else backend
    if selected == "macos":
        return _speak_macos(text, voice)
    if selected == "piper":
        return _speak_piper(text, piper_model)
    if selected == "gtts":
        return _speak_gtts(text)
    raise ValueError(f"TTS backend không hợp lệ: {backend}")


class SpeechCoordinator:
    """Phát TTS tuần tự; cảnh báo an toàn có thể ngắt ``say`` đang đọc.

    Trên macOS, câu thường chạy trong process riêng để một cảnh báo an toàn có
    thể dừng câu đó ngay. Piper/gTTS vẫn được tuần tự hóa nhưng không có API
    ngắt an toàn tương đương.
    """

    def __init__(self, *, backend: str = "auto", voice: str | None = None, piper_model: str | None = None):
        self.backend = backend
        self.voice = voice
        self.piper_model = piper_model
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None

    def _uses_macos_say(self) -> bool:
        return platform.system() == "Darwin" and self.backend in {"auto", "macos"}

    def speak(self, text: str, *, priority: str = "normal") -> bool:
        """Yêu cầu phát câu nói; câu safety được ưu tiên hơn câu thường."""
        if not text or not text.strip():
            return False
        if priority not in {"normal", "safety"}:
            raise ValueError("priority phải là 'normal' hoặc 'safety'.")
        with self._lock:
            if self._uses_macos_say():
                if self._process is not None and self._process.poll() is None:
                    if priority != "safety":
                        return False
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                self._process = subprocess.Popen(["say", "-v", self.voice or "Linh", text])
                return True
            speak_text(text, backend=self.backend, voice=self.voice, piper_model=self.piper_model)
            return True

    def wait_until_idle(self) -> None:
        """Không thu mic khi chính trợ lý vẫn đang đọc, tránh STT nghe tiếng TTS."""
        while True:
            with self._lock:
                process = self._process
            if process is None or process.poll() is not None:
                return
            time.sleep(0.05)

    def close(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()


def moondream_vlm() -> Callable:
    """Placeholder for a local/offline VLM provider."""
    raise RuntimeError(
        "Moondream provider chua duoc cai dat trong repo nay. "
        "Co the them sau bang transformers/moondream hoac dung Gemini/OpenAI cho MVP."
    )


__all__ = [
    "easyocr_fn",
    "gemini_vlm",
    "openai_vlm",
    "openai_ocr",
    "faster_whisper_stt",
    "speak_text",
    "SpeechCoordinator",
    "synthesize_macos_wav",
    "moondream_vlm",
]
