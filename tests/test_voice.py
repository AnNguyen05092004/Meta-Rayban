from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app.cli import Assistant
from app.interaction_log import InteractionLogger
from app.orchestrator import VisionAssistant
from app.voice import VoiceController
from perception.audio import record_until_silence_wav, record_wav
from skills import providers


def test_record_wav_rejects_non_positive_duration(tmp_path):
    with pytest.raises(ValueError, match="lớn hơn 0"):
        record_wav(tmp_path / "audio.wav", 0)


def test_vad_rejects_invalid_frame_configuration(tmp_path):
    with pytest.raises(ValueError, match="WebRTC VAD"):
        record_until_silence_wav(tmp_path / "audio.wav", frame_ms=25)


def test_speak_text_macos_builds_safe_argument_list(monkeypatch):
    called = {}
    monkeypatch.setattr(providers.platform, "system", lambda: "Darwin")

    def fake_run(command, **kwargs):
        called["command"] = command

    monkeypatch.setattr(providers.subprocess, "run", fake_run)
    providers.speak_text("Xin chào", backend="macos", voice="Linh")
    assert called["command"] == ["say", "-v", "Linh", "Xin chào"]


def test_speak_text_macos_uses_vietnamese_voice_by_default(monkeypatch):
    called = {}
    monkeypatch.setattr(providers.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(providers.subprocess, "run", lambda command, **kwargs: called.setdefault("command", command))
    providers.speak_text("Xin chào", backend="macos")
    assert called["command"] == ["say", "-v", "Linh", "Xin chào"]


def test_speech_coordinator_safety_interrupts_active_macos_speech(monkeypatch):
    processes = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command
            self.running = True
            self.terminated = False

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated = True
            self.running = False

        def wait(self, timeout):
            return 0

        def kill(self):
            self.running = False

    monkeypatch.setattr(providers.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(providers.subprocess, "Popen", lambda command: processes.append(FakeProcess(command)) or processes[-1])
    speaker = providers.SpeechCoordinator(backend="macos", voice="Linh")
    assert speaker.speak("Câu trả lời bình thường") is True
    assert speaker.speak("Câu khác") is False
    assert speaker.speak("Cảnh báo vật cản", priority="safety") is True
    assert processes[0].terminated is True
    assert processes[1].command == ["say", "-v", "Linh", "Cảnh báo vật cản"]


def _voice_controller(*, logger=None, timeout=30.0):
    core = Assistant(embedder_kind="synthetic")
    vision = VisionAssistant(core=core)
    return VoiceController(vision, stt_factory=lambda: lambda _path: "đây là Lan", logger=logger, confirmation_timeout_seconds=timeout)


def test_voice_teach_requires_confirmation_before_persisting():
    voice = _voice_controller()
    frame = "person_lan"
    pending = voice.process_text("Đây là Lan", frame)
    assert pending.route == "pending_teach"
    assert voice.assistant.core.cpm["face"].labels() == []

    confirmed = voice.process_text("Đúng.", frame, pending.state)
    assert confirmed.route == "confirm_teach"
    assert voice.assistant.core.cpm["face"].labels() == ["Lan"]


def test_voice_cancel_and_expired_confirmation_do_not_write_memory():
    voice = _voice_controller(timeout=0.01)
    pending = voice.process_text("Đây là Lan", "person_lan")
    cancelled = voice.process_text("Hủy", "person_lan", pending.state)
    assert cancelled.route == "confirm_cancel"
    assert voice.assistant.core.cpm["face"].labels() == []

    expired_state = voice.process_text("Đây là Lan", "person_lan").state
    time.sleep(0.02)
    expired = voice.process_text("đúng", "person_lan", expired_state)
    assert expired.route == "confirm_expired"
    assert voice.assistant.core.cpm["face"].labels() == []


def test_voice_recognition_without_frame_is_explicit():
    turn = _voice_controller().process_text("Ai đây?", None)
    assert turn.route == "recognition_missing_frame"
    assert "chưa có khung hình" in turn.response.lower()


def test_voice_safety_override_blocks_general_skill():
    voice = _voice_controller()
    voice.assistant.obstacle.set_distance(0.7)
    turn = voice.process_text("Đọc chữ trong ảnh", "person_lan")
    assert turn.route == "safety_override"
    assert "cảnh báo" in turn.response.lower()


def test_voice_safety_override_blocks_confirmation_and_recognition():
    voice = _voice_controller()
    frame = "person_lan"
    pending = voice.process_text("Đây là Lan", frame)
    voice.assistant.obstacle.set_distance(0.7)

    confirm = voice.process_text("Đúng", frame, pending.state)
    assert confirm.route == "safety_override"
    assert confirm.state["pending"]["label"] == "Lan"
    assert voice.assistant.core.cpm["face"].labels() == []

    recognition = voice.process_text("Ai đây?", frame)
    assert recognition.route == "safety_override"


def test_interaction_log_is_jsonl_and_user_id_is_safe(tmp_path):
    logger = InteractionLogger(tmp_path, user_id="demo_01")
    logger.write(transcript="Ai đây", route="recognition", latency={"stt": 0.1})
    record = json.loads(logger.path.read_text(encoding="utf-8"))
    assert record["transcript"] == "Ai đây"
    assert "timestamp" in record
    with pytest.raises(ValueError, match="user_id"):
        InteractionLogger(tmp_path, user_id="../escape")


def test_orchestrator_returns_friendly_message_when_face_is_missing(monkeypatch):
    assistant = VisionAssistant(embedder_kind="synthetic")
    monkeypatch.setattr(
        assistant,
        "_embed",
        lambda frame, modality: (_ for _ in ()).throw(ValueError("Không phát hiện khuôn mặt nào trong ảnh.")),
    )
    result = assistant.handle("Ai đây?", frame=np.zeros((4, 4, 3), dtype=np.uint8))
    assert "chưa thấy rõ" in result.lower()


def test_orchestrator_converts_real_face_rgb_to_bgr():
    class FakeEmbedder:
        pass

    class FakeCore:
        embedder = FakeEmbedder()

        def _embed(self, modality, image):
            return image

    assistant = VisionAssistant(core=FakeCore())
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    bgr = assistant._embed(rgb, "face")
    assert bgr.tolist() == [[[30, 20, 10]]]
