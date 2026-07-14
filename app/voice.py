"""Lõi hội thoại giọng nói dùng chung cho terminal và UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app.interaction_log import InteractionLogger
from app.orchestrator import VisionAssistant


_YES = {"đúng", "vâng", "có", "đồng ý", "xác nhận", "yes"}
_NO = {"không", "không phải", "hủy", "huỷ", "bỏ qua", "no"}


@dataclass
class VoiceTurn:
    transcript: str
    response: str
    state: dict
    latency: dict[str, float]
    route: str


class VoiceController:
    """STT lazy + state xác nhận trước mọi ghi/sửa CPM."""

    def __init__(
        self,
        assistant: VisionAssistant,
        *,
        stt_factory: Callable[[], Callable[[str], str]],
        logger: InteractionLogger | None = None,
        confirm_threshold: float = 0.80,
        confirmation_timeout_seconds: float = 30.0,
    ):
        self.assistant = assistant
        self._stt_factory = stt_factory
        self._stt = None
        self.logger = logger
        self.confirm_threshold = confirm_threshold
        self.confirmation_timeout_seconds = confirmation_timeout_seconds

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        started = time.perf_counter()
        if self._stt is None:
            self._stt = self._stt_factory()
        return self._stt(audio_path).strip(), time.perf_counter() - started

    def process_audio(self, audio_path: str, frame, state: dict | None = None) -> VoiceTurn:
        transcript, stt_seconds = self.transcribe(audio_path)
        if not transcript:
            turn = VoiceTurn("", "Tôi chưa nghe rõ. Bạn nói lại gần microphone hơn nhé.", state or {}, {"stt": stt_seconds}, "empty")
            if self.logger:
                self.logger.write(transcript="", response=turn.response, route="empty", latency=turn.latency)
            return turn
        turn = self.process_text(transcript, frame, state)
        turn.latency["stt"] = stt_seconds
        return turn

    def process_text(self, transcript: str, frame, state: dict | None = None) -> VoiceTurn:
        state = dict(state or {})
        started = time.perf_counter()
        text = transcript.strip()
        low = text.lower().strip(" .?!,;:")

        # Safety phải thắng cả xác nhận ghi/sửa lẫn nhận diện. Giữ nguyên state
        # pending để người dùng có thể xác nhận lại sau khi tình huống an toàn.
        safety = self.assistant.check_safety(frame)
        if safety.get("danger"):
            return self._finish(text, safety["text"], state, "safety_override", started, safety=True)

        pending = state.get("pending")
        if pending:
            if time.time() - pending.get("created_at", 0) > self.confirmation_timeout_seconds:
                state.pop("pending", None)
                return self._finish(text, "Xác nhận đã hết hạn nên tôi chưa thay đổi bộ nhớ. Bạn hãy nói lại từ đầu.", state, "confirm_expired", started)
            if low in _YES:
                if frame is None:
                    response, route = "Tôi chưa có khung hình để ghi nhớ. Hãy hướng camera vào đối tượng rồi nói lại.", "confirm_missing_frame"
                else:
                    try:
                        embedding = self.assistant._embed(frame, pending["modality"])
                        memory = self.assistant.core.cpm[pending["modality"]]
                        if pending["action"] == "teach":
                            memory.write(embedding, pending["label"])
                            verb = "ghi nhớ"
                        else:
                            memory.correct(embedding, pending["label"])
                            verb = "sửa"
                        self.assistant.core.persist_memory()
                        response, route = f"Đã {verb} {pending['label']}.", f"confirm_{pending['action']}"
                    except Exception as exc:
                        response, route = f"Tôi chưa thể lưu lúc này: {exc}", "confirm_error"
                state.pop("pending", None)
                return self._finish(text, response, state, route, started)
            if low in _NO:
                state.pop("pending", None)
                return self._finish(text, "Đã hủy, tôi sẽ không thay đổi bộ nhớ.", state, "confirm_cancel", started)
            return self._finish(text, "Bạn chỉ cần nói 'đúng' để xác nhận hoặc 'hủy' để bỏ qua.", state, "confirm_wait", started)

        action = "teach" if any(word in low for word in self.assistant.TEACH_KW) else "correct" if any(word in low for word in self.assistant.CORRECT_KW) else None
        if action:
            label = self.assistant._parse_label(text)
            if label:
                modality = self.assistant._modality(text)
                state["pending"] = {"action": action, "label": label, "modality": modality, "created_at": time.time()}
                verb = "ghi nhớ" if action == "teach" else "sửa"
                return self._finish(text, f"Bạn muốn {verb} {label} cho {modality}, đúng không?", state, f"pending_{action}", started)

        if any(word in low for word in self.assistant.RECOG_KW):
            if frame is None:
                return self._finish(text, "Tôi chưa có khung hình để nhận diện. Hãy bật webcam hoặc tải ảnh lên rồi thử lại.", state, "recognition_missing_frame", started)
            modality = self.assistant._modality(text)
            try:
                result = self.assistant.core.cpm[modality].recall(self.assistant._embed(frame, modality))[0]
                if not result["known"]:
                    response = "Tôi chưa nhận ra đối tượng này. Bạn có muốn dạy tôi không?"
                elif result["confidence"] < self.confirm_threshold:
                    response = f"Tôi đoán là {result['label']} (độ tin cậy {result['confidence']:.2f}). Bạn xác nhận hoặc nói 'Sửa, đây là ...'."
                else:
                    response = f"Đây là {result['label']} (độ tin cậy {result['confidence']:.2f})."
                return self._finish(text, response, state, "recognition", started, confidence=result["confidence"], safety=False)
            except Exception as exc:
                return self._finish(text, f"Tôi chưa xử lý được khung hình: {exc}", state, "recognition_error", started)

        response = self.assistant.handle(text, frame=frame)
        return self._finish(text, response, state, "assistant", started)

    def _finish(self, transcript, response, state, route, started, **extra) -> VoiceTurn:
        latency = {"assistant": time.perf_counter() - started}
        if self.logger:
            self.logger.write(transcript=transcript, response=response, route=route, latency=latency, **extra)
        return VoiceTurn(transcript, response, state, latency, route)
