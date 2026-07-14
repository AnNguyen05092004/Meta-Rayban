"""Vòng lặp nói-nghe cho trợ lý thị giác.

Mỗi lượt: nhấn Enter -> thu microphone -> STT -> lấy frame webcam mới nhất ->
VisionAssistant -> TTS. Dùng Enter thay vì global hotkey để không cần quyền
Accessibility trên macOS; khi chuyển sang app điện thoại, nút tai nghe/wake word
sẽ thay bước này.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2

from app.cli import Assistant
from app.interaction_log import InteractionLogger
from app.orchestrator import VisionAssistant
from app.voice import VoiceController
from perception.audio import list_input_devices, record_until_silence_wav, record_wav
from perception.capture import CameraStream
from skills import SafetyMonitor
from skills.providers import SpeechCoordinator, faster_whisper_stt


def _frame_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _print_devices() -> int:
    devices = list_input_devices()
    if not devices:
        print("Không tìm thấy microphone đầu vào.")
        return 1
    for item in devices:
        print(f"[{item['index']}] {item['name']} ({item['channels']} kênh)")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.list_devices:
        return _print_devices()

    core = Assistant(embedder_kind=args.embedder, user_id=args.user_id, memory_dir=args.memory_dir)
    assistant = VisionAssistant(
        embedder_kind=args.embedder, skills_mode=args.skills_mode, obstacle_mode=args.obstacle_mode, core=core
    )
    logger = None if args.no_log else InteractionLogger(args.log_dir, user_id=args.user_id)
    voice = VoiceController(
        assistant,
        stt_factory=lambda: faster_whisper_stt(model=args.stt_model, device=args.stt_device),
        logger=logger,
    )
    camera = CameraStream(args.cam, width=args.width, height=args.height, fps=args.fps)
    monitor = None
    speaker = SpeechCoordinator(backend=args.tts, voice=args.tts_voice, piper_model=args.piper_model)

    print("Đang mở webcam. Lần đầu nạp Whisper có thể mất vài phút để tải model.")
    try:
        camera.start()
        if args.obstacle_mode == "real":
            def on_alert(assessment):
                print(f"[An toàn] {assessment['text']}")
                if args.safety_audio:
                    speaker.speak(assessment["text"], priority="safety")

            monitor = SafetyMonitor(
                assistant.obstacle,
                camera.latest,
                interval=args.safety_interval,
                on_alert=on_alert,
            ).start()
            assistant.set_safety_provider(monitor.latest)
            print(f"Safety monitor đang chạy mỗi {args.safety_interval:.2f}s.")
        print("Sẵn sàng. Nhấn Enter để nói, gõ q rồi Enter để thoát.")
        with tempfile.TemporaryDirectory(prefix="meta_rayban_voice_") as tmp:
            wav_path = Path(tmp) / "utterance.wav"
            while True:
                command = input("> ").strip().lower()
                if command in {"q", "quit", "exit"}:
                    break
                speaker.wait_until_idle()

                if args.no_vad:
                    print(f"Đang nghe trong {args.seconds:.1f} giây...")
                    record_wav(wav_path, args.seconds, device=args.mic)
                    recording = None
                else:
                    print("Đang chờ bạn nói; sẽ tự dừng khi bạn im lặng...")
                    recording = record_until_silence_wav(
                        wav_path, device=args.mic, max_seconds=args.seconds, silence_seconds=args.silence_seconds
                    )
                    if not recording.speech_detected:
                        message = "Tôi chưa nghe thấy lời nói. Bạn thử lại gần microphone hơn nhé."
                        print(f"[Trợ lý] {message}")
                        speaker.speak(message)
                        if logger:
                            logger.write(route="vad_no_speech", transcript="", response=message, latency={}, vad_stop=recording.stop_reason)
                        continue
                frame = camera.latest()
                if frame is None:
                    raise RuntimeError("Webcam chưa trả frame. Kiểm tra quyền Camera rồi thử lại.")
                turn = voice.process_audio(str(wav_path), _frame_rgb(frame))
                if not turn.transcript:
                    message = turn.response
                    print(f"[Trợ lý] {message}")
                    speaker.speak(message)
                    continue
                print(f"[Bạn] {turn.transcript}")
                print(f"[Trợ lý] {turn.response}")
                speaker.speak(turn.response, priority="safety" if turn.route == "safety_override" else "normal")
                if logger and recording is not None:
                    logger.write(route="vad", transcript=turn.transcript, response="", latency={}, vad_stop=recording.stop_reason, vad_duration=recording.duration_seconds)
                print(f"[Độ trễ] STT {turn.latency.get('stt', 0):.2f}s | xử lý {turn.latency['assistant']:.2f}s | route={turn.route}")
    finally:
        if monitor is not None:
            monitor.stop()
        speaker.close()
        camera.stop()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Mic -> STT -> trợ lý thị giác -> TTS")
    p.add_argument("--cam", type=int, default=0, help="chỉ số webcam")
    p.add_argument("--mic", type=int, default=None, help="chỉ số microphone, xem --list-devices")
    p.add_argument("--list-devices", action="store_true", help="in các microphone rồi thoát")
    p.add_argument("--seconds", type=float, default=12.0, help="tối đa một lượt nói; VAD tự dừng sớm")
    p.add_argument("--silence-seconds", type=float, default=1.0, help="im lặng bao lâu thì VAD dừng")
    p.add_argument("--no-vad", action="store_true", help="dùng thu cố định --seconds (chỉ debug)")
    p.add_argument("--stt-model", default="small", help="model faster-whisper, ví dụ base/small")
    p.add_argument("--stt-device", default="auto", choices=["auto", "cpu"], help="faster-whisper hiện chạy CPU")
    p.add_argument("--tts", default="auto", choices=["auto", "macos", "piper", "gtts"])
    p.add_argument("--tts-voice", default=None, help="tên giọng macOS, ví dụ Linh")
    p.add_argument("--piper-model", default=None, help="đường dẫn file .onnx của giọng Piper")
    p.add_argument("--embedder", default="auto", choices=["synthetic", "real", "auto"])
    p.add_argument("--user-id", default="demo", help="dùng chung CPM local với UI")
    p.add_argument("--memory-dir", default=".local_memory", help="thư mục CPM local dùng chung UI")
    p.add_argument("--log-dir", default=".local_logs", help="JSONL transcript/latency cục bộ")
    p.add_argument("--no-log", action="store_true", help="không ghi log nghiệm thu")
    p.add_argument("--skills-mode", default="real", choices=["stub", "real"])
    p.add_argument("--obstacle-mode", default="stub", choices=["stub", "real"])
    p.add_argument("--safety-interval", type=float, default=0.75, help="chu kỳ kiểm tra vật cản (giây)")
    p.add_argument("--safety-audio", action="store_true", help="đọc cảnh báo an toàn ngay khi monitor phát hiện")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=15.0)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
