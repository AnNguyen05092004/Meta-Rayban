"""
Gradio UI for the CPM visual assistant demo.

Run:
  python -m app.demo_gradio
  python -m app.demo_gradio --embedder real
  python -m app.demo_gradio --embedder real --server-port 7861
"""

from __future__ import annotations

import argparse
import os

# numpy: thư viện xử lý mảng số, dùng để thao tác ảnh (ảnh = mảng các con số).
import numpy as np

# Assistant: bộ khung dạy/hỏi/sửa (chứa đôi mắt + trí nhớ CPM).
from app.cli import Assistant
from app.bootstrap_memory import bootstrap_directory
from app.interaction_log import InteractionLogger
from app.orchestrator import VisionAssistant
from app.voice import VoiceController
# SyntheticEmbedder: đôi mắt GIẢ; dùng để nhận biết đang chạy chế độ giả hay thật.
from perception.embed import SyntheticEmbedder
from skills.providers import faster_whisper_stt, synthesize_macos_wav


# _normalize_image_rgb: chuẩn hoá ảnh Gradio gửi vào về dạng thống nhất.
# Vào: ảnh bất kỳ (xám/RGB/RGBA, kiểu số khác nhau)
# Ra: mảng ảnh RGB, mỗi điểm ảnh là số nguyên 0..255 (uint8), 3 kênh màu.
# Mục đích: các bước sau luôn nhận đúng một định dạng ảnh, tránh lỗi vặt.
def _normalize_image_rgb(image_rgb) -> np.ndarray:
    """Convert Gradio image input to contiguous uint8 RGB."""
    if image_rgb is None:
        raise ValueError("Chưa có ảnh. Hãy chụp webcam hoặc tải ảnh lên trước.")

    arr = np.asarray(image_rgb)
    if arr.size == 0:
        raise ValueError("Ảnh rỗng. Hãy chụp hoặc tải lại ảnh khác.")
    # Ảnh xám (2 chiều, không có kênh màu) -> nhân thành 3 kênh cho giống ảnh màu.
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    # Phải là ảnh 3 chiều với 3 (RGB) hoặc 4 (RGBA) kênh, không thì báo lỗi.
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Định dạng ảnh không hợp lệ: shape={arr.shape}.")
    # Ảnh RGBA (4 kênh) -> bỏ kênh trong suốt, chỉ giữ 3 kênh màu RGB.
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    # Ép mọi điểm ảnh về số nguyên 0..255 (kẹp giá trị lạc ra ngoài về trong khoảng).
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Trả mảng "liền mạch trong bộ nhớ" để các thư viện ảnh xử lý nhanh, không lỗi.
    return np.ascontiguousarray(arr)


# _friendly_error: đổi lỗi kỹ thuật khó hiểu thành câu tiếng Việt dễ hiểu cho người dùng.
# Vào: lỗi (exception) + loại đối tượng -> Ra: câu nhắc rõ ràng nên làm gì.
def _friendly_error(exc: Exception, modality: str) -> str:
    msg = str(exc)
    lower = msg.lower()
    # Trường hợp hay gặp nhất: không tìm thấy mặt -> hướng dẫn chỉnh lại camera/ánh sáng.
    if "không phát hiện" in lower or "khong phat hien" in lower:
        return "Không phát hiện khuôn mặt rõ. Hãy nhìn thẳng camera, đủ sáng, chỉ một mặt chính."
    # Các lỗi về ảnh đầu vào vốn đã dễ hiểu -> giữ nguyên câu.
    if "chưa có ảnh" in lower or "ảnh rỗng" in lower or "định dạng ảnh" in lower:
        return msg
    # Còn lại: ghi rõ đang xử lý mặt hay đồ vật để dễ đoán nguyên nhân.
    if modality == "face":
        return f"Lỗi xử lý khuôn mặt: {msg}"
    return f"Lỗi xử lý đồ vật: {msg}"


# _to_embedding: biến ảnh của giao diện -> dấu vân tay số, chọn ĐÚNG cách theo
# đôi mắt đang dùng (giả hay thật) và theo loại đối tượng (mặt hay đồ vật).
# Lý do phải phân nhánh: mỗi model thật đòi ảnh ở một định dạng riêng.
def _to_embedding(assistant: Assistant, image_rgb, modality: str) -> np.ndarray:
    # Chuẩn hoá ảnh trước cho chắc chắn (RGB, uint8).
    image_rgb = _normalize_image_rgb(image_rgb)
    emb = assistant.embedder

    # Đôi mắt GIẢ: nhận thẳng ảnh RGB, không cần đổi định dạng.
    if isinstance(emb, SyntheticEmbedder):
        return emb.embed_face(image_rgb) if modality == "face" else emb.embed_object(image_rgb)

    # Đôi mắt THẬT + mặt: InsightFace/ArcFace cần ảnh theo thứ tự màu BGR (OpenCV).
    if modality == "face":
        import cv2

        # cvtColor: đổi thứ tự màu RGB -> BGR cho đúng ý model.
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return emb.embed_face(bgr)

    # Đôi mắt THẬT + đồ vật: CLIP nhận ảnh dạng PIL, nên đổi mảng numpy -> ảnh PIL.
    from PIL import Image

    return emb.embed_object(Image.fromarray(image_rgb))


# build_demo: dựng toàn bộ giao diện web Gradio (webcam live/upload + các nút).
# Trả về đối tượng "demo" để ở dưới gọi .launch() mở trang web.
def build_demo(
    embedder_kind: str = "synthetic",
    *,
    user_id: str = "demo",
    memory_dir: str = ".local_memory",
    bootstrap_objects: str | None = None,
    bootstrap_faces: str | None = None,
    obstacle_mode: str = "stub",
    voice_model: str = "small",
    log_dir: str = ".local_logs",
):
    # import tại đây (lazy) để chỉ nạp Gradio khi thật sự mở giao diện.
    import gradio as gr

    # Một trợ lý dùng chung cho cả 3 nút; nhớ những gì đã dạy trong suốt phiên.
    assistant = Assistant(embedder_kind=embedder_kind, user_id=user_id, memory_dir=memory_dir)
    for modality, directory in (("object", bootstrap_objects), ("face", bootstrap_faces)):
        if directory:
            if assistant.cpm[modality].labels():
                raise RuntimeError(
                    f"Bộ nhớ {modality} của user '{user_id}' đã có nhãn. "
                    "Không bootstrap lại để tránh ghi trùng dữ liệu."
                )
            imported = bootstrap_directory(assistant, directory, modality)
            print(f"[memory] Đã nạp {modality}: {imported}")
    # Dùng chung `assistant` để các nút VLM/OCR và CPM thấy cùng một phiên nhớ.
    # Không có OpenAI key thì dùng Stub ngay, tránh EasyOCR tự tải model khi mở UI.
    # EasyOCR vẫn có thể bật chủ động cho CLI bằng ENABLE_EASYOCR=1.
    skills_mode = "real" if os.environ.get("OPENAI_API_KEY") else "stub"
    vision = VisionAssistant(
        embedder_kind=embedder_kind, skills_mode=skills_mode, core=assistant, obstacle_mode=obstacle_mode
    )
    voice = VoiceController(
        vision,
        stt_factory=lambda: faster_whisper_stt(voice_model),
        logger=InteractionLogger(log_dir, user_id=user_id),
    )

    # _stats: câu tóm tắt đang nhớ bao nhiêu nhãn, hiện dưới giao diện.
    def _stats() -> str:
        return "Bộ nhớ - " + assistant.stats()

    def _select_image(live_image, upload_image, image_source):
        if image_source == "upload":
            return upload_image
        return live_image

    def _safety_warning(image_rgb) -> str | None:
        """Chặn thao tác CPM trực tiếp của UI khi safety đang báo nguy hiểm."""
        assessment = vision.check_safety(image_rgb)
        return assessment["text"] if assessment.get("danger") else None

    # teach: xử lý nút "Ghi nhớ" — biến ảnh thành dấu vân tay rồi dạy CPM tên = label.
    def teach(live_image, upload_image, image_source, modality, label):
        # Chưa nhập tên thì nhắc, không ghi bừa.
        if not label or not label.strip():
            return "Nhập nhãn trước khi Ghi nhớ.", _stats()
        try:
            image = _normalize_image_rgb(_select_image(live_image, upload_image, image_source))
            warning = _safety_warning(image)
            if warning:
                return warning, _stats()
            key = _to_embedding(assistant, image, modality)
            # write() = dạy CPM.
            assistant.cpm[modality].write(key, label.strip())
            assistant.persist_memory()
            return f"Đã ghi nhớ ({modality}): {label.strip()}", _stats()
        # Bọc lỗi lại thành câu dễ hiểu để không hiện lỗi kỹ thuật lên giao diện.
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    # ask: xử lý nút "Hỏi" — nhận diện ảnh đang xem là ai/cái gì đã học chưa.
    def ask(live_image, upload_image, image_source, modality):
        try:
            image = _normalize_image_rgb(_select_image(live_image, upload_image, image_source))
            warning = _safety_warning(image)
            if warning:
                return warning, _stats()
            key = _to_embedding(assistant, image, modality)
            # recall() trả danh sách phỏng đoán; [0] = khớp nhất.
            res = assistant.cpm[modality].recall(key)[0]
            # known=False: chưa đủ giống -> báo là chưa biết kèm độ tương đồng.
            if not res["known"]:
                return f"Chưa biết ({modality}) - độ tương đồng {res['confidence']:.2f}", _stats()
            # known=True: báo tên, độ tin cậy và tầng bộ nhớ đã trả lời.
            return (
                f"Đây là **{res['label']}** "
                f"(tin cậy {res['confidence']:.2f}, tầng {res['tier']})",
                _stats(),
            )
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    # fix: xử lý nút "Sửa" — khi trợ lý đoán sai, dạy lại tên ĐÚNG cho ảnh này.
    def fix(live_image, upload_image, image_source, modality, label):
        if not label or not label.strip():
            return "Nhập nhãn đúng trước khi Sửa.", _stats()
        try:
            image = _normalize_image_rgb(_select_image(live_image, upload_image, image_source))
            warning = _safety_warning(image)
            if warning:
                return warning, _stats()
            key = _to_embedding(assistant, image, modality)
            # correct() = sửa CPM, nhấn mạnh ảnh này thuộc về tên đúng.
            assistant.cpm[modality].correct(key, label.strip())
            assistant.persist_memory()
            return f"Đã sửa ({modality}): đây là {label.strip()}", _stats()
        except Exception as exc:
            return f"Lỗi: {_friendly_error(exc, modality)}", _stats()

    def describe(live_image, upload_image, image_source):
        try:
            image = _normalize_image_rgb(_select_image(live_image, upload_image, image_source))
            return vision.handle("Mô tả ngắn gọn khung cảnh trước mặt", frame=image), _stats()
        except Exception as exc:
            return f"Lỗi mô tả cảnh: {_friendly_error(exc, 'object')}", _stats()

    def read_text(live_image, upload_image, image_source):
        try:
            image = _normalize_image_rgb(_select_image(live_image, upload_image, image_source))
            return vision.handle("Đọc chữ trong khung hình", frame=image), _stats()
        except Exception as exc:
            return f"Lỗi đọc chữ: {_friendly_error(exc, 'object')}", _stats()

    def live_ask(live_image, image_source, modality, recognize_enabled, obstacle_enabled):
        if image_source != "webcam" or not (recognize_enabled or obstacle_enabled):
            return gr.skip(), gr.skip()
        try:
            # Khi đang tự nhận diện, safety luôn phải được kiểm tra trước; không
            # phụ thuộc checkbox hiển thị vật cản để tránh bỏ qua cảnh báo.
            safety = vision.check_safety(live_image)
            if safety.get("danger"):
                return f"Live: {safety['text']}", _stats()
            if recognize_enabled:
                key = _to_embedding(assistant, live_image, modality)
                res = assistant.cpm[modality].recall(key)[0]
                if not res["known"]:
                    message = f"Live: chưa biết ({modality}) - độ tương đồng {res['confidence']:.2f}"
                else:
                    message = f"Live: đây là **{res['label']}** (tin cậy {res['confidence']:.2f}, tầng {res['tier']})"
                return message, _stats()
            return f"Live: {safety['text']}", _stats()
        except Exception as exc:
            return f"Live: {_friendly_error(exc, modality)}", _stats()

    def voice_submit(audio_path, live_image, upload_image, image_source, state):
        try:
            if not audio_path:
                return "", "Hãy ghi âm một câu trước khi gửi.", None, state or {}, _stats()
            image = _select_image(live_image, upload_image, image_source)
            frame = _normalize_image_rgb(image) if image is not None else None
            turn = voice.process_audio(audio_path, frame, state)
            audio_out = synthesize_macos_wav(turn.response) if turn.response else None
            return turn.transcript, turn.response, audio_out, turn.state, _stats()
        except Exception as exc:
            return "", f"Lỗi voice: {_friendly_error(exc, 'object')}", None, state or {}, _stats()

    # Từ đây trở xuống là "xếp hình" giao diện: tiêu đề, ô ảnh, các nút, ô kết quả.
    with gr.Blocks(title="Trợ lý CPM - nhận diện và ghi nhớ") as demo:
        gr.Markdown(
            "# Trợ lý thị giác - nhận diện và ghi nhớ (CPM / Nested Learning)\n"
            "Dạy hệ thống người quen và đồ cá nhân, hỏi lại, sửa sai - học liên tục, không quên."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_source = gr.Radio(
                    ["webcam", "upload"],
                    value="webcam",
                    label="Nguồn ảnh",
                )
                live_image = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam live",
                    visible=True,
                )
                upload_image = gr.Image(
                    sources=["upload", "clipboard"],
                    type="numpy",
                    label="Ảnh upload",
                    visible=False,
                )
                modality = gr.Radio(["face", "object"], value="face", label="Loại đối tượng")
                auto_live = gr.Checkbox(value=False, label="Tự nhận diện live")
                auto_obstacle = gr.Checkbox(value=False, label="Theo dõi vật cản live")
                label = gr.Textbox(label="Nhãn", placeholder="VD: Lan / ví của tôi")
                with gr.Row():
                    btn_teach = gr.Button("Ghi nhớ", variant="primary")
                    btn_ask = gr.Button("Hỏi", variant="secondary")
                    btn_fix = gr.Button("Sửa")
                with gr.Row():
                    btn_describe = gr.Button("Mô tả cảnh")
                    btn_read_text = gr.Button("Đọc chữ")
                voice_audio = gr.Audio(sources=["microphone"], type="filepath", format="wav", label="Nói với trợ lý")
                btn_voice = gr.Button("Xử lý lời nói", variant="primary")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Kết quả")
                stats = gr.Markdown(_stats())
                transcript = gr.Textbox(label="Bạn nói", interactive=False)
                voice_reply = gr.Textbox(label="Trợ lý", interactive=False)
                voice_output = gr.Audio(label="Phản hồi giọng nói", autoplay=True, type="filepath")
                voice_state = gr.State({})

        # Nối mỗi nút với hàm xử lý: bấm nút -> chạy hàm với các ô đầu vào ->
        # ghi kết quả ra ô [out] (kết quả) và [stats] (tóm tắt bộ nhớ).
        def switch_source(src):
            return gr.update(visible=src == "webcam"), gr.update(visible=src == "upload")

        image_source.change(switch_source, [image_source], [live_image, upload_image])
        live_image.stream(
            live_ask,
            [live_image, image_source, modality, auto_live, auto_obstacle],
            [out, stats],
            stream_every=1.0,
            trigger_mode="always_last",
            show_progress="hidden",
        )
        btn_teach.click(teach, [live_image, upload_image, image_source, modality, label], [out, stats])
        btn_ask.click(ask, [live_image, upload_image, image_source, modality], [out, stats])
        btn_fix.click(fix, [live_image, upload_image, image_source, modality, label], [out, stats])
        btn_describe.click(describe, [live_image, upload_image, image_source], [out, stats])
        btn_read_text.click(read_text, [live_image, upload_image, image_source], [out, stats])
        btn_voice.click(
            voice_submit,
            [voice_audio, live_image, upload_image, image_source, voice_state],
            [transcript, voice_reply, voice_output, voice_state, stats],
            concurrency_limit=1,
        )

    return demo.queue(default_concurrency_limit=1)


# main: điểm vào khi chạy "python -m app.demo_gradio".
# Đọc tuỳ chọn dòng lệnh, dựng giao diện rồi mở trang web.
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    p.add_argument("--user-id", default="demo", help="user ID local, chỉ dùng chữ/số/_/-")
    p.add_argument("--memory-dir", default=".local_memory", help="thư mục lưu CPM local an toàn")
    p.add_argument("--bootstrap-objects", help="nạp một lần data/objects/<nhãn>/* vào CPM")
    p.add_argument("--bootstrap-faces", help="nạp một lần data/faces/<nhãn>/* vào CPM")
    p.add_argument("--obstacle-mode", default="stub", choices=["stub", "real"])
    p.add_argument("--voice-model", default="small", help="model faster-whisper cho UI")
    p.add_argument("--log-dir", default=".local_logs")
    # --share: tạo link công khai tạm thời để người khác truy cập từ xa.
    p.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    # --server-port: đổi cổng nếu cổng mặc định 7860 đang bận.
    p.add_argument("--server-port", type=int, default=None, help="use another port if 7860 is busy")
    args = p.parse_args()
    # launch(): mở máy chủ web và hiển thị giao diện.
    build_demo(
        args.embedder,
        user_id=args.user_id,
        memory_dir=args.memory_dir,
        bootstrap_objects=args.bootstrap_objects,
        bootstrap_faces=args.bootstrap_faces,
        obstacle_mode=args.obstacle_mode,
        voice_model=args.voice_model,
        log_dir=args.log_dir,
    ).launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
