# Voice - Hướng dẫn nghiệm thu

Tài liệu này mô tả voice slice hiện có trên laptop/M4. Nó dùng **cùng** CPM local với UI: cùng `--user-id` và `--memory-dir` nghĩa là những gì đã dạy ở UI sẽ được nhận ra trong terminal voice, và câu dạy đã xác nhận từ voice cũng được lưu để UI thấy ở lần mở sau.

## 1. Mở đúng môi trường

```bash
cd "/Users/an/Documents/AI GLASS/Meta-Rayban"
deactivate 2>/dev/null || true
source .venv312-arm64/bin/activate
source ~/.zshrc
python -c 'import webrtcvad; print("WebRTC VAD OK")'
```

Vào **System Settings > Privacy & Security > Camera/Microphone**, cấp quyền cho đúng ứng dụng chạy lệnh (Terminal, iTerm hoặc VS Code), rồi đóng và mở lại ứng dụng đó một lần.

## 2. Chạy voice dùng chung bộ nhớ UI

UI mặc định dùng `user-id=demo`, `memory-dir=.local_memory`; chạy terminal cùng giá trị đó:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.voice_loop \
  --cam 0 --user-id demo --memory-dir .local_memory \
  --stt-model small --tts macos
```

Lần đầu `faster-whisper` tải model `small`; các lần sau dùng cache. Nhấn Enter để bắt đầu một lượt. VAD chờ tiếng nói, giữ một đoạn đầu ngắn để không mất âm tiết đầu, rồi tự dừng sau khoảng 1 giây im lặng. `--seconds 12` là giới hạn tối đa, không phải thời lượng ghi cố định. Gõ `q` rồi Enter để dừng.

Nếu cần xem microphone:

```bash
python -m scripts.voice_loop --list-devices
```

Sau đó thêm `--mic <số-thứ-tự>` vào lệnh chính nếu macOS chọn nhầm input. `--no-vad` chỉ dành cho debug; nó quay về thu cố định theo `--seconds`.

## 3. Kịch bản xác nhận CPM

Đặt mặt hoặc đồ vật rõ trong khung hình, rồi thử theo đúng thứ tự:

1. Nói: `Đây là An`.
2. Hệ thống phải hỏi lại: `Bạn muốn ghi nhớ An cho face, đúng không?`.
3. Nói `Đúng` trong vòng 30 giây. Terminal phải báo đã ghi nhớ; khởi động lại UI cùng `--user-id demo` để thấy nhãn còn đó.
4. Lặp lại nhưng nói `Hủy`. Nhãn mới không được xuất hiện.
5. Hỏi `Ai đây?` khi không có frame hợp lệ. Hệ thống phải nói chưa có khung hình, không suy đoán.
6. Nếu độ tin cậy CPM dưới `0.80`, hệ thống chỉ nói `Tôi đoán là ...` và yêu cầu xác nhận/sửa, không tự ghi bộ nhớ.

Để dạy đồ vật, nói câu có gợi ý đồ vật, ví dụ `Đây là chiếc ví của tôi`; để sửa, nói `Sửa, đây là ví của An`, rồi xác nhận `Đúng`.

## 4. Voice trong UI

Mở UI bình thường với chính user và memory dir ở trên:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio \
  --embedder real --user-id demo --memory-dir .local_memory \
  --server-port 7863
```

Ở khối **Nói với trợ lý**, bấm biểu tượng record của browser để bắt đầu, bấm lại để kết thúc, rồi bấm **Xử lý lời nói**. UI hiển thị transcript, câu trả lời và phát WAV TTS trong browser. Đây là điều khiển record chuẩn của Gradio, chưa phải press-and-hold native; phần đó và wake-word được để lại cho app mobile/kính vì cần kiểm thử kích hoạt nhầm trên thiết bị thật.

Để các câu nhận diện/dạy được dùng frame đúng, giữ webcam đang hiển thị rõ đối tượng hoặc chọn ảnh upload trước khi gửi audio.

## 5. Safety và không chồng tiếng

Chạy safety nền cùng voice:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.voice_loop \
  --cam 0 --user-id demo --memory-dir .local_memory \
  --obstacle-mode real --safety-interval 0.75 --safety-audio
```

Trên TTS macOS mặc định, `SpeechCoordinator` đảm bảo chỉ có một câu được đọc. Khi có cảnh báo mới, nó dừng câu thường đang phát trước khi đọc cảnh báo. Khi monitor đang báo nguy hiểm, voice chặn VLM/OCR và chỉ trả cảnh báo; không mô tả hay OCR tiếp trong tình huống đó. Vẫn cần test C0-C5 trong `SPRINT_2_NGHIEM_THU.md` với người quan sát, không dùng cảnh báo như thiết bị an toàn độc lập.

## 6. Log nghiệm thu và riêng tư

Mỗi lượt được ghi tại `.local_logs/voice_<user-id>.jsonl`. Một dòng có timestamp, transcript, response, route/skill, độ trễ STT/xử lý, confidence CPM (khi nhận diện), trạng thái safety và lý do dừng VAD. Log **không** chứa ảnh, WAV hay API key; thư mục đã bị ignore bởi Git.

Xem nhanh 10 lượt mới nhất:

```bash
tail -n 10 .local_logs/voice_demo.jsonl
```

Đừng commit log giọng nói thật. Chỉ trích các metric đã ẩn danh vào báo cáo nghiệm thu.

## 7. Những việc cố ý chưa làm

- Wake-word tiếng Việt và press-and-hold native: làm sau khi push-to-talk/VAD ổn định trên phone/kính, có đo false trigger/false reject.
- STT chạy `faster-whisper` CPU int8 trên Apple Silicon; CTranslate2 không dùng PyTorch MPS. CPM là NumPy CPU, còn CLIP/YOLO/Depth dùng MPS khi backend hỗ trợ.
- UI là demo một user local. Multi-user đăng nhập, backend và sync thuộc Sprint 3-5; local JSON/NPZ hiện tại không thay thế backend đó.
