# Sprint 2 - Nghiệm thu vật cản, camera stream và safety

Sprint 2 đưa camera từ nguồn ảnh liên tục sang cảnh báo vật cản chạy nền. Bản hiện tại dùng **YOLO11n + Depth Anything V2 Small** trên M4/MPS.

> Quan trọng: camera RGB đơn cho **độ sâu tương đối**. Vì vậy hệ thống chỉ nói `gần` hoặc `rất gần`; không được báo số mét hay dùng như thiết bị an toàn y tế/đi đường cho đến khi nhóm hiệu chuẩn và đánh giá false negative trên cảnh thật.

## 1. Trạng thái code

- `perception.capture.CameraStream`: giữ webcam mở, cập nhật frame mới nhất trên thread nền.
- `skills.RealObstacle`: lọc vật trong hành lang giữa ảnh, kết hợp YOLO11n và depth tương đối để chọn vật gần nhất.
- `skills.SafetyMonitor`: chạy inference định kỳ, giữ kết quả mới nhất và rate-limit cảnh báo lặp.
- `VisionAssistant(..., obstacle_mode="real")`: dùng kết quả monitor để SAFETY override đứng trước mọi trả lời khác.
- UI Gradio có checkbox **Theo dõi vật cản live**. `scripts.voice_loop` có `--obstacle-mode real --safety-audio` để đọc cảnh báo nền.

Các checkpoint đã được cache sau smoke test đầu tiên:

- YOLO11n: `.model_cache/yolo11n.pt` (đã ignore Git).
- Depth Anything: Hugging Face cache của user.

Benchmark kỹ thuật ban đầu trên M4, frame tổng hợp 320x240, sau khi model nạp: trung bình **85.5 ms/inference**. Nghiệm thu webcam #0 ngày 2026-07-14 đã có 5 lượt: lượt nạp đầu khoảng 5.98 s, các lượt ấm **188-206 ms**. Đây chưa phải benchmark cảnh thật; phải đo lại bên dưới.

Lời nói cho người dùng luôn là `vật cản` thay vì nhãn YOLO. Nhãn có thể sai trên ảnh thật (ví dụ model có thể gọi nhầm một điện thoại là lớp COCO khác); nhãn chỉ xuất hiện trong overlay/log cho mục đích debug.

### Lưu ý môi trường macOS

Khi chung process có OpenCV và PyAV, macOS có thể in cảnh báo `AVFFrameReceiver implemented in both ...`. PyAV là dependency của `faster-whisper`; cảnh báo đã xuất hiện từ lúc import cả hai thư viện, không phải kết quả inference. Monitor webcam 10 giây vẫn chạy ổn định, nhưng phải ghi lại ngay nếu voice+safety chạy dài bị crash. Nếu có crash, tách voice và safety sang hai process là fallback kỹ thuật, không được bỏ qua cảnh báo này.

## 2. Chạy safety monitor từ webcam

```bash
cd "/Users/an/Documents/AI GLASS/Meta-Rayban"
deactivate 2>/dev/null || true
source .venv312-arm64/bin/activate
source ~/.zshrc

HF_HUB_DISABLE_XET=1 python -m scripts.safety_monitor \
  --cam 0 --show --seconds 60 --interval 0.75
```

Lần đầu tải/nạp model có thể chậm. Những lần sau terminal phải in `infer=...ms`; cửa sổ preview phải có frame liên tục và chỉ hiện bounding box cho vật trong vùng đi giữa ảnh. Nhấn `Esc` hoặc `Ctrl-C` để dừng.

Nếu camera bị từ chối, cấp quyền cho Terminal hoặc VS Code tại **System Settings > Privacy & Security > Camera**, đóng hoàn toàn ứng dụng terminal/IDE rồi mở lại.

## 3. Kịch bản test bắt buộc

Quay video màn hình hoặc ghi bảng quan sát. Không thử khi đang đi lại thật; người test đứng yên và có người quan sát.

| Kịch bản | Thiết lập | Kỳ vọng tối thiểu | Ghi lại |
|---|---|---|---|
| C0 | Hành lang/trước mặt trống | Không có cảnh báo nguy hiểm | false alarm, latency |
| C1 | Ghế/thùng ở giữa, xa khoảng 2-3 m | Có thể phát hiện nhưng không `rất gần` | nhãn, gần/xa, latency |
| C2 | Ghế/thùng ở giữa, gần khoảng 0.5-1 m | Cảnh báo `gần` hoặc `rất gần` trong vài chu kỳ | thời gian phát hiện, bỏ sót |
| C3 | Cùng vật gần nhưng lệch hẳn trái/phải | Không được ưu tiên hơn vật nằm giữa | bounding box, cảnh báo sai |
| C4 | C2 với sáng yếu và ngược sáng | Ghi rõ cảnh báo đúng/sai; không được bỏ qua lỗi | điều kiện sáng, latency |
| C5 | 5 phút stream | Không sập, FPS camera ổn định, không rò camera | FPS, lỗi/log |

Mục tiêu compute ban đầu là `<300 ms` cho mỗi inference **sau khi model nạp**. Nếu lớn hơn, thử `--interval 1.0`; không tăng FPS bằng mọi giá vì an toàn cần kết quả ổn định hơn tần suất danh nghĩa.

## 4. Tích hợp voice/UI

Voice có monitor chạy nền và tùy chọn đọc cảnh báo ngay:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.voice_loop \
  --cam 0 --seconds 5 --stt-model small --tts macos \
  --obstacle-mode real --safety-interval 0.75 --safety-audio
```

Mở UI có monitor thật:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio \
  --embedder real --obstacle-mode real --server-port 7863
```

Trong UI, chọn webcam rồi bật **Theo dõi vật cản live**. Đây là stream do browser cung cấp, nên không dùng `CameraStream` của server; nó kiểm tra theo từng callback Gradio. Voice/`safety_monitor.py` mới là đường kiểm thử camera stream nền đầy đủ.

## 5. Chưa được đánh dấu xong

- **Depth sang mét:** cần protocol hiệu chuẩn riêng bằng vật có kích thước/khoảng cách biết trước. Không thay đổi câu trả lời sang `~1 m` chỉ từ Depth Anything relative.
- **Độ chính xác an toàn:** cần thống kê false negative/false positive ở các cảnh C0-C4, đặc biệt vật thấp, kính/trong suốt, cầu thang và ánh sáng ngược.
- **Ngưỡng CPM cá nhân:** cần thêm người/vật đã đồng ý và calibrate riêng, không liên quan trực tiếp đến ngưỡng vật cản.
- **Wake-word:** là task độc lập sau khi đường camera+safety đã ổn định.
