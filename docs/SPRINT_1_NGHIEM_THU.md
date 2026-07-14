# Sprint 1 - Hướng dẫn nghiệm thu trên M4

Tài liệu này phân tách rõ: phần đã được code/test trong repo và các bước bắt buộc cần thao tác trên máy, tài khoản, camera, microphone hay ảnh thật. Không đưa API key, ảnh khuôn mặt hay dữ liệu cá nhân vào Git.

## Chạy nhanh: các lệnh cần dùng

Luôn chạy các lệnh bên dưới từ terminal trong thư mục project. Nếu prompt còn hiện `(.venv)` cũ, thoát nó trước; môi trường đúng duy nhất là `.venv312-arm64`.

```bash
cd "/Users/an/Documents/AI GLASS/Meta-Rayban"
deactivate 2>/dev/null || true
source .venv312-arm64/bin/activate
source ~/.zshrc
python --version
python -c 'import os; print("OPENAI_API_KEY loaded:", bool(os.environ.get("OPENAI_API_KEY")))'
```

Phải thấy `Python 3.12.10` và `OPENAI_API_KEY loaded: True` trước khi chạy OpenAI VLM/OCR.

### Mở UI lần đầu, nạp sẵn đồ vật từ dataset

Chỉ dùng khi `.local_memory/demo/` chưa có bộ nhớ object. Nếu UI cũ đang chiếm cổng, nhấn `Ctrl-C` trong terminal đang chạy nó. Nếu không còn terminal đó, tìm PID rồi dừng đúng PID:

```bash
lsof -tiTCP:7863 -sTCP:LISTEN
kill <PID>
```

Sau đó chạy:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio \
  --embedder real \
  --bootstrap-objects data/objects \
  --server-port 7863
```

Mở `http://127.0.0.1:7863`. Phần thống kê phải hiển thị `object: 2 nhãn` nếu hiện có `chia_khoa` và `vi_cua_toi`.

### Mở UI ở các lần sau

Không dùng lại `--bootstrap-objects`; CPM local đã tự lưu sau mỗi lần **Ghi nhớ**/**Sửa**:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio --embedder real --server-port 7863
```

### Test VLM/OCR, voice và dữ liệu

```bash
# Mô tả cảnh / đọc chữ thật qua OpenAI (thay bằng file ảnh thật)
python -m scripts.try_vlm --image "/đường/dẫn/ảnh-cảnh.jpg" --query "Trước mặt có gì?"
python -m scripts.try_vlm --image "/đường/dẫn/ảnh-chữ.jpg" --query "Đọc chữ trong ảnh giúp tôi"

# Kiểm tra mic và chạy vòng lặp nói-nghe
python -m scripts.voice_loop --list-devices
HF_HUB_DISABLE_XET=1 python -m scripts.voice_loop \
  --cam 0 --user-id demo --memory-dir .local_memory --stt-model small --tts macos

# Tạo số liệu thực nghiệm; không nạp dữ liệu này vào UI
HF_HUB_DISABLE_XET=1 python -m experiments.real_data --data_dir data/objects --modality object

# Với đúng 2 nhãn object hiện tại: chừa 1 nhãn làm impostor, 1 nhãn quen
python -m scripts.calibrate_threshold --data_dir data/objects --modality object --impostor_split 1
```

Khi có ít nhất 4 nhãn object hoặc face, có thể dùng `--impostor_split 3`: 3 nhãn làm mẫu lạ và vẫn còn ít nhất 1 nhãn quen để đo genuine probes. Tuy nhiên, nên thu thêm dữ liệu để calibration có ý nghĩa hơn, thay vì chỉ dựa trên một nhãn quen.

## 1. Trạng thái sau khi cập nhật

- Đã có `scripts/voice_loop.py`: microphone + WebRTC VAD → faster-whisper tiếng Việt → state machine xác nhận CPM → TTS.
- Voice loop và UI dùng chung `.local_memory/<user-id>/`; transcript/route/latency được ghi local JSONL, không lưu ảnh/WAV/key.
- TTS macOS `say` với giọng `Linh` (vi_VN) là mặc định để chạy được ngay trên M4. Piper (offline) và gTTS (online) là backend tùy chọn.
- Gradio có nút `Mô tả cảnh` và `Đọc chữ`; khi process có `OPENAI_API_KEY`, hai nút gọi OpenAI vision. Không có key thì hệ thống fallback theo provider hiện có, không tự nhận là kết quả thật.
- Lỗi không thấy mặt, frame không hợp lệ, microphone/camera bị từ chối quyền được trả về bằng câu tiếng Việt thay vì traceback.

## 2. Mở terminal đúng môi trường và kiểm tra key

```bash
cd "/Users/an/Documents/AI GLASS/Meta-Rayban"
source .venv312-arm64/bin/activate
python -c 'import os; print("OPENAI_API_KEY loaded:", bool(os.environ.get("OPENAI_API_KEY")))'
```

Kết quả cần là `True`. Nếu là `False`, key đã được đặt trong terminal/IDE khác; đặt lại trong **cùng terminal sẽ chạy app**, không cần và không nên gửi key cho Codex:

```bash
export OPENAI_API_KEY='sk-...'
```

Để lưu cho các terminal zsh mới, tự thêm biến này vào `~/.zshrc` trên máy của bạn, mở terminal mới, rồi lặp lại lệnh kiểm tra. Không tạo hay commit `.env` thật; repo đã ignore `.env`, và chỉ có [`.env.example`](../.env.example) làm mẫu.

## 3. Cấp quyền macOS (bắt buộc một lần)

Vào **System Settings > Privacy & Security**.

1. Mở **Camera**, bật cho ứng dụng đang chạy lệnh: Terminal, iTerm hoặc VS Code. Nếu chạy trong VS Code Integrated Terminal, cấp cho VS Code.
2. Mở **Microphone**, bật cho cùng ứng dụng đó.
3. Đóng hoàn toàn terminal/VS Code và mở lại sau khi đổi quyền.
4. Nếu có nhiều webcam/mic, liệt kê mic bằng lệnh ở bước 5; webcam thử `--cam 1`.

Không cần cấp Accessibility cho bản Sprint 1: voice loop dùng Enter trong terminal, không dùng global hotkey. Đây là chủ ý để tránh thêm một quyền nhạy cảm.

## 4. Nghiệm thu OCR/VLM thật bằng key OpenAI

Chọn hai ảnh không nhạy cảm: một ảnh có cảnh và một ảnh có chữ tiếng Việt rõ. Chạy:

```bash
python -m scripts.try_vlm --image /đường/dẫn/ảnh-cảnh.jpg --query "Trước mặt có gì?"
python -m scripts.try_vlm --image /đường/dẫn/ảnh-chữ.jpg --query "Đọc chữ trong ảnh giúp tôi"
```

Thay `/đường/dẫn/...` bằng đường dẫn ảnh thật trên máy. Cần thấy log `Scene/VQA: dùng VLM OpenAI` và `OCR: dùng OpenAI vision`, không phải `Stub`. Lưu lại: ảnh đầu vào (nếu đã được đồng ý), câu hỏi, câu trả lời, thời gian chạy và lỗi nếu có.

Để thử trên giao diện:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio --embedder real --server-port 7863
```

Mở `http://127.0.0.1:7863`, cho phép Camera, chọn webcam hoặc upload, sau đó bấm `Mô tả cảnh` và `Đọc chữ`. Khi key chưa được nạp lúc server khởi động, dừng server bằng `Ctrl-C`, export key trong cùng terminal và chạy lại.

Nếu cần OCR offline thay vì OpenAI, đặt `ENABLE_EASYOCR=1` trước khi chạy. Đây là opt-in vì EasyOCR có thể tải checkpoint lớn lần đầu; Sprint 1 mặc định dùng OpenAI khi đã có key.

## 5. Nghiệm thu voice loop

Đảm bảo đang dùng đúng môi trường `.venv312-arm64`, không phải `.venv` cũ. Liệt kê microphone trước:

```bash
python -m scripts.voice_loop --list-devices
```

Chạy luồng thật. Model `small` của Whisper được tải từ Hugging Face lần đầu, cần mạng và vài trăm MB dung lượng cache. Không đóng terminal khi đang tải.

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.voice_loop \
  --cam 0 --user-id demo --memory-dir .local_memory --stt-model small --tts macos
```

Nhấn Enter, nói một trong hai câu sau. VAD tự dừng sau khoảng 1 giây im lặng (`--seconds 12` chỉ là giới hạn tối đa):

- `Trước mặt có gì?`
- `Đọc chữ trong khung hình giúp tôi.`

Sau đó kiểm tra terminal in được `[Bạn]`, `[Trợ lý]`, và máy đọc câu trả lời. Lưu hai dòng `[Độ trễ]`; mục tiêu Sprint 1 là tổng STT + xử lý online khoảng 3-5 giây, không tính thời gian người dùng nói. Whisper `small` đã được kiểm tra nạp thành công trên M4; độ chính xác tiếng Việt vẫn bắt buộc nghiệm thu bằng giọng người thật, không dùng giọng máy tổng hợp làm bằng chứng.

Nếu giọng macOS không có tiếng Việt tốt, kiểm tra danh sách giọng bằng `say -v '?' | rg -i 'vi|linh'`, rồi chạy lại với `--tts-voice <tên-giọng>`. Piper offline chỉ dùng khi đã có file model `.onnx` tiếng Việt: `--tts piper --piper-model /đường/dẫn/voice.onnx`.

Hướng dẫn chi tiết cho shared memory UI/voice, xác nhận trước khi ghi CPM, UI microphone, safety priority và JSONL log nằm trong [VOICE_NGHIEM_THU.md](VOICE_NGHIEM_THU.md).

## 6. Dữ liệu nhóm: ArcFace/CLIP và ngưỡng thật

Bạn hoặc người đã có đồng ý của các thành viên cần tự chụp dữ liệu. Mỗi người/vật: 10-20 ảnh, thay đổi góc, khoảng cách và ánh sáng; không dùng ảnh của người chưa đồng ý.

```bash
python -m scripts.capture_faces --name Lan --n 12 --out data/faces
python -m scripts.capture_faces --name Huy --n 12 --out data/faces
python -m scripts.capture_objects --name vi_cua_toi --n 12 --out data/objects
python -m scripts.capture_objects --name chia_khoa --n 12 --out data/objects
```

Sau khi có ít nhất 4 nhãn mỗi modality, có thể dùng 3 nhãn làm impostor và chạy các lệnh sau. Lưu bảng CSV/ảnh kết quả vào hồ sơ thực nghiệm của nhóm:

```bash
HF_HUB_DISABLE_XET=1 python -m experiments.real_data --data_dir data/faces --modality face
HF_HUB_DISABLE_XET=1 python -m experiments.real_data --data_dir data/objects --modality object
python -m scripts.calibrate_threshold --data_dir data/faces --modality face --impostor_split 3
python -m scripts.calibrate_threshold --data_dir data/objects --modality object --impostor_split 3
```

Nếu hiện chỉ có 2 nhãn object, dùng `--impostor_split 1` để lệnh chạy được; đây chỉ là kiểm tra kỹ thuật, chưa đủ dữ liệu để chốt ngưỡng deploy.

Đọc diff của `configs/thresholds.json`, ghi ngày, số ảnh/nhãn, model, máy và metric. Chỉ thay ngưỡng deploy sau khi nhóm xem lại false accept/false reject trên dữ liệu đã chụp.

### Nạp dữ liệu đã dạy vào UI và nhớ qua lần khởi động sau

`experiments.real_data` chỉ tạo số liệu thực nghiệm, không đưa ảnh vào UI. Để tạo bộ nhớ CPM local từ dataset đã chụp, chạy **một lần**:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio --embedder real --bootstrap-objects data/objects --server-port 7863
```

Lệnh này đọc từng thư mục `data/objects/<nhãn>/`, tạo embedding CLIP thật và ghi prototype vào CPM user `demo`. Khi mở UI, phần thống kê phải hiện `object: <số nhãn> nhãn`. Không chạy lại cùng cờ bootstrap trên cùng user vì sẽ ghi lặp mẫu; UI chủ động chặn tình huống này.

Sau lần đầu, chỉ cần chạy bình thường và CPM tự khôi phục từ `.local_memory/demo/`:

```bash
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio --embedder real --server-port 7863
```

Mỗi lần bấm **Ghi nhớ** hoặc **Sửa** đều tự lưu. Bộ nhớ face và object được tách riêng, lưu bằng JSON + NPZ (không dùng pickle), và `.local_memory/` đã được bỏ qua khỏi Git. Có thể tách người dùng local bằng `--user-id an`, khi đó bộ nhớ nằm trong `.local_memory/an/`.

## 7. Cổng quyết định Meta Ray-Ban (cần tài khoản/thiết bị của nhóm)

Tình trạng kiểm tra hiện tại: Device Access Toolkit công khai mô tả camera streaming, video frames, photo capture và audio/microphone cho app iOS/Android; Mock Device Kit có thể giả lập stream khi chưa có kính. Tuy vậy, nhóm vẫn phải tự xác nhận quyền truy cập, khu vực, dòng kính và test trên điện thoại mục tiêu, vì đây là phụ thuộc SDK bên ngoài.

1. Đăng nhập/tạo Meta developer account và vào Wearables Developer Center.
2. Đọc README/quickstart của repository chính thức: [iOS toolkit](https://github.com/facebook/meta-wearables-dat-ios) và [Android toolkit](https://github.com/facebook/meta-wearables-dat-android).
3. Chốt Android hay iOS. Cài sample của đúng nền tảng, bật Developer Mode/permission theo hướng dẫn Meta.
4. Không có kính: chạy Mock Device Kit và xác nhận app nhận video frame liên tục, permission state và audio path.
5. Có kính: test 10 phút camera stream + mic + loa, ghi lại FPS/resolution, độ trễ, mất kết nối, quyền hiện trên Meta AI app và SDK version.
6. Kết luận bằng một trong hai câu: `GO: có video frame liên tục + audio đủ cho Sprint 3` hoặc `NO-GO: kích hoạt camera đeo DIY/phone lanyard`. Đính kèm log/screenshot, không ghi nhớ chủ quan.

Nguồn để đối chiếu: [Meta Mock Device Kit](https://wearables.developer.meta.com/docs/mock-device-kit/) mô tả giả lập media streaming và permissions; README chính thức của toolkit liệt kê camera streaming/video frames. OpenAI hiện hỗ trợ image input qua API, phù hợp cho hai nút VLM/OCR của Sprint 1: [OpenAI Models](https://developers.openai.com/api/docs/models).

## 8. Checklist báo lại cho mình

- [ x] `OPENAI_API_KEY loaded: True` trong terminal chạy project.
- [ x] Hai lệnh `try_vlm` trả kết quả thật, không phải Stub.
- [ x] Camera và microphone đã được cấp quyền.
- [ x] Voice loop nghe/nhận/đáp được ít nhất 3 lượt; có số đo độ trễ.
- [ x] Đã chụp data có đồng ý và chạy `real_data` cho face/object.
- [ ] Đã calibrate threshold trên data nhóm và review kết quả.
- [ x] Đã bootstrap `data/objects` vào CPM local và mở lại UI để xác nhận bộ nhớ được khôi phục.
- [ ] Đã có kết luận GO/NO-GO về Meta Wearables Toolkit kèm bằng chứng.

# Bạn đang có 2 nhãn object, nên hãy dừng server UI cũ rồi chạy một lần trong .venv312-arm64:
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio \
  --embedder real \
  --bootstrap-objects data/objects \
  --server-port 7863

HF_HUB_DISABLE_XET=1 python -m app.demo_gradio \
  --embedder real \
  --bootstrap-faces data/faces \
  --server-port 7863
# Khi mở http://127.0.0.1:7863, phần bộ nhớ phải hiện object: 2 nhãn. Sau này chỉ chạy:
HF_HUB_DISABLE_XET=1 python -m app.demo_gradio --embedder real --server-port 7863
# Nó sẽ tự khôi phục ví và chìa khóa. Đừng chạy lại --bootstrap-objects trên user demo trừ khi chủ động làm mới bộ nhớ; app sẽ chặn để tránh ghi trùng.
