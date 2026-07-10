# Hướng dẫn setup MacBook Pro M4 + kịch bản test ảnh thật

> Mục tiêu: tối về mở máy là **chạy được liền**. Làm tuần tự từ trên xuống, copy-paste từng lệnh.
> Máy dev: MacBook Pro M4 (Apple Silicon). Cập nhật: 2026-07-09.

**Checklist nhanh (đánh dấu khi xong):**
- [ ] 0. Đưa code lên M4
- [ ] 1. Xcode CLT + `uv` (hoặc Python 3.12)
- [ ] 2. Tạo môi trường + cài thư viện lõi → chạy test/smoke/thí nghiệm (KHÔNG cần model)
- [ ] 3. Cài stack perception (torch/insightface/clip…)
- [ ] 4. Chuẩn bị ảnh thật
- [ ] 5. Chạy demo webcam thật + thí nghiệm retention ảnh thật

---

## 0. Đưa code từ máy hiện tại lên M4

**Cách A — Git (khuyến nghị).** Trên **máy hiện tại (Windows)**, trong thư mục `Meta-Rayban/`:
```bash
cd Meta-Rayban
git init && git add . && git commit -m "core demo"
# Tạo 1 repo trống trên GitHub (private) rồi:
git remote add origin https://github.com/<user>/meta-rayban.git
git branch -M main && git push -u origin main
```
Trên **M4**:
```bash
git clone https://github.com/<user>/meta-rayban.git
cd meta-rayban
```

**Cách B — Không dùng Git.** Nén thư mục `Meta-Rayban/` → đưa qua Google Drive/AirDrop/USB → giải nén trên M4.
> Lưu ý: **không cần copy** thư mục `data/`, `__pycache__/`, `.venv/` (sẽ tạo lại trên M4).

---

## 1. Công cụ nền (Xcode CLT + uv)

```bash
# Công cụ biên dịch (một số thư viện cần) — nếu đã có sẽ báo skip
xcode-select --install

# uv: quản lý Python + venv nhanh gọn (khuyến nghị)
curl -LsSf https://astral.sh/uv/install.sh | sh
# mở lại terminal, hoặc: source $HOME/.local/bin/env
uv --version
```
> **Vì sao Python 3.12?** Một số wheel (insightface/onnxruntime/torch) **chưa có sẵn cho Python 3.14**.
> Dùng **3.12** cho chắc. `uv` sẽ tự tải đúng phiên bản Python, bạn không phải cài tay.

---

## 2. Môi trường + thư viện lõi → chạy phần KHÔNG cần model trước

```bash
cd meta-rayban            # thư mục dự án trên M4
uv venv --python 3.12
source .venv/bin/activate

# Thư viện lõi (đã đủ để chạy test/thí nghiệm/UI synthetic)
uv pip install numpy pytest matplotlib gradio
```

Kiểm tra **mọi thứ đã có sẵn vẫn chạy** (chưa cần model):
```bash
python -m pytest tests/ -v          # kỳ vọng: 6 passed
python -m app.cli --smoke           # kỳ vọng: ✅ SMOKE PASSED
python -m experiments.retention     # tạo experiments/results/retention.png
python -m experiments.ablation      # tạo experiments/results/ablation.png
```
> Nếu 4 lệnh trên chạy tốt → nền tảng OK, sang bước cài model.

---

## 3. Cài stack perception (model thật — Apple Silicon)

```bash
uv pip install torch torchvision            # có sẵn MPS cho Apple Silicon
uv pip install onnxruntime                  # backend cho InsightFace
uv pip install insightface                  # ArcFace (face embedding)
uv pip install open_clip_torch              # CLIP (object embedding)
uv pip install opencv-python pillow
```

Kiểm tra MPS (tăng tốc trên Apple Silicon):
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# kỳ vọng: MPS: True
```

Kiểm tra model tải & chạy (lần đầu sẽ **tải model ~300MB**, cần mạng):
```bash
python -c "
from perception.embed import RealEmbedder
import numpy as np, PIL.Image as Image
e = RealEmbedder()
# CLIP object trên 1 ảnh trắng (chỉ để chắc chắn pipeline nạp được)
v = e.embed_object(Image.new('RGB',(224,224),'white'))
print('CLIP embed OK, dim =', v.shape)
"
```
> InsightFace tải `buffalo_l` vào `~/.insightface` ở lần gọi `embed_face` đầu tiên.

---

## 4. Chuẩn bị ảnh thật

Cấu trúc thư mục cần tạo (mỗi người/đồ = 1 thư mục con, tên thư mục = nhãn):
```
data/faces/
├─ Lan/    img1.jpg img2.jpg ... (5–10 ảnh, góc/ánh sáng khác nhau)
├─ Huy/    ...
└─ Nam/    ...
data/objects/
├─ vi_cua_toi/   ...
└─ chia_khoa/    ...
```

**Cách nhanh nhất — chụp bằng webcam** (script kèm sẵn):
```bash
# chụp 10 ảnh khuôn mặt cho "Lan": nhấn PHÍM CÁCH để chụp, ESC để thoát
python -m scripts.capture_faces --name Lan --n 10
python -m scripts.capture_faces --name Huy --n 10
python -m scripts.capture_faces --name Nam --n 10
```
> **Quyền camera macOS:** lần đầu macOS sẽ hỏi cấp quyền cho Terminal/VS Code.
> Nếu bị chặn: *System Settings → Privacy & Security → Camera* → bật cho app bạn đang dùng, rồi mở lại.

**Hoặc** dùng ảnh có sẵn (điện thoại chụp) copy vào các thư mục trên. Cần ≥ 2 ảnh/người.

---

## 5. Chạy thật

### 5A. Demo webcam trực quan (định tính)
```bash
python -m app.demo_gradio --embedder real
# mở link http://127.0.0.1:7860 hiện trong terminal
```
**Kịch bản test (làm đúng thứ tự để thấy "học liên tục, không quên"):**
1. Chọn **face** → webcam vào mặt bạn → gõ nhãn "Toi" → **📝 Ghi nhớ** → thấy "Đã ghi nhớ".
2. Vẫn mặt đó → **❓ Hỏi** → phải ra "Đây là **Toi**".
3. Nhờ 2–3 người khác lần lượt: Ghi nhớ tên họ.
4. Chọn **object** → đưa cái ví/chìa khoá → nhãn "vi cua toi" → **Ghi nhớ** → **Hỏi** lại.
5. **Quay lại mặt bạn** (người đầu tiên) → **Hỏi** → vẫn ra "Toi" ✅ (không quên).
6. Thử một người **CHƯA dạy** → **Hỏi** → "Chưa biết" ✅.
7. Với người bị nhận nhầm → nhập tên đúng → **🛠️ Sửa** → **Hỏi** lại → đúng ✅.

### 5B. Thí nghiệm retention trên ẢNH THẬT (định lượng cho báo cáo)
```bash
# face
python -m experiments.real_data --data_dir data/faces --modality face
# object
python -m experiments.real_data --data_dir data/objects --modality object
```
Sinh ra `experiments/results/retention_real_face.png` (+ CSV) — bản **retention.png nhưng bằng
embedding ArcFace thật**, so sánh CPM vs kNN vs fine-tune. **Đây là số liệu chính danh để đưa vào báo cáo.**

> Cần ≥ 4–5 người, mỗi người ≥ 5 ảnh để đường cong có ý nghĩa.

---

## Xử lý sự cố (troubleshooting)

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `No matching distribution` khi cài | Python 3.14 chưa có wheel | Dùng `uv venv --python 3.12` (bước 2) |
| `MPS: False` | torch/macOS cũ | `uv pip install -U torch`; vẫn chạy được trên CPU (chậm hơn) |
| InsightFace cài lỗi (build) | thiếu Xcode CLT | `xcode-select --install`; rồi `uv pip install cython` trước |
| "Không phát hiện khuôn mặt" | ảnh không rõ mặt/quá tối | ảnh chính diện, đủ sáng, 1 mặt rõ |
| Webcam đen / không mở | chưa cấp quyền camera | System Settings → Privacy → Camera → bật cho Terminal/VS Code, mở lại |
| `SSL: CERTIFICATE_VERIFY_FAILED` khi tải model | thiếu cert | chạy `/Applications/Python*/Install Certificates.command` hoặc `uv pip install -U certifi` |
| Gradio báo cổng bận | 7860 đang dùng | `python -m app.demo_gradio --embedder real` rồi đổi cổng, hoặc tắt tiến trình cũ |
| Model tải chậm/timeout | mạng | thử lại; model chỉ tải 1 lần rồi cache |

---

### 5C. Orchestrator đa kỹ năng (mô tả / OCR / vật cản / nhận diện + SAFETY)
```bash
python -m app.orchestrator --smoke     # chạy ngay với stub (không cần model)
```
Để bật **kỹ năng thật** (trên M4), sửa `app/orchestrator.py` (`__init__`) thay Stub bằng Real:
- `RealScene(vlm=...)` — `vlm(frame_rgb, prompt)->str`: gắn Gemini/GPT-4o/Claude **hoặc** VLM on-device (Moondream/PaliGemma).
- `RealOCR(ocr_fn=...)` — `ocr_fn(frame_rgb)->str`: VietOCR/PaddleOCR hoặc VLM.
- `RealObstacle()` — hiện thực `check()` bằng YOLO (`pip install ultralytics`) + Depth Anything (ước lượng khoảng cách).
> Nhận diện người/đồ đã chạy thật qua CPM + ArcFace/CLIP (mục 5A/5B).

## Sau khi 5A + 5B chạy được = phần CORE đã "test thực tế" xong ✅
Lúc đó báo tôi kết quả (ảnh nhận đúng không, retention_real.png trông thế nào) để tính bước tiếp:
thêm baseline chuẩn (NCM/iCaRL/EWC), hybrid recall, hoặc mở sang OCR/obstacle/kính.

---

## Task 3 update — chụp đồ vật thật + Gradio webcam

### Chụp object bằng webcam
Script riêng cho đồ vật đã có:
```bash
python -m scripts.capture_objects --name vi_cua_toi --n 10
python -m scripts.capture_objects --name chia_khoa --n 10
python -m scripts.capture_objects --name coc_nuoc --n 10
```

Ảnh sẽ nằm trong:
```text
data/objects/<ten_do_vat>/
```

Mẹo chụp để CLIP/object ổn hơn:
- Mỗi đồ vật nên có 8-15 ảnh.
- Đổi góc, khoảng cách, nền, ánh sáng.
- Tránh để nhiều đồ vật chính trong cùng khung hình.
- Tạo thêm thư mục đồ/vật lạ để calibrate impostor nếu muốn test open-set chắc hơn.

Sau khi chụp:
```bash
python -m scripts.calibrate_threshold --data_dir data/objects --modality object --impostor_split 2
python -m experiments.real_data --data_dir data/objects --modality object
```

### Chạy Gradio webcam
```bash
python -m app.demo_gradio --embedder real
```

Nếu port 7860 bận:
```bash
python -m app.demo_gradio --embedder real --server-port 7861
```

Checklist nghiệm thu:
- Chọn `face`, chụp mặt bạn, nhập tên, bấm `Ghi nho`.
- Bấm `Hoi` với cùng người → phải nhận đúng.
- Đưa người chưa dạy vào → nên ra `Chua biet` sau khi đã calibrate ngưỡng.
- Chọn `object`, đưa ví/chìa khóa/cốc, nhập nhãn, bấm `Ghi nho`, rồi `Hoi`.
- Nếu ảnh chưa chụp hoặc không thấy mặt, UI phải báo lỗi thân thiện thay vì crash.

Smoke headless không cần camera đã có trong test:
```bash
python -m pytest tests/test_demo_gradio.py -v
```

---

## Task 4 update — bật OCR/VLM thật

Repo đã có wiring provider thật nhưng lazy/fallback: thiếu thư viện hoặc thiếu API key thì orchestrator tự dùng stub, không crash.

### Cài provider tuỳ chọn
```bash
uv pip install easyocr google-generativeai pillow
```

### Gemini VLM
Đặt API key:
```bash
export GEMINI_API_KEY="your_key_here"          # macOS/Linux
# Windows PowerShell:
# $env:GEMINI_API_KEY="your_key_here"
```

Chạy orchestrator với skill thật:
```bash
python -m app.orchestrator --skills-mode real
```

Smoke vẫn chạy không cần key/model:
```bash
python -m app.orchestrator --smoke
python -m pytest tests/test_providers.py -v
```

Ghi chú riêng tư: Gemini/VLM cloud sẽ gửi ảnh lên dịch vụ ngoài máy. Nếu demo cần offline, để `skills_mode=stub` hoặc thay provider bằng VLM local sau.
