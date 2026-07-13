# Understand.md — Hiểu toàn bộ project từ số 0

> Tài liệu này viết cho **người chưa có nhiều kiến thức AI**. Mục tiêu: đọc xong bạn hiểu
> đồ án đang làm gì, mỗi thư mục/file để làm gì, và dữ liệu "chạy" qua hệ thống như thế nào.
> Không cần biết toán. Mọi thuật ngữ đều được giải thích bằng ví dụ đời thường.
> Cập nhật: 2026-07-13.

---

## 0. Một câu tóm tắt đồ án

> Chúng ta làm một **trợ lý cho người khiếm thị**: người dùng chỉ camera vào ai đó/đồ gì đó và
> **dạy** trợ lý ("đây là chị Lan", "đây là ví của tôi"). Lần sau gặp lại, trợ lý **nhận ra và
> nói tên**. Điểm đặc biệt: trợ lý **học ngay tại chỗ, nhớ lâu, và không quên** những gì đã học
> trước đó — giống một người bạn có trí nhớ tốt, chứ không phải một cỗ máy "mỗi lần dùng là quên sạch".

Phần "nhìn" (nhận ra khuôn mặt, đọc chữ) dùng các model AI **có sẵn** của thế giới. Phần
**nghiên cứu tự làm** của nhóm là cái "**trí nhớ**" đó — gọi tắt là **CPM**. Đây là trái tim đồ án.

---

## 1. Bức tranh lớn bằng ẩn dụ

Hãy tưởng tượng trợ lý gồm **3 nhân vật** làm việc cùng nhau:

| Nhân vật | Trong code gọi là | Việc của họ |
|---|---|---|
| 👁️ **Đôi mắt** | `perception/` (Perception) | Nhìn ảnh, biến khuôn mặt/đồ vật thành một "dấu vân tay số" |
| 🧠 **Trí nhớ** | `cpm/` (CPM) ⭐ | Ghi nhớ "dấu vân tay này tên là Lan", và nhận ra khi gặp lại |
| 🎧 **Người điều phối** | `app/orchestrator.py` | Nghe câu hỏi tiếng Việt, quyết định gọi mắt/trí nhớ/kỹ năng nào, rồi trả lời |

Cộng thêm vài **kỹ năng phụ** (`skills/`): mô tả khung cảnh, đọc chữ, cảnh báo vật cản.

**Ví dụ một lượt trò chuyện:**
```
Bạn: "Hãy nhớ đây là Lan"  (camera đang chỉ vào mặt chị Lan)
  → 🎧 Điều phối hiểu: đây là lệnh DẠY, tên = "Lan"
  → 👁️ Mắt biến ảnh mặt thành dấu vân tay số
  → 🧠 Trí nhớ lưu: "dấu vân tay này = Lan"
  → Trả lời: "Đã ghi nhớ: Lan."

(lát sau) Bạn: "Ai đây?"  (camera lại chỉ vào chị Lan)
  → 🎧 hiểu: lệnh HỎI
  → 👁️ tạo dấu vân tay mới từ ảnh
  → 🧠 so khớp với những gì đã nhớ → giống "Lan" nhất
  → Trả lời: "Đây là Lan (tin cậy 0.95)."
```

---

## 2. Từ điển thuật ngữ (giải thích bằng ví dụ)

Đọc phần này như tra từ điển — không cần thuộc, cứ quay lại khi gặp từ lạ.

### Nhóm A — về "biến ảnh thành số"

- **Embedding (véc-tơ đặc trưng)** = **"dấu vân tay số" của một ảnh**. Model AI nhìn ảnh khuôn mặt
  và trả về một dãy **512 con số**. Điều kỳ diệu: **cùng một người** → dãy số **rất giống nhau**;
  **hai người khác nhau** → dãy số **khác nhau**. Nhờ vậy máy tính "so mặt" bằng cách so hai dãy số.
  *(512 con số này gọi là một véc-tơ 512 chiều — đừng sợ chữ "chiều", nó chỉ nghĩa là "dãy dài 512 số".)*

- **Cosine similarity (độ tương đồng cô-sin)** = **thước đo hai dấu vân tay giống nhau bao nhiêu**,
  cho ra số từ **-1 đến 1**. Gần **1** = rất giống (chắc cùng người); gần **0** = chẳng liên quan.
  Cả đồ án dùng con số này để quyết định "có phải người quen không".

- **Chuẩn hoá (normalize / L2 norm)** = "co dãn dấu vân tay về cùng độ dài chuẩn" trước khi so,
  để phép so công bằng. Trong code là hàm `_unit(...)`. Bạn không cần quan tâm chi tiết.

- **Modality (kiểu đối tượng)** = **loại dữ liệu**: ở đây có 2 loại — **face** (khuôn mặt) và
  **object** (đồ vật). Hai loại dùng **model khác nhau** nên dấu vân tay của chúng **không cùng
  "ngôn ngữ"** → phải để trí nhớ **riêng cho từng loại**, không trộn lẫn.

- **ArcFace / InsightFace** = model AI có sẵn, chuyên biến **khuôn mặt** thành dấu vân tay 512 số.
- **CLIP (OpenCLIP)** = model AI có sẵn, chuyên biến **đồ vật/ảnh bất kỳ** thành dấu vân tay.
- **VLM (Vision-Language Model)** = model AI "nhìn ảnh và nói bằng chữ" — ví dụ GPT-4o, Gemini.
  Dùng để **mô tả khung cảnh** và **đọc chữ**.
- **OCR** = "đọc chữ trong ảnh" (biển báo, tờ giấy, hạn sử dụng).
- **YOLO** = model phát hiện vật thể nhanh, dùng cho **cảnh báo vật cản**.

### Nhóm B — về "trí nhớ học liên tục" (trái tim đồ án)

- **Continual learning (học liên tục)** = học **thêm cái mới mà không phải học lại từ đầu**, và
  **không quên cái cũ**. Giống bạn học thêm tên người mới mà vẫn nhớ tên bạn cũ.

- **Catastrophic forgetting (quên thảm khốc)** = "bệnh" của nhiều AI: dạy nó cái mới thì nó
  **quên sạch cái cũ**. Đồ án muốn **tránh** bệnh này.

- **Prototype (nguyên mẫu) / NCM (Nearest Class Mean)** = cách nhớ **đơn giản mà rất mạnh**:
  với mỗi người, ta lưu **một dấu vân tay "trung bình"** của tất cả ảnh đã thấy. Nhận diện =
  "câu hỏi giống dấu-vân-tay-trung-bình của ai nhất?". Đây là **xương sống nhận diện** của CPM.
  *(NCM = tên khoa học của đúng cách này. CPM dùng chính nó làm nền.)*

- **Delta-rule / associative memory (bộ nhớ liên kết)** = cơ chế **"chỉ ghi phần sai lệch/mới"**
  lấy từ lý thuyết **Nested Learning**. Ý tưởng: khi có thông tin mới, chỉ cập nhật **phần khác
  biệt** so với cái đã biết, nên **không đè mất kiến thức cũ**. Trong CPM đây là **thành phần tuỳ
  chọn** (mặc định TẮT) — dùng để nghiên cứu/thử nghiệm, không phải cái chạy hằng ngày.

- **Nested Learning (NL)** = bài báo/lý thuyết (Google, 2025) truyền cảm hứng cho đồ án. Ý cốt lõi
  ta mượn: **bộ nhớ nhiều tầng cập nhật ở nhiều tốc độ khác nhau** (nhanh/vừa/chậm) để vừa thích
  nghi nhanh vừa nhớ lâu.

- **Ba tầng fast / medium / slow (Continuum Memory System)** = ẩn dụ trí nhớ người:
  - **fast** = trí nhớ tạm ("vừa mới thấy") — thích nghi nhanh, quên nhanh.
  - **medium** = việc gần đây.
  - **slow** = trí nhớ dài hạn ("mặt người thân") — bền, khó quên.

- **EMA (trung bình trượt luỹ thừa)** = một kiểu prototype **"bám theo cái gần đây"**: nếu ngoại
  hình một người **đổi dần theo thời gian** (đeo kính, để râu, già đi), EMA cập nhật để **theo kịp**,
  trong khi trung bình thường bị mẫu cũ "kéo lùi". Đây là **tầng nhanh** của CPM (mặc định TẮT,
  chỉ bật trong thí nghiệm drift).

- **Drift (trôi ngoại hình)** = tình huống **ngoại hình đổi dần theo thời gian**. Đây là chỗ CPM
  (có EMA) **thắng** NCM thường — điểm nhấn nghiên cứu.

### Nhóm C — về "quen hay lạ" và đo lường

- **Novelty gate / recall_threshold (cổng quen-lạ / ngưỡng nhận)** = **vạch quyết định**: nếu độ
  giống ≥ ngưỡng → "người quen, đây là X"; nếu < ngưỡng → "**chưa biết**, có muốn dạy tôi không?".
  Chọn ngưỡng sai thì hoặc nhận nhầm người lạ, hoặc từ chối cả người quen.

- **Calibrate (hiệu chuẩn ngưỡng)** = **tìm con số ngưỡng đúng** cho từng model bằng dữ liệu thật.
  (Đồ án từng đặt nhầm 0.35 → nhận nhầm ~60% người lạ; đã sửa bằng cách tính lại từ dữ liệu.)

- **FAR / FRR** = 2 kiểu lỗi của cổng quen-lạ:
  - **FAR (False Accept Rate)** = tỉ lệ **nhận nhầm người lạ thành quen** (nguy hiểm hơn).
  - **FRR (False Reject Rate)** = tỉ lệ **từ chối nhầm người quen**.
  Đồ án chọn chính sách **`far1`** = "giữ nhận-nhầm-người-lạ ≤ 1%" (an toàn cho người khiếm thị).

- **ROC / AUC / EER** = các thước đo "hệ thống phân biệt quen-lạ tốt cỡ nào" (càng cao/càng thấp
  tuỳ chỉ số càng tốt). Bạn chỉ cần biết: **AUC gần 1.0 = phân biệt rất tốt**.

- **Footprint (dung lượng bộ nhớ)** = **số lượng con số** hệ thống phải lưu. Quan trọng vì thiết bị
  đeo có bộ nhớ hạn chế. "**Bị chặn (bounded)**" = không phình to vô hạn dù dùng nhiều; "**phình
  (unbounded)**" = càng dùng càng to (điểm yếu của kNN).

- **kNN (vector-DB)** = cách nhớ "lưu **hết mọi ảnh**" rồi so với tất cả. Nhận diện tốt nhưng
  **bộ nhớ phình vô hạn** — đây là lý do CPM/NCM tốt hơn cho thiết bị đeo.

### Nhóm D — về code

- **Orchestrator (bộ điều phối)** = "nhạc trưởng" đọc câu tiếng Việt, đoán ý định (dạy? hỏi? sửa?
  đọc chữ? mô tả cảnh?) rồi gọi đúng bộ phận.
- **Stub vs Real** = **bản giả để chạy thử ngay** (Stub, không cần cài model/mạng) vs **bản thật**
  (Real, cần model/API key). Có cả hai để code **luôn chạy được** kể cả khi chưa cài gì.
- **Synthetic vs Real embedder** = dấu-vân-tay **giả mô phỏng** (để kiểm tra hệ thống chạy đúng
  mạch, không cần camera/model) vs **thật** (từ ArcFace/CLIP).
- **Lazy import** = "chỉ nạp model nặng **khi thật sự cần**" — nhờ vậy chạy thử phần nhẹ vẫn nhanh.
- **Smoke test** = "chạy thử nhanh cho có khói" — kiểm tra cả mạch nối có thông không.
- **pytest / unit test** = các bài kiểm tra tự động, đảm bảo sửa code không làm hỏng thứ khác.

---

## 3. Cấu trúc thư mục (bản đồ tổng)

```
Meta-Rayban/
├─ cpm/              ⭐ TRÁI TIM: trí nhớ học liên tục (CPM)
├─ perception/          👁️ đôi mắt: biến ảnh → dấu vân tay số (embedding)
├─ skills/              🛠️ kỹ năng phụ: mô tả cảnh, đọc chữ, cảnh báo vật cản
├─ app/                 🎧 giao tiếp: điều phối + giao diện (CLI, web Gradio)
├─ experiments/         🔬 thí nghiệm + đo lường (số liệu để viết báo cáo)
│   └─ results/         📊 biểu đồ + CSV kết quả (ảnh .png)
├─ scripts/             ⚙️ công cụ chạy tay (chụp ảnh, hiệu chuẩn ngưỡng, thử VLM)
├─ tests/               ✅ bài kiểm tra tự động (35 bài)
├─ configs/             🔧 cấu hình đã lưu (ngưỡng đã hiệu chuẩn)
├─ data/                🗂️ ảnh mẫu / dataset (ảnh lớn không lưu vào git)
├─ docs/                📚 tài liệu (kế hoạch, kết quả, file này)
├─ requirements.txt        danh sách thư viện cần cài
└─ .venv312/               môi trường Python (Python 3.12) — KHÔNG sửa
```

**Nếu chỉ được đọc 3 thư mục:** đọc `cpm/` (trái tim), `app/` (cách mọi thứ nối lại), `experiments/`
(bằng chứng số liệu).

---

## 4. Vai trò từng file (chi tiết)

### 4.1. `cpm/` — Trí nhớ học liên tục ⭐ (quan trọng nhất)

| File | Vai trò (dễ hiểu) |
|---|---|
| [cpm/memory.py](../cpm/memory.py) | **Bộ não chính.** Chứa lớp `ContinualPersonalizationMemory`: hàm `write` (dạy), `recall` (nhận diện), `correct` (sửa sai), `consolidate` (bền hoá), `snapshot/load` (lưu/khôi phục). Xương sống = **prototype trung-bình-thật**. |
| [cpm/config.py](../cpm/config.py) | **Bảng cấu hình.** Các "núm vặn": số chiều (512), ngưỡng quen-lạ, có bật ma trận đa tầng không, có bật EMA không, tham số 3 tầng fast/medium/slow. |
| [cpm/thresholds.py](../cpm/thresholds.py) | **Quản lý ngưỡng quen-lạ.** Lưu/đọc ngưỡng đã hiệu chuẩn theo từng (model, loại đối tượng) vào file `configs/thresholds.json`. |
| [cpm/__init__.py](../cpm/__init__.py) | "Cửa vào" gói `cpm` — liệt kê những thứ cho phép import từ ngoài. |

### 4.2. `perception/` — Đôi mắt 👁️

| File | Vai trò |
|---|---|
| [perception/embed.py](../perception/embed.py) | **Biến ảnh → dấu vân tay số.** `RealEmbedder` (ArcFace cho mặt, CLIP cho đồ — model thật) và `SyntheticEmbedder` (giả, chạy ngay không cần model). Hàm `get_embedder("synthetic"/"real"/"auto")` chọn loại. |
| [perception/capture.py](../perception/capture.py) | **Lấy ảnh vào.** Đọc ảnh từ file hoặc chụp 1 khung từ webcam; chuyển đổi định dạng màu (BGR ↔ RGB) cho đúng model. |

### 4.3. `cpm/` dùng chung với `skills/` — Kỹ năng phụ 🛠️

| File | Vai trò |
|---|---|
| [skills/base.py](../skills/base.py) | **Khuôn mẫu chung** cho mọi kỹ năng: mỗi kỹ năng có `keywords` (từ khoá tiếng Việt để nhận biết) + `run()`. Định nghĩa **mức ưu tiên** (an toàn = cao nhất → chen ngang). |
| [skills/scene.py](../skills/scene.py) | **Mô tả cảnh / hỏi-đáp (VQA).** `StubScene` (câu mẫu) và `RealScene` (gọi VLM thật như GPT-4o/Gemini). |
| [skills/ocr.py](../skills/ocr.py) | **Đọc chữ.** `StubOCR` (câu mẫu) và `RealOCR` (gọi OCR/VLM thật). |
| [skills/obstacle.py](../skills/obstacle.py) | **Cảnh báo vật cản (an toàn).** `StubObstacle` (chỉnh khoảng cách bằng tay để demo) và `RealObstacle` (YOLO+Depth — khung, hiện thực trên M4). Có `check()` để **chủ động cảnh báo chen ngang**. |
| [skills/providers.py](../skills/providers.py) | **Nhà cung cấp model thật** cho OCR/VLM: `openai_vlm`/`openai_ocr` (GPT-4o), `gemini_vlm` (Gemini), `easyocr_fn` (OCR ngoại tuyến). Tất cả **lazy** + báo lỗi rõ ràng nếu thiếu key/thư viện. |

### 4.4. `app/` — Giao tiếp & điều phối 🎧

| File | Vai trò |
|---|---|
| [app/orchestrator.py](../app/orchestrator.py) | **Nhạc trưởng.** `VisionAssistant`: đọc câu tiếng Việt → đoán ý định (dạy/sửa/hỏi/đọc chữ/mô tả) → gọi CPM + kỹ năng → trả lời. Có **safety override** (vật cản chen ngang) và tự chọn provider VLM/OCR (OpenAI→Gemini→Stub). |
| [app/cli.py](../app/cli.py) | **Bộ khung `Assistant`** nối Perception→CPM cho 2 modality, tự nạp ngưỡng đã hiệu chuẩn. Chạy dòng lệnh: `--smoke` (thử nhanh) hoặc gõ lệnh teach/ask/fix. |
| [app/demo_gradio.py](../app/demo_gradio.py) | **Giao diện web** (thư viện Gradio): webcam/tải ảnh + 3 nút **Ghi nhớ / Hỏi / Sửa**. *(Hiện chỉ có phần CPM; chưa có nút mô tả cảnh/đọc chữ.)* |

### 4.5. `experiments/` — Thí nghiệm & đo lường 🔬 (bằng chứng cho báo cáo)

| File | Vai trò |
|---|---|
| [experiments/baselines.py](../experiments/baselines.py) | **Các phương pháp so sánh** cùng một khuôn: `CPMAdapter` (của nhóm), `CPMEMAAdapter` (có EMA), `NCMAdapter`, `KNNAdapter`, `EWCAdapter`, `FineTuneAdapter`. Để chứng minh CPM hơn/bằng ở điểm nào. |
| [experiments/metrics.py](../experiments/metrics.py) | **Hộp công cụ đo:** tính ROC, AUC, EER, FAR/FRR... |
| [experiments/retention.py](../experiments/retention.py) | **Thí nghiệm chống quên + bộ nhớ:** dạy tuần tự nhiều danh tính, đo độ chính xác danh tính cũ + footprint. Xuất `retention.png`. |
| [experiments/ablation.py](../experiments/ablation.py) | **Mổ xẻ cơ chế:** sức chứa ma trận, vai trò từng tầng, độ bền theo "độ giống danh tính". Xuất `ablation.png`. |
| [experiments/drift.py](../experiments/drift.py) | **Thí nghiệm trôi ngoại hình (điểm nhấn):** nơi CPM-EMA vượt NCM. Có `run_ablation()` quét tham số. Xuất `drift.png`, `drift_ablation.png`. |
| [experiments/calibration.py](../experiments/calibration.py) | **Hiệu chuẩn ngưỡng:** từ điểm số quen/lạ, tính ngưỡng theo chính sách `far1`/`far10`/`eer`. |
| [experiments/data.py](../experiments/data.py) | Sinh **dữ liệu tổng hợp** (giả, có kiểm soát) cho các thí nghiệm cơ chế. |
| [experiments/real_data.py](../experiments/real_data.py) | Chạy thí nghiệm trên **ảnh thật** trong thư mục (đọc ảnh → embedding → retention). |
| [experiments/real_lfw.py](../experiments/real_lfw.py) | Thí nghiệm trên **bộ mặt công khai LFW** (dữ liệu thật): nhận diện + open-set + vẽ điểm ngưỡng. |
| [experiments/results/](../experiments/results/) | **Kết quả:** các biểu đồ `.png` + `.csv` để dán vào báo cáo. |

### 4.6. `scripts/` — Công cụ chạy tay ⚙️

| File | Vai trò |
|---|---|
| [scripts/calibrate_threshold.py](../scripts/calibrate_threshold.py) | **Hiệu chuẩn ngưỡng thật:** trỏ vào thư mục ảnh của bạn → tính ngưỡng quen-lạ → ghi `configs/thresholds.json`. |
| [scripts/capture_faces.py](../scripts/capture_faces.py) | Chụp **dataset khuôn mặt** từ webcam. |
| [scripts/capture_objects.py](../scripts/capture_objects.py) | Chụp **dataset đồ vật** từ webcam (SPACE = lưu, ESC = thoát). |
| [scripts/try_vlm.py](../scripts/try_vlm.py) | **Thử OCR/VLM thật trên 1 ảnh** bằng 1 lệnh (không cần webcam/UI) — dùng để kiểm chứng key OpenAI/Gemini. |

### 4.7. `tests/`, `configs/`, `data/`

| File/Thư mục | Vai trò |
|---|---|
| [tests/](../tests/) | 6 file test tự động (CPM, orchestrator, calibration, drift, demo, providers) — **35 bài, tất cả pass.** Chạy `pytest` để yên tâm code không hỏng. |
| [configs/thresholds.json](../configs/thresholds.json) | Ngưỡng quen-lạ đã hiệu chuẩn (hiện là số trên LFW/Caltech; cần chạy lại trên ảnh cá nhân trước demo). |
| [data/](../data/) | Ảnh mẫu & dataset. Ảnh lớn **không** lưu vào git (xem `.gitignore`). |

---

## 5. Luồng hoạt động (theo bước)

### 5.1. Ba lệnh cốt lõi của CPM

```
DẠY (teach):
  ảnh ──[👁️ embed]──▶ dấu vân tay k ──▶ 🧠 CPM.write(k, "Lan")
       → cập nhật "dấu vân tay trung bình" của Lan   → "Đã ghi nhớ: Lan"

HỎI (ask):
  ảnh ──[👁️ embed]──▶ dấu vân tay q ──▶ 🧠 CPM.recall(q)
       → so q với trung bình của TỪNG người → chọn người giống nhất
       → độ giống ≥ ngưỡng?  Có → "Đây là Lan (0.95)"   Không → "Chưa biết"

SỬA (correct):
  ảnh + tên đúng ──▶ 🧠 CPM.correct(q, "Huy")
       → ghi mạnh mẫu này vào người "Huy" → lần sau nhận đúng
```

**Vì sao không quên?** Mỗi người có **ô nhớ riêng** (trung bình riêng). Dạy người mới chỉ **thêm ô
mới**, không đụng ô của người cũ → người cũ vẫn nguyên. (Khác với vài AI khác dùng chung một "bộ
tham số", học mới đè lên cũ → quên.)

### 5.2. Luồng đầy đủ qua Orchestrator (câu hỏi tiếng Việt)

```
     Câu tiếng Việt ("Ai đây?", "Đọc chữ giúp tôi", "Trước mặt có gì?")
                          │
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │ 🎧 ORCHESTRATOR (app/orchestrator.py)                │
   │ 0. Sửa lỗi font (mojibake) nếu có                    │
   │ 1. ⚠️ Kiểm tra vật cản trước → nếu nguy hiểm, CHEN   │
   │    NGANG cảnh báo (safety override)                  │
   │ 2. Đoán ý định theo TỪ KHOÁ:                         │
   │    "nhớ/đây là"  → DẠY   (CPM.write)                 │
   │    "sửa/nhầm"    → SỬA   (CPM.correct)               │
   │    "ai/cái gì"   → HỎI   (CPM.recall)                │
   │    "đọc/chữ"     → OCR    (skills/ocr)               │
   │    "mô tả/có gì" → SCENE  (skills/scene → VLM)       │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
                  Câu trả lời tiếng Việt (đọc lên bằng loa — bước sau)
```

### 5.3. "Thật" vs "Mô phỏng" — vì sao code luôn chạy được

Mỗi bộ phận có **2 chế độ**, để nhóm phát triển được **mọi lúc kể cả không có camera/model/mạng**:

| Bộ phận | Bản mô phỏng (mặc định, chạy ngay) | Bản thật (cần cài đặt) |
|---|---|---|
| Đôi mắt | `SyntheticEmbedder` (dấu vân tay giả tất định) | `RealEmbedder` (ArcFace + CLIP, cần M4) |
| Mô tả cảnh | `StubScene` (câu mẫu) | `RealScene` + OpenAI/Gemini (cần API key) |
| Đọc chữ | `StubOCR` (câu mẫu) | `RealOCR` + OpenAI/EasyOCR |
| Vật cản | `StubObstacle` (đặt khoảng cách tay) | `RealObstacle` (YOLO+Depth, cần M4) |

Khi bật bản thật mà thiếu key/thư viện → hệ thống **tự lùi về bản mô phỏng + in cảnh báo**, **không
sập**. Đây là lý do bạn thấy nhiều chỗ "Stub" trong code.

---

## 6. Cách chạy thử (copy–paste được)

> Trên Windows, **luôn** dùng Python trong môi trường ảo: `.\.venv312\Scripts\python.exe`

```powershell
# 1) Thử nhanh phần trí nhớ CPM (không cần model/camera) — nên chạy đầu tiên
.\.venv312\Scripts\python.exe -m app.cli --smoke

# 2) Thử toàn bộ điều phối (dạy/hỏi/vật cản) bằng dữ liệu giả
.\.venv312\Scripts\python.exe -m app.orchestrator --smoke

# 3) Chạy toàn bộ 35 bài kiểm tra tự động (yên tâm không hỏng gì)
.\.venv312\Scripts\python.exe -m pytest tests/ -q

# 4) Mở giao diện web (webcam + nút Ghi nhớ/Hỏi/Sửa)
.\.venv312\Scripts\python.exe -m app.demo_gradio

# 5) Vẽ lại các biểu đồ kết quả cho báo cáo
.\.venv312\Scripts\python.exe -m experiments.retention   # → results/retention.png
.\.venv312\Scripts\python.exe -m experiments.drift       # → results/drift.png

# 6) Thử OCR/VLM thật (cần: pip install openai + đặt $env:OPENAI_API_KEY)
.\.venv312\Scripts\python.exe -m scripts.try_vlm --image data\anh.jpg --query "Trước mặt có gì?"
```

---

## 7. Tình trạng hiện tại & điều cần nhớ khi báo cáo

**Đã xong (chạy được, có test):** lõi CPM + đôi mắt + điều phối + kỹ năng phụ + giao diện web +
bộ thí nghiệm; **35/35 test pass**. Đây là **Core Demo #1** — vòng lặp "dạy → nhớ → không quên".

**3 điều TRUNG THỰC phải nói khi bảo vệ** (rất quan trọng để không bị hội đồng bắt bẻ):
1. Đóng góp **KHÔNG phải "nhận diện chính xác hơn"** (baseline NCM cũng ngang) — mà là **cả hệ
   thống trí nhớ cá nhân hoá liên tục** (học 1 lần, không train lại, bộ nhớ bị chặn, đa người dùng,
   cổng quen-lạ) + **triển khai đa tầng edge–cloud** + **ứng dụng vào trợ thị**.
2. Số liệu hiện chủ yếu trên **dữ liệu giả/công khai** để kiểm chứng cơ chế — cần lặp lại với ảnh thật.
3. Thí nghiệm "trôi ngoại hình" (nơi CPM thắng NCM) là **kịch bản dựng có kiểm soát** — phải trình
   bày đúng phạm vi, không thổi phồng.

**Còn phải làm (theo kế hoạch 5 tháng):** giọng nói (nói lệnh/đọc trả lời), đồng bộ edge–cloud thật,
app điện thoại, tích hợp kính Meta Ray-Ban, thử nghiệm với người khiếm thị, viết luận văn/bài báo.

---

## 8. Đọc tiếp ở đâu

| Muốn hiểu | Đọc file |
|---|---|
| Kế hoạch tổng thể 5 tháng, kiến trúc, phân công | [KE_HOACH_DO_AN.md](KE_HOACH_DO_AN.md) |
| Đặc tả demo đầu tiên + task list chi tiết | [PLAN_CORE_DEMO.md](PLAN_CORE_DEMO.md) |
| Kết quả thí nghiệm + cách định vị đóng góp (cho báo cáo) | [KET_QUA_THI_NGHIEM.md](KET_QUA_THI_NGHIEM.md) |
| Các ý tưởng mở rộng để dành sau | [MO_RONG_VA_BACKLOG.md](MO_RONG_VA_BACKLOG.md) |
| **File này** — hiểu tổng thể từ số 0 | Understand.md |
```
