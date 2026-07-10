# Kế hoạch chi tiết — 4 task tiếp theo

> Lập trước khi code, để bạn duyệt. Bám sát API thật hiện có:
> `cpm/memory.py` (CPM v2, recall trả `confidence=cos(query,prototype)`), `cpm/config.py`
> (`recall_threshold=0.35`), `experiments/metrics.py` (đã có `eer`, `tar_at_far`, `open_set_summary`),
> `experiments/baselines.py` (CPM/NCM/kNN/EWC/FineTune), `experiments/retention.py` (`run_forgetting`),
> `app/orchestrator.py` + `skills/` (Stub/Real, tiêm callable). Cập nhật: 2026-07-10.

## Tổng quan & thứ tự ưu tiên

| # | Task | Loại | Giá trị | Ai code / ai chạy | Ước lượng |
|---|------|------|---------|-------------------|-----------|
| 1 | Calibrate ngưỡng quen/lạ theo embedder thật | **Sửa lỗi thật** | Cao (demo đúng) | Tôi code + test synthetic; bạn chạy trên LFW | ~0.5 ngày |
| 2 | Thí nghiệm appearance-drift (CPM > NCM) | **Nghiên cứu** | Cao nhất (điểm mạnh còn thiếu) | Tôi code + test; bạn chạy | ~1–1.5 ngày |
| 3 | Test đồ vật thật + webcam Gradio | Sản phẩm/demo | Trung bình | Tôi enable + test mock; **bạn chụp ảnh & bấm webcam** | ~0.5 ngày (tôi) |
| 4 | Gắn OCR/VLM thật (2 skill còn stub) | Sản phẩm | Trung bình | Tôi viết provider + wiring; **bạn cắm API key/cài lib** | ~1 ngày |

**Phụ thuộc & thứ tự đề xuất:** `1 → 2 → 3 → 4`.
- Task 1 độc lập, sửa lỗi thật → làm trước.
- Task 2 cần thêm 1 thành phần nhỏ vào CPM (EMA đa tầng) → làm sau khi CPM "đứng yên" ở task 1.
- Task 3 & 4 hướng sản phẩm, phụ thuộc máy/dữ liệu/API của bạn → tôi làm phần code + test bằng mock/synthetic, bạn nghiệm thu trên máy thật.

**Nguyên tắc chung:** mọi thứ tôi viết đều có test chạy được **không cần model/mạng** (synthetic hoặc mock); phần cần dữ liệu/model/API thật thì bạn chạy trên `.venv312`.

---

## Task 1 — Calibrate ngưỡng quen/lạ theo embedder thật

### Mục tiêu
Thay ngưỡng cứng `recall_threshold=0.35` (suy ra từ dữ liệu synthetic) bằng ngưỡng **hiệu chỉnh từ ROC thật** của từng embedder + từng modality. Sửa lỗi hiện tại: trên facenet thật, 0.35 cho **FAR≈0.6** (nhận nhầm ~60% người lạ) dù AUC=0.985.

### Vì sao có lỗi
`confidence = cos(query, prototype)`. Trên facenet/ArcFace thật, cos giữa 2 người *khác nhau* ~0.3–0.5 (không phải ~0 như synthetic). Ngưỡng phải nằm ở vùng phân tách thật (≈ ngưỡng EER, có thể ~0.5–0.6), và **khác nhau giữa face (ArcFace) và object (CLIP)** → phải calibrate riêng theo `(embedder, modality)`.

### Input / Output
- **Input:** tập có nhãn: người/đồ **quen** (mỗi nhãn vài mẫu → dạy + giữ lại test) và người/đồ **lạ** (impostor, không dạy).
- **Output:**
  - `threshold` tối ưu theo chính sách chọn + bảng số (EER, TAR@FAR=1%, TAR@FAR=10%).
  - File lưu ngưỡng theo `(embedder, modality)` để nạp lại ở runtime.

### Thiết kế
**File mới `cpm/calibration.py`** (thuần numpy, test được bằng synthetic):
```python
def collect_open_set_scores(cpm, known_test, impostor_embs):
    """genuine = confidence recall của mẫu test người QUEN;
       impostor = confidence recall của mẫu người LẠ (max cos qua mọi prototype)."""
    genuine  = [cpm.recall(e)[0]["confidence"] for label, embs in known_test for e in embs]
    impostor = [cpm.recall(e)[0]["confidence"] for e in impostor_embs]
    return np.array(genuine), np.array(impostor)

def calibrate_threshold(genuine, impostor, policy="far1"):
    """policy: 'far1' (TAR@FAR=1%, an toàn) | 'far10' | 'eer' (cân bằng).
       Trả dict {threshold, policy, eer, auc, tar@far=1%, ...} — dùng metrics.py sẵn có."""
```
**Lưu/nạp ngưỡng** — file `configs/thresholds.json`:
```json
{ "arcface_buffalo_l:face": 0.52, "openclip_vitb32:object": 0.78 }
```
- `save_thresholds(path, mapping)` / `load_thresholds(path)`.
- `app/cli.py::Assistant.__init__`: sau khi tạo `cpm_face`/`cpm_object`, nếu có file ngưỡng khớp `(embedder_name, modality)` thì set `cpm.cfg.recall_threshold = <đã calibrate>`. Không có file → giữ mặc định 0.35 (synthetic) + in cảnh báo "chưa calibrate".

**Refactor `experiments/real_lfw.py`:** dùng `collect_open_set_scores` + `calibrate_threshold` thay vì tính tay; vẽ **điểm ngưỡng đã calibrate** (thay chấm đỏ 0.35 sai) và in ngưỡng ra để bạn lưu vào `configs/thresholds.json`.

### Chính sách chọn ngưỡng (khuyến nghị)
Cho **trợ thị**, nhận nhầm người lạ thành người quen (chào nhầm) là lỗi UX/an toàn → ưu tiên **FAR thấp**.
- **Mặc định đề xuất: `far1` (TAR@FAR=1%)** — chắc chắn, hiếm khi nhận nhầm.
- Thay thế: `eer` (cân bằng FAR/FRR) nếu FAR=1% khiến bỏ sót người quen quá nhiều.
> Đây là **quyết định thiết kế #1** cần bạn chốt (mặc định tôi đề xuất `far1`).

### Các bước
1. `cpm/calibration.py`: `collect_open_set_scores`, `calibrate_threshold`, `save/load_thresholds`.
2. `tests/test_calibration.py`: synthetic genuine ~N(0.8), impostor ~N(0.3) → ngưỡng nằm giữa; FAR tại ngưỡng `far1` ≤ 0.01 + 1e-6; ngưỡng `eer` cho FAR≈FRR.
3. Sửa `app/cli.py`: nạp ngưỡng theo `(embedder, modality)`; cảnh báo nếu thiếu.
4. Refactor `real_lfw.py`: dùng hàm calibrate, vẽ điểm đúng, in ngưỡng đề xuất.
5. Cập nhật `docs/KET_QUA_THI_NGHIEM.md`: giải thích lỗi 0.35 + số sau calibrate.

### Tiêu chí nghiệm thu
- Test synthetic: ngưỡng `far1` cho FAR ≤ 1% trên tập impostor; TAR còn cao.
- Bạn chạy `real_lfw`: chấm đỏ dịch về vùng FAR thấp; in ra ngưỡng face thật (kỳ vọng ~0.5–0.6).
- Orchestrator: người lạ → "chưa nhận ra" (không còn chào nhầm) sau khi nạp ngưỡng thật.

### Rủi ro / lưu ý
- Cần đủ impostor để ước lượng FAR ổn định (≥ vài chục mẫu). Ít mẫu → ngưỡng nhiễu; ghi rõ N trong output.
- Ngưỡng phụ thuộc embedder: đổi model (buffalo_l ↔ facenet) phải calibrate lại → dùng khoá `(embedder_name, modality)`.

---

## Task 2 — Thí nghiệm appearance-drift (nơi CPM > NCM)

### Mục tiêu
Tạo **bằng chứng CPM vượt NCM** — mảnh còn thiếu (hiện mọi kết quả là parity). Câu chuyện: khi **ngoại hình đổi theo thời gian** (đổi kiểu tóc/kính/ánh sáng; đồ vật đổi bối cảnh), prototype **trung-bình-thật của NCM bị kéo lùi bởi mẫu cũ** → tụt; còn CPM **đa tầng (fast/slow)** bám được ngoại hình *hiện tại* mà vẫn giữ danh tính.

### Vì sao cần thêm 1 thành phần nhỏ vào CPM
Hiện `proto_keys` của CPM = **trung bình thật = NCM y hệt** → trên dữ liệu dừng (stationary) hai bên **bằng nhau** (đã xác nhận). Muốn CPM thắng khi có drift, CPM cần một **tầng nhanh có thiên lệch gần đây (recency)**. Đề xuất: thêm **prototype EMA trong không gian embedding** (đúng tinh thần Continuum Memory System, tránh điểm yếu của ma trận liên kết đã biết).

**Bổ sung CPM (`cpm/memory.py` + `cpm/config.py`):**
```python
# config: use_recency_ema=False (mặc định TẮT → không đổi hành vi cũ),
#         ema_beta=0.3 (tầng nhanh), recency_w=0.5 (trọng số trộn khi recall)
# memory: song song proto_keys (SLOW, trung bình thật) thêm:
#   proto_ema[label] = (1-beta)*proto_ema[label] + beta*unit(key)   # FAST, bám gần đây
# recall khi bật EMA: score = (1-recency_w)*cos(q, proto_slow) + recency_w*cos(q, proto_ema)
# confidence/known VẪN = cos(q, proto_slow)  (cổng danh tính ổn định) — hoặc max(2 tầng), sẽ chốt.
```
> **Quyết định thiết kế #2:** thêm **EMA đa tầng** (khuyến nghị — đơn giản, embedding-space, đúng NL, tránh điểm yếu ma trận) **so với** hồi sinh ma trận liên kết fast/medium/slow (đã biết yếu với danh tính tương quan). Mặc định tôi đề xuất **EMA**.

### Dữ liệu drift (synthetic — chính)
`experiments/data.py::make_drift_stream(...)`: mỗi danh tính có quỹ đạo ngoại hình **quay dần** từ `proto_0` tới `proto_target` bằng **slerp** theo thời gian:
```
proto_t = slerp(proto_0, proto_target, t/T)      # góc drift tối đa vd 45–60°
mẫu quan sát tại t: sample_with_cos(proto_t, cos_intra, seed)   # (đã có sẵn hàm)
```
Sinh **luồng thời gian** đan xen N danh tính qua T bước; dạy tuần tự theo thời gian; **test = nhận diện ngoại hình HIỆN TẠI (mới nhất)**.

### So sánh & metric
- Phương pháp: **NCM** (trung bình đều — tụt), **kNN** (giữ hết — bám tốt nhưng bộ nhớ phình), **CPM-EMA** (bám + bộ nhớ chặn), FineTune, (tuỳ chọn EWC).
- Metric: **accuracy trên ngoại hình mới nhất** theo mức drift tích luỹ; phụ: footprint (nhắc lại kNN phình).
- **Câu chuyện thắng của CPM:** *độ chính xác cao khi có drift* **VÀ** *bộ nhớ bị chặn* — góc phần tư mà NCM (chặn nhưng tụt) và kNN (chính xác nhưng phình) đều không có.

### Output
`experiments/drift.py` → `results/drift.png` (accuracy-vs-drift cho các PP) + `drift.csv`. In bảng: acc@drift lớn của NCM vs CPM-EMA vs kNN + footprint.

### Các bước
1. `cpm/config.py`: thêm `use_recency_ema`, `ema_beta`, `recency_w` (mặc định TẮT).
2. `cpm/memory.py`: `proto_ema` (update ở `write`), nhánh recall khi bật EMA, cập nhật `snapshot/load` + `stats.footprint` (+dim mỗi nhãn).
3. `experiments/data.py`: `slerp`, `make_drift_stream`.
4. `experiments/baselines.py`: `CPMDriftAdapter` (bật `use_recency_ema`) — giữ `CPMAdapter` cũ nguyên vẹn.
5. `experiments/drift.py`: chạy + vẽ + CSV.
6. `tests/test_drift.py`: trên luồng drift, **acc(CPM-EMA) − acc(NCM) ≥ biên** (vd ≥ 0.15) ở drift lớn; trên dữ liệu **dừng**, CPM-EMA ≈ NCM (không hại trường hợp cũ).
7. `docs/KET_QUA_THI_NGHIEM.md`: thêm mục "CPM thắng ở đâu" + hình.

### Tiêu chí nghiệm thu
- Test: có drift → CPM-EMA > NCM rõ; không drift → xấp xỉ (không hồi quy).
- `drift.png` kể đúng 3 đường: NCM tụt, kNN cao (phình), CPM-EMA cao (chặn).
- Không phá 13/13 test cũ (EMA mặc định tắt).

### Rủi ro / lưu ý
- Phải **trung thực**: CPM thắng NCM **chỉ khi có drift**; dừng thì bằng. Ghi rõ trong báo cáo.
- `ema_beta`, `recency_w`, góc drift là siêu tham số → thêm 1 panel **ablation nhạy cảm** (beta nhỏ→giống NCM; lớn→nhiễu) để tránh "chọn số đẹp".
- Drift **thật** khó lấy (LFW không có mốc thời gian) → bản real là *stretch*: giả lập bằng augmentation tăng dần (sáng/xoay/che) hoặc dùng dữ liệu có lão hoá; ghi chú rõ synthetic là chính.

---

## Task 3 — Test đồ vật thật + webcam Gradio

### Mục tiêu
Hoàn thiện phần **demo sản phẩm**: (a) chạy retention trên **đồ vật thật** (không phải mặt-qua-CLIP như hiện tại), (b) test **webcam trong Gradio** với người thật.

### 3A. Đồ vật thật
- `experiments/real_data.py --modality object --data_dir data/objects` đã có; cần **ảnh đồ thật** trong `data/objects/<nhãn>/` (ví, chìa khoá, cốc… mỗi loại ≥5 ảnh, nền/góc khác nhau).
- **Tôi làm:** rà nhánh `object` trong `real_data.py` + `perception/embed.py` (CLIP nhận `PIL.Image`), thêm script chụp đồ **`scripts/capture_objects.py`** (giống `capture_faces` nhưng **bỏ bước dò khuôn mặt**), test headless `_to_embedding` với ảnh giả để bắt lỗi crash.
- **Bạn làm:** chụp/tải ảnh đồ, chạy `real_data.py` → `retention_real_object.png`.

### 3B. Webcam Gradio
- `app/demo_gradio.py --embedder real`. Chưa test người thật.
- **Tôi làm:** rà cấu hình component webcam (streaming/ảnh tĩnh), xử lý khung `None`/không có mặt, thông báo lỗi thân thiện; thêm **smoke headless** dựng `build_demo()` + chạy 1 lần embed ảnh giả (bắt lỗi import/logic mà không cần camera).
- **Bạn làm:** bấm theo kịch bản 5A trong `HUONG_DAN_M4.md` (Ghi nhớ → Hỏi → người lạ → Sửa), xác nhận không crash + cấp quyền camera.

### Output
- `scripts/capture_objects.py`, smoke webcam headless, `retention_real_object.png` (bạn tạo).
- Cập nhật `HUONG_DAN_M4.md`: kịch bản test đồ vật + lưu ý webcam trên Windows.

### Tiêu chí nghiệm thu
- Smoke headless demo chạy xanh (không cần camera).
- Bạn: webcam mở, Ghi nhớ/Hỏi/Sửa đúng luồng, người lạ → "chưa biết" (dùng ngưỡng đã calibrate ở Task 1).

### Rủi ro / lưu ý
- Quyền camera Windows (Settings → Privacy → Camera). Cổng 7860 bận → đổi cổng.
- CLIP coi *đồ khác loại* cũng khá giống → **ngưỡng object phải calibrate riêng** (Task 1) mới phân biệt "đồ quen/lạ" tốt.

---

## Task 4 — Gắn OCR/VLM thật (2 skill còn stub)

### Mục tiêu
Biến `StubOCR`/`StubScene` thành **thật**, giữ kiến trúc tiêm callable (`ocr_fn`, `vlm`) để không khoá cứng nhà cung cấp; bật qua `skills_mode="real"` trong orchestrator.

### Thiết kế
**File mới `skills/providers.py`** — factory lazy-import, đọc key từ biến môi trường:
```python
def easyocr_fn(langs=("vi","en")):   # -> callable(frame_rgb)->str   (đơn giản, có tiếng Việt)
def gemini_vlm(model="gemini-1.5-flash"):  # -> callable(frame_rgb, prompt)->str  (free tier, GEMINI_API_KEY)
def moondream_vlm():                  # -> callable(...)  (on-device, offline — stretch)
```
- **OCR (khuyến nghị MVP): EasyOCR** `readtext` (cài đơn giản, hỗ trợ `vi`+`en`). Thay thế chính xác hơn: PaddleOCR/VietOCR (cần detector) — để sau.
- **VLM (khuyến nghị MVP): Gemini free tier** (`gemini-1.5-flash`, quota free rộng, tiếng Việt tốt). Offline: **Moondream2** on-device (stretch). Vẫn có thể cắm GPT-4o/Claude qua cùng interface.
- `app/orchestrator.py`: hiện thực nhánh `skills_mode="real"` → dựng `RealOCR(ocr_fn=...)`, `RealScene(vlm=...)` từ provider (chọn qua tham số/env); thiếu key/lib → fallback Stub + cảnh báo.
- `requirements.txt`: thêm (comment/optional) `easyocr`, `google-generativeai`.

### Các bước
1. `skills/providers.py` (easyocr, gemini, moondream) — lazy import, key từ `os.environ`.
2. `skills/__init__.py`: export provider.
3. `orchestrator.__init__`: nhánh `real` dựng skill thật; fallback Stub nếu thiếu.
4. `tests/test_providers.py`: **mock** `ocr_fn`/`vlm` (callable trả chuỗi cố định) → orchestrator định tuyến đúng "đọc chữ"/"mô tả", **không gọi mạng**.
5. `docs`: hướng dẫn `GEMINI_API_KEY`, `pip install easyocr`, mẹo quota free.

### Tiêu chí nghiệm thu
- Test mock: intent "đọc chữ" → gọi `ocr_fn`; "trước mặt có gì" → gọi `vlm`; SAFETY vẫn chen ngang.
- Bạn (có key/lib): ảnh có chữ → đọc ra text; ảnh cảnh → mô tả tiếng Việt.

### Rủi ro / lưu ý
- Gemini free có **giới hạn tần suất** → thêm timeout + thông báo lỗi thân thiện; không chặn luồng khi API lỗi.
- **Riêng tư**: VLM cloud gửi ảnh lên server — nêu rõ trong báo cáo; hướng offline (Moondream) là điểm cộng cho người khiếm thị.
- EasyOCR kéo torch (đã có). Bản đầu có thể chậm trên CPU — chấp nhận cho demo.
- (Ngoài phạm vi 4 task) Obstacle thật (YOLO+Depth) có thể làm cùng đợt này nếu muốn — cùng kiểu provider.

---

## Quyết định đã chốt (2026-07-10)
1. **Ngưỡng (Task 1):** ✅ **`far1` (TAR@FAR≤1%, an toàn)** — ưu tiên không chào nhầm người lạ.
2. **Drift (Task 2):** ✅ **EMA đa tầng** — thêm prototype EMA (tầng nhanh) cạnh trung-bình-thật.
3. **Thứ tự:** ✅ **1 → 2 → 3 → 4**, bắt đầu Task 1.

## Trạng thái thực hiện
- **Task 1 — XONG (2026-07-10):** `cpm/thresholds.py`, `experiments/calibration.py`,
  `scripts/calibrate_threshold.py`, embedder `.name`, `Assistant` tự nạp ngưỡng,
  `real_lfw.py` vẽ điểm ngưỡng đúng, `tests/test_calibration.py` (8 test).
  Đã chạy calibration benchmark với dữ liệu công khai:
  `real:face=0.66098` trên `faces_lfw_12x20` và `real:object=0.65889`
  trên `objects_caltech_subset`. **24/24 pytest pass**, 2 smoke pass.
  *Còn lại trước deploy thật: chạy lại `scripts.calibrate_threshold` trên ảnh cá nhân/vật thể của bạn.*
- **Task 2 — XONG (2026-07-10):** thêm EMA đa tầng trong `cpm/memory.py`
  (`proto_ema`, score trộn slow/EMA, snapshot/load, footprint), `CPMEMAAdapter`,
  `experiments/drift.py` sinh `results/drift.png` + `results/drift.csv`,
  `tests/test_drift.py` (3 test). Kết quả drift mạnh: NCM=0.50, CPM-EMA=1.00,
  kNN=1.00; CPM-EMA footprint bị chặn (6,144 floats) còn kNN phình (49,152 floats).
  **24/24 pytest pass**, CLI/orchestrator smoke pass, retention/ablation pass.
- **Task 3 — CODE/SMOKE XONG (2026-07-10):** thêm `scripts/capture_objects.py`,
  làm sạch `app/demo_gradio.py` để xử lý ảnh rỗng/không đúng định dạng/lỗi không thấy mặt thân thiện hơn,
  thêm `--server-port`, thêm `tests/test_demo_gradio.py` (headless, không cần webcam).
  Đã có object benchmark thật hơn từ Caltech101 subset:
  `data/objects_caltech_subset` (12 lớp x 20 ảnh), `retention_real_object.png`.
  **28/28 pytest pass**. *Còn lại: bạn bấm webcam/browser và chụp đồ cá nhân thật để nghiệm thu UX.*
- **Task 4 — CODE/MOCK XONG (2026-07-10):** thêm `skills/providers.py`
  (`easyocr_fn`, `gemini_vlm`, placeholder `moondream_vlm`), wiring `app/orchestrator.py`
  với `--skills-mode real` và fallback stub nếu thiếu lib/API key, export providers,
  thêm `tests/test_providers.py` để kiểm tra OCR/VLM callable và routing không cần mạng.
  **32/32 pytest pass**. *Còn lại: bạn cài optional provider + đặt `GEMINI_API_KEY` nếu muốn chạy OCR/VLM thật.*
