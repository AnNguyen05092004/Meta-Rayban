# Plan hoàn thiện đồ án — từ trạng thái hiện tại đến demo trên kính Meta Ray-Ban

> Tài liệu này nối tiếp [KE_HOACH_DO_AN.md](KE_HOACH_DO_AN.md) (plan 5 tháng gốc). Nó **không thay thế** plan gốc, mà biến phần "còn lại" thành **task-list làm được ngay** theo sprint, dựa trên trạng thái code THẬT (tháng 7/2026) và 4 quyết định nhóm vừa chốt.

**Ngày lập:** 2026-07-14 · **Deadline cuối:** ~T12/2026 (~4.5 tháng còn lại)

---

## 0. Bốn quyết định nhóm đã chốt (14/07/2026)

| # | Câu hỏi | Trả lời | Hệ quả cho plan |
|---|---|---|---|
| 1 | Trọng tâm 3–4 tuần tới | **Vòng lặp giọng nói (WS-A)** | Bắt đầu bằng luồng nói–nghe trên laptop/M4 |
| 2 | Dạng demo lúc bảo vệ | **BẮT BUỘC kính Meta Ray-Ban** | Kéo WS-E (app hub) + WS-C (backend) + WS-F (kính) vào **đường găng**; SDK kính thành rủi ro số 1 phải kiểm chứng SỚM |
| 3 | Deadline | **Chỉ có mốc cuối ~T12/2026** | Đủ thời gian làm đầy đủ; vẫn cần de-risk sớm |
| 4 | Đã verify gì trên máy thật | **Mới có MacBook M4** (chưa chạy ArcFace/CLIP trên ảnh nhóm, chưa chạy OCR/VLM bằng key OpenAI) | 3 việc "nghiệm thu thật" đưa vào Sprint 1 |

---

## 1. Trạng thái hiện tại (đã đối chiếu code)

**Xong & có test (35/35 pass):**
- Lõi **CPM v2**: prototype backbone + delta-rule/associative (tuỳ chọn) + EMA drift + 3 tầng + cô lập theo `user_id` + novelty gate + calibrate ngưỡng + snapshot/load.
- **Orchestrator**: định tuyến intent tiếng Việt (dạy/sửa/hỏi) + SAFETY override + sửa mojibake.
- **Perception ảnh**: face=ArcFace, object=OpenCLIP; VLM scene/VQA + OCR (OpenAI/Gemini/EasyOCR) — có bản Real, thiếu key thì fallback Stub.
- **Thực nghiệm**: retention, ablation, drift, calibration, real_lfw + baselines (CPM/NCM/kNN/EWC/fine-tune). Diễn giải trung thực trong [KET_QUA_THI_NGHIEM.md](KET_QUA_THI_NGHIEM.md).

**Chưa có / mới là khung:**
- ❌ Giọng nói (STT/TTS/wake-word) · ❌ App điện thoại · ❌ Backend/sync/multi-user server
- 🟡 Vật cản `RealObstacle.check()` còn `NotImplementedError` (chỉ Stub chạy) · ❌ On-device/offline packaging · ❌ Tích hợp kính · ❌ User study

**Định vị đóng góp (giữ nguyên, trung thực):** đóng góp KHÔNG phải "độ chính xác nhận diện" (trên embedding tốt thì NCM/kNN cũng ~1.0) mà là **hệ thống bộ nhớ cá nhân hoá LIÊN TỤC** (dạy/sửa 1-shot online, bộ nhớ bị chặn, đa người dùng, novelty gate) + **triển khai edge-cloud đa tần số** + **ứng dụng NL vào assistive tech**. Nơi CPM kỳ vọng vượt NCM = **appearance drift** (tầng nhanh thích nghi, tầng chậm giữ gốc).

---

## 2. ⚠️ PHỤ THUỘC SỐ 1 — SDK kính Meta Ray-Ban (phải kiểm chứng NGAY)

Vì demo bảo vệ **bắt buộc** chạy trên kính, rủi ro **R1** (🔴 Cao trong plan gốc) trở thành **đường găng**: nếu SDK **Wearables Device Access Toolkit** không cho app bên thứ ba **truy cập camera/audio real-time**, thì toàn bộ giả định "demo trên kính" sụp — và ta phải biết điều đó **trong 2 tuần đầu**, không phải tháng 11.

**Cổng quyết định (Sprint 1):**
1. Đăng ký/khảo sát Wearables Toolkit: có cho **video stream real-time** không? Waitlist? Giới hạn khu vực (VN)? Cần tài khoản/ thiết bị gì?
2. Nếu **CÓ** → tiếp tục lộ trình kính (WS-F) như dưới.
3. Nếu **KHÔNG / chưa mở** → **kích hoạt phương án dự phòng** (giữ đúng tinh thần "kiến trúc tách rời phần cứng" của plan §4.1):
   - **B1:** Camera đeo DIY (ESP32-CAM / camera kẹp mũ-kính) → stream vào điện thoại. Vẫn là "thiết bị đeo".
   - **B2:** Điện thoại đeo trước ngực/cổ (lanyard) làm camera+mic+loa. Chắc ăn nhất.
   - Báo cáo nêu rõ: kính là mục tiêu, phương án đeo khác là fallback do ràng buộc SDK — đây là **lập luận kỹ thuật hợp lệ**, không phải điểm trừ.

> **Khuyến nghị:** đừng mua kính cho tới khi bước 1 xác nhận SDK dùng được (plan §12 R1). Cả TV-A/B/C cùng làm bước này tuần đầu.

---

## 3. Kiến trúc mục tiêu cho bản chạy trên kính

Vì kính chỉ là cảm biến, topology cuối bám theo plan §6.2 (3 tầng vật lý):

```
┌─────────────┐  BLE/Wi-Fi ┌───────────────────────────────┐  Wi-Fi/4G ┌────────────────────┐
│    KÍNH      │ ─────────▶ │   ĐIỆN THOẠI (HUB / EDGE)      │ ────────▶ │  BACKEND (M4/CLOUD) │
│ camera/mic/ │            │  - STT + TTS tiếng Việt        │           │  - VQA/OCR nặng     │
│ loa         │ ◀───────── │  - Orchestrator + CPM fast     │ ◀──────── │  - consolidation    │
└─────────────┘   audio    │  - vật cản on-device (YOLO-n)  │  adapter  │  - memory per-user  │
                           └───────────────────────────────┘           └────────────────────┘
```

- **Tầng FAST (điện thoại/kính):** phản hồi tức thì, offline được — vật cản + nhận diện người/đồ đã học + CPM delta-rule.
- **Tầng SLOW (backend = M4 qua ngrok, hoặc cloud):** VQA/OCR nặng + consolidation + kho memory theo `user_id`.
- Luồng laptop/M4 ở Sprint 1 là **bản thu nhỏ của topology này** (máy tính đóng cả 3 vai) → chạy trước để de-risk phần mềm, rồi mới tách ra kính+phone+backend.

---

## 4. Roadmap theo sprint (2 tuần/sprint) — task-list chi tiết

Ký hiệu owner: **A** = TV-A (mobile/voice/on-device/kính) · **B** = TV-B (perception/backend/sync) · **C** = TV-C (CPM/thực nghiệm/viết). `[ ]` = việc cần làm.

### 🏁 Sprint 1 (nay → ~đầu T8) — Voice slice + de-risk kính + nghiệm thu model thật
**Mục tiêu:** nói–nghe chạy trên M4; biết chắc SDK kính dùng được không; số liệu chạy trên dữ liệu nhóm.
- [ ] **A** STT tiếng Việt: tích hợp `faster-whisper`/`whisper.cpp` (hoặc PhoWhisper) → `transcribe(audio)->text` (đặt `perception/audio.py` hoặc thêm vào `skills/providers.py`)
- [ ] **A** TTS tiếng Việt: Piper (offline) hoặc gTTS (online nhanh) → `speak(text)`
- [ ] **A** `scripts/voice_loop.py`: mic → STT → `VisionAssistant.handle()` → TTS → loa (space-to-talk trước, wake-word sau)
- [ ] **A+B+C** ⚠️ **Kiểm chứng SDK kính** (mục §2) → **CỔNG QUYẾT ĐỊNH**: đi tiếp kính hay bật fallback
- [ ] **B** Nghiệm thu OCR/VLM THẬT: `pip install openai pillow`, set `OPENAI_API_KEY`, chạy `python -m scripts.try_vlm --image <ảnh> --query "Trước mặt có gì?"` và "Đọc chữ giúp tôi"
- [ ] **C** Chạy ArcFace/CLIP trên **ảnh của nhóm** trên M4 (dùng `scripts/capture_faces.py` + `experiments/real_data.py`) → xác nhận số liệu trên dữ liệu cá nhân
- [ ] **A** Thêm nút **"Mô tả cảnh"/"Đọc chữ"** vào [app/demo_gradio.py](../app/demo_gradio.py) (hiện chỉ teach/ask/fix)
- [ ] **A/B** **Xử lý lỗi thực tế** khi ghép real embedder: KHÔNG phát hiện mặt/vật (`RealEmbedder.embed_face` **raise `ValueError`**), khung mờ, người dùng im lặng → trả lời nhẹ nhàng ("chưa thấy rõ, thử lại nhé") thay vì sập chương trình
- **Xong khi:** nói "trước mặt có gì?" trên M4 → nghe mô tả tiếng Việt (**vòng lặp < ~3–5s online**); có kết luận SDK kính; có 1 bảng số trên ảnh nhóm.

### Sprint 2 (~T8 nửa đầu) — Vật cản thật + dữ liệu cá nhân
- [ ] **B** Hiện thực [skills/obstacle.py](../skills/obstacle.py) `RealObstacle.check()`: YOLOv11n → boxes; Depth Anything v2 small → depth; vật gần nhất vùng giữa khung
- [ ] **B** Hiệu chỉnh depth→mét (vật biết kích thước / homography sàn); đo độ trễ (mục tiêu <300ms)
  - ⚠️ **Rủi ro:** Depth Anything cho **depth TƯƠNG ĐỐI**, không phải mét tuyệt đối. Dùng checkpoint **Metric** (indoor) nhưng vẫn sai số lớn theo cảnh. **Fallback an toàn:** cảnh báo theo mức "**gần / rất gần**" (ngưỡng trên depth tương đối + kích thước bbox) thay vì hứa đo mét chính xác — ưu tiên "không bỏ sót vật cản" hơn là con số đẹp.
- [ ] **A/B** **Vòng lặp camera liên tục** (thay "chụp 1 khung" `grab_webcam_bgr`): capture stream + safety monitor chạy nền mỗi N khung → nền cho vật cản real-time (khắc phục mục "Camera stream 🟡")
- [ ] **B** Nối RealObstacle vào safety-override (giữ Stub cho test); thêm test kịch bản gần/xa
- [ ] **C** Recalibrate ngưỡng trên **data cá nhân** (`configs/thresholds.json` đang là số LFW/Caltech): `python -m scripts.calibrate_threshold --data_dir data/faces --modality face --impostor_split 3` (+ object)
- [ ] **A** Wake-word đơn giản (openWakeWord) thay space-to-talk
- **Xong khi:** camera thật → cảnh báo "vật cản ~1m" real-time; ngưỡng nhận diện là số của nhóm.

### Sprint 3 (~T8 nửa sau) — Khung app điện thoại + khung backend
- [ ] **A** Flutter skeleton: camera stream + thu/phát âm + kết nối WebSocket (chọn Android/iOS — xem §6 câu hỏi mở)
- [ ] **B** Backend skeleton: FastAPI + WebSocket + định tuyến theo `user_id`; đẩy VQA/OCR nặng lên backend
- [ ] **B** Chuyển RealScene/RealOCR chạy phía backend; điện thoại gọi qua WS
- **Xong khi:** app điện thoại gửi ảnh+giọng nói tới backend (M4 qua ngrok) và nghe trả lời.

### Sprint 4 (~T9 nửa đầu) — CPM per-user trên backend + UX giọng nói trên phone
- [ ] **C** Bổ sung `export_delta()` / `import_adapter()` cho CPM (plan §5.5 — hiện mới có snapshot/load)
  - ⚠️ **Sửa:** `snapshot/load` hiện dùng **pickle** — cho backend nhận dữ liệu client là **rủi ro bảo mật (thực thi code tuỳ ý)** + giòn theo phiên bản. Đổi sang **npz (mảng) + JSON (metadata)** hoặc safetensors cho phần export/sync.
- [ ] **B** Memory store per-user: SQLite metadata + snapshot theo `user_id`
- [ ] **A** App: UX giọng nói hoàn chỉnh (wake-word, ngắt lời, đọc trả lời); thao tác không cần nhìn màn hình
- **Xong khi:** 2 user khác nhau đăng nhập → mỗi người bộ nhớ riêng, không rò.

### Sprint 5 (~T9 nửa sau) — Sync edge-cloud + on-device offline
- [ ] **B+C** Sync protocol §6.4: push delta+correction → consolidate → pull → merge (không đè fast-weight); versioning last-writer-wins
- [ ] **C** Lịch chạy `consolidate()` nền (replay mục tin cậy cao → slow tier; prune nhiễu)
- [ ] **A** On-device: YOLO-nano + CPM fast-weight chạy offline trên phone/M4; VLM nhỏ (Moondream) cho mô tả offline
- **Xong khi:** "sạc → sync" hoạt động; rút mạng vẫn cảnh báo vật cản + nhận diện người đã học.

### Sprint 6 (~T10 nửa đầu) — Tích hợp kính + thí nghiệm drift
- [ ] **A** Tích hợp kính (nếu SDK OK ở Sprint 1): kính = camera/mic/loa → phone hub qua BLE; hoặc chạy fallback B1/B2
- [ ] **A** Kịch bản đầu-cuối trên kính: "nhìn → hỏi → nghe trả lời"
- [ ] **C** Thí nghiệm **appearance drift bán-thật** (augment ảnh nhóm theo thời gian) — bằng chứng CPM-EMA > NCM
- [ ] **C** Chốt **cấu hình CPM khi deploy**: prototype-mean (mặc định) vs **+EMA** (`use_ema`) vs **+associative-matrix/delta-rule** (`use_associative_matrix`) — có thí nghiệm biện minh; nếu bật EMA phải calibrate lại ngưỡng (khắc phục mục "delta-rule 🟡" + "associative recall 🟡")
- [ ] **C** Ghi rõ trong báo cáo: associative/delta-rule **TẮT mặc định là lựa chọn CÓ CHỦ Ý** từ ablation trung thực (prototype-mean bền hơn khi danh tính giống nhau) — KHÔNG phải thiếu sót; giá trị đa tầng thể hiện ở kịch bản appearance drift
- **Xong khi:** demo chạy trên kính (hoặc thiết bị đeo fallback); có biểu đồ drift trên dữ liệu bán-thật.

### Sprint 7 (~T10 nửa sau) — Hardening + đo lường + freeze tính năng
- [ ] **A** Ổn định demo kính (độ trễ, mất kết nối, pin); haptic/âm báo an toàn
- [ ] **C** Đo sync/consolidation cost + online update cost (metric §11.1); bảng task-success kịch bản chuẩn
- [ ] **A+B+C** **Tự test kỹ** toàn bộ kịch bản (bước tiền đề trước khi mời người khiếm thị)
- **Xong khi:** đóng băng tính năng; hệ thống chạy ổn định end-to-end.

### Sprint 8 (~T11 nửa đầu) — User study
- [ ] **C** Liên hệ hội/trường người khiếm thị; chuẩn bị phiếu SUS + kịch bản phỏng vấn
- [ ] **A+B+C** Chạy test thực tế; thu SUS (mục tiêu ≥68) + phản hồi định tính
- [ ] **A+B** Sửa lỗi & tối ưu độ trễ theo phản hồi
- **Xong khi:** có số liệu user study + danh sách cải tiến.

### Sprint 9 (~T11 nửa sau → T12) — Viết + video + bảo vệ
- [ ] **C** Báo cáo đồ án: gom từ docs/ + [Understand.md](Understand.md) + [KET_QUA_THI_NGHIEM.md](KET_QUA_THI_NGHIEM.md)
- [ ] **C** Draft bài báo (định vị đóng góp = hệ thống liên tục + edge-cloud + ứng dụng)
- [ ] **A** Video demo + slide bảo vệ
- [ ] Buffer T12: polish, tập bảo vệ.

### Gantt rút gọn
```
Sprint       1    2    3    4    5    6    7    8    9
Voice (A)    ██   ░
Kính-SDK⚠   ██ ─────────────────▶ tích hợp: ██(S6)  ██(S7)
Vật cản (B)       ██   ░
App phone(A)           ██   ██   ░
Backend (B)            ██   ██   ██
CPM sync (C)                ██   ██
On-device(A)                     ██   ░
Drift/exp(C)      ░              ░    ██   ██
User study(C)                                   ██
Viết (C)     ·    ·    ·    ·    ▓    ▓    ▓    ▓   ██
```

---

## 4b. Truy vết — mỗi mục 🟡/❌ trong sơ đồ §4.2 đều có task

**Nhóm 1 — các ô 🟡/❌ ngay trong sơ đồ kiến trúc:**

| Thành phần (màu cũ) | Task xử lý | Sprint |
|---|---|---|
| Camera stream 🟡 | WS-A frame source + **WS-B vòng lặp camera liên tục** | 1–2 |
| Mic / STT ❌ | WS-A STT tiếng Việt | 1 |
| Loa / TTS ❌ | WS-A TTS tiếng Việt | 1 |
| Nút / wake-word ❌ | WS-A space-to-talk → openWakeWord | 1–2 |
| Object/Obstacle YOLO+Depth 🟡 | WS-B hiện thực `RealObstacle.check()` + depth→mét | 2 |
| delta-rule self-modifying 🟡 | WS-G chốt cấu hình CPM deploy + biện minh (chủ ý) | 6 |
| associative recall k→v 🟡 | WS-G (như trên) | 6 |
| SAFETY override 🟡 (đọc từ Stub) | WS-B nối RealObstacle vào safety-override | 2 |

**Nhóm 2 — các phần ❌ để "đủ như hình đầy đủ" (ngoài 4 ô, thuộc §6/§7):**

| Phần thiếu | Task xử lý | Sprint |
|---|---|---|
| Backend + đa người dùng | WS-C FastAPI + memory store per-user | 3–4 |
| Sync edge-cloud + consolidation lịch | WS-C sync §6.4 + `export_delta/import_adapter` | 4–5 |
| On-device / offline | WS-D YOLO-nano + CPM fast-weight + VLM nhỏ | 5–6 |
| App điện thoại (hub) | WS-E Flutter | 3–4 |
| Kính Meta Ray-Ban | WS-F (sau khi kiểm chứng SDK ở S1) | 6–7 |
| User study | WS-G mời hội/trường người khiếm thị | 8 |
| Số liệu trên dữ liệu nhóm + ngưỡng thật | WS-I + WS-G recalibrate | 1–2 |

→ **Kết luận: mọi mục 🟡/❌ đều có task tương ứng.** Hai mục CPM (delta-rule/associative) là **lựa chọn thiết kế có chủ ý** (giữ tắt), nên "xử lý" ở đây = ra quyết định + biện minh trong báo cáo, không nhất thiết là bật lên.

---

## 5. Đường găng, rủi ro & điểm cắt (nếu chậm)

**Đường găng (critical path):** SDK kính (S1) → app hub (S3–4) → backend (S3–5) → tích hợp kính (S6–7) → user study (S8) → viết (S9). Nếu một mắt trượt, cắt theo thứ tự dưới.

| Ưu tiên giữ | Có thể cắt nếu thiếu thời gian | Lý do |
|---|---|---|
| Voice slice + CPM + vật cản + 1 thiết bị đeo chạy được | | Đủ để bảo vệ "sản phẩm + nghiên cứu" |
| Demo trên **1** thiết bị đeo (kính **hoặc** fallback) | Kính "xịn" nếu SDK vướng → dùng fallback B1/B2 | R1; kiến trúc tách rời cho phép |
| Thực nghiệm drift + retention (đã có nền) | | Là bằng chứng cho đóng góp |
| | **Edge-cloud sync (WS-C)** → làm **on-device-only** trước | R4: sync là phần nâng cao, cắt được |
| | **On-device offline (WS-D)** → chạy qua backend M4 | Vẫn demo được khi có mạng |
| | Multi-user quy mô → demo 2 user là đủ minh hoạ | |

**Rủi ro cần theo dõi:** ① SDK kính (§2) — de-risk S1; ② STT/TTS tiếng Việt kém → fallback cloud (PhoWhisper/Piper trước, Google STT/TTS sau); ③ vật cản depth→mét sai số → nêu rõ giới hạn, ưu tiên "không bỏ sót"; ④ ôm quá nhiều (R6) → bám điểm cắt ở trên.

---

## 6. Câu hỏi còn mở (không chặn, nhưng cần trả lời sớm)

- [ ] **Điện thoại hub là Android hay iOS?** → quyết định on-device stack (Core ML nếu iOS / TFLite/NNAPI nếu Android) và ảnh hưởng Sprint 3. *Cần chốt trước Sprint 3.*
- [ ] Giảng viên kỳ vọng "novelty" ở mức nào (ứng dụng NL + hệ thống edge-cloud đã đủ chưa)? → ảnh hưởng độ sâu WS-C.
- [ ] Có ngân sách mua kính không (nếu SDK OK, ~300–380$)? → ảnh hưởng cổng quyết định §2.

---

## 7. Việc làm NGAY tuần này (top 5)

1. **A** Dựng `scripts/voice_loop.py` với STT+TTS tiếng Việt → có bản nói–nghe chạy trên M4.
2. **A+B+C** Khảo sát/đăng ký **Wearables Toolkit** — trả lời câu "kính có cho video real-time không?".
3. **B** Chạy `try_vlm.py` với key OpenAI → nghiệm thu OCR/VLM thật.
4. **C** Chụp ảnh mặt/đồ của nhóm + chạy `real_data.py` trên M4 → số liệu dữ liệu cá nhân + recalibrate ngưỡng.
5. Chốt **Android hay iOS** cho điện thoại hub.

> Cập nhật trạng thái các `[ ]` ở trên trực tiếp trong file này, hoặc mirror sang GitHub Projects theo sprint (plan §14.2).

---

## 8. Ghi chú rà soát code (14/07/2026) — điểm sửa/cải thiện

Sau khi đọc lại code thật (cpm/memory.py, config.py, app/cli.py, skills/*, requirements.txt, configs/thresholds.json):

**Đã xác nhận plan mô tả ĐÚNG:** CPM v2 (prototype backbone + delta-rule/associative/EMA tắt mặc định + per-user `user_id` + `consolidate()` chỉ tác dụng khi bật ma trận + KHÔNG có `export_delta/import_adapter`); obstacle chỉ Stub; orchestrator safety đọc từ Stub; `thresholds.json` = số LFW/Caltech (far1: face 0.661, object 0.659) cần recalibrate; Real scene/OCR tiêm callable + fallback Stub. 35/35 test pass.

**Đã vá vào plan (mục tương ứng ở trên):**
- Depth→mét là **tương đối, không tuyệt đối** → thêm fallback "gần/rất gần" (WS-B).
- `snapshot/load` dùng **pickle** → đổi serialization an toàn cho backend/sync (WS-C).
- Ghép real embedder dễ **sập khi không thấy mặt/vật** → thêm task xử lý lỗi (WS-A).
- Thêm **ngân sách độ trễ voice loop < ~3–5s** (WS-A).

**Cần bổ sung khi tới sprint tương ứng (chưa phải task cứng bây giờ):**
- [ ] **Phụ thuộc:** `requirements.txt` hiện mới liệt kê `ultralytics`; khi làm WS-B cần thêm **model depth** (Depth Anything), khi làm WS-A cần **STT/TTS** (đang để mục "stretch"). Bỏ comment đúng lúc, tránh cài thừa.
- [ ] **Đạo đức/đồng thuận (TRƯỚC user study, Sprint 8):** phiếu đồng thuận, quy tắc xử lý ảnh khuôn mặt (ưu tiên on-device, mã hoá, cho xoá) — plan gốc §7.3 có nguyên tắc, cần biến thành checklist cụ thể trước khi mời người thật.
- [ ] **Duy trì CI test xanh** xuyên suốt (repo đã là git); mỗi WS xong phải giữ pytest pass.

**Khung hoá trung thực cho báo cáo (quan trọng khi bảo vệ):**
- Nếu **cắt WS-C** (sync edge-cloud, theo điểm cắt R4) thì đóng góp "edge-cloud đa tầng-thời-gian" trong paper phải khung lại là **"thiết kế + phân tích"**, KHÔNG tuyên bố "đã đo đạc trên hệ thống thật" — nếu không sẽ hớ khi phản biện.
- Với nhóm 3 người / 1 người mobile, kịch bản **kính-bắt-buộc + full-stack (voice+obstacle+backend+sync+on-device+kính)** là bản THAM VỌNG NHẤT. Khuyến nghị **khoá "core chắc chắn bảo vệ được"** = voice + obstacle + CPM + **1** thiết bị đeo chạy được + thí nghiệm + viết; coi backend/sync/on-device/kính-polish là **stretch có điểm cắt**.

**Cải thiện tuỳ chọn (backlog, không bắt buộc):**
- Định tuyến intent hiện bằng **so khớp từ khoá** (dễ nhầm khi câu dài) → có thể nâng lên **phân loại ý định bằng LLM** (đã có key OpenAI) khi cần bền hơn cho demo.
