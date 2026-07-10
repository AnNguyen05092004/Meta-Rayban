# Kết quả thí nghiệm & ĐỊNH VỊ ĐỒ ÁN (bản nháp cho báo cáo)

> Cập nhật: 2026-07-09. Tài liệu này ghi lại **hành trình thí nghiệm trung thực** và
> **định vị đóng góp đúng** — quan trọng để không bị hội đồng bác ở phần bảo vệ.
>
> ⚠️ Số liệu dùng **embedding TỔNG HỢP có kiểm soát** để kiểm chứng *cơ chế*. Số chính danh cần
> chạy lại với embedding thật (ArcFace/CLIP) trên M4 — xem [HUONG_DAN_M4.md](HUONG_DAN_M4.md).

## Tái lập
```bash
python -m experiments.retention   # -> results/retention.png
python -m experiments.ablation    # -> results/ablation.png
python -m pytest tests/ -v        # 6/6 pass
```

---

## 0. TL;DR — 3 kết luận trung thực (đọc trước)
1. **Catastrophic forgetting KHÔNG phải vấn đề** khi thao tác trên embedding *tách biệt* (frozen
   ArcFace/CLIP): mọi phương pháp (kể cả fine-tune) giữ ~100%. ⇒ **Đừng định vị đồ án là "chống quên".**
2. **NCM (nearest class mean) là baseline rất mạnh & đơn giản** — bền cả khi các danh tính giống nhau.
   Bản CPM ban đầu (ma trận liên kết + anchor ngẫu nhiên) **YẾU HƠN NCM** khi người giống nhau ⇒ đã
   **đổi thiết kế**: CPM v2 dùng **prototype trung-bình-thật làm xương sống** (mạnh như NCM, bộ nhớ gọn).
3. ⇒ **Đóng góp thật của đồ án KHÔNG phải "độ chính xác nhận diện"**, mà là **hệ thống bộ nhớ cá nhân
   hoá liên tục** (đa tần số + consolidation + novelty gate + triển khai edge–cloud) cho **ứng dụng
   trợ thị** — cùng phân tích tradeoff trung thực.

---

## 1. Hành trình (điều tưởng vs điều tìm ra)
| Giả thuyết ban đầu | Thí nghiệm | Sự thật | Hệ quả |
|---|---|---|---|
| "Fine-tune quên thảm khốc, CPM thì không" | retention id đầu | Chỉ id_0 = 0 do **artifact 1-lớp** (softmax 1-logit → gradient 0). Sửa (thêm logit background) → **fine-tune giữ 100%** | Bỏ luận điểm "chống quên" trên embedding tách biệt |
| "Ma trận liên kết NL là xương sống nhận diện" | độ giống danh tính | Khi người **giống nhau** (cos cao): NCM=1.0 nhưng **CPM-ma-trận≈0.02** | Đổi xương sống sang **prototype true-mean** |
| "CPM tiết kiệm bộ nhớ" | footprint | Bản cũ (3 ma trận 512²) **to hơn** kNN ở quy mô nhỏ | v2 bỏ ma trận mặc định → **bounded & gọn như NCM** |

---

## 2. Thí nghiệm chống quên & bộ nhớ (`retention.png`)
5 phương pháp (CPM v2 / NCM / kNN / EWC / fine-tune), dạy tuần tự 20 danh tính.

| | Acc (đã học) | Acc (id đầu) | Footprint @4000 quan sát |
|---|---|---|---|
| CPM v2 | 1.00 | 1.00 | **10,240** (bị chặn) |
| NCM | 1.00 | 1.00 | 10,240 (bị chặn) |
| EWC / fine-tune | 1.00 | 1.00 | ~10,240 (bị chặn) |
| kNN | 1.00 | 1.00 | **2,048,000** (PHÌNH) |

**Kết luận:** trên embedding tách biệt, **retention không phân biệt** các phương pháp (Panel A phẳng).
Điểm phân biệt duy nhất là **chi phí bộ nhớ** (Panel B): **kNN phình vô hạn**, còn CPM/NCM **bị chặn**.

---

## 3. Bộ thí nghiệm cơ chế (`ablation.png`)
### (1) Sức chứa ma trận liên kết ~ dim
Accuracy của **assoc-matrix** giảm khi N > dim ⇒ ma trận có giới hạn sức chứa. (dim=512 vẫn đủ cho use case, nhưng đây là lý do không dùng nó làm xương sống.)
### (2) Vai trò tầng (phân tích ma trận)
fast-only (α<1) sụp; cần tầng bền vững ⇒ chứng minh nguyên lý đa tần số (khi dùng ma trận).
### (3) assoc vs proto vs hybrid
proto & hybrid giữ ~100%; assoc giảm sau N=dim ⇒ **không dùng assoc đơn độc**.
### (4) ⭐ Bền theo ĐỘ GIỐNG danh tính (phát hiện quan trọng nhất)
Khi các danh tính giống nhau (shared cao): **CPM-proto = NCM (bền)**; **assoc-matrix & fine-tune SỤP**.
⇒ Đây là **lý do quyết định** CPM v2 dùng **prototype làm xương sống**.
### (5) Open-set (quen/lạ)
Trên **synthetic** AUC≈1.0, EER≈0, ngưỡng 0.35 → FAR≈0.01. **NHƯNG trên embedding THẬT (facenet)
ngưỡng 0.35 SAI**: cos giữa 2 người khác nhau ~0.3–0.5 nên 0.35 cho **FAR≈0.6** (nhận nhầm ~60%
người lạ) dù AUC vẫn 0.985. ⇒ **phải calibrate ngưỡng theo từng embedder + modality từ ROC thật.**

**Đã sửa (Task 1):** thêm `experiments/calibration.py` (`calibrate_threshold`, policy mặc định
`far1` = TAR@FAR≤1%, an toàn cho trợ thị) + `cpm/thresholds.py` (lưu/nạp theo `(embedder, modality)`).
- `python -m scripts.calibrate_threshold --data_dir data/faces --modality face --impostor_split 3`
  → ghi `configs/thresholds.json`; `app.cli.Assistant` tự nạp khi dùng embedder thật.
- `experiments/real_lfw.py` giờ vẽ **điểm ngưỡng đã calibrate** (đúng) cạnh chấm 0.35 cũ (minh hoạ lỗi).
- Ngưỡng phụ thuộc model: facenet ≠ ArcFace(buffalo_l) ≠ CLIP ⇒ calibrate lại khi đổi model/modality.

---

## 4. THIẾT KẾ CPM v2 (điều chỉnh theo bằng chứng)
- **Xương sống nhận diện = prototype trung-bình-thật** mỗi nhãn (bền như NCM; bộ nhớ O(#nhãn)).
- **Cổng novelty** (quen/lạ) = cos(query, prototype) với ngưỡng chọn từ ROC.
- **Cập nhật online** = 1 lần ghi (dạy/sửa tức thì, không train lại).
- **Ma trận liên kết đa tầng (fast/medium/slow) + consolidation** = **thành phần TUỲ CHỌN** để
  phân tích / thích nghi; **tắt mặc định** (config `use_associative_matrix=False`).

---

## 5. ĐỊNH VỊ ĐÓNG GÓP (đưa vào chương Mở đầu & Thảo luận)
**KHÔNG nói:** "CPM cho độ chính xác nhận diện cao hơn các phương pháp khác." (Sai — NCM ngang bằng.)

**NÊN nói — đóng góp là HỆ THỐNG + ỨNG DỤNG:**
1. **Hệ thống bộ nhớ cá nhân hoá liên tục** cho thiết bị trợ thị: dạy/sửa online 1-shot, không train
   lại cloud, bộ nhớ bị chặn, cô lập đa người dùng, có cổng quen/lạ.
2. **Triển khai đa tần số edge–cloud** (§6 plan tổng): tầng nhanh on-device real-time, tầng chậm
   consolidation trên backend — hiện thực nguyên lý multi-timescale của Nested Learning.
3. **Ứng dụng NL vào assistive tech** + **phân tích tradeoff trung thực** (khi nào associative memory
   vs prototype thắng; giới hạn sức chứa; độ bền theo độ giống).
4. **Bộ benchmark + protocol** cá nhân hoá liên tục tiếng Việt (dạy–nhớ–sửa–thu hồi–open-set).

**Trả lời câu hỏi hội đồng "sao không dùng NCM?":** *"NCM là baseline nhận diện mạnh và chúng tôi
DÙNG chính nó làm xương sống. Đóng góp không nằm ở phép phân loại, mà ở hệ thống bộ nhớ liên tục
đa tần số, novelty gating, triển khai edge–cloud và ứng dụng — những thứ NCM không giải quyết."*

---

## 6. Nơi CPM có thể VƯỢT NCM (thí nghiệm nên làm tiếp)
NCM dùng **một** trung bình cố định/nhãn ⇒ kém khi **ngoại hình thay đổi theo thời gian** (ánh sáng,
đeo kính, già đi, đội mũ). CPM đa tầng: **tầng nhanh thích nghi phiên hiện tại**, **tầng chậm giữ danh
tính gốc** ⇒ **kỳ vọng vượt NCM trong kịch bản DRIFT/đổi miền theo thời gian**. → Đề xuất thí nghiệm
"appearance drift": mỗi phiên đổi dần phân phối embedding, đo accuracy CPM vs NCM. **Đây là chỗ chứng
minh giá trị riêng của multi-timescale.**

**Đã làm (Task 2):** thêm EMA đa tầng (`proto_ema`) cạnh prototype trung-bình-thật; mặc định tắt để
không đổi hành vi cũ, bật trong `CPMEMAAdapter`/thí nghiệm drift. Kết quả synthetic drift mạnh:
NCM=0.50, CPM-EMA=1.00, kNN=1.00; footprint CPM-EMA=6,144 floats, kNN=49,152 floats.
Đã lưu `experiments/results/drift.png` và `experiments/results/drift.csv`.

> ⚠️ **KHUNG HOÁ TRUNG THỰC (phải nói rõ khi bảo vệ).** Đây là **kịch bản dựng có kiểm soát
> (existence proof)**, KHÔNG phải "CPM chính xác hơn NCM nói chung". Khoảng cách chỉ xuất hiện khi
> có **một danh tính gây nhiễu (distractor)** nằm gần vùng ngoại hình đã drift tới: khi đó trung
> bình cũ của NCM bị mẫu cũ kéo lùi và thua distractor, còn EMA bám mẫu gần đây nên đúng. Không có
> distractor gây nhiễu ⇒ mọi phương pháp đều đúng (không có khoảng cách). Câu chốt đúng để trích
> dẫn: **EMA = chính xác khi drift *và* bộ nhớ bị chặn; NCM = chặn nhưng thua drift; kNN = chính
> xác nhưng phình** — góc phần tư "chính xác + bị chặn" là chỗ EMA thắng thật.

**Ablation độ nhạy (`drift_ablation.png`)** — bằng chứng kết quả **bền trên một dải tham số**, không
phải chọn đúng một điểm may mắn (trung bình 3 seed, drift mạnh):
- **(A) theo `ema_alpha`:** thấp (bám mẫu mới) → acc≈1.00; cao (giữ mẫu cũ, tiến về NCM) → acc≈0.50.
- **(B) theo `ema_weight`:** 0 (thuần NCM) → 0.50; vùng trọng số cao **rộng và đều** đạt ≈1.00.
- **(C) theo `confusability`:** distractor càng gần ngoại hình mới, NCM càng sụp; EMA giữ ≈1.00 cho
  tới khi confusability→1 (distractor **trùng hẳn** ngoại hình mới ⇒ hai danh tính nhập một, ai cũng
  thua ≈0.50 — đúng vì bài toán vô nghiệm). Đây là ranh giới trung thực của phương pháp.
- **Hạn chế còn lại:** drift ở đây là synthetic; bản bán-thật (augment ảnh LFW mờ/sáng/xoay tăng dần
  theo "thời gian") là việc nên làm để củng cố. EMA hiện **chỉ bật trong thí nghiệm**, chưa vào demo
  deploy (bật cần calibrate lại ngưỡng vì `confidence` thành điểm trộn slow+fast).

## 7. Hạn chế & việc cần làm
- [x] **Calibrate ngưỡng quen/lạ theo embedder thật** (Task 1) — sửa lỗi FAR cao của 0.35. *(2026-07-10)*
- [x] Chạy calibration benchmark công khai: `real:face=0.66098` trên LFW 12x20,
  `real:object=0.65889` trên Caltech101 subset. *(2026-07-10)*
- [ ] Trước deploy/demo cá nhân: chạy lại `scripts.calibrate_threshold` trên `data/faces`/`data/objects`
  của bạn + impostor thật → lưu ngưỡng deploy cuối.
- [x] Làm **thí nghiệm drift** (§6, Task 2) — thêm EMA đa tầng, nơi CPM > NCM. *(2026-07-10)*
- [x] Task 3 phần code/smoke: thêm `scripts/capture_objects.py`, Gradio headless test,
  object benchmark Caltech101 subset (`CPM=0.99`, `kNN=0.98`, `Fine-tune=0.41`). *(2026-07-10)*
- [ ] Task 3 nghiệm thu thủ công: bạn mở webcam trong browser, test Ghi nhớ/Hỏi/Sửa với người thật
  và đồ cá nhân thật.
- [ ] Chạy lại retention/open-set với **embedding thật** (đã có real_lfw/real_data).
- [ ] Đo **correction efficiency** (số lần sửa để đúng) và **đa người dùng** (cô lập).
- [ ] Thêm iCaRL vào bảng nếu muốn đầy đủ (đã có NCM/EWC/kNN/fine-tune).

### Task 4 update (2026-07-10)
- [x] Code/mock xong: `skills/providers.py` co `easyocr_fn`, `gemini_vlm`, `moondream_vlm` placeholder.
- [x] `app/orchestrator.py` co `--skills-mode real`, tu fallback Stub neu thieu lib/API key.
- [x] `tests/test_providers.py` kiem tra OCR/VLM callable va routing khong can mang.
- [ ] Chay that khi co moi truong/key: cai `easyocr google-generativeai`, dat `GEMINI_API_KEY`, thu anh co chu/canh that.
