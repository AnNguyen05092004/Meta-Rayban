# Đồ án tốt nghiệp — Kính trợ thị thông minh dựa trên Nested Learning

**Tên đề tài (đề xuất):** *"Trợ lý thị giác đeo được cho người khiếm thị với bộ nhớ cá nhân hoá học liên tục lấy cảm hứng từ Nested Learning"*
*(A Wearable Vision Assistant for the Visually Impaired with a Nested-Learning-Inspired Continual Personalization Memory)*

| | |
|---|---|
| **Nhóm** | 3 thành viên |
| **Thời lượng** | ~5 tháng (bắt đầu 2026-07) |
| **Nền tảng** | Điện thoại trước → Meta Ray-Ban sau (kiến trúc **tách rời phần cứng**) |
| **Máy phát triển** | **MacBook Pro M4** (Apple Silicon — chạy on-device rất mạnh) + Google Colab |
| **Ngôn ngữ chính** | Tiếng Việt |
| **Định hướng** | Cân bằng: sản phẩm demo chạy được + đóng góp nghiên cứu để viết báo |
| **Vai trò của NL** | Lớp **bộ nhớ cá nhân hoá học liên tục** (CPM) trên nền perception có sẵn — **KHÔNG** train lại HOPE từ đầu |
| **Lộ trình vận hành** | Online-first (độ trễ 1–3s) → **offline real-time on-device** (mục tiêu cuối) |

> ✅ Tài liệu này đã cập nhật theo các điều kiện nhóm xác nhận (xem [§13](#13-điều-kiện--giả-định)) và chính thức hoá **kiến trúc edge–cloud đa tầng-thời-gian** do nhóm đề xuất ([§6](#6-kiến-trúc-triển-khai-edgecloud-đa-tầng-thời-gian)).

---

## Mục lục
1. [Phân tích đề tài & định vị](#1-phân-tích-đề-tài--định-vị)
2. [Mục tiêu đồ án](#2-mục-tiêu-đồ-án)
3. [Đóng góp khoa học (điểm để viết báo)](#3-đóng-góp-khoa-học-điểm-để-viết-báo)
4. [Thiết kế tổng thể & kiến trúc phân lớp](#4-thiết-kế-tổng-thể--kiến-trúc-phân-lớp)
5. [Thiết kế lõi: NL Continual Personalization Memory (CPM)](#5-thiết-kế-lõi-nl-continual-personalization-memory-cpm)
6. [Kiến trúc triển khai: Edge–Cloud đa tầng-thời-gian](#6-kiến-trúc-triển-khai-edgecloud-đa-tầng-thời-gian)
7. [Đa người dùng (multi-tenancy) & riêng tư](#7-đa-người-dùng-multi-tenancy--riêng-tư)
8. [Tech stack](#8-tech-stack)
9. [Tối ưu chi phí / ngân sách](#9-tối-ưu-chi-phí--ngân-sách)
10. [Kế hoạch theo phase (5 tháng)](#10-kế-hoạch-theo-phase-5-tháng)
11. [Thực nghiệm & tiêu chí đánh giá](#11-thực-nghiệm--tiêu-chí-đánh-giá)
12. [Rủi ro & phương án dự phòng](#12-rủi-ro--phương-án-dự-phòng)
13. [Điều kiện & giả định](#13-điều-kiện--giả-định)
14. [Phân công & công cụ](#14-phân-công--công-cụ)
15. [Phụ lục: bản đồ 2 project tham khảo](#phụ-lục-bản-đồ-2-project-tham-khảo--dùng-vào-đâu)

---

## 1. Phân tích đề tài & định vị

### 1.1. Vấn đề thực tế
Người khiếm thị gặp khó ở 4 nhóm nhu cầu hằng ngày:
- **Hiểu bối cảnh xung quanh** ("trước mặt tôi có gì?").
- **Đọc chữ** (biển báo, tài liệu, hạn sử dụng, mệnh giá tiền).
- **Di chuyển an toàn** (tránh vật cản, tìm cửa/ghế/bậc thang).
- **Nhận ra người quen & tìm đồ cá nhân** (ai đang đến, ví/chìa khoá của tôi đâu).

Các app hiện có (Seeing AI, Be My Eyes, Envision, Meta AI trên kính…) làm tốt phần "nhìn" **nhưng gần như *không có trí nhớ cá nhân bền vững***: mỗi lần dùng là "mới tinh". Chúng không nhớ *"đây là vợ tôi tên Lan"*, *"cái ví da nâu là của tôi"*, *"bếp ở bên trái phòng khách"*, và khi bị sửa sai thì lần sau vẫn sai. Đây đúng là căn bệnh **"anterograde amnesia"** mà bài báo NL mô tả cho LLM.

### 1.2. Vì sao Nested Learning hợp với bài toán này
NL không phải công nghệ "nhìn" — phần nhìn là của **VLM/Computer Vision**. Giá trị cốt lõi của NL (NeurIPS 2025, Google) là **học liên tục không quên (continual learning)** qua:
- **Continuum Memory System (CMS):** bộ nhớ trải phổ nhiều **tần số cập nhật** — nhanh (thích nghi tức thời, quên nhanh) → chậm (kiến thức bền vững, chống quên).
- **Self-modifying / delta-rule:** khi có thông tin mới, chỉ ghi **phần sai lệch/mới** (error `v − kM`) với **cổng quên α** và **tốc độ học η** → **học online mà không đè kiến thức cũ**, **không cần train lại trên cloud**.
- **Associative memory:** ánh xạ *key → value* (embedding khuôn mặt/vật thể → danh tính/ghi chú).

→ Ghép **perception (mắt) + NL-CPM (trí nhớ cá nhân học liên tục)** tạo ra thứ các sản phẩm hiện tại thiếu, và đúng chỗ NL toả sáng. Thêm nữa, tinh thần "đa tần số" của NL ánh xạ **rất tự nhiên** vào mô hình triển khai **edge–cloud** (§6) — vừa là điểm mạnh sản phẩm, vừa là đóng góp hệ thống cho bài báo.

### 1.3. Ranh giới phạm vi (để không "ôm" quá sức)
| Làm (in-scope) | Không làm (out-of-scope) |
|---|---|
| Dùng VLM/CV **có sẵn** cho phần nhìn | Tự pretrain VLM/HOPE-LM từ đầu (15B tokens, GPU lớn) |
| Cài **module CPM nhỏ** (vài triệu tham số) theo CMS + delta-rule | Cạnh tranh SOTA language modeling |
| Điện thoại-first; kính là bước tích hợp cuối | Phụ thuộc sống-còn vào SDK kính |
| Edge–cloud, on-device real-time về sau | Federated learning quy mô lớn (chỉ nêu như hướng mở rộng) |
| Tiếng Việt, 4 use case ở mức demo tốt | Thiết bị y tế được chứng nhận, điều hướng ngoài trời phức tạp |

---

## 2. Mục tiêu đồ án

### 2.1. Mục tiêu sản phẩm
Trợ lý thị giác đeo được (điện thoại → kính), tương tác **giọng nói tiếng Việt**, 4 năng lực:
1. **Mô tả cảnh & hỏi-đáp thị giác (VQA).**
2. **Đọc chữ / OCR** (biển báo, tài liệu, tiền, hạn sử dụng).
3. **Cảnh báo vật cản & hỗ trợ di chuyển** (real-time, on-device).
4. **Nhận diện người quen & đồ vật cá nhân** (dùng CPM — năng lực khác biệt).

### 2.2. Mục tiêu nghiên cứu
Chứng minh bằng **thực nghiệm định lượng** rằng lớp **NL-CPM**:
- **Cải thiện độ chính xác cá nhân hoá theo thời gian** (càng dùng càng đúng).
- **Chống quên (anti-catastrophic-forgetting):** học danh tính mới không làm mất danh tính cũ — vượt fine-tuning ngây thơ và RAG đơn tầng.
- **Chạy online trên thiết bị yếu**, học không cần train lại trên cloud, và **triển khai được theo mô hình edge–cloud đa tầng-thời-gian** (real-time on-device + consolidation trên backend).

### 2.3. Tiêu chí "hoàn thành tốt"
- Demo end-to-end chạy thật trên điện thoại (và nếu kịp, trên kính), **có nhánh chạy on-device offline real-time**.
- Bộ thực nghiệm + biểu đồ so sánh CPM vs baselines (ablation) + đo forgetting.
- Báo cáo đồ án sạch + **1 bản draft bài báo**.

---

## 3. Đóng góp khoa học (điểm để viết báo)

**Luận điểm:** *"Áp dụng Continuum Memory System + cập nhật self-modifying (delta-rule) của Nested Learning làm lớp bộ nhớ cá nhân hoá on-device cho thiết bị trợ thị, triển khai theo mô hình edge–cloud đa tầng-thời-gian, giúp học liên tục danh tính/đồ vật/bối cảnh của người dùng mà không quên và không cần train lại trên cloud."*

**Các đóng góp cụ thể:**
1. **NL-CPM:** bộ nhớ cá nhân hoá đa tầng-thời-gian (fast/medium/slow) + delta-rule, thao tác trên **embedding tri giác** (mặt/vật thể/địa điểm) thay vì token — *ứng dụng mới* của NL sang assistive tech.
2. **Triển khai edge–cloud đa tầng-thời-gian (§6):** ánh xạ "tần số cập nhật khác nhau" của NL thành phân tầng thiết bị–backend: tầng nhanh chạy on-device real-time, tầng trung/chậm consolidation trên backend khi sạc/có mạng, đồng bộ adapter về thiết bị. **Đây là đóng góp hệ thống**.
3. **Consolidation "offline":** hợp nhất trí nhớ nhanh→chậm (giống consolidation lúc ngủ) → tăng độ bền, giảm quên.
4. **Bộ benchmark cá nhân hoá liên tục tiếng Việt** + protocol đo **retention/forgetting** cho trợ thị.
5. **Hệ thống demo hoàn chỉnh** + hỗ trợ **đa người dùng** (mỗi người một bộ nhớ cá nhân, cô lập & riêng tư).

**Baselines — xương sống thực nghiệm:**
| Baseline | Mô tả | Điểm yếu kỳ vọng |
|---|---|---|
| B0. Stateless VLM | VLM thuần, không nhớ | Không cá nhân hoá |
| B1. Naive fine-tune | Fine-tune mỗi lần dạy mới | **Catastrophic forgetting**, tốn kém |
| B2. RAG / vector-DB (FAISS) đơn tầng | Lưu embedding + truy hồi kNN | Không đa tần số, không forget-gate, nhiễu tích luỹ |
| **B3. NL-CPM (của nhóm)** | CMS đa tầng + delta-rule + consolidation | Kỳ vọng tốt nhất về cá nhân hoá + retention |

---

## 4. Thiết kế tổng thể & kiến trúc phân lớp

### 4.1. Nguyên tắc thiết kế
- **Tách rời phần cứng cảm biến:** camera/mic/loa có thể là điện thoại **hoặc** kính → đổi được, không sửa lõi.
- **Perception thay thế được:** cloud VLM (chất lượng) ⇄ on-device (offline) qua cùng interface.
- **An toàn ưu tiên:** cảnh báo vật cản chạy **cục bộ, độ trễ thấp**, chen ngang mọi luồng.
- **CPM là lõi độc lập**, test được riêng, là nơi tập trung đóng góp NL.
- **Đa tầng-thời-gian xuyên suốt:** cùng một nguyên lý NL chi phối cả *thuật toán* (§5) lẫn *triển khai* (§6).

### 4.2. Kiến trúc phân lớp (logic)
```
┌──────────────────────────────────────────────────────────────────────┐
│  (1) SENSING / INTERFACE  —  Điện thoại (Phase 1) → Kính (Phase 2)     │
│   Camera stream · Mic (STT) · Loa/tai nghe (TTS) · Nút/wake-word       │
└───────────────┬───────────────────────────────────┬──────────────────┘
                │ ảnh/khung hình + lệnh giọng nói    │ phản hồi giọng nói
                ▼                                    ▲
┌──────────────────────────────────────────────────────────────────────┐
│  (4) ORCHESTRATOR (agent điều phối intent)                            │
│   - Nhận diện ý định (mô tả? đọc chữ? vật cản? "đây là ai?")           │
│   - Gọi Perception + truy vấn/ghi CPM → soạn câu trả lời tiếng Việt     │
│   - SAFETY OVERRIDE: cảnh báo vật cản chen ngang                       │
└───────┬───────────────────────────────────────────────┬──────────────┘
        ▼                                                 ▼
┌───────────────────────────────┐        ┌──────────────────────────────┐
│  (2) PERCEPTION (mắt)          │        │  (3) ★ NL-CPM (trí nhớ) ★     │
│   - VLM: mô tả cảnh + VQA       │  embed │   CMS: fast → medium → slow   │
│   - OCR tiếng Việt              │───────▶│   delta-rule self-modifying   │
│   - Object/Obstacle: YOLO+Depth │  keys  │   associative recall (k→v)    │
│   - Face embed (ArcFace)        │◀───────│   consolidation (offline)     │
│   - Object embed (CLIP/DINO)    │ labels │   per-user isolation          │
└───────────────────────────────┘        └──────────────────────────────┘
        │                                                 │
        └───────────────► FEEDBACK LOOP ◄─────────────────┘
             người dùng xác nhận/sửa → ghi vào CPM (học liên tục)
```
> Lưu ý: đây là kiến trúc **logic**. Cách các lớp này *nằm ở đâu* (thiết bị vs backend) được mô tả ở **§6** — và chính điểm đó tạo ra khả năng offline real-time + đóng góp hệ thống.

### 4.3. Luồng dữ liệu tiêu biểu
**A. "Ai đang ở trước mặt tôi?" (dùng CPM, ưu tiên on-device):**
```
Mic → STT("ai đang ở trước mặt") → Orchestrator
  → Perception: face detect → ArcFace embedding (query key)
  → CPM.recall(key) [tầng fast/slow on-device] → "Lan (vợ)"
  → TTS: "Chị Lan đang ở phía trước, khoảng 2 mét."
```
**B. "Nhớ nhé, đây là ví của tôi" (dạy — học liên tục):**
```
Voice + camera → Object detect + CLIP embed (key)
  → CPM.write(key, value="ví của tôi", tier=slow, delta-rule α,η) [on-device tức thì]
  → (nền) đánh dấu để đồng bộ/consolidate trên backend khi sạc
  → TTS: "Đã ghi nhớ ví của bạn."
```
**C. Cảnh báo vật cản (real-time, cục bộ, kể cả offline):**
```
Camera (liên tục) → YOLO + Depth (on-device)
  → vật cản < ngưỡng → SAFETY OVERRIDE → TTS/haptic: "Vật cản phía trước 1 mét."
```

---

## 5. Thiết kế lõi: NL Continual Personalization Memory (CPM)

> Phần **đóng góp NL** — tái sử dụng/chuyển thể cơ chế từ 2 repo tham khảo (`nested-learning/src/core/memory.py` — CMS; delta-rule fast-weight trong `HOPE-nested-learning`), nhưng thao tác trên **embedding tri giác** thay vì token ngôn ngữ.

### 5.1. Ba tầng bộ nhớ theo tần số (ánh xạ CMS)
| Tầng | Vai trò | Tần số cập nhật | Ví dụ | Đặc tính |
|---|---|---|---|---|
| **Fast (working)** | Ngữ cảnh phiên hiện tại | Mỗi bước | "vừa thấy cái cốc" | Nhỏ, quên nhanh (α thấp) |
| **Medium (episodic)** | Sự việc gần đây | Trung bình | "quán cà phê tuần này" | Quên có kiểm soát |
| **Slow (semantic/long-term)** | Danh tính bền vững | Hiếm | mặt người thân, ví, bố cục nhà | Lớn, chống quên (α cao) |

### 5.2. Cập nhật (delta-rule self-modifying)
Với mỗi cặp `(key k, value v)` (key = embedding tri giác, value = nhãn/ghi chú/embedding đích), cập nhật ma trận nhớ `M` (rút gọn từ Eq. 96 của bài báo):
```
v̂     = M·k                         # dự đoán hiện tại của bộ nhớ
error = v − v̂                        # chỉ phần sai lệch/mới
M     = α·M  +  η · error · kᵀ       # α: cổng quên, η: tốc độ học (học được hoặc heuristic)
```
- **Chỉ ghi error** → không đè kiến thức cũ đã đúng → **chống quên**.
- α, η **theo tầng** (fast: quên nhanh; slow: giữ lâu) và có thể **phụ thuộc dữ liệu**.

### 5.3. Truy hồi liên kết (associative recall)
```
Cho query key q:
  ứng viên = argmax_i sim(q, key_i) trên cả 3 tầng (ưu tiên slow > medium > fast khi mạnh)
  sim ≥ ngưỡng → trả value + độ tin cậy;  sim < ngưỡng → "chưa biết" → hỏi để học
```
> Có thể kết hợp **FAISS** để truy hồi kNN nhanh khi nhiều mục, nhưng phần "học" (delta-rule, đa tầng, consolidation) mới là đóng góp NL — không phải bản thân vector-DB (đó là baseline B2).

### 5.4. Consolidation "offline"
Mô phỏng consolidation lúc ngủ: **replay** mục fast/medium ổn định (được xác nhận nhiều, tin cậy cao) → **ghi vào slow tier**; **prune** mục nhiễu. → tăng retention, giảm phình bộ nhớ. (Bước này chính là việc "nặng" đẩy lên backend ở §6.)

### 5.5. API mô-đun (đề xuất)
```python
class ContinualPersonalizationMemory:
    def __init__(self, user_id): ...                                 # cô lập theo người dùng
    def write(self, key, value, tier="auto", confidence=1.0): ...    # dạy/ sửa
    def recall(self, query_key, top_k=1) -> list[(value, score)]: ...# hỏi "đây là gì/ai"
    def correct(self, query_key, new_value): ...                     # người dùng sửa sai
    def consolidate(self): ...                                       # chạy khi idle/sạc (backend)
    def export_delta(self) / import_adapter(self, adapter): ...      # đồng bộ edge↔cloud (§6)
    def snapshot(self) / load(self): ...                             # lưu/khôi phục state
```

---

## 6. Kiến trúc triển khai: Edge–Cloud đa tầng-thời-gian

> **Đây là ý tưởng nhóm đề xuất — và nó rất đúng tinh thần NL.** Bên dưới tôi chính thức hoá + tinh chỉnh để vừa khả thi vừa thành một đóng góp hệ thống viết được vào báo.

### 6.1. Ý tưởng gốc của nhóm (được giữ nguyên cốt lõi)
```
[Kính] → [Điện thoại]
   ├─ Inference nhanh, model nén (adapter tần số cao) → chạy ngay, real-time
   └─ Khi có mạng/đang sạc → đồng bộ dữ liệu tích luỹ lên backend
          → backend tinh chỉnh lớp tần số trung + thấp (nặng hơn)
          → gửi adapter đã cập nhật về lại điện thoại
```
**Đánh giá:** ✅ Hợp lý và hiệu quả. Nó ánh xạ 1–1 với NL: *"các thành phần cập nhật ở tần số khác nhau"* → *"các tầng nằm ở nơi khác nhau, cập nhật theo nhịp khác nhau"*.

### 6.2. Bản tinh chỉnh (3 tầng vật lý)
```
┌──────────────┐   BLE    ┌─────────────────────────────┐   Wi-Fi/4G   ┌────────────────────┐
│    KÍNH      │ ───────▶ │        ĐIỆN THOẠI (EDGE)     │ ───────────▶ │   BACKEND (CLOUD)  │
│ camera/mic/  │          │  TẦNG NHANH (real-time)      │  khi sạc/    │  TẦNG TRUNG + CHẬM │
│ loa          │ ◀─────── │  - perception nén on-device  │  có mạng     │  - consolidation   │
└──────────────┘  audio   │    (YOLO-nano, Depth, OCR,    │             │  - tinh chỉnh      │
                          │     face/obj embed, VLM nhỏ) │ ◀────────── │    adapter (nhẹ)   │
                          │  - CPM fast-weight (delta)   │  adapter     │  - kho memory/user │
                          │  → phản hồi tức thì, offline │  cập nhật    │  - hàng đợi job    │
                          └─────────────────────────────┘             └────────────────────┘
```

| Tầng NL | Nằm ở đâu | Nhịp cập nhật | Việc | Yêu cầu |
|---|---|---|---|---|
| **Fast** | Điện thoại (& kính) | Mỗi tương tác | Inference + fast-weight session memory, delta-rule | **Real-time, offline được** |
| **Medium** | Điện thoại (nền) | Vài phút–giờ | Episodic memory, gom dữ liệu chờ đồng bộ | Nhẹ |
| **Slow** | Backend | Khi sạc/có mạng | Consolidation + tinh chỉnh adapter, đẩy về | Nặng, không cần real-time |

### 6.3. Ba tinh chỉnh quan trọng
1. **"Train lại" → "consolidation + tinh chỉnh adapter nhẹ", KHÔNG full-retrain.** Backend không train lại cả model mỗi user (quá tốn). Thay vào đó: **replay + hợp nhất memory** (§5.4) và tuỳ chọn **fine-tune một adapter nhỏ kiểu LoRA** trên các *correction* tích luỹ. Đúng ý "tinh chỉnh lớp tần số trung + thấp".
2. **Riêng tư trước hết:** dữ liệu thô (ảnh khuôn mặt) **ưu tiên ở lại thiết bị**. Chỉ đồng bộ **embedding/adapter/delta** (nhỏ hơn + riêng tư hơn) lên backend; nếu bắt buộc gửi dữ liệu thì **mã hoá + xin đồng thuận** (xem §7).
3. **Offline real-time đạt được nhờ:** tầng fast + perception **nén on-device** (YOLO-nano cho vật cản, VLM nhỏ như Moondream/PaliGemma, OCR on-device) chạy hẳn trên máy (Neural Engine của M4 / NPU điện thoại). **Cloud chỉ dùng cho VQA nặng khi có mạng.** → mất mạng vẫn cảnh báo vật cản + nhận diện người/đồ đã học.

> 🔭 **Lưu ý mở rộng (xử lý sau):** việc "nén model xuống điện thoại" cho tầng fast **không phải cách duy nhất** để đạt real-time. Có phương án hay/nhẹ công hơn cho giai đoạn demo (**edge server = M4 qua Wi‑Fi**, **CPM recall local**, **streaming TTS**, **cache khung hình**); nén model chỉ thực sự cần khi muốn *đồng thời offline + mobility ngoài trời*. Toàn bộ phân tích + khuyến nghị đã ghi tại **[MO_RONG_VA_BACKLOG.md §1](MO_RONG_VA_BACKLOG.md)**.

### 6.4. Cơ chế đồng bộ (sync protocol)
```
Điện thoại (khi sạc & có Wi-Fi):
  1. push:  gửi delta memory + correction log (đã mã hoá) lên backend
  2. backend: consolidate + (tuỳ chọn) tinh chỉnh adapter cho user đó
  3. pull:  tải adapter/slow-memory đã cập nhật về
  4. merge: hợp nhất vào CPM cục bộ (không ghi đè fast-weight đang dùng)
Xung đột: dùng versioning + last-writer-wins theo timestamp; ưu tiên correction mới nhất của user.
```

### 6.5. Chế độ suy giảm mượt (graceful degradation)
| Tình huống | Hành vi |
|---|---|
| Có mạng | VQA/OCR chất lượng cao qua cloud; sync khi sạc |
| Mất mạng | Chạy hoàn toàn on-device: vật cản + nhận diện đã học + OCR nhẹ + mô tả bằng VLM nhỏ |
| Chưa sync lâu | Vẫn hoạt động; memory mới nằm ở fast/medium tới khi sạc thì consolidate |

---

## 7. Đa người dùng (multi-tenancy) & riêng tư

> Nhóm yêu cầu tính đến "nhiều người dùng". Đây vừa là yêu cầu hệ thống, vừa là điểm cộng nghiên cứu (mỗi người là một bài toán continual learning cô lập).

### 7.1. Cô lập theo người dùng
- **Mỗi user = một instance CPM riêng** (fast/medium/slow + faces/objects của riêng họ), khoá theo `user_id`. **Không trộn** bộ nhớ giữa các user (vừa đúng vừa an toàn).
- **Backend lưu state theo user:** một snapshot memory (`.pt`) + metadata trong DB, phân vùng theo `user_id`. Consolidation chạy **theo từng user** (hàng đợi job).
- **Compute backend stateless:** node xử lý không giữ state; state nằm ở kho memory → dễ scale ngang (thêm worker khi nhiều user).

### 7.2. Kiến trúc backend cho nhiều user
```
[N điện thoại] → API Gateway (auth, user_id) → Worker pool (stateless, FastAPI)
                                                  │
                                       ┌──────────┴───────────┐
                                       ▼                      ▼
                             Memory Store (per-user)   Job Queue (consolidation)
                             (object storage +.pt,     (chạy nền khi user sạc)
                              DB metadata theo user_id)
```

### 7.3. Riêng tư & đạo đức (bắt buộc nêu trong báo cáo)
- **Sinh trắc học (khuôn mặt) ưu tiên xử lý & lưu on-device**, mã hoá tại chỗ.
- **Đồng thuận** khi thu ảnh người khác; cơ chế **"xoá dữ liệu của tôi"**.
- Backend chỉ giữ embedding/adapter (không ảnh thô) khi có thể; mã hoá khi truyền & khi lưu; khoá theo user.
- **Hướng mở rộng (future work, không bắt buộc):** *federated learning* — cải thiện model chung từ nhiều user mà không chia sẻ dữ liệu thô. Nêu như tương lai để tăng chiều sâu bài báo, **không làm trong 5 tháng**.

---

## 8. Tech stack

> Nguyên tắc: **miễn phí/on-device trước (tận dụng M4) → cloud chỉ khi cần chất lượng** → dần đưa mọi thứ về on-device cho offline.

### 8.1. Tận dụng MacBook Pro M4 (rất quan trọng cho chi phí)
- 2 repo tham khảo **hỗ trợ sẵn Mac M1–M4** (MPS) → nhóm **chạy/huấn luyện CMS + delta-rule ngay trên M4, miễn phí**.
- M4 có **Neural Engine + unified memory** → chạy được perception on-device (YOLO, Depth, CLIP, ArcFace, whisper.cpp, Piper) và cả VLM nhỏ → dựng nhánh **offline real-time** ngay trên máy dev.
- Backend dev chạy **local trên M4** (FastAPI) → phơi ra cho điện thoại qua `ngrok`/`cloudflared` → **0đ hạ tầng** giai đoạn đầu.

### 8.2. Ứng dụng đeo (client)
| Thành phần | Đề xuất | Thay thế |
|---|---|---|
| App | **Flutter** (1 codebase, camera/audio tốt) | React Native · Android native (Kotlin) |
| STT tiếng Việt | **PhoWhisper / whisper.cpp** (on-device) | Google STT · Groq/Gemini free tier |
| TTS tiếng Việt | **Piper** (on-device, free) | Google/Azure TTS · gTTS |
| Kết nối backend | **WebSocket** (stream ảnh/âm) | gRPC · REST |

### 8.3. Perception (mắt)
| Năng lực | Đề xuất (chất lượng) | On-device/offline (M4/điện thoại) |
|---|---|---|
| Mô tả cảnh + VQA | **Gemini Flash** (free tier tốt, đa ngữ) / GPT-4o-mini | **Moondream · PaliGemma · Florence-2 · LLaVA-tiny** |
| OCR tiếng Việt | VLM cloud | **VietOCR / PaddleOCR** |
| Object/Obstacle | **YOLOv11** | YOLO-nano (Core ML/TFLite) |
| Khoảng cách | **Depth Anything v2** | Depth Anything small |
| Face embedding | **InsightFace/ArcFace** | MediaPipe + embedding nhẹ |
| Object embedding | **CLIP** / DINOv2 | MobileCLIP |

### 8.4. Lõi & backend
| Thành phần | Lựa chọn |
|---|---|
| Lõi CPM | **PyTorch** (tái dùng code CMS/delta-rule từ 2 repo) |
| Truy hồi nhanh | FAISS (khi số mục lớn) |
| Backend | **Python + FastAPI + WebSocket**; local M4 → về sau VM nhỏ/HF Spaces |
| Orchestrator | State machine đơn giản **hoặc** LLM function-calling |
| Lưu state | SQLite + snapshot `.pt` per-user |
| Đóng gói on-device | **Core ML** (Apple) / ONNX / TFLite / ExecuTorch |

### 8.5. Kính (Phase 4, tuỳ chọn)
- **Meta Ray-Ban** qua **Wearables Device Access Toolkit** — *kiểm chứng quyền truy cập camera/audio real-time trước khi mua* (§12 R1).
- Kính = nguồn camera+mic+loa; điện thoại = hub tính toán (BLE).

---

## 9. Tối ưu chi phí / ngân sách

> Nhóm chưa chốt ngân sách → đây là phương án **tối ưu (gần như 0đ giai đoạn dev)** nhờ M4 + free tier.

### 9.1. Chiến lược "free-first"
| Khoản | Cách làm | Chi phí |
|---|---|---|
| Perception nặng (VLM/OCR) | **Gemini free tier**; vượt hạn thì chạy VLM nhỏ trên M4 | **~0đ** (vài $ nếu tràn) |
| STT/TTS | whisper.cpp + Piper **on-device** trên M4 | **0đ** |
| YOLO/Depth/Face/CLIP/CPM | Mã nguồn mở, chạy M4/Colab | **0đ** |
| Huấn luyện/thí nghiệm | Colab **free**; Colab Pro chỉ khi cần chạy dài | **0–10$/tháng** |
| Backend | Local M4 + ngrok (dev) → VM nhỏ/HF Spaces free (demo) | **0đ → ~5$/tháng** |
| Kính Meta Ray-Ban | **Chỉ mua ở Phase 4 nếu SDK OK** | ~300–380$ (một lần, tuỳ chọn) |

### 9.2. Khuyến nghị
1. **Làm tất cả free-first trên M4** trong Phase 0–3. M4 đủ mạnh để chạy cả nhánh on-device.
2. **Chỉ chi tiền khi cần:** (a) Colab Pro nếu một lần train cần GPU lâu; (b) ít $ cho Gemini nếu vượt free tier; (c) kính ở cuối nếu SDK cho phép.
3. **Ước tính tổng thực tế:** dưới ~**50$ API** trong 5 tháng + (tuỳ chọn) Colab Pro + (tuỳ chọn) kính. Rất khả thi với đồ án SV.

---

## 10. Kế hoạch theo phase (5 tháng)

> Chiến lược: **"vertical slice" sớm** — dựng 1 luồng end-to-end mỏng ở Tháng 1 để de-risk toàn bộ pipeline, rồi mới đắp dày. Nhánh **on-device** làm song song từ sớm để đảm bảo mục tiêu offline real-time.

### 📌 Tháng 0 (Tuần 1–2): Nền tảng & nghiên cứu
- Đọc kỹ NL.pdf (§7 CMS, §8 HOPE); **chạy lại** CMS + delta-rule từ 2 repo **trên M4** (MPS) trên dữ liệu toy.
- Khảo sát tài liệu: assistive tech, continual learning, forgetting metrics.
- **Kiểm chứng SDK Meta Ray-Ban** (Wearables Toolkit: có cho video real-time không, waitlist, khu vực).
- Chốt scope, dựng **mono-repo** + CI, viết **đề cương** + **protocol đánh giá + định nghĩa metric**.
- ✅ *Deliverable:* Đề cương + demo CMS chạy lại trên M4 + spec kiến trúc.

### 📌 Tháng 1: Vertical slice + khung app
- App Flutter stream camera + thu giọng nói → backend (local M4) → **1 tính năng** "mô tả trước mặt" bằng VLM → **TTS tiếng Việt** → phản hồi.
- Chốt giao thức WebSocket; đo độ trễ round-trip (online).
- ✅ *Deliverable:* Demo "hỏi bằng giọng nói → nghe mô tả cảnh tiếng Việt" chạy thật.

### 📌 Tháng 2: CPM v1 — lõi nghiên cứu ★
- Cài **ContinualPersonalizationMemory**: 3 tầng + delta-rule + associative recall + **cô lập theo user**.
- Tích hợp **nhận diện người quen & đồ cá nhân**: face/object embed → CPM (dạy/hỏi/sửa).
- Unit test lõi CPM (dạy A, dạy B, kiểm tra **không quên** A).
- ✅ *Deliverable:* Demo "dạy tên người/đồ → hỏi lại đúng"; CPM có test.

### 📌 Tháng 3: Hoàn thiện tính năng + an toàn + nhánh on-device
- **OCR tiếng Việt**; **cảnh báo vật cản** (YOLO+Depth) + **SAFETY OVERRIDE** + haptic.
- Orchestrator hợp nhất perception + CPM; UX giọng nói mượt (wake-word, ngắt lời).
- **Bắt đầu nhánh on-device** (M4/điện thoại): YOLO-nano + CPM fast-weight chạy offline real-time.
- ✅ *Deliverable:* App đủ 4 tính năng; obstacle chạy offline.

### 📌 Tháng 4: Edge–cloud sync + multi-user + thực nghiệm + (tuỳ chọn) kính
- Hoàn thiện **sync protocol §6.4** + **consolidation trên backend** + **multi-tenancy §7**.
- Chạy **benchmark cá nhân hoá liên tục** + đo **forgetting** vs B0–B2 (ablation, biểu đồ).
- Nếu ổn định → **mua & tích hợp Meta Ray-Ban** (hoặc giữ điện thoại nếu SDK không cho).
- **Tự test kỹ** các kịch bản (bước tiền đề trước khi mời người khiếm thị).
- ✅ *Deliverable:* Bảng kết quả + biểu đồ + demo edge–cloud + (tuỳ chọn) kính.

### 📌 Tháng 5: User study + hoàn thiện + viết
- Liên hệ **hội/trường người khiếm thị** → phỏng vấn + test thực tế (SUS + định tính).
- Polish, tối ưu độ trễ; viết **báo cáo đồ án** + **draft bài báo**; video demo + slide bảo vệ.
- ✅ *Deliverable:* Đồ án hoàn chỉnh + draft paper + demo bảo vệ.

### Gantt rút gọn
```
Tháng        0    1    2    3    4    5
Nền tảng     ██
VerticalS.        ██
CPM lõi ★          ██   ██  ····(cải tiến dần)
Tính năng               ██   ██
An toàn                      ██
On-device                    ██   ██
Edge-cloud sync                   ██
Multi-user                        ██
Thực nghiệm                       ██   ▓
User study                             ██
Viết báo      ·    ·    ·    ▓    ▓    ██
```

---

## 11. Thực nghiệm & tiêu chí đánh giá

### 11.1. Metric nghiên cứu (quan trọng nhất để viết báo)
| Metric | Ý nghĩa | Cách đo |
|---|---|---|
| **Personalization accuracy@session** | Càng dùng càng đúng | % nhận diện đúng người/đồ theo số phiên |
| **Retention / Forgetting rate** | Học mới có làm quên cũ? | Độ chính xác trên danh tính cũ sau khi học N cái mới (so B1/B2) |
| **Correction efficiency** | Sửa 1 lần có nhớ không? | % đúng ở lần gặp kế sau khi được sửa |
| **Online update cost** | Học có rẻ không? | thời gian/RAM cho 1 lần `write` (so fine-tune) |
| **Sync/consolidation cost** | Chi phí edge–cloud | thời gian consolidate/user, kích thước adapter đồng bộ |

### 11.2. Metric sản phẩm
| Metric | Mục tiêu tham khảo |
|---|---|
| Độ trễ hỏi–đáp (VQA/OCR) online | < 2–3 s |
| **Độ trễ on-device (vật cản)** | **< 200–300 ms** (real-time) |
| Obstacle detection recall | cao, ưu tiên không bỏ sót |
| Task success rate (kịch bản thực) | đo trên bộ kịch bản chuẩn |
| **SUS** (usability) | ≥ 68 |
| Hoạt động khi mất mạng | các tính năng on-device vẫn chạy |

### 11.3. Bộ dữ liệu / kịch bản
- **Tự xây bộ kịch bản cá nhân hoá tiếng Việt**: dạy–nhớ–sửa–thu hồi (người thân, đồ cá nhân, địa điểm trong nhà); **nhiều user** để test cô lập.
- Perception dùng dữ liệu công khai để sanity-check (OCR tiếng Việt, tập vật cản/indoor).
- **Đạo đức:** đồng thuận khi thu ảnh khuôn mặt; ẩn danh; nêu rõ trong báo cáo.

---

## 12. Rủi ro & phương án dự phòng

| # | Rủi ro | Mức | Dự phòng |
|---|---|---|---|
| **R1** | **SDK Meta Ray-Ban không cho truy cập camera/audio real-time** | 🔴 Cao | Kiến trúc **tách rời** → điện thoại-first vẫn là sản phẩm hoàn chỉnh; kính là "nice-to-have". Dự phòng: camera glasses khác/DIY (ESP32-CAM), webcam gắn mũ. |
| R2 | Độ trễ cloud cao khi đi lại | 🟠 TB | Obstacle + nhận diện đã học chạy **on-device**; chỉ VQA nặng gọi cloud. |
| R3 | CPM khó vượt baseline RAG | 🟠 TB | Benchmark **nhấn vào forgetting & correction** — điểm yếu cố hữu của RAG/fine-tune. |
| R4 | Đồng bộ edge–cloud phức tạp/không kịp | 🟠 TB | Làm **on-device-only trước** (đã đủ demo + đo được); sync là phần nâng cao, cắt được nếu thiếu thời gian. |
| R5 | Riêng tư dữ liệu khuôn mặt / nhiều user | 🟠 TB | On-device + mã hoá + đồng thuận + xoá được + cô lập theo user (§7). |
| R6 | Ôm quá nhiều tính năng | 🟠 TB | **Vertical slice sớm**; ưu tiên use case #4 (personalization) làm điểm nhấn nghiên cứu. |
| R7 | Thiếu người khiếm thị test sớm | 🟡 Thấp | **Tự test trước** (đã chốt), sau mới mời hội/trường + chuyên gia đánh giá định tính. |
| R8 | Tiếng Việt STT/TTS/OCR kém | 🟡 Thấp | PhoWhisper/VietOCR/Piper-VN; fallback cloud. |

---

## 13. Điều kiện & giả định

**Đã xác nhận với nhóm (2026-07-09):**
- ✅ **Kỹ năng:** PyTorch cơ bản + 1 người làm mobile; kiến thức khác tự học thêm khi cần.
- ✅ **Tính toán:** Google Colab (mua Pro nếu cần) + **MacBook Pro M4** (ưu tiên chạy on-device khi đủ nhẹ).
- ✅ **Online-first** giai đoạn đầu (độ trễ 1–3s), **mục tiêu cuối là offline real-time** (đã đưa vào §6).
- ✅ **Kiến trúc edge–cloud đa tầng-thời-gian** do nhóm đề xuất → được chính thức hoá ở §6.
- ✅ **Nhiều người dùng** → thiết kế multi-tenancy ở §7.
- ✅ **User test:** tự test thành công trước → sau mới liên hệ hội/trường người khiếm thị.
- ✅ **Ngân sách:** chưa chốt → dùng phương án free-first tối ưu ở §9.

**Còn cần xác nhận (nhỏ, không chặn tiến độ):**
- [ ] Deadline giữa kỳ cụ thể & mốc cần bản chạy đầu tiên?
- [ ] Kỳ vọng "novelty" của giảng viên (ứng dụng NL vào assistive tech + đóng góp hệ thống edge–cloud có đủ không)?
- [ ] Nền tảng điện thoại test chính: Android hay iOS? (ảnh hưởng lựa chọn on-device: Core ML nếu iOS, TFLite/NNAPI nếu Android).

---

## 14. Phân công & công cụ

### 14.1. Gợi ý chia vai (3 người, vẫn hỗ trợ chồng lấn)
| Người | Trọng tâm | Phụ |
|---|---|---|
| **TV-A** | Mobile app + UX giọng nói + on-device (Core ML/TFLite) + (Phase 4) kính | Orchestrator |
| **TV-B** | Perception pipeline (VLM/OCR/YOLO/Depth/face-object embed) + backend + sync | Multi-tenancy |
| **TV-C** | **Lõi NL-CPM + consolidation + thực nghiệm/viết báo** (điểm nhấn nghiên cứu) | Data/kịch bản |

### 14.2. Công cụ
- **Mã nguồn:** GitHub mono-repo: `app/`, `backend/`, `cpm/`, `perception/`, `experiments/`, `docs/`.
- **Quản lý việc:** GitHub Projects theo phase.
- **Thí nghiệm:** Weights & Biases / MLflow (log metric, biểu đồ cho báo cáo).
- **AI hỗ trợ:** dùng cho code/scaffold/viết, nhưng **tự kiểm chứng** lõi CPM & số liệu.

---

## Phụ lục: bản đồ 2 project tham khảo → dùng vào đâu
| Repo tham khảo | Lấy gì | Dùng cho |
|---|---|---|
| `nested-learning/src/core/memory.py` (CMS) | Cấu trúc tầng đa tần số, aggregation | Khung 3 tầng của **CPM §5.1** |
| `nested-learning/src/models/titans.py` + `HOPE-nested-learning` (delta-rule fast-weight) | Công thức `M = M(αI − ηkkᵀ) − η∇L` | Cơ chế **delta-rule §5.2** |
| Cả hai (là **LM**, hỗ trợ Mac M1–M4) | Cơ chế + xác nhận chạy được trên **M4** | Chuyển thể: **thay token → embedding tri giác**; chạy dev on-device |

> **Khác biệt cốt lõi cần nhấn trong báo cáo:** 2 repo áp dụng NL cho *ngôn ngữ*; nhóm áp dụng NL cho *bộ nhớ cá nhân hoá đa phương thức, triển khai edge–cloud, đa người dùng, trên thiết bị trợ thị* — đó là tính mới.
