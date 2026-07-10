# Plan chi tiết — Core Demo #1: Trợ lý nhận diện & ghi nhớ cá nhân hoá (CPM)

> Bản đặc tả cho **phiên bản demo đầu tiên**, tập trung vào lõi nghiên cứu: **NL Continual Personalization Memory (CPM)** — *"dạy → nhớ → không quên"*.
> Chạy trọn vẹn trên **MacBook Pro M4**, không cần điện thoại/kính/cloud. Bổ trợ cho [KE_HOACH_DO_AN.md](KE_HOACH_DO_AN.md).
> Cập nhật: 2026-07-09.

---

## 1. Mục tiêu

### 1.1. Mục tiêu chính
Chứng minh **on M4, chạy được thật**: hệ thống có thể **học online** danh tính người/đồ vật (one-shot), **nhận lại đúng**, **sửa sai chỉ 1 lần là nhớ**, và **KHÔNG quên** các danh tính cũ khi học thêm nhiều danh tính mới — bằng cơ chế **delta-rule + bộ nhớ đa tầng** của Nested Learning (không phải chỉ kNN).

### 1.2. Mục tiêu cụ thể (Definition of Done)
| ID | Mục tiêu | Đo bằng |
|---|---|---|
| G1 | Vòng lặp **dạy → nhận lại** chạy được | Dạy người/đồ → hỏi lại ra đúng nhãn + độ tin cậy |
| G2 | **Sửa online** | Sửa 1 lần → lần gặp sau nhận đúng |
| G3 | **Chống quên (định lượng)** | Retention@N: dạy thêm N nhãn, độ chính xác trên nhãn đầu **gần như giữ nguyên**; baseline fine-tune thì tụt |
| G4 | **Trung thành với NL** | Có delta-rule associative memory + 3 tầng fast/medium/slow (không phải kNN thuần) |
| G5 | **Demo nhìn thấy được** | UI Gradio bấm nút Dạy/Hỏi/Sửa qua webcam + 1 biểu đồ retention cho báo cáo |

### 1.3. KHÔNG làm trong demo này (out-of-scope)
App điện thoại · kính · obstacle/navigation · OCR · VQA mô tả cảnh · edge–cloud sync · multi-user runtime · tối ưu real-time · nén model on-device. *(Giọng nói STT/TTS là tuỳ chọn "stretch".)*

---

## 2. Input / Output

### 2.1. Input
| Loại | Chi tiết |
|---|---|
| **Ảnh** | Khung hình webcam **hoặc** file ảnh (jpg/png) |
| **Lệnh** | 1 trong 3 intent: `TEACH(label)` · `RECALL` · `CORRECT(label)` (gõ text; giọng nói là stretch) |
| **Nhãn (label)** | Chuỗi tiếng Việt: tên người ("Lan"), tên đồ ("ví của tôi") |
| **(Thí nghiệm)** | Bộ ảnh nhỏ N danh tính (ảnh thành viên nhóm + đồ vật, hoặc subset LFW) chia teach/test |

### 2.2. Output
| Loại | Chi tiết |
|---|---|
| **RECALL** | `{label, confidence, tier_matched}` hoặc `"chưa biết"` nếu dưới ngưỡng τ |
| **TEACH** | Xác nhận: "Đã ghi nhớ: Lan" |
| **CORRECT** | Xác nhận: "Đã sửa: đây là Lan" |
| **Artifacts** | Snapshot bộ nhớ `.pt` · log · CSV/biểu đồ metric · số mục trong từng tầng |

---

## 3. Kiến trúc demo (tối giản)

```
 Webcam/Ảnh ─▶ Perception (detect + embed) ─▶ key k (embedding chuẩn hoá)
                                                     │
 Lệnh (TEACH/RECALL/CORRECT + label) ─▶ Orchestrator │
                                                     ▼
                                   ┌─────────────────────────────────┐
                                   │      CPM (★ lõi nghiên cứu)       │
                                   │  M_fast / M_medium / M_slow       │
                                   │  delta-rule write · recall ·      │
                                   │  correct · consolidate            │
                                   └─────────────────────────────────┘
                                                     │
                                     {label, conf, tier} ─▶ Text / Gradio (+TTS stretch)
```

### 3.1. Cấu trúc thư mục (dựng ở `Meta-Rayban/`)
```
Meta-Rayban/
├─ docs/                 # các plan (đã có)
├─ perception/
│   ├─ embed.py          # embed_face(), embed_object()
│   └─ capture.py        # webcam / image loader
├─ cpm/
│   ├─ memory.py         # ContinualPersonalizationMemory, TierMemory, LabelRegistry
│   └─ config.py         # d, α, η, ngưỡng τ, cadence từng tầng
├─ app/
│   ├─ demo_gradio.py    # UI demo
│   └─ cli.py            # chạy dòng lệnh
├─ experiments/
│   ├─ retention.py      # kịch bản dạy tuần tự + đo retention
│   └─ baselines.py      # B1 fine-tune head (để so sánh forgetting)
├─ tests/
│   └─ test_cpm.py
├─ data/                 # ảnh mẫu (ảnh lớn để .gitignore)
├─ requirements.txt
└─ README.md
```

---

## 4. Đặc tả lõi CPM (phần quan trọng nhất)

### 4.1. Ký hiệu
- `k` = embedding tri giác đã chuẩn hoá L2 (face 512-d ArcFace, hoặc object CLIP ~512-d). Dùng làm **key**.
- `ℓ` = nhãn (label). Mỗi nhãn có **anchor** `a_ℓ` — một vector đơn vị cố định (sinh ngẫu nhiên tất định khi nhãn xuất hiện lần đầu), lưu trong `LabelRegistry`. Đây là **value** đích.
- `M` = ma trận associative memory của một tầng, `M ∈ R^{d×d}`, khởi tạo 0.

### 4.2. Cơ chế chính (NL-faithful) — associative memory + delta-rule
**Ghi (write) 1 cặp `(k, ℓ)` vào một tầng:**
```
a      = registry.get_or_create(ℓ)      # anchor của nhãn
error  = a − M·k                        # chỉ phần sai lệch (delta rule)
M      = α·M + η · error · kᵀ           # α: cổng quên, η: tốc độ học (theo tầng)
```
**Nhận (recall) cho query `q`:**
```
v̂     = M·q
ℓ*     = argmax_ℓ  cos(v̂, a_ℓ)
conf   = cos(v̂, a_{ℓ*})
→ nếu conf ≥ τ: trả (ℓ*, conf);  ngược lại: "chưa biết"
```
**Sửa (correct) `(q, ℓ_đúng)`:** ghi mạnh (η lớn) vào **fast + slow**; đánh dấu ưu tiên nhãn đúng.

### 4.3. Ba tầng (Continuum Memory System)
| Tầng | α (giữ) | η (học) | Nhịp cập nhật | Vai trò |
|---|---|---|---|---|
| **fast** | thấp | cao | mỗi lần | thích nghi tức thời, phiên hiện tại |
| **medium** | TB | TB | vài lần | sự việc gần đây |
| **slow** | cao | thấp | hiếm (qua consolidate) | danh tính bền vững, **chống quên** |

**Recall tổng hợp:** lấy `v̂` từ cả 3 tầng, kết hợp theo độ tin cậy với ưu tiên `slow > medium > fast` (hoặc cộng `v̂` có trọng số). **Consolidate:** cặp nào `hit_count ≥ c` và `conf ≥ τ_c` thì ghi lại vào `M_slow` (η cao) → bền hoá.

### 4.4. Phương án dự phòng (nếu associative memory bị nhiễu/capacity)
Nếu ma trận `M` bị **giao thoa (interference)** khi nhiều nhãn → chuyển sang **prototype-delta** (bounded, vẫn NL-flavored): mỗi nhãn giữ 1 vector prototype `p_ℓ` cập nhật bằng delta-rule `p_ℓ ← α p_ℓ + η(k − p_ℓ)`; recall = `argmax_ℓ cos(q, p_ℓ)`; vẫn 3 tầng. → **Khuyến nghị:** làm associative-memory trước (trung thành/mới hơn), giữ prototype-delta làm lưới an toàn.

### 4.5. API
```python
class ContinualPersonalizationMemory:
    def __init__(self, dim, user_id="default"): ...
    def write(self, key, label, tier="fast", confidence=1.0): ...
    def recall(self, query_key, top_k=1) -> list[dict]:   # {label, conf, tier}
    def correct(self, query_key, new_label): ...
    def consolidate(self): ...
    def snapshot(self, path) / load(self, path): ...
    def export_delta(self): ...        # stub cho sync edge–cloud (làm sau)
    def stats(self) -> dict: ...        # số nhãn, số mục/tầng, footprint
```

---

## 5. Các bước triển khai (build order) + Task list

> Ước tính cho nhóm 3 người làm song song. Ký hiệu owner: **A** = mobile/UX (ở đây làm UI+I/O), **B** = perception/backend, **C** = lõi CPM + thí nghiệm.

### WP0 — Setup & chạy lại cơ chế *(B0)*
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T0.1 | Tạo venv/uv trên M4; cài `torch`(MPS), `insightface`/`onnxruntime`, `open_clip_torch`, `gradio`, `numpy`, `matplotlib`, `pytest` | B | 0.5d | — | `import torch; torch.backends.mps.is_available()` = True |
| T0.2 | Chạy lại delta-rule/CMS từ `nested-learning`/`HOPE-nested-learning` trên M4 | C | 0.5d | T0.1 | 1 script toy delta-rule chạy trên MPS |
| T0.3 | Dựng khung thư mục (mục §3.1) + `requirements.txt` + `pytest` khung | A | 0.5d | — | `pytest` chạy (0 test) |

### WP1 — Perception embedding
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T1.1 | `embed_face(image)->512d` (InsightFace buffalo_l), chuẩn hoá L2 | B | 1d | T0.1 | Cùng người cos cao, khác người cos thấp |
| T1.2 | `embed_object(image)->vec` (open_clip ViT-B/32) | B | 0.5d | T0.1 | Phân biệt được vài đồ vật mẫu |
| T1.3 | `capture.py`: webcam frame + load ảnh file | A | 0.5d | T0.1 | Lấy được frame trên M4 |
| T1.4 | Test sanity embedding | B | 0.5d | T1.1–1.3 | Ngưỡng cos phân tách người/đồ hợp lý |

### WP2 — CPM core ★
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T2.1 | `LabelRegistry` (label↔anchor) + `TierMemory` (M, α, η, counter) | C | 1d | T0.2 | Tạo/đọc anchor tất định |
| T2.2 | `write` + `recall` (1 tầng) theo delta-rule §4.2 | C | 1d | T2.1 | Dạy A → recall A đúng |
| T2.3 | Mở rộng **3 tầng** + tổng hợp recall + ngưỡng "chưa biết" | C | 1d | T2.2 | Recall ưu tiên slow; dưới τ trả "chưa biết" |
| T2.4 | `correct` + `consolidate` (promote fast→slow) | C | 1d | T2.3 | Sửa 1 lần → nhớ; consolidate chạy |
| T2.5 | `snapshot/load` + `stats` + `export_delta` (stub) | C | 0.5d | T2.3 | Lưu/khôi phục ra kết quả như cũ |
| T2.6 | **Tests** `test_cpm.py`: (a) dạy A→recall A; (b) dạy A..J→vẫn recall A (**retention**); (c) correct→fixed; (d) unknown<τ | C | 1d | T2.4 | Tất cả test pass; retention@10 cao |

### WP3 — Orchestrator + I/O
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T3.1 | Parse intent (TEACH/RECALL/CORRECT + label) | A | 0.5d | — | Nhận đúng 3 intent |
| T3.2 | Ghép perception→CPM→câu trả lời tiếng Việt (`cli.py`) | A | 0.5d | T1.*, T2.* | Dạy/hỏi/sửa qua CLI chạy |

### WP4 — Gradio demo UI
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T4.1 | UI: webcam + ô nhãn + 3 nút (Ghi nhớ / Hỏi / Sửa) | A | 1d | T3.2 | Bấm nút chạy đúng luồng |
| T4.2 | Hiển thị label+conf+tier + ảnh crop + `stats()` bộ nhớ | A | 0.5d | T4.1 | Demo mượt trên M4 |

### WP5 — Thí nghiệm & metric (retention — "money shot")
| ID | Task | Owner | Ước tính | Phụ thuộc | Nghiệm thu |
|---|---|---|---|---|---|
| T5.1 | Bộ dữ liệu nhỏ (ảnh nhóm + đồ vật, hoặc subset LFW) chia teach/test | C | 0.5d | T1.1 | ≥10 nhãn, mỗi nhãn vài ảnh |
| T5.2 | `retention.py`: dạy tuần tự N nhãn, đo accuracy nhãn cũ sau mỗi bước | C | 1d | T2.6, T5.1 | Ra bảng accuracy@bước |
| T5.3 | `baselines.py`: **B1 fine-tune head** tuần tự (để lộ forgetting) | C | 1d | T5.1 | B1 chạy, thấy tụt accuracy |
| T5.4 | Vẽ **retention curve** CPM vs B1 + lưu `docs/` | C | 0.5d | T5.2–5.3 | Biểu đồ rõ CPM giữ, B1 quên |

### WP6 — Giọng nói *(stretch, làm nếu còn thời gian)*
| ID | Task | Owner | Ước tính | Nghiệm thu |
|---|---|---|---|---|
| T6.1 | STT tiếng Việt (whisper.cpp/PhoWhisper) cho lệnh | A | 1d | Nói lệnh → ra intent |
| T6.2 | TTS tiếng Việt (Piper) đọc kết quả | A | 0.5d | Nghe câu trả lời |

**Tổng thời gian:** WP0–WP5 ≈ **1–1.5 tuần** (song song 3 người). Giọng nói cộng thêm ~1.5 ngày.

### ✅ Tiến độ thực tế (2026-07-09)
| WP | Trạng thái | Ghi chú |
|---|---|---|
| WP0 khung + môi trường | ✅ | `cpm/ perception/ app/ experiments/ tests/`, requirements, README |
| WP1 perception | ✅ khung | `SyntheticEmbedder` chạy ngay; `RealEmbedder` (InsightFace+CLIP, lazy) cần M4 |
| WP2 CPM core | ✅ verify | 6/6 test pass; retention 100% tới 200 nhãn |
| WP3 orchestrator | ✅ | `app/cli.py` smoke end-to-end pass |
| WP4 Gradio UI | ✅ | `app/demo_gradio.py` build OK; webcam/real cần M4 |
| WP5 thí nghiệm | ✅ | `experiments/` — biểu đồ retention + footprint + CSV (CPM vs kNN vs fine-tune) |
| WP6 giọng nói | ⏳ stretch | chưa làm |
| Chạy model thật M4 | ⏳ | cần cài stack perception trên M4 |

---

## 6. Kịch bản demo & tiêu chí thành công

### 6.1. Kịch bản trình diễn (cho giảng viên)
1. Chỉ webcam vào người A → gõ "Lan" → **Ghi nhớ** → "Đã ghi nhớ: Lan".
2. Chỉ lại người A → **Hỏi** → "Đây là **Lan** (tin cậy 0.9x)". ✅ G1
3. Cố tình để nó đoán sai người B → **Sửa** "Huy" → lần sau **Hỏi** ra "Huy". ✅ G2
4. Dạy thêm 8–10 người/đồ → quay lại người A đầu tiên → **vẫn ra "Lan"**. ✅ G3
5. Mở biểu đồ **retention CPM vs fine-tune**: CPM phẳng, fine-tune tụt dốc. ✅ G3/G5

### 6.2. Tiêu chí định lượng
| Metric | Ngưỡng mục tiêu (demo) |
|---|---|
| Recall accuracy trên ~10 nhãn (one-shot) | ≥ 90% |
| Retention@10 (accuracy nhãn đầu sau khi dạy 10 nhãn) | tụt < 10% (CPM); B1 tụt mạnh |
| Correction efficiency | đúng ở lần gặp kế sau khi sửa (100% ca demo) |
| Footprint bộ nhớ | **cố định** theo kích thước ma trận (không phình theo số ảnh) |

---

## 7. Rủi ro & cách xử lý (riêng cho demo)
| Rủi ro | Xử lý |
|---|---|
| InsightFace khó cài trên ARM/M4 | fallback `facenet-pytorch` hoặc MediaPipe FaceMesh + embedding nhẹ |
| Associative memory nhiễu khi nhiều nhãn | chuyển sang **prototype-delta §4.4** (bounded, ổn định) |
| Lỗi vặt MPS trong torch | fallback CPU (model nhỏ, vẫn nhanh trên M4) |
| Nhầm lẫn "CPM hơn kNN" | **Trung thực:** kNN cũng không quên; điểm hơn của CPM là **bộ nhớ bị chặn (không phình), học 1-pass, xử lý correction, đa tầng**. So sánh chính trong demo là **vs fine-tune (B1)** — nơi khác biệt rõ nhất. (Ablation vs kNN để dành Tháng 4.) |

---

## 8. Deliverables của demo #1
- ✅ Demo **Gradio chạy trên M4** (dạy/hỏi/sửa qua webcam).
- ✅ Module `cpm/` + `perception/` + **tests pass**.
- ✅ **Biểu đồ retention** CPM vs fine-tune + ghi chú kết quả ngắn trong `docs/`.
- ✅ Snapshot bộ nhớ `.pt` + `README.md` hướng dẫn chạy.

---

## 9. Quyết định đã chốt (2026-07-09)
- [x] **Đối tượng:** làm **cả face + object cùng lúc**. → Lưu ý: 2 embedding ở **không gian khác nhau** → CPM giữ **bộ nhớ riêng theo modality** (2 instance `cpm_face`, `cpm_object`), không trộn chung 1 ma trận.
- [x] **Dữ liệu:** **cả hai** — ảnh thành viên nhóm (demo trực quan) + subset LFW (đo số liệu/retention cho báo cáo).
- [x] **Giọng nói:** **text trước**, STT/TTS là stretch (WP6).
- [x] **Bắt đầu code:** WP0 → WP2 (dựng khung + CPM core + test "không quên"), verify bằng embedding tổng hợp trước khi chạy model thật trên M4.
```
