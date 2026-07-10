# Mở rộng & Backlog — xử lý sau

> Tài liệu ghi lại các ý tưởng/kỹ thuật **chưa làm ngay** nhưng đáng tham khảo khi mở rộng đồ án. Bổ trợ cho [KE_HOACH_DO_AN.md](KE_HOACH_DO_AN.md).
> Cập nhật: 2026-07-09.

---

## 1. Chiến lược đảm bảo REAL-TIME mà không cần nén model xuống điện thoại

**Bối cảnh:** Trong plan §6, tầng "fast" ban đầu dự kiến chạy **model nén trên điện thoại** để phản hồi real-time. Câu hỏi đặt ra: *nếu không nén model xuống điện thoại thì còn cách nào khác hay & hiệu quả hơn để đảm bảo real-time không?* → Ghi lại phân tích + khuyến nghị bên dưới để **xử lý sau** (giai đoạn tối ưu / mở rộng mobility-offline).

### 1.1. Nguyên tắc gốc (khung tư duy)
> **"Real-time" ≠ "nén model xuống điện thoại".** Chỉ có 3 đòn bẩy để phản hồi nhanh:
> 1. **Đưa compute lại gần cảm biến** (on-device *hoặc* edge server gần).
> 2. **Giảm khối lượng cần real-time** (tách theo mức độ khẩn, cache).
> 3. **Giảm độ trễ *cảm nhận*** (streaming, trả lời sớm, xử lý nền).

Lưu ý về NL: tinh thần "đa tần số" nói về **nhịp cập nhật**, KHÔNG bắt buộc inference chạy ở đâu. Nên "tầng nhanh" có thể phục vụ bằng **edge server + cache + CPM recall** thay vì model nén — câu chuyện NL vẫn nguyên vẹn.

### 1.2. Các phương án (không cần nén model on-device)

| # | Phương án | Ý chính | Ưu | Nhược |
|---|---|---|---|---|
| ① | **Edge server qua Wi‑Fi (dùng M4 làm edge node)** ⭐ | Điện thoại là thin client stream khung hình; **M4 chạy full model**. RTT nội mạng ~10–40ms (cloud là 300ms–2s) | Nhanh + **chất lượng đầy đủ, không phải nén**; ít công sức | Chỉ nhanh khi edge box ở gần (nhà/lab); mobility ngoài trời cần mini‑PC (Jetson Orin Nano) / 5G‑MEC |
| ② | **Tách theo mức độ khẩn** | Chỉ vật cản cần real-time cứng → **model bé chuyên dụng (YOLO‑nano)**; VQA/OCR để cloud | Pragmatic; model nhỏ ≠ nén model lớn | Cần phân loại luồng cẩn thận |
| ③ | **CPM recall local** ⭐ | "Ai/cái gì?" = **tra cứu vector (vài ms)**, chỉ tính embedding local (~10–50ms), không chạm model nặng | Nhận diện tức thì, offline, riêng tư | Chỉ áp dụng cho nhận diện đã học |
| ④ | **Streaming + trả lời sớm** ⭐ | **Đọc TTS ngay từ vài từ đầu**; mô tả nền liên tục để câu trả lời "ấm" sẵn | Giảm độ trễ **cảm nhận** mạnh; dễ làm | Không giảm tổng thời gian thực |
| ⑤ | **Cache theo độ giống khung hình** | Cảnh gần như không đổi → tái dùng mô tả trước | Rất hiệu quả khi đứng yên/di chuyển chậm | Cần ngưỡng so sánh embedding tốt |
| ⑥ | **Tối ưu đường mạng** | WebSocket giữ sẵn; ảnh downscale/nén; region gần (Singapore); provider/model nhanh (Groq, Gemini Flash‑Lite, GPT‑4o‑mini) | Dễ, nhanh có kết quả | Vẫn phụ thuộc mạng, không offline |
| ⑦ | **Racing: đường nhanh + đường chất lượng** | Bắn song song local(thô)+cloud(tinh); đọc đáp án local ngay rồi bổ sung/sửa | Vừa nhanh vừa dần chính xác | Logic hợp nhất phức tạp hơn |

### 1.3. So sánh tổng hợp
| Cách | Độ trễ | Offline? | Mobility | Chất lượng | Công sức |
|---|---|---|---|---|---|
| Nén model xuống ĐT | Rất thấp | ✅ | ✅ | ↓ (vì nén) | Cao (convert/quantize) |
| **① Edge server (M4/Wi‑Fi)** | **Rất thấp** | ⚠️ khi có edge | ⚠️ trong nhà | **Cao** | **Thấp** |
| ② Tách theo khẩn | Thấp (obstacle) | ✅ phần local | ✅ | Cao | TB |
| ③ CPM recall local | ~ms | ✅ | ✅ | Cao | Thấp |
| ④ Streaming TTS | Cảm nhận ↓↓ | — | — | Nguyên vẹn | Thấp |
| ⑤ Cache khung hình | ↓ khi cảnh tĩnh | ✅ | ✅ | Nguyên vẹn | Thấp |
| ⑥ Tối ưu mạng | TB | ❌ | ✅ | Cao | Thấp |

### 1.4. ⭐ KHUYẾN NGHỊ (hybrid, không cần nén model) — để áp dụng khi tối ưu real-time
1. **Giữ 1 model bé cho vật cản chạy local** — an toàn phải hoạt động cả khi mất mạng (ngoại lệ bắt buộc, nhưng là model nhỏ chuyên dụng, KHÔNG phải nén model lớn).
2. **Edge server = M4 qua Wi‑Fi** cho toàn bộ phần nặng (VLM/OCR/CPM) giai đoạn demo → nhanh + chất lượng đầy đủ + **né hẳn việc nén/quantize** (tiết kiệm rất nhiều công sức).
3. **CPM recall + embedding chạy local** → nhận diện người/đồ tức thì.
4. **Streaming TTS + cache khung hình** → cắt độ trễ cảm nhận.

→ Đạt real-time cho việc quan trọng **mà không nén model nào**. Việc nén xuống điện thoại chỉ cần khi muốn **đồng thời offline + mobility ngoài trời** → xếp vào mở rộng.

### 1.5. Đánh đổi cần nhớ
- Edge-server: nhanh nhưng phụ thuộc có edge ở gần (hợp demo trong nhà).
- Chỉ **on-device (nén model)** mới cho **đồng thời** offline + di động + riêng tư.
- Sản phẩm chín thường **hybrid** → chọn theo kịch bản (trong nhà vs ngoài trời).

### 1.6. Việc cần làm khi xử lý sau (checklist)
- [ ] Đo thực tế RTT + độ trễ: cloud vs edge-server(M4/Wi‑Fi) vs on-device, trên đúng kịch bản.
- [ ] Cài streaming TTS (đọc theo token đầu) + mô tả nền + cache khung hình (ngưỡng similarity).
- [ ] Chốt "model bé cho vật cản chạy local" (YOLO‑nano) chạy offline ổn định.
- [ ] Chỉ khi cần mobility+offline: chuyển sang nén model on-device (Core ML/TFLite/ExecuTorch, quantize) — coi như hướng mở rộng.
- [ ] Cân nhắc mini‑PC đeo (Jetson Orin Nano) / 5G‑MEC cho edge di động (ngoài scope 5 tháng).

---

## 2. Các hướng mở rộng khác (ghi sẵn, xử lý sau)
- **Federated learning:** cải thiện model chung từ nhiều user mà không chia sẻ dữ liệu thô (đã nêu ở plan §7 — future work).
- **Nén model on-device đầy đủ** (quantization/distillation) cho nhánh offline + mobility ngoài trời.
- **Edge di động:** mini‑PC đeo hoặc 5G‑MEC.
- **(Thêm dần các mục khác vào đây khi phát sinh.)**
