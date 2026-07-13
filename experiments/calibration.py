"""
Hiệu chỉnh ngưỡng quen/lạ (novelty gate) từ ROC THẬT của một embedder + modality.

Bối cảnh (xem docs/KET_QUA_THI_NGHIEM.md §3.5): ngưỡng cứng 0.35 lấy từ synthetic. Trên
embedding thật (facenet/ArcFace), cos giữa 2 người KHÁC nhau ~0.3–0.5 nên 0.35 cho FAR rất cao
(nhận nhầm người lạ). Ngưỡng đúng = chọn từ phân bố điểm genuine (người quen) vs impostor (người lạ).

Quy trình:
    1) dạy các danh tính QUEN vào 1 CPM
    2) collect_open_set_scores -> (genuine, impostor)  [dùng chính confidence recall = max cos]
    3) calibrate_threshold(genuine, impostor, policy) -> {threshold, far, frr, ...}

Chính sách (policy):
    - "far1"  (mặc định, AN TOÀN cho trợ thị): ngưỡng cao nhất giữ FAR ≤ 1% -> hiếm chào nhầm người lạ.
    - "far10" : nới FAR ≤ 10% (nhận đúng người quen nhiều hơn).
    - "eer"   : cân bằng FAR ≈ FRR.

Phần TÍNH dùng experiments/metrics.py; phần LƯU/NẠP dùng cpm/thresholds.py.
"""

from __future__ import annotations

import numpy as np

from experiments.metrics import open_set_summary

# Bảng tra: tên chính sách người dùng chọn -> tên trường ngưỡng trong kết quả open_set_summary.
# far1  = chọn ngưỡng sao cho nhận-nhầm-người-lạ (FAR) <= 1% (an toàn cho trợ thị).
# far10 = nới lỏng, FAR <= 10%. eer = điểm cân bằng FAR ~ FRR.
_POLICY_TO_KEY = {
    "far1": "threshold@far=1%",
    "far10": "threshold@far=10%",
    "eer": "eer_threshold",
}


# collect_open_set_scores: gom "điểm số" để về sau dựng đường ROC và chọn ngưỡng.
# genuine  = điểm của các mẫu người QUEN (đã dạy) -> kỳ vọng CAO.
# impostor = điểm của các mẫu người LẠ (chưa dạy) -> kỳ vọng THẤP (cổng quen-lạ phải chặn).
# Điểm ở đây = độ giống (cosine) với prototype gần nhất mà CPM tìm được khi recall.
def collect_open_set_scores(cpm, known_probes, impostor_embs):
    """
    Thu điểm confidence để dựng ROC.

    known_probes : iterable (label, [emb, ...]) — mẫu KIỂM của danh tính ĐÃ dạy (genuine).
    impostor_embs: iterable emb — mẫu của danh tính CHƯA dạy (impostor / người lạ).

    Điểm = confidence recall = cos(query, prototype gần nhất). Với genuine, đây ≈ cos tới đúng
    prototype; với impostor, đây = cos tới prototype quen GẦN nhất (điểm mà cổng novelty phải chặn).
    """
    # Với mỗi ảnh kiểm của người ĐÃ dạy: hỏi CPM, lấy confidence (độ giống prototype khớp nhất).
    genuine = [float(cpm.recall(e)[0]["confidence"]) for _label, embs in known_probes for e in embs]
    # Với mỗi ảnh người LẠ: lấy độ giống với prototype quen GẦN nhất (điểm mà cổng phải chặn).
    impostor = [float(cpm.recall(e)[0]["confidence"]) for e in impostor_embs]
    return np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)


# calibrate_threshold: từ 2 tập điểm (quen vs lạ), chọn CON SỐ NGƯỠNG quen-lạ theo chính sách,
# rồi báo cáo FAR/FRR THỰC TẾ tại ngưỡng đó để biết ngưỡng này an toàn tới đâu.
def calibrate_threshold(genuine, impostor, policy: str = "far1") -> dict:
    """
    Chọn ngưỡng theo policy + báo cáo FAR/FRR THỰC TẾ tại ngưỡng đó.

    Trả dict: threshold, policy, far, frr, tar, auc, eer, n_genuine, n_impostor
              (+ các trường thô của open_set_summary).
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    # Bắt buộc phải có CẢ hai tập: thiếu genuine hoặc impostor thì không vẽ được ROC -> báo lỗi rõ.
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError(
            f"Cần cả genuine ({genuine.size}) lẫn impostor ({impostor.size}) > 0 để calibrate. "
            "Impostor = mẫu của người/đồ CHƯA dạy — xem scripts/calibrate_threshold.py."
        )
    if policy not in _POLICY_TO_KEY:
        raise ValueError(f"policy phải thuộc {list(_POLICY_TO_KEY)}, nhận '{policy}'.")

    summ = open_set_summary(genuine, impostor)  # tính hết chỉ số open-set (ROC/AUC/EER/TAR@FAR)
    thr = float(summ[_POLICY_TO_KEY[policy]])  # lấy đúng ngưỡng ứng với chính sách đã chọn

    far = float((impostor >= thr).mean())     # người lạ bị nhận nhầm là quen
    frr = float((genuine < thr).mean())        # người quen bị coi là lạ (bỏ sót)
    tar = 1.0 - frr  # TAR = tỉ lệ NHẬN ĐÚNG người quen = 1 trừ tỉ lệ bỏ sót (FRR)
    return {
        "threshold": thr,
        "policy": policy,
        "far": far,
        "frr": frr,
        "tar": tar,
        "auc": float(summ["auc"]),
        "eer": float(summ["eer"]),
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        **{k: float(v) for k, v in summ.items()},
    }
