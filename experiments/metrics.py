"""
Metric chuẩn cho continual learning + open-set recognition (dùng chung cho các thí nghiệm).

- accuracy
- average_forgetting / BWT: từ ma trận R[i][j] = accuracy trên danh tính j sau khi học tới i
- open-set: ROC, AUC, EER, TAR@FAR (đánh giá cổng "quen/lạ")
"""

from __future__ import annotations

import numpy as np


# accuracy: độ chính xác = tỉ lệ đoán đúng (số lần pred == label chia tổng số lần).
# Nếu danh sách rỗng thì trả 0.0 để khỏi lỗi chia cho 0.
def accuracy(preds: list, labels: list) -> float:
    return float(np.mean([p == y for p, y in zip(preds, labels)])) if preds else 0.0


# ----------------------------------------------------------- continual learning
# average_forgetting: đo mức "quên thảm khốc" (học nhãn mới làm quên nhãn cũ).
# R là bảng điểm: R[i][j] = độ chính xác trên danh tính j sau khi đã học tới bước i.
# Quên của một danh tính = điểm CAO NHẤT từng đạt trừ điểm CUỐI CÙNG. Số càng lớn = quên càng nhiều.
def average_forgetting(R: np.ndarray) -> float:
    """
    R (T×T, lower-tri): R[i][j] = accuracy trên danh tính j sau khi đã học tới bước i (j<=i).
    Forgetting_j = max_{i>=j} R[i][j] - R[T-1][j]. Trả trung bình trên j<T-1.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    # Với mỗi danh tính cũ j: max(R[j:, j]) = điểm tốt nhất từng đạt; R[T-1, j] = điểm ở cuối.
    # Hiệu hai giá trị = mức tụt điểm (đã quên bao nhiêu). fs gom mức tụt của mọi danh tính cũ.
    fs = [np.max(R[j:, j]) - R[T - 1, j] for j in range(T - 1)]
    return float(np.mean(fs))  # trả trung bình mức quên


# backward_transfer (BWT): học cái mới ẢNH HƯỞNG tới nhãn cũ thế nào.
# Âm = học mới làm điểm nhãn cũ tụt (quên); Dương = học mới còn giúp nhãn cũ tốt hơn.
def backward_transfer(R: np.ndarray) -> float:
    """BWT = mean_{j<T-1} (R[T-1][j] - R[j][j]). Âm = quên."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    # So điểm CUỐI của nhãn j (R[T-1, j]) với điểm NGAY KHI vừa học xong j (R[j, j]).
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))


# ------------------------------------------------------------------- open-set
# roc_points: dựng đường ROC để đánh giá cổng "quen/lạ".
# Ý tưởng: thử NHIỀU mức ngưỡng khác nhau; tại mỗi ngưỡng đo 2 con số:
#   TAR = tỉ lệ bắt ĐÚNG người quen, FAR = tỉ lệ nhận NHẦM người lạ.
# Trả 3 mảng (far, tar, thresholds) để vẽ đồ thị hoặc tính AUC/EER.
def roc_points(genuine: np.ndarray, impostor: np.ndarray, n: int = 300):
    """Trả (far, tar, thresholds) sắp theo threshold giảm dần."""
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    lo = min(genuine.min(), impostor.min())  # điểm thấp nhất trong cả 2 tập
    hi = max(genuine.max(), impostor.max())  # điểm cao nhất trong cả 2 tập
    thr = np.linspace(hi, lo, n)  # tạo n mức ngưỡng, quét từ cao xuống thấp
    tar = np.array([(genuine >= t).mean() for t in thr])  # tỉ lệ người quen có điểm >= ngưỡng
    far = np.array([(impostor >= t).mean() for t in thr])  # tỉ lệ người lạ bị lọt (điểm >= ngưỡng)
    return far, tar, thr


# auc: diện tích dưới đường ROC. Gần 1.0 = phân biệt quen-lạ RẤT tốt; ~0.5 = đoán mò.
def auc(far: np.ndarray, tar: np.ndarray) -> float:
    order = np.argsort(far)  # sắp theo FAR tăng dần để tính diện tích cho đúng thứ tự
    # Dùng np.trapezoid (numpy mới) nếu có, không thì np.trapz (numpy cũ) -> tương thích 2 phiên bản.
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(tar[order], far[order]))  # tích phân TAR theo FAR = diện tích dưới ROC


# eer (Equal Error Rate): điểm ngưỡng nơi hai kiểu lỗi bằng nhau (FAR ~ FRR).
# EER càng THẤP thì hệ thống phân biệt quen-lạ càng tốt. Trả (eer, ngưỡng tại đó).
def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Trả (eer, threshold) — nơi FAR ≈ FRR."""
    far, tar, thr = roc_points(genuine, impostor)
    frr = 1 - tar  # FRR = tỉ lệ TỪ CHỐI NHẦM người quen (ngược với TAR)
    i = int(np.argmin(np.abs(far - frr)))  # tìm ngưỡng nơi FAR và FRR gần bằng nhau nhất
    return float((far[i] + frr[i]) / 2), float(thr[i])  # EER = trung bình 2 lỗi tại đó + ngưỡng


# tar_at_far: tìm ngưỡng "bắt đúng người quen nhiều nhất" (TAR lớn nhất) mà VẪN giữ
# nhận-nhầm-người-lạ (FAR) không vượt far_target. Đây là cách chọn ngưỡng AN TOÀN cho trợ thị.
def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far_target: float) -> tuple[float, float]:
    """TAR lớn nhất với FAR <= far_target. Trả (tar, threshold)."""
    far, tar, thr = roc_points(genuine, impostor)
    ok = far <= far_target  # đánh dấu các ngưỡng thoả điều kiện FAR <= mục tiêu
    if not ok.any():
        return 0.0, float(thr[0])  # không ngưỡng nào đạt -> trả TAR = 0
    i = int(np.argmax(tar[ok]))  # trong các ngưỡng đạt, chọn cái cho TAR cao nhất
    idx = np.where(ok)[0][i]  # đổi lại về chỉ số trong mảng gốc
    return float(tar[idx]), float(thr[idx])


# open_set_summary: gom mọi chỉ số open-set vào 1 dict cho gọn: AUC, EER, và
# TAR (kèm ngưỡng) tại 2 mức an toàn FAR = 1% và FAR = 10%.
def open_set_summary(genuine, impostor) -> dict:
    far, tar, _ = roc_points(genuine, impostor)  # đường ROC để tính AUC
    e, e_thr = eer(genuine, impostor)  # điểm cân bằng lỗi
    tar1, thr1 = tar_at_far(genuine, impostor, 0.01)  # chính sách an toàn nhất: FAR <= 1%
    tar10, thr10 = tar_at_far(genuine, impostor, 0.10)  # nới lỏng hơn: FAR <= 10%
    return {
        "auc": auc(far, tar),
        "eer": e,
        "eer_threshold": e_thr,
        "tar@far=1%": tar1,
        "threshold@far=1%": thr1,
        "tar@far=10%": tar10,
        "threshold@far=10%": thr10,
    }
