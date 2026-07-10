"""
Metric chuẩn cho continual learning + open-set recognition (dùng chung cho các thí nghiệm).

- accuracy
- average_forgetting / BWT: từ ma trận R[i][j] = accuracy trên danh tính j sau khi học tới i
- open-set: ROC, AUC, EER, TAR@FAR (đánh giá cổng "quen/lạ")
"""

from __future__ import annotations

import numpy as np


def accuracy(preds: list, labels: list) -> float:
    return float(np.mean([p == y for p, y in zip(preds, labels)])) if preds else 0.0


# ----------------------------------------------------------- continual learning
def average_forgetting(R: np.ndarray) -> float:
    """
    R (T×T, lower-tri): R[i][j] = accuracy trên danh tính j sau khi đã học tới bước i (j<=i).
    Forgetting_j = max_{i>=j} R[i][j] - R[T-1][j]. Trả trung bình trên j<T-1.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    fs = [np.max(R[j:, j]) - R[T - 1, j] for j in range(T - 1)]
    return float(np.mean(fs))


def backward_transfer(R: np.ndarray) -> float:
    """BWT = mean_{j<T-1} (R[T-1][j] - R[j][j]). Âm = quên."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))


# ------------------------------------------------------------------- open-set
def roc_points(genuine: np.ndarray, impostor: np.ndarray, n: int = 300):
    """Trả (far, tar, thresholds) sắp theo threshold giảm dần."""
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thr = np.linspace(hi, lo, n)
    tar = np.array([(genuine >= t).mean() for t in thr])
    far = np.array([(impostor >= t).mean() for t in thr])
    return far, tar, thr


def auc(far: np.ndarray, tar: np.ndarray) -> float:
    order = np.argsort(far)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(tar[order], far[order]))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Trả (eer, threshold) — nơi FAR ≈ FRR."""
    far, tar, thr = roc_points(genuine, impostor)
    frr = 1 - tar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far_target: float) -> tuple[float, float]:
    """TAR lớn nhất với FAR <= far_target. Trả (tar, threshold)."""
    far, tar, thr = roc_points(genuine, impostor)
    ok = far <= far_target
    if not ok.any():
        return 0.0, float(thr[0])
    i = int(np.argmax(tar[ok]))
    idx = np.where(ok)[0][i]
    return float(tar[idx]), float(thr[idx])


def open_set_summary(genuine, impostor) -> dict:
    far, tar, _ = roc_points(genuine, impostor)
    e, e_thr = eer(genuine, impostor)
    tar1, thr1 = tar_at_far(genuine, impostor, 0.01)
    tar10, thr10 = tar_at_far(genuine, impostor, 0.10)
    return {
        "auc": auc(far, tar),
        "eer": e,
        "eer_threshold": e_thr,
        "tar@far=1%": tar1,
        "threshold@far=1%": thr1,
        "tar@far=10%": tar10,
        "threshold@far=10%": thr10,
    }
