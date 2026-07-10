"""
Sinh dữ liệu embedding cho thí nghiệm retention.

- Mặc định: embedding TỔNG HỢP (tái lập được, không cần model) — mỗi danh tính có 1 prototype
  và nhiều 'lần chụp' với cos điều khiển được (mô phỏng biến thiên ảnh-cùng-người của ArcFace).
- Hook `load_real_embeddings()` để sau này cắm embedding thật (LFW/ảnh nhóm) qua perception.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def make_prototypes(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, dim))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def sample_with_cos(proto: np.ndarray, cos_target: float, seed: int) -> np.ndarray:
    """Một mẫu có cos(proto, mẫu) = cos_target (biến thiên intra-class thực tế)."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(proto.shape[0])
    n = n - (n @ proto) * proto
    n = _unit(n)
    return _unit(cos_target * proto + np.sqrt(max(0.0, 1.0 - cos_target**2)) * n)


def make_correlated_prototypes(n: int, dim: int, shared: float, seed: int = 0) -> np.ndarray:
    """
    n prototype có TƯƠNG QUAN (mô phỏng người/đồ giống nhau). shared=0 -> gần trực giao;
    shared cao -> các danh tính chia sẻ một thành phần chung (cos giữa 2 danh tính khác nhau cao).
    """
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal(dim))
    P = []
    for _ in range(n):
        v = _unit(rng.standard_normal(dim))
        P.append(_unit(shared * base + (1.0 - shared) * v))
    return np.array(P)


def make_dataset(
    n_ids: int = 20,
    n_train: int = 8,
    n_test: int = 5,
    dim: int = 512,
    cos_target: float = 0.75,
    seed: int = 42,
) -> list[dict]:
    """Trả list theo danh tính: {label, train:[emb...], test:[emb...]}."""
    protos = make_prototypes(n_ids, dim, seed)
    data = []
    for i in range(n_ids):
        train = [sample_with_cos(protos[i], cos_target, seed=1000 * i + j) for j in range(n_train)]
        test = [sample_with_cos(protos[i], cos_target, seed=5000 * i + j) for j in range(n_test)]
        data.append({"label": f"id_{i:02d}", "train": train, "test": test})
    return data
