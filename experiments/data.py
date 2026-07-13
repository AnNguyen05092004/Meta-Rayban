"""
Sinh dữ liệu embedding cho thí nghiệm retention.

- Mặc định: embedding TỔNG HỢP (tái lập được, không cần model) — mỗi danh tính có 1 prototype
  và nhiều 'lần chụp' với cos điều khiển được (mô phỏng biến thiên ảnh-cùng-người của ArcFace).
- Hook `load_real_embeddings()` để sau này cắm embedding thật (LFW/ảnh nhóm) qua perception.
"""

from __future__ import annotations

import numpy as np


# _unit: chuẩn hoá véc-tơ về độ dài 1 (chia véc-tơ cho chính độ dài của nó).
# Vì sao cần? Để so hai "dấu vân tay số" (embedding) công bằng bằng độ giống cosine.
# +1e-8 (số cực nhỏ): phòng khi độ dài = 0 thì không bị lỗi chia cho 0.
def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


# make_prototypes: tạo n "dấu vân tay tâm" (prototype) ngẫu nhiên, mỗi cái là véc-tơ dim chiều.
# Mỗi prototype coi như "tâm" đại diện cho 1 danh tính (người/đồ) giả trong thí nghiệm.
# seed cố định -> chạy lại vẫn ra đúng bộ số cũ (thí nghiệm tái lập được).
def make_prototypes(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)  # bộ sinh số ngẫu nhiên, gắn seed để lặp lại được
    p = rng.standard_normal((n, dim))  # n véc-tơ ngẫu nhiên (phân phối chuẩn)
    return p / np.linalg.norm(p, axis=1, keepdims=True)  # chuẩn hoá từng hàng về độ dài 1


# sample_with_cos: tạo 1 "lần chụp" giả của CÙNG một danh tính (quanh tâm proto),
# sao cho độ giống (cosine) giữa mẫu này và tâm đúng bằng cos_target.
# cos_target cao (vd 0.9) = ảnh rất giống tâm; thấp = ảnh lệch nhiều.
# Mục đích: mô phỏng "cùng 1 người nhưng mỗi lần chụp hơi khác nhau".
def sample_with_cos(proto: np.ndarray, cos_target: float, seed: int) -> np.ndarray:
    """Một mẫu có cos(proto, mẫu) = cos_target (biến thiên intra-class thực tế)."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(proto.shape[0])  # tạo một hướng nhiễu ngẫu nhiên
    n = n - (n @ proto) * proto  # bỏ phần trùng hướng proto -> nhiễu VUÔNG GÓC với tâm
    n = _unit(n)  # chuẩn hoá nhiễu về độ dài 1
    # Trộn tâm (tỉ lệ cos_target) với nhiễu vuông góc (tỉ lệ sqrt(1-cos^2)) rồi chuẩn hoá.
    # Nhờ hai phần vuông góc nhau, cos(kết quả, proto) rơi đúng bằng cos_target.
    return _unit(cos_target * proto + np.sqrt(max(0.0, 1.0 - cos_target**2)) * n)


# make_correlated_prototypes: tạo n tâm nhưng CỐ Ý cho chúng giống nhau một phần
# (mô phỏng tình huống nhiều người/đồ trông na ná nhau -> nhận diện khó hơn).
# shared = mức chia sẻ "thành phần chung": 0 = mỗi tâm một hướng riêng; càng cao càng giống nhau.
def make_correlated_prototypes(n: int, dim: int, shared: float, seed: int = 0) -> np.ndarray:
    """
    n prototype có TƯƠNG QUAN (mô phỏng người/đồ giống nhau). shared=0 -> gần trực giao;
    shared cao -> các danh tính chia sẻ một thành phần chung (cos giữa 2 danh tính khác nhau cao).
    """
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal(dim))  # một hướng "chung" mà mọi tâm cùng pha vào
    P = []
    for _ in range(n):
        v = _unit(rng.standard_normal(dim))  # hướng riêng của từng danh tính
        # Mỗi tâm = trộn phần chung (shared*base) + phần riêng ((1-shared)*v).
        P.append(_unit(shared * base + (1.0 - shared) * v))
    return np.array(P)


# make_dataset: dựng cả BỘ dữ liệu giả cho thí nghiệm (không cần camera/model thật).
# Tạo n_ids danh tính; mỗi danh tính có n_train ảnh để DẠY và n_test ảnh để KIỂM.
# cos_target điều khiển độ "giống nhau" giữa các ảnh của cùng một người.
def make_dataset(
    n_ids: int = 20,
    n_train: int = 8,
    n_test: int = 5,
    dim: int = 512,
    cos_target: float = 0.75,
    seed: int = 42,
) -> list[dict]:
    """Trả list theo danh tính: {label, train:[emb...], test:[emb...]}."""
    protos = make_prototypes(n_ids, dim, seed)  # tạo 1 tâm cho mỗi danh tính
    data = []
    for i in range(n_ids):
        # Sinh các "lần chụp" quanh tâm của danh tính i. Seed train/test khác nhau
        # (1000*i vs 5000*i) để ảnh KIỂM không trùng ảnh DẠY.
        train = [sample_with_cos(protos[i], cos_target, seed=1000 * i + j) for j in range(n_train)]
        test = [sample_with_cos(protos[i], cos_target, seed=5000 * i + j) for j in range(n_test)]
        # Gói lại theo nhãn dạng id_00, id_01, ... kèm danh sách ảnh train và test.
        data.append({"label": f"id_{i:02d}", "train": train, "test": test})
    return data
