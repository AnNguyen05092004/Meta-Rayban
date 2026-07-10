"""
Test lõi CPM bằng embedding tổng hợp (không cần model thật / M4).

Kịch bản then chốt của đồ án:
- dạy 1 -> nhận lại đúng
- dạy nhiều nhãn -> KHÔNG quên nhãn đầu (retention)
- sửa 1 lần -> nhớ
- query lạ -> "chưa biết"
"""

import numpy as np
import pytest

from cpm import CPMConfig, ContinualPersonalizationMemory


def make_identities(n: int, dim: int = 512, seed: int = 0) -> np.ndarray:
    """n prototype gần trực giao (mô phỏng embedding các danh tính khác nhau)."""
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((n, dim))
    return protos / np.linalg.norm(protos, axis=1, keepdims=True)


def noisy(proto: np.ndarray, cos_target: float = 0.75, seed: int = 0) -> np.ndarray:
    """
    Một 'lần chụp' của cùng danh tính, có cos(proto, sample) = cos_target đúng như đặt.
    Mô phỏng biến thiên ảnh-cùng-người thực tế (ArcFace: same-person cos ~0.4-0.8).
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(proto.shape[0])
    n = n - (n @ proto) * proto           # thành phần trực giao với proto
    n = n / np.linalg.norm(n)
    v = cos_target * proto + np.sqrt(1.0 - cos_target**2) * n
    return v / np.linalg.norm(v)


def new_cpm(dim: int = 512) -> ContinualPersonalizationMemory:
    return ContinualPersonalizationMemory(dim=dim, modality="test", config=CPMConfig(dim=dim))


def test_teach_then_recall():
    """G1: dạy 1 người -> nhận lại (kể cả có nhiễu) đúng nhãn."""
    dim = 512
    ids = make_identities(1, dim)
    cpm = new_cpm(dim)
    cpm.write(ids[0], "Lan")

    res = cpm.recall(noisy(ids[0], seed=1))[0]
    assert res["label"] == "Lan"
    assert res["known"] is True
    assert res["confidence"] >= 0.3


def test_retention_no_forgetting():
    """G3: dạy 10 nhãn tuần tự, mọi nhãn đã học vẫn nhận đúng (không quên)."""
    dim = 512
    n = 10
    ids = make_identities(n, dim, seed=42)
    names = [f"id_{i}" for i in range(n)]

    cpm = new_cpm(dim)
    for i in range(n):
        cpm.write(ids[i], names[i])  # one-shot mỗi nhãn

    # sau khi học hết, kiểm tra lại TẤT CẢ (đặc biệt nhãn đầu tiên)
    correct = 0
    for i in range(n):
        res = cpm.recall(noisy(ids[i], seed=100 + i))[0]
        if res["label"] == names[i]:
            correct += 1
    acc = correct / n
    assert acc >= 0.9, f"retention accuracy quá thấp: {acc}"

    # nhãn ĐẦU TIÊN vẫn phải nhận đúng (điểm nhấn chống quên)
    first = cpm.recall(noisy(ids[0], seed=7))[0]
    assert first["label"] == names[0]


def test_correction():
    """G2: sửa 1 lần -> lần sau nhớ đúng."""
    dim = 512
    ids = make_identities(2, dim, seed=5)
    cpm = new_cpm(dim)
    cpm.write(ids[0], "A")

    # ids[1] chưa dạy; người dùng sửa: đây là B
    cpm.correct(ids[1], "B")
    res = cpm.recall(noisy(ids[1], seed=3))[0]
    assert res["label"] == "B"
    assert res["known"] is True

    # sửa không được làm hỏng nhãn A cũ
    res_a = cpm.recall(noisy(ids[0], seed=9))[0]
    assert res_a["label"] == "A"


def test_unknown_below_threshold():
    """Query lạ (chưa từng dạy) -> known=False."""
    dim = 512
    ids = make_identities(3, dim, seed=11)
    cpm = new_cpm(dim)
    for i in range(3):
        cpm.write(ids[i], f"known_{i}")

    rng = np.random.default_rng(999)
    stranger = rng.standard_normal(dim)
    stranger = stranger / np.linalg.norm(stranger)
    res = cpm.recall(stranger)[0]
    assert res["known"] is False


def test_bounded_footprint():
    """Bộ nhớ bị chặn: footprint không phụ thuộc số lần dạy (khác kNN)."""
    dim = 128
    ids = make_identities(5, dim, seed=1)
    cpm = new_cpm(dim)

    for i in range(5):
        cpm.write(ids[i], f"id_{i}")
    fp1 = cpm.stats()["footprint_floats"]

    # dạy lại nhiều lần (nhiều 'ảnh') nhưng KHÔNG thêm nhãn mới
    for rep in range(50):
        for i in range(5):
            cpm.write(noisy(ids[i], seed=rep * 10 + i), f"id_{i}")
    fp2 = cpm.stats()["footprint_floats"]

    assert fp1 == fp2, "footprint phải cố định theo số NHÃN, không theo số ảnh"


def test_snapshot_roundtrip(tmp_path):
    """Lưu/khôi phục cho cùng kết quả."""
    dim = 256
    ids = make_identities(4, dim, seed=2)
    cpm = new_cpm(dim)
    for i in range(4):
        cpm.write(ids[i], f"id_{i}")

    path = tmp_path / "cpm.pkl"
    cpm.snapshot(str(path))
    loaded = ContinualPersonalizationMemory.load(str(path))

    q = noisy(ids[2], seed=4)
    assert cpm.recall(q)[0]["label"] == loaded.recall(q)[0]["label"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
