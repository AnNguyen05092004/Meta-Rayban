"""
Test hiệu chỉnh ngưỡng quen/lạ (Task 1) — chạy thuần numpy, không cần model/mạng.

Bao trùm:
  - calibrate_threshold: far1 giữ FAR ≤ 1%; eer cho FAR ≈ FRR; sửa đúng lỗi "0.35 quá thấp".
  - collect_open_set_scores: genuine (đã dạy) > impostor (chưa dạy).
  - thresholds: save/load/resolve roundtrip + fallback default + entry dạng dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpm import (
    CPMConfig,
    ContinualPersonalizationMemory,
    load_thresholds,
    resolve_threshold,
    save_thresholds,
    threshold_key,
)
from experiments.calibration import calibrate_threshold, collect_open_set_scores
from experiments.data import make_prototypes, sample_with_cos


def _scores(mean, std, n, seed):
    return np.clip(np.random.default_rng(seed).normal(mean, std, n), -1.0, 1.0)


def test_far1_keeps_far_below_1pct():
    genuine = _scores(0.75, 0.06, 500, 1)
    impostor = _scores(0.30, 0.06, 500, 2)
    r = calibrate_threshold(genuine, impostor, policy="far1")
    assert r["far"] <= 0.01 + 1e-9          # đúng ràng buộc an toàn
    assert r["tar"] > 0.9                     # vẫn nhận ra hầu hết người quen
    assert 0.30 < r["threshold"] < 0.75       # nằm giữa 2 phân bố


def test_eer_balances_far_frr():
    genuine = _scores(0.70, 0.08, 500, 3)
    impostor = _scores(0.35, 0.08, 500, 4)
    r = calibrate_threshold(genuine, impostor, policy="eer")
    assert abs(r["far"] - r["frr"]) < 0.05    # cân bằng


def test_calibrate_fixes_035_being_too_low_for_facenet():
    """Mô phỏng facenet: người khác nhau cos ~0.4. Ngưỡng 0.35 cho FAR cao;
    calibrate far1 phải nâng ngưỡng lên và hạ FAR mạnh."""
    genuine = _scores(0.70, 0.08, 600, 5)
    impostor = _scores(0.40, 0.08, 600, 6)
    r = calibrate_threshold(genuine, impostor, policy="far1")
    far_at_035 = float((impostor >= 0.35).mean())
    assert far_at_035 > 0.5                    # 0.35 nhận nhầm > 50% người lạ (lỗi hiện tại)
    assert r["threshold"] > 0.35               # ngưỡng đúng cao hơn
    assert r["far"] < far_at_035               # và an toàn hơn nhiều


def test_calibrate_requires_both_sets():
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([0.8, 0.9]), np.array([]), policy="far1")


def test_collect_scores_genuine_above_impostor():
    dim = 128
    protos = make_prototypes(4, dim, seed=0)
    cpm = ContinualPersonalizationMemory(dim=dim, config=CPMConfig(dim=dim))
    # dạy id 0,1,2 ; giữ id 3 làm impostor
    for i in range(3):
        for j in range(6):
            cpm.write(sample_with_cos(protos[i], 0.85, seed=100 * i + j), f"id{i}")
    known_probes = [
        (f"id{i}", [sample_with_cos(protos[i], 0.85, seed=900 + 10 * i + j) for j in range(4)])
        for i in range(3)
    ]
    impostor_embs = [sample_with_cos(protos[3], 0.85, seed=7000 + j) for j in range(12)]

    genuine, impostor = collect_open_set_scores(cpm, known_probes, impostor_embs)
    assert genuine.size == 12 and impostor.size == 12
    assert genuine.mean() > impostor.mean() + 0.1   # người quen rõ ràng cao hơn người lạ


def test_thresholds_save_load_resolve(tmp_path):
    path = str(tmp_path / "thresholds.json")
    key_face = threshold_key("real", "face")
    save_thresholds({key_face: {"threshold": 0.52, "policy": "far1"}}, path)
    save_thresholds({threshold_key("real", "object"): 0.78}, path)   # merge, không ghi đè face

    mapping = load_thresholds(path)
    assert set(mapping) == {key_face, threshold_key("real", "object")}
    assert resolve_threshold(mapping, "real", "face", default=0.35) == 0.52     # entry dict
    assert resolve_threshold(mapping, "real", "object", default=0.35) == 0.78   # entry số
    assert resolve_threshold(mapping, "synthetic", "face", default=0.35) == 0.35  # thiếu -> default


def test_load_thresholds_missing_file_returns_empty(tmp_path):
    assert load_thresholds(str(tmp_path / "khong-ton-tai.json")) == {}


def test_assistant_applies_calibrated_threshold(monkeypatch):
    """Assistant phải NẠP và ÁP ngưỡng calibrate theo (embedder, modality)."""
    import app.cli as cli

    monkeypatch.setattr(cli, "load_thresholds", lambda *a, **k: {threshold_key("synthetic", "face"): 0.66})
    a = cli.Assistant(embedder_kind="synthetic")
    assert a.cpm["face"].cfg.recall_threshold == 0.66     # áp ngưỡng đã calibrate
    assert a.cpm["object"].cfg.recall_threshold == 0.35   # thiếu khoá -> giữ mặc định
