from __future__ import annotations

from cpm import CPMConfig, ContinualPersonalizationMemory
from experiments.baselines import CPMEMAAdapter, NCMAdapter
from experiments.drift import accuracy, make_drift_stream


def _trained(adapter, train):
    adapter.reset()
    for emb, label in train:
        adapter.teach(emb, label)
    return adapter


def test_cpm_ema_beats_ncm_under_strong_drift():
    dim = 128
    train, test = make_drift_stream(n_pairs=10, n_steps=16, dim=dim, drift=1.0, seed=1)

    ncm = _trained(NCMAdapter(dim), train)
    ema = _trained(CPMEMAAdapter(dim), train)

    assert accuracy(ema, test) - accuracy(ncm, test) >= 0.15
    assert ema.footprint() < 0.2 * 10 * 16 * 2 * dim  # much smaller than storing every observation


def test_cpm_ema_does_not_hurt_stationary_data():
    dim = 128
    train, test = make_drift_stream(n_pairs=10, n_steps=16, dim=dim, drift=0.0, seed=2)

    ncm = _trained(NCMAdapter(dim), train)
    ema = _trained(CPMEMAAdapter(dim), train)

    assert abs(accuracy(ema, test) - accuracy(ncm, test)) <= 0.05


def test_cpm_ema_snapshot_roundtrip(tmp_path):
    dim = 64
    train, test = make_drift_stream(n_pairs=3, n_steps=8, dim=dim, drift=1.0, seed=3)
    cpm = ContinualPersonalizationMemory(dim=dim, config=CPMConfig(dim=dim, use_ema=True))
    for emb, label in train:
        cpm.write(emb, label)

    path = tmp_path / "cpm_ema.pkl"
    cpm.snapshot(str(path))
    loaded = ContinualPersonalizationMemory.load(str(path))

    assert loaded.cfg.use_ema is True
    assert loaded.stats()["footprint_floats"] == cpm.stats()["footprint_floats"]
    assert [cpm.recall(x)[0]["label"] for x, _ in test] == [loaded.recall(x)[0]["label"] for x, _ in test]
