from __future__ import annotations

import json

import numpy as np

from app.cli import Assistant
from cpm import CPMConfig, ContinualPersonalizationMemory, LocalMemoryStore


def _unit(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


def test_local_store_roundtrip_uses_json_and_npz_only(tmp_path):
    cpm = ContinualPersonalizationMemory(dim=16, user_id="an", modality="object", config=CPMConfig(dim=16))
    cpm.write(_unit(1), "ví")
    cpm.write(_unit(2), "chìa_khóa")
    store = LocalMemoryStore(tmp_path)
    store.save(cpm)

    metadata = json.loads((tmp_path / "an" / "object.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert not list((tmp_path / "an").glob("*.pkl"))

    loaded = store.load("an", "object")
    assert loaded is not None
    assert loaded.recall(_unit(1))[0]["label"] == "ví"
    assert loaded.stats()["hit_counts"] == cpm.stats()["hit_counts"]


def test_assistant_restores_local_memory_across_instances(tmp_path):
    first = Assistant("synthetic", user_id="demo", memory_dir=tmp_path)
    first.teach("object", "my_wallet", "ví của tôi")

    restarted = Assistant("synthetic", user_id="demo", memory_dir=tmp_path)
    assert restarted.ask("object", "my_wallet").startswith("Đây là ví của tôi")


def test_local_store_rejects_unsafe_user_id(tmp_path):
    store = LocalMemoryStore(tmp_path)
    try:
        store.exists("../other", "object")
    except ValueError as exc:
        assert "user_id" in str(exc)
    else:
        raise AssertionError("unsafe user_id must be rejected")
