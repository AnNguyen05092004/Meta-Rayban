"""Lưu CPM cục bộ an toàn bằng JSON + NPZ, không dùng pickle.

Định dạng này dành cho demo một máy và cũng là nền để chuyển sang backend per-user:
metadata đọc được bằng JSON, embedding/prototype là mảng NPZ với ``allow_pickle=False``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import CPMConfig, TierConfig
from .memory import ContinualPersonalizationMemory


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCHEMA_VERSION = 1


class LocalMemoryStore:
    """Kho CPM cục bộ theo ``user_id`` và modality."""

    def __init__(self, root: str | Path = ".local_memory"):
        self.root = Path(root)

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        if not _SAFE_ID.fullmatch(user_id):
            raise ValueError("user_id chỉ gồm chữ, số, dấu gạch dưới hoặc gạch nối (tối đa 64 ký tự).")
        return user_id

    def _paths(self, user_id: str, modality: str) -> tuple[Path, Path]:
        user_id = self._validate_user_id(user_id)
        if modality not in {"face", "object"}:
            raise ValueError(f"Modality không hợp lệ: {modality}")
        folder = self.root / user_id
        return folder / f"{modality}.json", folder / f"{modality}.npz"

    def exists(self, user_id: str, modality: str) -> bool:
        meta, arrays = self._paths(user_id, modality)
        return meta.is_file() and arrays.is_file()

    def save(self, memory: ContinualPersonalizationMemory) -> None:
        """Ghi state prototype hiện tại; file hoàn tất được thay nguyên tử."""
        meta_path, array_path = self._paths(memory.user_id, memory.modality)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        labels = sorted(memory.labels())
        n = len(labels)
        dim = memory.dim
        arrays: dict[str, np.ndarray] = {
            "labels": np.asarray(labels, dtype="U"),
            "proto_sum": np.stack([memory.proto_sum[x] for x in labels]) if n else np.empty((0, dim)),
            "proto_count": np.asarray([memory.proto_count[x] for x in labels], dtype=np.int64),
            "proto_keys": np.stack([memory.proto_keys[x] for x in labels]) if n else np.empty((0, dim)),
            "proto_ema": np.stack([memory.proto_ema[x] for x in labels]) if n else np.empty((0, dim)),
            "hit_counts": np.asarray([memory.hit_counts[x] for x in labels], dtype=np.int64),
        }
        for name, tier in memory.tiers.items():
            arrays[f"tier_{name}"] = tier.M
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "user_id": memory.user_id,
            "modality": memory.modality,
            "config": asdict(memory.cfg),
        }

        with tempfile.NamedTemporaryFile(dir=meta_path.parent, suffix=".npz", delete=False) as tmp:
            tmp_array = Path(tmp.name)
            np.savez_compressed(tmp, **arrays)
        try:
            os.replace(tmp_array, array_path)
            with tempfile.NamedTemporaryFile(
                dir=meta_path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
            ) as tmp:
                tmp_meta = Path(tmp.name)
                json.dump(metadata, tmp, ensure_ascii=False, indent=2, sort_keys=True)
                tmp.write("\n")
            os.replace(tmp_meta, meta_path)
        finally:
            for path in (locals().get("tmp_array"), locals().get("tmp_meta")):
                if path and Path(path).exists():
                    Path(path).unlink()

    def load(self, user_id: str, modality: str) -> ContinualPersonalizationMemory | None:
        """Nạp state đã kiểm tra cấu trúc; không có state thì trả ``None``."""
        meta_path, array_path = self._paths(user_id, modality)
        if not meta_path.is_file() and not array_path.is_file():
            return None
        if not meta_path.is_file() or not array_path.is_file():
            raise RuntimeError("Bộ nhớ cục bộ không đầy đủ; hãy xóa cặp file JSON/NPZ lỗi rồi khởi động lại.")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError("phiên bản schema không hỗ trợ")
            if metadata.get("user_id") != user_id or metadata.get("modality") != modality:
                raise ValueError("user_id hoặc modality không khớp")
            config_data = dict(metadata["config"])
            config_data["tiers"] = [TierConfig(**item) for item in config_data["tiers"]]
            config = CPMConfig(**config_data)
            with np.load(array_path, allow_pickle=False) as data:
                labels = [str(x) for x in data["labels"].tolist()]
                proto_sum = np.asarray(data["proto_sum"], dtype=np.float64)
                proto_count = np.asarray(data["proto_count"], dtype=np.int64)
                proto_keys = np.asarray(data["proto_keys"], dtype=np.float64)
                proto_ema = np.asarray(data["proto_ema"], dtype=np.float64)
                hit_counts = np.asarray(data["hit_counts"], dtype=np.int64)
                n, dim = len(labels), config.dim
                if len(set(labels)) != n or any(not label for label in labels):
                    raise ValueError("nhãn trùng hoặc rỗng")
                for name, arr in {
                    "proto_sum": proto_sum,
                    "proto_keys": proto_keys,
                    "proto_ema": proto_ema,
                }.items():
                    if arr.shape != (n, dim) or not np.isfinite(arr).all():
                        raise ValueError(f"mảng {name} không hợp lệ")
                if proto_count.shape != (n,) or hit_counts.shape != (n,) or np.any(proto_count < 1):
                    raise ValueError("bộ đếm không hợp lệ")
                memory = ContinualPersonalizationMemory(dim=dim, user_id=user_id, modality=modality, config=config)
                memory.proto_sum = {label: proto_sum[i].copy() for i, label in enumerate(labels)}
                memory.proto_count = {label: int(proto_count[i]) for i, label in enumerate(labels)}
                memory.proto_keys = {label: proto_keys[i].copy() for i, label in enumerate(labels)}
                memory.proto_ema = {label: proto_ema[i].copy() for i, label in enumerate(labels)}
                memory.hit_counts = {label: int(hit_counts[i]) for i, label in enumerate(labels)}
                for name, tier in memory.tiers.items():
                    key = f"tier_{name}"
                    if key in data:
                        matrix = np.asarray(data[key], dtype=np.float64)
                        if matrix.shape != (dim, dim) or not np.isfinite(matrix).all():
                            raise ValueError(f"ma trận {name} không hợp lệ")
                        tier.M = matrix
                return memory
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Không thể nạp bộ nhớ cục bộ {meta_path}: {exc}") from exc
