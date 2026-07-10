"""
Continual Personalization Memory (CPM) — v2
===========================================

Lõi Nested-Learning của đồ án: bộ nhớ cá nhân hoá học liên tục, thao tác trên
*embedding tri giác* (khuôn mặt / vật thể) thay vì token ngôn ngữ.

THIẾT KẾ (điều chỉnh theo bằng chứng thí nghiệm — xem docs/KET_QUA_THI_NGHIEM.md):
- XƯƠNG SỐNG NHẬN DIỆN = **prototype trung-bình-thật** cho mỗi nhãn (như NCM): bền vững kể cả
  khi các danh tính giống nhau, và bộ nhớ BỊ CHẶN (O(#nhãn), không phình như kNN).
- CƠ CHẾ NL xếp lên trên: cổng novelty (quen/lạ) trong không gian embedding; delta-rule &
  ma trận liên kết đa tầng (fast/medium/slow) là THÀNH PHẦN TUỲ CHỌN để phân tích/adaptation
  (tắt mặc định vì ma trận giới hạn sức chứa ~dim và yếu với danh tính tương quan cao).

Thuần numpy -> chạy mọi nền tảng, không phụ thuộc torch.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np

from .config import CPMConfig, TierConfig


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + eps)


class LabelRegistry:
    """Sổ nhãn: mỗi nhãn -> một anchor (value) là vector đơn vị sinh TẤT ĐỊNH theo tên nhãn."""

    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.seed = seed
        self.anchors: dict[str, np.ndarray] = {}

    def get_or_create(self, label: str) -> np.ndarray:
        if label not in self.anchors:
            h = int.from_bytes(hashlib.sha256(f"{self.seed}:{label}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(h)
            self.anchors[label] = _unit(rng.standard_normal(self.dim))
        return self.anchors[label]


class TierMemory:
    """Một tầng associative memory cập nhật bằng delta-rule (thành phần tuỳ chọn)."""

    def __init__(self, dim: int, cfg: TierConfig):
        self.cfg = cfg
        self.M = np.zeros((dim, dim), dtype=np.float64)

    def write(self, k: np.ndarray, a: np.ndarray, eta: float | None = None) -> None:
        eta = self.cfg.eta if eta is None else eta
        error = a - self.M @ k
        self.M = self.cfg.alpha * self.M + eta * np.outer(error, k)

    def predict(self, q: np.ndarray) -> np.ndarray:
        return self.M @ q


class ContinualPersonalizationMemory:
    """
    CPM cho MỘT modality (một không gian embedding).

        cpm_face = ContinualPersonalizationMemory(dim=512, modality="face")
        cpm_obj  = ContinualPersonalizationMemory(dim=512, modality="object")
    """

    def __init__(
        self,
        dim: int = 512,
        user_id: str = "default",
        modality: str = "generic",
        config: CPMConfig | None = None,
    ):
        self.cfg = config or CPMConfig(dim=dim)
        self.dim = self.cfg.dim
        self.user_id = user_id
        self.modality = modality

        # Xương sống: prototype trung-bình-thật cho mỗi nhãn
        self.proto_sum: dict[str, np.ndarray] = {}
        self.proto_count: dict[str, int] = {}
        self.proto_keys: dict[str, np.ndarray] = {}  # unit(mean) — dùng để recall
        self.proto_ema: dict[str, np.ndarray] = {}   # tầng nhanh: prototype EMA, dùng khi cfg.use_ema=True
        self.hit_counts: dict[str, int] = {}

        # Thành phần tuỳ chọn: ma trận liên kết đa tầng
        self.use_matrix = self.cfg.use_associative_matrix
        self.registry = LabelRegistry(self.dim, self.cfg.anchor_seed)
        self.tiers: dict[str, TierMemory] = (
            {t.name: TierMemory(self.dim, t) for t in self.cfg.tiers} if self.use_matrix else {}
        )

    def labels(self) -> list[str]:
        return list(self.proto_keys.keys())

    # ------------------------------------------------------------------ write
    def write(self, key, label: str, eta_scale: float = 1.0) -> None:
        """Dạy 1 cặp (embedding, nhãn)."""
        k = _unit(key)

        # cập nhật prototype trung-bình-thật (online, bounded)
        if label in self.proto_sum:
            self.proto_sum[label] += k
            self.proto_count[label] += 1
        else:
            self.proto_sum[label] = k.copy()
            self.proto_count[label] = 1
        self.proto_keys[label] = _unit(self.proto_sum[label] / self.proto_count[label])
        if label in self.proto_ema:
            a = self.cfg.ema_alpha
            self.proto_ema[label] = _unit(a * self.proto_ema[label] + (1.0 - a) * k)
        else:
            self.proto_ema[label] = k.copy()
        self.hit_counts[label] = self.hit_counts.get(label, 0) + 1

        # (tuỳ chọn) ghi vào ma trận liên kết đa tầng
        if self.use_matrix:
            a = self.registry.get_or_create(label)
            for t in self.cfg.tiers:
                self.tiers[t.name].write(k, a, eta=t.eta * eta_scale)
            if self.hit_counts[label] >= self.cfg.consolidate_hit_count and "slow" in self.tiers:
                self.tiers["slow"].write(self.proto_keys[label], a, eta=self.cfg.tier("slow").eta)

    # ----------------------------------------------------------------- recall
    def recall(self, query_key, top_k: int = 1, mode: str | None = None) -> list[dict]:
        """
        Nhận diện. `mode`:
          - "proto"  (mặc định): argmax cos(query, prototype) — bền, bộ nhớ bị chặn.
          - "assoc"  : associative memory NL (cần bật ma trận).
          - "hybrid" : kết hợp assoc + proto (cần bật ma trận).
        "known"/confidence luôn = cos(query, prototype) (cổng novelty embedding-space).

        Trả list dict {label, confidence, proto_score, assoc_score, tier, known}.
        """
        mode = mode or self.cfg.default_recall_mode
        q = _unit(query_key)
        labels = self.labels()
        if not labels:
            return [{"label": None, "confidence": 0.0, "proto_score": 0.0,
                     "assoc_score": 0.0, "tier": None, "known": False}]

        proto = np.array([float(self.proto_keys[l] @ q) for l in labels])
        ema = None
        if self.cfg.use_ema:
            ema = np.array([float(self.proto_ema.get(l, self.proto_keys[l]) @ q) for l in labels])

        assoc = None
        if mode in ("assoc", "hybrid"):
            if not self.use_matrix:
                mode = "proto"  # không có ma trận -> fallback
            else:
                A = np.stack([self.registry.get_or_create(l) for l in labels])
                vhat = np.zeros(self.dim)
                for t in self.cfg.tiers:
                    vt = _unit(self.tiers[t.name].predict(q))
                    conf_t = float(np.max(A @ vt))
                    vhat += t.weight * max(conf_t, 0.0) ** 2 * vt
                vhat = _unit(vhat)
                assoc = A @ vhat

        if mode == "assoc":
            score = assoc
        elif mode == "hybrid":
            score = self.cfg.hybrid_w_assoc * assoc + self.cfg.hybrid_w_proto * proto
        else:
            score = proto
            if ema is not None:
                w = float(np.clip(self.cfg.ema_weight, 0.0, 1.0))
                score = (1.0 - w) * proto + w * ema

        order = np.argsort(-score)[:top_k]
        out = []
        for i in order:
            out.append(
                {
                    "label": labels[i],
                    "confidence": float(score[i]),
                    "proto_score": float(proto[i]),
                    "ema_score": float(ema[i]) if ema is not None else float(proto[i]),
                    "assoc_score": float(assoc[i]) if assoc is not None else float(proto[i]),
                    "tier": "ema" if (mode == "proto" and ema is not None) else ("proto" if mode == "proto" else "assoc"),
                    "known": float(score[i]) >= self.cfg.recall_threshold,
                }
            )
        return out

    # ---------------------------------------------------------------- correct
    def correct(self, query_key, new_label: str) -> None:
        """Người dùng sửa sai -> ghi mẫu vào prototype nhãn đúng (và ma trận nếu bật)."""
        self.write(query_key, new_label, eta_scale=self.cfg.correct_eta_boost)

    # ------------------------------------------------------------ consolidate
    def consolidate(self) -> None:
        """Bền hoá (chỉ có tác dụng khi bật ma trận): replay prototype vào tầng slow; xả tầng fast."""
        if not self.use_matrix:
            return
        if "slow" in self.tiers:
            for label, proto in self.proto_keys.items():
                self.tiers["slow"].write(proto, self.registry.get_or_create(label), eta=self.cfg.tier("slow").eta)
        if "fast" in self.tiers:
            self.tiers["fast"].M *= 0.5

    # ------------------------------------------------------------------ utils
    def stats(self) -> dict:
        n = len(self.labels())
        footprint = self.dim * n  # prototype (bounded)
        if self.cfg.use_ema:
            footprint += self.dim * n
        if self.use_matrix:
            footprint += sum(t.M.size for t in self.tiers.values())
        return {
            "user_id": self.user_id,
            "modality": self.modality,
            "dim": self.dim,
            "n_labels": n,
            "labels": self.labels(),
            "hit_counts": dict(self.hit_counts),
            "use_matrix": self.use_matrix,
            "footprint_floats": int(footprint),
        }

    def snapshot(self, path: str) -> None:
        state = {
            "cfg": self.cfg,
            "user_id": self.user_id,
            "modality": self.modality,
            "proto_sum": self.proto_sum,
            "proto_count": self.proto_count,
            "proto_keys": self.proto_keys,
            "proto_ema": self.proto_ema,
            "hit_counts": self.hit_counts,
            "anchors": self.registry.anchors,
            "tiers": {name: tm.M for name, tm in self.tiers.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "ContinualPersonalizationMemory":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(dim=state["cfg"].dim, user_id=state["user_id"], modality=state["modality"], config=state["cfg"])
        obj.proto_sum = state["proto_sum"]
        obj.proto_count = state["proto_count"]
        obj.proto_keys = state["proto_keys"]
        obj.proto_ema = state.get("proto_ema", {k: v.copy() for k, v in obj.proto_keys.items()})
        obj.hit_counts = state["hit_counts"]
        obj.registry.anchors = state.get("anchors", {})
        for name, M in state.get("tiers", {}).items():
            if name in obj.tiers:
                obj.tiers[name].M = M
        return obj
