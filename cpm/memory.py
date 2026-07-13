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


# Chuẩn hoá véc-tơ về độ dài 1 để phép so cosine công bằng.
# Vào: một dãy số v (dấu vân tay số / embedding). Ra: cùng hướng nhưng độ dài = 1.
# (eps là số rất nhỏ cộng thêm để tránh chia cho 0 khi véc-tơ toàn số 0.)
def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)  # ép về mảng số thực, 1 hàng
    return v / (np.linalg.norm(v) + eps)  # chia cho độ dài -> véc-tơ đơn vị


# Sổ "mỏ neo" (anchor) cho ma trận liên kết — CHỈ dùng khi bật thành phần tuỳ chọn này.
# Mỗi nhãn (vd "Lan") được gán một véc-tơ cố định sinh TỪ TÊN -> cùng tên luôn ra cùng véc-tơ.
class LabelRegistry:
    """Sổ nhãn: mỗi nhãn -> một anchor (value) là vector đơn vị sinh TẤT ĐỊNH theo tên nhãn."""

    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.seed = seed
        self.anchors: dict[str, np.ndarray] = {}

    # Lấy mỏ neo của nhãn; chưa có thì tạo mới (TẤT ĐỊNH: cùng tên -> cùng véc-tơ).
    def get_or_create(self, label: str) -> np.ndarray:
        if label not in self.anchors:
            # băm tên nhãn thành một số -> làm hạt giống ngẫu nhiên -> luôn tái lập được
            h = int.from_bytes(hashlib.sha256(f"{self.seed}:{label}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(h)
            self.anchors[label] = _unit(rng.standard_normal(self.dim))  # véc-tơ ngẫu nhiên đã chuẩn hoá
        return self.anchors[label]


# Một tầng "ma trận liên kết" M (thành phần TUỲ CHỌN, mặc định TẮT).
# Cập nhật theo delta-rule: M = αM + η·error·kᵀ — chỉ ghi phần SAI LỆCH (error = a - M·k)
# nên KHÔNG đè kiến thức cũ -> chống quên.
class TierMemory:
    """Một tầng associative memory cập nhật bằng delta-rule (thành phần tuỳ chọn)."""

    def __init__(self, dim: int, cfg: TierConfig):
        self.cfg = cfg
        self.M = np.zeros((dim, dim), dtype=np.float64)  # ma trận dim×dim, khởi tạo toàn số 0

    # Dạy ma trận: khi thấy khoá k thì nên trả ra giá trị a. Không trả về gì.
    def write(self, k: np.ndarray, a: np.ndarray, eta: float | None = None) -> None:
        eta = self.cfg.eta if eta is None else eta  # không truyền eta thì lấy tốc độ học mặc định
        error = a - self.M @ k  # phần SAI LỆCH: giá trị mong muốn a trừ đi cái ma trận đang đoán (M·k)
        # delta-rule: giữ lại αM (retention) rồi cộng thêm phần sửa lỗi -> chỉ ghi vào chỗ sai
        self.M = self.cfg.alpha * self.M + eta * np.outer(error, k)

    # Truy hồi: đưa khoá q vào ma trận để đoán giá trị liên kết (phép nhân M·q).
    def predict(self, q: np.ndarray) -> np.ndarray:
        return self.M @ q


# LỚP CHÍNH của đồ án: bộ nhớ cá nhân hoá học liên tục (CPM) cho MỘT modality.
# Nó nhớ "dấu vân tay số này tên gì" và nhận ra khi gặp lại; học thêm mà KHÔNG quên cái cũ.
# modality = loại đối tượng: face (mặt) hay object (đồ vật) — 2 không gian khác nhau nên tách riêng.
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
        # Nhận cấu hình sẵn, hoặc tự tạo cấu hình mặc định theo số chiều dim.
        self.cfg = config or CPMConfig(dim=dim)
        self.dim = self.cfg.dim
        self.user_id = user_id
        self.modality = modality  # "face" (mặt) hay "object" (đồ vật) — mỗi loại một bộ nhớ riêng

        # Xương sống: prototype trung-bình-thật cho mỗi nhãn
        # prototype = dấu vân tay TRUNG BÌNH của mỗi nhãn (giống baseline NCM) — xương sống nhận diện.
        # Để tính trung bình "online", ta lưu TỔNG các véc-tơ và ĐẾM số lần, khi cần thì chia ra.
        self.proto_sum: dict[str, np.ndarray] = {}  # tổng các dấu vân tay đã dạy của mỗi nhãn
        self.proto_count: dict[str, int] = {}  # số lần đã dạy mỗi nhãn
        # proto_keys = unit(mean): dấu vân tay trung bình ĐÃ chuẩn hoá, dùng để so khi nhận diện.
        self.proto_keys: dict[str, np.ndarray] = {}  # unit(mean) — dùng để recall
        # proto_ema = tầng NHANH: bám ngoại hình GẦN ĐÂY, theo kịp khi ngoại hình đổi dần (drift).
        self.proto_ema: dict[str, np.ndarray] = {}   # tầng nhanh: prototype EMA, dùng khi cfg.use_ema=True
        self.hit_counts: dict[str, int] = {}  # đếm số lần mỗi nhãn được xác nhận (để quyết định bền hoá)

        # Thành phần tuỳ chọn: ma trận liên kết đa tầng
        # Mặc định TẮT (use_associative_matrix=False) vì thí nghiệm cho thấy nó giới hạn sức chứa
        # ~dim và yếu khi các danh tính giống nhau; chỉ bật để nghiên cứu/ablation.
        self.use_matrix = self.cfg.use_associative_matrix
        self.registry = LabelRegistry(self.dim, self.cfg.anchor_seed)
        self.tiers: dict[str, TierMemory] = (
            {t.name: TierMemory(self.dim, t) for t in self.cfg.tiers} if self.use_matrix else {}
        )

    # Trả về danh sách tất cả nhãn (tên người/đồ) đang được nhớ.
    def labels(self) -> list[str]:
        return list(self.proto_keys.keys())

    # ------------------------------------------------------------------ write
    # write() = DẠY: đưa vào một dấu vân tay (key) kèm nhãn (tên người/đồ) -> cập nhật bộ nhớ.
    # eta_scale: hệ số nhân tốc độ học của ma trận (dùng khi correct để ghi mạnh hơn). Không trả về gì.
    def write(self, key, label: str, eta_scale: float = 1.0) -> None:
        """Dạy 1 cặp (embedding, nhãn)."""
        k = _unit(key)  # chuẩn hoá dấu vân tay về độ dài 1 trước khi ghi

        # cập nhật prototype trung-bình-thật (online, bounded)
        if label in self.proto_sum:
            # nhãn đã có: cộng dồn vào tổng và tăng bộ đếm
            self.proto_sum[label] += k
            self.proto_count[label] += 1
        else:
            # nhãn mới: tạo ô nhớ RIÊNG -> dạy người mới không đè người cũ -> chống quên
            self.proto_sum[label] = k.copy()
            self.proto_count[label] = 1
        # prototype = trung bình (tổng / số lần) rồi chuẩn hoá lại về độ dài 1
        self.proto_keys[label] = _unit(self.proto_sum[label] / self.proto_count[label])
        # cập nhật tầng NHANH (EMA): trộn ký ức cũ với mẫu mới; a nhỏ -> bám mẫu mới nhanh hơn
        if label in self.proto_ema:
            a = self.cfg.ema_alpha
            self.proto_ema[label] = _unit(a * self.proto_ema[label] + (1.0 - a) * k)
        else:
            self.proto_ema[label] = k.copy()  # lần đầu: EMA chính là mẫu vừa thấy
        self.hit_counts[label] = self.hit_counts.get(label, 0) + 1  # đếm thêm 1 lần xác nhận nhãn này

        # (tuỳ chọn) ghi vào ma trận liên kết đa tầng — chỉ chạy khi bật use_matrix
        if self.use_matrix:
            a = self.registry.get_or_create(label)  # lấy mỏ neo cố định của nhãn
            # ghi cặp (dấu vân tay k -> mỏ neo a) vào cả 3 tầng fast/medium/slow
            for t in self.cfg.tiers:
                self.tiers[t.name].write(k, a, eta=t.eta * eta_scale)
            # đã xác nhận đủ nhiều lần -> bền hoá prototype vào tầng slow (trí nhớ dài hạn)
            if self.hit_counts[label] >= self.cfg.consolidate_hit_count and "slow" in self.tiers:
                self.tiers["slow"].write(self.proto_keys[label], a, eta=self.cfg.tier("slow").eta)

    # ----------------------------------------------------------------- recall
    # recall() = NHẬN DIỆN: đưa vào một dấu vân tay câu hỏi -> trả về nhãn giống nhất kèm điểm.
    # So độ giống (cosine, từ -1 đến 1; gần 1 = rất giống) giữa câu hỏi và từng prototype.
    # "known" = True nếu điểm ≥ recall_threshold (ngưỡng quen-lạ): coi là người/đồ QUEN.
    def recall(self, query_key, top_k: int = 1, mode: str | None = None) -> list[dict]:
        """
        Nhận diện. `mode`:
          - "proto"  (mặc định): argmax cos(query, prototype) — bền, bộ nhớ bị chặn.
          - "assoc"  : associative memory NL (cần bật ma trận).
          - "hybrid" : kết hợp assoc + proto (cần bật ma trận).
        "known"/confidence luôn = cos(query, prototype) (cổng novelty embedding-space).

        Trả list dict {label, confidence, proto_score, assoc_score, tier, known}.
        """
        mode = mode or self.cfg.default_recall_mode  # không chỉ định thì dùng chế độ mặc định ("proto")
        q = _unit(query_key)  # chuẩn hoá câu hỏi để so cosine công bằng
        labels = self.labels()
        if not labels:
            # bộ nhớ còn rỗng (chưa dạy gì) -> trả về kết quả "chưa biết"
            return [{"label": None, "confidence": 0.0, "proto_score": 0.0,
                     "assoc_score": 0.0, "tier": None, "known": False}]

        # điểm cosine giữa câu hỏi q và prototype của TỪNG nhãn (chỉ cần tích vô hướng vì đều đã chuẩn hoá)
        proto = np.array([float(self.proto_keys[l] @ q) for l in labels])
        # nếu bật EMA: tính thêm điểm cosine với tầng NHANH (bám ngoại hình gần đây)
        ema = None
        if self.cfg.use_ema:
            ema = np.array([float(self.proto_ema.get(l, self.proto_keys[l]) @ q) for l in labels])

        # (tuỳ chọn) tính điểm bằng ma trận liên kết đa tầng — chỉ khi mode là "assoc"/"hybrid"
        assoc = None
        if mode in ("assoc", "hybrid"):
            if not self.use_matrix:
                mode = "proto"  # không có ma trận -> fallback
            else:
                A = np.stack([self.registry.get_or_create(l) for l in labels])  # xếp các mỏ neo thành ma trận
                vhat = np.zeros(self.dim)
                # mỗi tầng đoán ra một véc-tơ; cộng lại theo trọng số (tầng nào tự tin hơn thì góp nhiều hơn)
                for t in self.cfg.tiers:
                    vt = _unit(self.tiers[t.name].predict(q))
                    conf_t = float(np.max(A @ vt))  # độ tự tin của tầng = điểm giống mỏ neo cao nhất
                    vhat += t.weight * max(conf_t, 0.0) ** 2 * vt
                vhat = _unit(vhat)
                assoc = A @ vhat  # điểm giống giữa dự đoán tổng hợp và từng mỏ neo

        # chọn điểm cuối cùng theo chế độ:
        if mode == "assoc":
            score = assoc  # chỉ dùng ma trận liên kết
        elif mode == "hybrid":
            score = self.cfg.hybrid_w_assoc * assoc + self.cfg.hybrid_w_proto * proto  # trộn ma trận + prototype
        else:
            score = proto  # MẶC ĐỊNH: dùng prototype trung bình (xương sống nhận diện)
            if ema is not None:
                # trộn thêm tầng nhanh EMA: w là trọng số của EMA (kẹp trong khoảng [0,1])
                w = float(np.clip(self.cfg.ema_weight, 0.0, 1.0))
                score = (1.0 - w) * proto + w * ema

        # sắp xếp điểm giảm dần rồi lấy top_k nhãn giống nhất (dấu trừ để đảo về thứ tự giảm dần)
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
                    "known": float(score[i]) >= self.cfg.recall_threshold,  # quen hay lạ: điểm có ≥ ngưỡng không?
                }
            )
        return out

    # ---------------------------------------------------------------- correct
    # correct() = SỬA SAI: người dùng chỉ ra nhãn ĐÚNG -> gọi write với eta mạnh hơn để ghi đậm mẫu này.
    def correct(self, query_key, new_label: str) -> None:
        """Người dùng sửa sai -> ghi mẫu vào prototype nhãn đúng (và ma trận nếu bật)."""
        self.write(query_key, new_label, eta_scale=self.cfg.correct_eta_boost)

    # ------------------------------------------------------------ consolidate
    # consolidate() = BỀN HOÁ: chép prototype vào tầng slow (nhớ lâu) và làm nhạt tầng fast.
    # Chỉ có tác dụng khi bật ma trận; không bật thì thoát ngay.
    def consolidate(self) -> None:
        """Bền hoá (chỉ có tác dụng khi bật ma trận): replay prototype vào tầng slow; xả tầng fast."""
        if not self.use_matrix:
            return
        if "slow" in self.tiers:
            # ghi lại từng prototype vào tầng slow để củng cố trí nhớ dài hạn
            for label, proto in self.proto_keys.items():
                self.tiers["slow"].write(proto, self.registry.get_or_create(label), eta=self.cfg.tier("slow").eta)
        if "fast" in self.tiers:
            self.tiers["fast"].M *= 0.5  # làm nhạt tầng nhanh (giảm ảnh hưởng của ký ức tạm)

    # ------------------------------------------------------------------ utils
    # stats() = báo cáo tình trạng bộ nhớ: số nhãn, footprint (dung lượng), có bật ma trận không...
    # footprint = dung lượng bộ nhớ = số float phải lưu (thiết bị đeo cần bộ nhớ nhỏ, bị chặn).
    def stats(self) -> dict:
        n = len(self.labels())
        footprint = self.dim * n  # prototype (bounded)
        if self.cfg.use_ema:
            footprint += self.dim * n  # bật EMA -> lưu thêm 1 prototype nhanh cho mỗi nhãn
        if self.use_matrix:
            footprint += sum(t.M.size for t in self.tiers.values())  # cộng thêm kích thước các ma trận tầng
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

    # snapshot() = LƯU toàn bộ trạng thái bộ nhớ ra file (path) để dùng lại sau (dạng pickle).
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

    # load() = KHÔI PHỤC bộ nhớ từ file đã snapshot; trả về một CPM mới đã nạp đầy đủ dữ liệu.
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
