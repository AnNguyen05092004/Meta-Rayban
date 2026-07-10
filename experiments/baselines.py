"""
Các phương pháp so sánh cho thí nghiệm retention, cùng một interface:

    adapter.teach(embedding, label)
    adapter.predict(embedding) -> label
    adapter.footprint() -> số float đang lưu (đo chi phí bộ nhớ)
    adapter.reset()

- CPMAdapter      : lõi NL của nhóm (associative memory + delta-rule + 3 tầng).
- KNNAdapter      : kNN/vector-DB (lưu mọi embedding) — retention tốt nhưng bộ nhớ PHÌNH.
- FineTuneAdapter : linear head học tuần tự (chỉ dữ liệu lớp mới) — CATASTROPHIC FORGETTING.
"""

from __future__ import annotations

import numpy as np

from cpm import CPMConfig, ContinualPersonalizationMemory


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-8)


def _ce_grad(W: np.ndarray, x: np.ndarray, y: int) -> np.ndarray:
    """
    Gradient (softmax cross-entropy) cho linear head, CÓ logit 'background'=0 để tránh suy biến
    khi chỉ có 1 lớp (nếu không, softmax 1-logit -> gradient 0 -> lớp đầu không được học).
    Trả g (len C) = softmax_real - onehot(y).
    """
    z = np.concatenate([W @ x, [0.0]])
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    g = p[:-1].copy()
    g[y] -= 1.0
    return g


class CPMAdapter:
    name = "CPM (NL)"

    def __init__(self, dim: int, use_ema: bool = False, ema_alpha: float = 0.35, ema_weight: float = 0.65):
        self.dim = dim
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.ema_weight = ema_weight
        self.reset()

    def reset(self):
        self.cpm = ContinualPersonalizationMemory(
            dim=self.dim,
            config=CPMConfig(
                dim=self.dim,
                use_ema=self.use_ema,
                ema_alpha=self.ema_alpha,
                ema_weight=self.ema_weight,
            ),
        )

    def teach(self, emb, label):
        self.cpm.write(emb, label)

    def predict(self, emb):
        return self.cpm.recall(emb)[0]["label"]

    def footprint(self) -> int:
        return self.cpm.stats()["footprint_floats"]


class CPMEMAAdapter(CPMAdapter):
    name = "CPM-EMA (NL)"

    def __init__(self, dim: int, ema_alpha: float = 0.35, ema_weight: float = 0.65):
        super().__init__(dim, use_ema=True, ema_alpha=ema_alpha, ema_weight=ema_weight)


class KNNAdapter:
    name = "kNN (vector-DB)"

    def __init__(self, dim: int):
        self.dim = dim
        self.reset()

    def reset(self):
        self.keys: list[np.ndarray] = []
        self.labels: list[str] = []

    def teach(self, emb, label):
        self.keys.append(_unit(emb))
        self.labels.append(label)

    def predict(self, emb):
        q = _unit(emb)
        sims = np.array([k @ q for k in self.keys])
        return self.labels[int(np.argmax(sims))]

    def footprint(self) -> int:
        return len(self.keys) * self.dim  # PHÌNH theo số ảnh


class NCMAdapter:
    """Nearest Class Mean — lưu 1 vector trung bình/lớp (bounded). Baseline mạnh & đơn giản."""

    name = "NCM (nearest mean)"

    def __init__(self, dim: int):
        self.dim = dim
        self.reset()

    def reset(self):
        self.sum: dict[str, np.ndarray] = {}
        self.count: dict[str, int] = {}

    def teach(self, emb, label):
        k = _unit(emb)
        self.sum[label] = self.sum.get(label, np.zeros(self.dim)) + k
        self.count[label] = self.count.get(label, 0) + 1

    def predict(self, emb):
        q = _unit(emb)
        best, best_s = None, -np.inf
        for l in self.sum:
            mean = _unit(self.sum[l] / self.count[l])
            s = float(mean @ q)
            if s > best_s:
                best_s, best = s, l
        return best

    def footprint(self) -> int:
        return len(self.sum) * self.dim  # bounded: 1 mean/lớp


class EWCAdapter:
    """
    Elastic Weight Consolidation (bản gọn): fine-tune head + phạt Fisher để giảm quên.
    Consolidate Fisher mỗi khi chuyển sang nhãn mới (mỗi nhãn = 1 'task').
    """

    name = "EWC (fine-tune+Fisher)"

    def __init__(self, dim: int, lr: float = 0.5, epochs: int = 5, lam: float = 30.0):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.reset()

    def reset(self):
        self.l2i: dict[str, int] = {}
        self.i2l: list[str] = []
        self.W = np.zeros((0, self.dim))
        self.W_star = np.zeros((0, self.dim))
        self.F = np.zeros((0, self.dim))
        self.cur = None
        self.buf: list[np.ndarray] = []

    def _ensure(self, label):
        if label not in self.l2i:
            self.l2i[label] = len(self.i2l)
            self.i2l.append(label)
            self.W = np.vstack([self.W, np.zeros((1, self.dim))])
            self.W_star = np.vstack([self.W_star, np.zeros((1, self.dim))])
            self.F = np.vstack([self.F, np.zeros((1, self.dim))])

    def _consolidate(self):
        if self.cur is None or not self.buf:
            return
        fisher = np.zeros_like(self.W)
        y = self.l2i[self.cur]
        for x in self.buf:
            fisher += np.outer(_ce_grad(self.W, x, y), x) ** 2
        self.F += fisher / max(1, len(self.buf))
        self.W_star = self.W.copy()
        self.buf = []

    def teach(self, emb, label):
        x = _unit(emb)
        if label != self.cur:
            self._consolidate()      # kết thúc task cũ -> ghi nhớ Fisher
            self.cur = label
        self._ensure(label)
        y = self.l2i[label]
        for _ in range(self.epochs):
            g = _ce_grad(self.W, x, y)
            grad = np.outer(g, x) + self.lam * self.F * (self.W - self.W_star)
            self.W -= self.lr * grad
        self.buf.append(x)

    def predict(self, emb):
        x = _unit(emb)
        return self.i2l[int(np.argmax(self.W @ x))]

    def footprint(self) -> int:
        return int(self.W.size)


class FineTuneAdapter:
    """Linear softmax head, học ONLINE chỉ trên dữ liệu lớp mới -> quên lớp cũ."""

    name = "Fine-tune head"

    def __init__(self, dim: int, lr: float = 0.5, epochs: int = 5):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.reset()

    def reset(self):
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: list[str] = []
        self.W = np.zeros((0, self.dim))

    def _ensure_class(self, label):
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.idx_to_label)
            self.idx_to_label.append(label)
            self.W = np.vstack([self.W, np.zeros((1, self.dim))])

    def teach(self, emb, label):
        self._ensure_class(label)
        x = _unit(emb)
        y = self.label_to_idx[label]
        for _ in range(self.epochs):
            g = _ce_grad(self.W, x, y)
            self.W -= self.lr * np.outer(g, x)   # cập nhật MỌI hàng -> lớp cũ có thể trôi đi

    def predict(self, emb):
        x = _unit(emb)
        return self.idx_to_label[int(np.argmax(self.W @ x))]

    def footprint(self) -> int:
        return int(self.W.size)
