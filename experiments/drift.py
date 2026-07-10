"""
Appearance-drift experiment for Task 2.

Goal: show where CPM-EMA is different from plain NCM/mean prototypes. When a
person/object appearance moves over time, a uniform class mean is pulled toward
old observations. The EMA tier tracks the recent appearance while keeping a
bounded footprint.

TRUNG THỰC (đọc kỹ trước khi đưa vào báo cáo): đây là một **kịch bản dựng có kiểm soát**
(existence proof), KHÔNG phải bằng chứng "CPM chính xác hơn NCM nói chung". Cơ chế thắng chỉ
xuất hiện khi có **một danh tính gây nhiễu (distractor)** nằm gần vùng ngoại hình đã drift tới —
lúc đó trung bình cũ của NCM bị mẫu cũ kéo lùi và thua distractor, còn EMA bám mẫu gần đây nên
đúng. Nếu KHÔNG có distractor gây nhiễu thì mọi phương pháp đều đúng (không có khoảng cách).
`run_ablation()` quét ema_alpha / ema_weight / confusability để chứng minh kết quả **bền trên
một dải tham số**, không phải chọn đúng một điểm may mắn — và cũng cho thấy khi distractor trùng
hẳn với ngoại hình mới (confusability→1) thì hai danh tính nhập một, ai cũng thua (bài toán vô
nghiệm), đúng như kỳ vọng.

Kết luận đúng để trích dẫn: EMA = *chính xác khi drift* VÀ *bộ nhớ bị chặn*; NCM = chặn nhưng
thua drift; kNN = chính xác nhưng phình. Góc phần tư "chính xác + bị chặn" là chỗ EMA thắng thật.

Run:
  python -m experiments.drift            # figure chính + ablation
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.baselines import CPMEMAAdapter, KNNAdapter, NCMAdapter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COLORS = {
    "NCM (nearest mean)": "#e7298a",
    "CPM-EMA (NL)": "#1b9e77",
    "kNN (vector-DB)": "#7570b3",
}


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-8)


def _orthonormal_pair(rng: np.random.Generator, dim: int) -> tuple[np.ndarray, np.ndarray]:
    a = _unit(rng.standard_normal(dim))
    b = rng.standard_normal(dim)
    b = _unit(b - (b @ a) * a)
    return a, b


def _mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return _unit((1.0 - t) * a + t * b)


def make_drift_stream(
    n_pairs: int = 12,
    n_steps: int = 16,
    dim: int = 128,
    drift: float = 1.0,
    confusability: float = 0.585,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, str]], list[tuple[np.ndarray, str]]]:
    """Create a stream with drifting identities and stable distractors.

    Each pair contains:
      - drift_i: observations move from old vector a to current vector b.
      - distractor_i: stable identity placed on the old->target arc at position
        `confusability` (0 = at old, 1 = exactly at current appearance).

    `confusability` điều khiển distractor gần ngoại hình-mới tới đâu (mặc định 0.585 ~ kịch bản
    gốc). Cao hơn = khó hơn cho NCM (mean cũ càng dễ thua distractor); →1 = distractor trùng
    ngoại hình mới nên hai danh tính nhập một, ai cũng thua.
    """
    rng = np.random.default_rng(seed)
    train: list[tuple[np.ndarray, str]] = []
    test: list[tuple[np.ndarray, str]] = []

    for i in range(n_pairs):
        old, target = _orthonormal_pair(rng, dim)
        current = _mix(old, target, drift)
        distractor = _mix(old, target, confusability)

        drift_label = f"drift_{i:02d}"
        distractor_label = f"distractor_{i:02d}"

        for step in range(n_steps):
            t = drift * (step / max(1, n_steps - 1))
            train.append((_mix(old, current, t), drift_label))
            train.append((distractor, distractor_label))

        test.append((current, drift_label))
        test.append((distractor, distractor_label))

    return train, test


def accuracy(adapter, test: list[tuple[np.ndarray, str]]) -> float:
    return float(np.mean([adapter.predict(x) == y for x, y in test]))


def run_once(drift: float, dim: int = 128, seed: int = 0) -> dict[str, float]:
    train, test = make_drift_stream(dim=dim, drift=drift, seed=seed)
    adapters = [NCMAdapter(dim), CPMEMAAdapter(dim), KNNAdapter(dim)]
    row: dict[str, float] = {}
    for adapter in adapters:
        adapter.reset()
        for emb, label in train:
            adapter.teach(emb, label)
        row[f"{adapter.name}:acc"] = accuracy(adapter, test)
        row[f"{adapter.name}:footprint"] = float(adapter.footprint())
    return row


def _acc_after_training(make_adapter, train, test) -> float:
    a = make_adapter()
    a.reset()
    for emb, label in train:
        a.teach(emb, label)
    return accuracy(a, test)


def run_ablation(dim: int = 128, n_pairs: int = 10, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Sensitivity ablation: chứng minh khoảng cách EMA>NCM bền trên một DẢI tham số,
    không phải một điểm may mắn. 3 quét: ema_alpha, ema_weight, confusability."""

    def avg(make_adapter, **stream_kw) -> float:
        vals = []
        for s in seeds:
            train, test = make_drift_stream(n_pairs=n_pairs, dim=dim, seed=s, **stream_kw)
            vals.append(_acc_after_training(make_adapter, train, test))
        return float(np.mean(vals))

    alphas = np.linspace(0.0, 0.95, 11)          # retention EMA: thấp = bám mẫu mới nhanh
    weights = np.linspace(0.0, 1.0, 11)          # trọng số trộn slow/fast (0 = thuần NCM)
    confus = np.linspace(0.30, 0.95, 14)         # distractor gần ngoại hình mới tới đâu

    res = {
        "alphas": alphas,
        "ema_vs_alpha": [avg(lambda a=a: CPMEMAAdapter(dim, ema_alpha=a), drift=1.0) for a in alphas],
        "ncm_ref": avg(lambda: NCMAdapter(dim), drift=1.0),
        "weights": weights,
        "ema_vs_weight": [avg(lambda w=w: CPMEMAAdapter(dim, ema_weight=w), drift=1.0) for w in weights],
        "confus": confus,
        "ncm_vs_confus": [avg(lambda: NCMAdapter(dim), drift=1.0, confusability=c) for c in confus],
        "ema_vs_confus": [avg(lambda: CPMEMAAdapter(dim), drift=1.0, confusability=c) for c in confus],
        "knn_vs_confus": [avg(lambda: KNNAdapter(dim), drift=1.0, confusability=c) for c in confus],
    }

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    green, pink, purple = COLORS["CPM-EMA (NL)"], COLORS["NCM (nearest mean)"], COLORS["kNN (vector-DB)"]

    a1.plot(alphas, res["ema_vs_alpha"], marker="o", ms=4, color=green, label="CPM-EMA")
    a1.axhline(res["ncm_ref"], ls="--", color=pink, label="NCM")
    a1.set_title("(A) Độ nhạy theo ema_alpha\n(thấp = bám mẫu mới; cao → về NCM)")
    a1.set_xlabel("ema_alpha (retention EMA)")
    a1.set_ylabel("Accuracy (drift mạnh)")

    a2.plot(weights, res["ema_vs_weight"], marker="o", ms=4, color=green, label="CPM-EMA")
    a2.axhline(res["ncm_ref"], ls="--", color=pink, label="NCM")
    a2.set_title("(B) Độ nhạy theo ema_weight\n(0 = thuần NCM; vùng cao rộng đều thắng)")
    a2.set_xlabel("ema_weight (trọng số tầng nhanh)")

    a3.plot(confus, res["ema_vs_confus"], marker="o", ms=4, color=green, label="CPM-EMA")
    a3.plot(confus, res["ncm_vs_confus"], marker="s", ms=4, color=pink, label="NCM")
    a3.plot(confus, res["knn_vs_confus"], marker="^", ms=4, color=purple, label="kNN")
    a3.set_title("(C) Theo độ gây nhiễu distractor\n(→1: hai danh tính nhập một, ai cũng thua)")
    a3.set_xlabel("confusability (distractor gần ngoại hình mới)")

    for ax in (a1, a2, a3):
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Ablation độ nhạy drift — khoảng cách EMA>NCM bền trên một DẢI tham số (không phải điểm may mắn)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "drift_ablation.png")
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")
    return res


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    drifts = np.linspace(0.0, 1.0, 11)
    rows = []
    for d in drifts:
        row = {"drift": float(d)}
        row.update(run_once(float(d)))
        rows.append(row)

    csv_path = os.path.join(RESULTS_DIR, "drift.csv")
    headers = ["drift"]
    for name in COLORS:
        headers.extend([f"{name}:acc", f"{name}:footprint"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, color in COLORS.items():
        ax1.plot(
            [r["drift"] for r in rows],
            [r[f"{name}:acc"] for r in rows],
            marker="o",
            ms=4,
            label=name,
            color=color,
        )
    ax1.set_title("(A) Appearance drift: EMA tracks recent appearance")
    ax1.set_xlabel("Drift strength")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    last = rows[-1]
    names = list(COLORS)
    ax2.bar(names, [last[f"{n}:footprint"] for n in names], color=[COLORS[n] for n in names])
    ax2.set_title("(B) Footprint at strongest drift")
    ax2.set_ylabel("Stored floats")
    ax2.tick_params(axis="x", rotation=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "drift.png")
    fig.savefig(png_path, dpi=130)

    print("\n=== Appearance drift @ strongest drift ===")
    print(f"{'Method':<22} {'Accuracy':>10} {'Footprint':>12}")
    for name in names:
        print(f"{name:<22} {last[f'{name}:acc']:>10.2f} {int(last[f'{name}:footprint']):>12,}")
    print(f"\nSaved: {png_path}")
    print(f"Saved: {csv_path}")

    # Ablation độ nhạy (trung thực: kết quả bền trên một dải tham số, không phải điểm may mắn)
    ab = run_ablation()
    print("\n=== Ablation (drift mạnh) ===")
    print(f"  NCM tham chiếu           : {ab['ncm_ref']:.2f}")
    print(f"  CPM-EMA theo ema_alpha   : min={min(ab['ema_vs_alpha']):.2f}  max={max(ab['ema_vs_alpha']):.2f}")
    print(f"  CPM-EMA theo ema_weight  : min={min(ab['ema_vs_weight']):.2f}  max={max(ab['ema_vs_weight']):.2f}")
    print(f"  confusability→1 (EMA)    : {ab['ema_vs_confus'][0]:.2f} -> {ab['ema_vs_confus'][-1]:.2f}")


if __name__ == "__main__":
    main()
