"""
Thí nghiệm cho báo cáo — kể 2 câu chuyện TRUNG THỰC:

  (A) CHỐNG QUÊN (continual learning): dạy danh tính tuần tự; đo accuracy trên danh tính
      ĐẦU TIÊN khi càng học thêm.  -> Fine-tune QUÊN (về 0); CPM & kNN GIỮ.

  (B) TĂNG TRƯỞNG BỘ NHỚ (dùng lâu dài): tiếp tục quan sát lại các danh tính nhiều lần;
      đo số float phải lưu.  -> kNN PHÌNH tuyến tính (lưu mọi quan sát); CPM & fine-tune BỊ CHẶN.
      (Trung thực: ở quy mô nhỏ kNN còn nhỏ hơn CPM; CPM chỉ thắng SAU điểm giao cắt.)

Kết luận: chỉ CPM đạt ĐỒNG THỜI không-quên + bộ nhớ-bị-chặn; fine-tune quên, kNN phình.

Chạy:  python -m experiments.retention
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

from experiments.baselines import CPMAdapter, EWCAdapter, FineTuneAdapter, KNNAdapter, NCMAdapter
from experiments.data import make_dataset, make_prototypes, sample_with_cos

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COLORS = {
    "CPM (NL)": "#1b9e77",
    "kNN (vector-DB)": "#7570b3",
    "Fine-tune head": "#d95f02",
    "NCM (nearest mean)": "#e7298a",
    "EWC (fine-tune+Fisher)": "#66a61e",
}


def run_forgetting(adapter, data) -> dict:
    """(A) Dạy tuần tự; sau mỗi danh tính, đo acc(đã học) và acc(danh tính đầu tiên)."""
    adapter.reset()
    overall, first_id = [], []
    for i, item in enumerate(data):
        for emb in item["train"]:
            adapter.teach(emb, item["label"])
        correct = total = 0
        for j in range(i + 1):
            for emb in data[j]["test"]:
                correct += adapter.predict(emb) == data[j]["label"]
                total += 1
        overall.append(correct / total)
        first_id.append(
            sum(adapter.predict(e) == data[0]["label"] for e in data[0]["test"]) / len(data[0]["test"])
        )
    return {"overall": overall, "first_id": first_id}


def run_memory_growth(adapters, dim, n_ids=20, max_obs=4000, cos=0.75, seed=7):
    """(B) Quan sát lại các danh tính lặp lại; ghi footprint theo tổng số quan sát."""
    protos = make_prototypes(n_ids, dim, seed)
    for a in adapters:
        a.reset()
    xs = []
    curves = {a.name: [] for a in adapters}
    obs = 0
    step = 0
    while obs < max_obs:
        for i in range(n_ids):
            emb = sample_with_cos(protos[i], cos, seed=step)
            for a in adapters:
                a.teach(emb, f"id_{i:02d}")
            obs += 1
            step += 1
            if obs % 100 == 0:
                xs.append(obs)
                for a in adapters:
                    curves[a.name].append(a.footprint())
            if obs >= max_obs:
                break
    return xs, curves


def main():
    dim = 512
    data = make_dataset(n_ids=20, n_train=8, n_test=5, dim=dim, cos_target=0.75, seed=42)
    x_ids = list(range(1, len(data) + 1))

    # (A) chống quên — 5 phương pháp
    forget_adapters = [CPMAdapter(dim), NCMAdapter(dim), KNNAdapter(dim), EWCAdapter(dim), FineTuneAdapter(dim)]
    forget = {a.name: run_forgetting(a, data) for a in forget_adapters}
    # (B) tăng trưởng bộ nhớ — đại diện: CPM (cố định) / NCM (bị chặn) / kNN (phình)
    xs_obs, growth = run_memory_growth([CPMAdapter(dim), NCMAdapter(dim), KNNAdapter(dim)], dim)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: acc trên danh tính đầu tiên
    for name, r in forget.items():
        ax1.plot(x_ids, r["first_id"], marker="o", ms=3, label=name, color=COLORS[name], alpha=0.8)
    ax1.set_title("(A) Retention: trên embedding tách biệt, MỌI PP giữ ~100%")
    ax1.set_xlabel("Số danh tính đã dạy (tuần tự)")
    ax1.set_ylabel("Accuracy (danh tính đầu)")
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # Panel B: footprint theo số quan sát
    for name in growth:
        ax2.plot(xs_obs, growth[name], label=name, color=COLORS[name])
    ax2.set_title("(B) Bộ nhớ khi dùng lâu dài")
    ax2.set_xlabel("Tổng số lần quan sát (dùng liên tục)")
    ax2.set_ylabel("Footprint (floats)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    # điểm giao cắt kNN vượt CPM
    cpm_fp = growth["CPM (NL)"][-1]
    cross = next((xs_obs[i] for i, v in enumerate(growth["kNN (vector-DB)"]) if v > cpm_fp), None)
    if cross:
        ax2.axvline(cross, ls="--", color="gray", alpha=0.6)
        ax2.text(cross, cpm_fp * 0.5, f"kNN vượt CPM\n~{cross} quan sát", fontsize=8, ha="center")

    fig.suptitle(
        "Continual Personalization — điểm phân biệt là BỘ NHỚ (B), không phải retention (A)", fontsize=12
    )
    fig.tight_layout()
    png = os.path.join(RESULTS_DIR, "retention.png")
    fig.savefig(png, dpi=130)

    # CSV (phần A)
    csv_path = os.path.join(RESULTS_DIR, "retention.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_ids", "method", "overall_acc", "first_id_acc"])
        for name, r in forget.items():
            for i in range(len(x_ids)):
                w.writerow([x_ids[i], name, f"{r['overall'][i]:.4f}", f"{r['first_id'][i]:.4f}"])

    # bảng tóm tắt
    print("\n=== (A) Sau khi dạy hết", len(data), "danh tính ===")
    print(f"{'Method':<20} {'Acc(đã học)':>12} {'Acc(id đầu)':>12}")
    for name, r in forget.items():
        print(f"{name:<20} {r['overall'][-1]:>12.2f} {r['first_id'][-1]:>12.2f}")
    print("\n=== (B) Footprint tại", xs_obs[-1], "quan sát ===")
    for name in growth:
        print(f"{name:<20} {growth[name][-1]:>12,} floats")
    if cross:
        print(f"\n-> kNN vượt footprint CPM sau ~{cross} lần quan sát (điểm giao cắt).")
    print(f"\nĐã lưu: {png}\nĐã lưu: {csv_path}")


if __name__ == "__main__":
    main()
