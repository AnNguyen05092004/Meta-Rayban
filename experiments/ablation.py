"""
Bộ thí nghiệm nghiên cứu cho CPM (chạy synthetic, verify được ngay).

Chạy ở CHẾ ĐỘ ÉP TẢI (dim nhỏ, nhiều danh tính, có nhiễu/impostor) để LỘ khác biệt
giữa các thành phần — ở dim=512 mọi thứ gần như hoàn hảo nên không phân biệt được.

Thí nghiệm:
  1) capacity   : accuracy theo số danh tính, cho nhiều dim  -> giới hạn ~ dim.
  2) tiers      : fast-only vs persistent-1tier vs 3-tier    -> vai trò tầng chậm/đa tần số.
  3) recall_mode: assoc (ma trận) vs proto (prototype-NN)    -> giới hạn của associative memory.
  4) robustness : accuracy theo nhiễu intra-class (cos)      -> độ bền.
  5) openset    : ROC quen/lạ + biện minh ngưỡng 0.35.

Chạy:  python -m experiments.ablation
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ép luồng in ra màn hình dùng bảng mã UTF-8 để tiếng Việt (có dấu) không bị lỗi font trên Windows.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

# "Agg" = chế độ vẽ THẲNG ra file ảnh (.png), không cần mở cửa sổ -> chạy được cả trên máy không màn hình.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpm import CPMConfig, ContinualPersonalizationMemory, TierConfig
from experiments.baselines import FineTuneAdapter, NCMAdapter
from experiments.data import make_correlated_prototypes, make_prototypes, sample_with_cos
from experiments.metrics import open_set_summary, roc_points

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# Tạo một bộ nhớ CPM để thí nghiệm, theo "biến thể" (variant) cấu hình tầng:
#   - "fast-only"        = chỉ 1 tầng nhanh (bám mẫu mới nhanh nhưng quên nhanh).
#   - "persistent-1tier" = chỉ 1 tầng chậm/bền (khó quên).
#   - "3-tier"           = đủ 3 tầng fast/medium/slow (mặc định).
# consolidate=False -> tắt bền hoá để quan sát vai trò riêng của tầng chậm.
def make_cpm(dim: int, variant: str = "3-tier", consolidate: bool = True):
    # Bộ thí nghiệm này PHÂN TÍCH ma trận liên kết -> bật use_associative_matrix.
    # (CPM mặc định của sản phẩm dùng prototype, ma trận TẮT.)
    if variant == "fast-only":
        tiers = [TierConfig("fast", alpha=0.90, eta=1.0, weight=1.0)]
        cfg = CPMConfig(dim=dim, tiers=tiers, use_associative_matrix=True)
    elif variant == "persistent-1tier":
        tiers = [TierConfig("slow", alpha=1.0, eta=0.5, weight=1.0)]
        cfg = CPMConfig(dim=dim, tiers=tiers, use_associative_matrix=True)
    else:  # 3-tier
        cfg = CPMConfig(dim=dim, use_associative_matrix=True)
    if not consolidate:
        cfg.consolidate_hit_count = 10**9  # đặt ngưỡng cực lớn = gần như KHÔNG bao giờ bền hoá
    return ContinualPersonalizationMemory(dim=dim, config=cfg)


# Đo accuracy: với mỗi danh tính, sinh n_test ảnh truy vấn, hỏi CPM đoán rồi đếm số lần đúng.
# mode = cách nhận diện: "assoc" (ma trận liên kết), "proto" (prototype), "hybrid" (trộn hai cái).
def eval_acc(cpm, protos, n_ids, n_test=5, cos=0.7, mode="assoc", seed0=90000) -> float:
    correct = total = 0
    for i in range(n_ids):
        for j in range(n_test):
            # tạo 1 ảnh truy vấn của người i (giống gốc ở mức cos); seed khác nhau mỗi ảnh.
            q = sample_with_cos(protos[i], cos, seed=seed0 + 1000 * i + j)
            # recall trả danh sách ứng viên; [0] là đoán tốt nhất; so nhãn với "id{i}".
            correct += cpm.recall(q, mode=mode)[0]["label"] == f"id{i}"
            total += 1
    return correct / total


# --------------------------------------------------------------- 1) capacity
# (1) SỨC CHỨA: nhồi ngày càng nhiều danh tính vào ma trận rồi đo accuracy.
# Kỳ vọng: accuracy tụt mạnh khi số danh tính vượt ~ số chiều (dim) -> ma trận có giới hạn chứa.
def exp_capacity(ax):
    dims = [32, 64, 128]
    for dim in dims:
        nmax = int(2.5 * dim)  # thử tới 2.5 lần dim để thấy rõ điểm sụp
        checks = list(range(dim // 4, nmax + 1, max(1, dim // 4)))  # các mốc số danh tính để chấm điểm
        protos = make_prototypes(nmax, dim, seed=1)
        cpm = make_cpm(dim, "3-tier")
        xs, ys = [], []
        for i in range(nmax):
            cpm.write(protos[i], f"id{i}")  # dạy thêm 1 danh tính
            if (i + 1) in checks:
                xs.append((i + 1) / dim)  # chuẩn hoá theo dim
                ys.append(eval_acc(cpm, protos, i + 1))
        ax.plot(xs, ys, marker="o", ms=3, label=f"dim={dim}")
    ax.axvline(1.0, ls="--", color="gray", alpha=0.6)  # vạch N = dim (ngưỡng sức chứa lý thuyết)
    ax.text(1.02, 0.15, "N = dim", fontsize=8)
    ax.set_title("(1) Sức chứa: accuracy theo #danh tính")
    ax.set_xlabel("Số danh tính / dim")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()


# ------------------------------------------------------------------ 2) tiers
# (2) VAI TRÒ TỪNG TẦNG: so 3 biến thể cấu hình tầng, đo độ nhớ danh tính ĐẦU khi học thêm.
# Cho thấy tầng chậm/bền giúp GIỮ nhãn cũ tốt hơn là chỉ có một tầng nhanh.
def exp_tiers(ax):
    dim, N = 96, 240
    protos = make_prototypes(N, dim, seed=2)
    variants = ["fast-only", "persistent-1tier", "3-tier"]
    checks = list(range(10, N + 1, 15))
    for v in variants:
        cpm = make_cpm(dim, v)
        xs, ys = [], []
        for i in range(N):
            cpm.write(protos[i], f"id{i}")
            if (i + 1) in checks:
                # accuracy trên danh tính ĐẦU TIÊN (đo quên)
                acc0 = eval_acc_single(cpm, protos, 0)
                xs.append(i + 1)
                ys.append(acc0)
        ax.plot(xs, ys, marker="o", ms=3, label=v)
    ax.set_title("(2) Vai trò tầng: acc danh tính ĐẦU khi học thêm")
    ax.set_xlabel("Số danh tính đã dạy")
    ax.set_ylabel("Accuracy (danh tính đầu)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()


# Giống eval_acc nhưng chỉ đo trên MỘT danh tính idx (thường là danh tính đầu, để đo mức quên).
def eval_acc_single(cpm, protos, idx, n_test=5, cos=0.7, mode="assoc", seed0=70000) -> float:
    c = sum(
        cpm.recall(sample_with_cos(protos[idx], cos, seed=seed0 + j), mode=mode)[0]["label"] == f"id{idx}"
        for j in range(n_test)
    )
    return c / n_test


# ------------------------------------------------------------- 3) recall_mode
# (3) SO CÁCH ĐỌC BỘ NHỚ: cùng 1 bộ nhớ nhưng nhận diện theo 3 kiểu (assoc / proto / hybrid).
# Chứng minh prototype (xương sống) bền hơn ma trận liên kết khi số danh tính lớn.
def exp_recall_mode(ax):
    dim = 64
    nmax = 200
    checks = list(range(10, nmax + 1, 15))
    protos = make_prototypes(nmax, dim, seed=3)
    cpm = make_cpm(dim, "3-tier")
    xs, assoc, proto, hybrid = [], [], [], []
    for i in range(nmax):
        cpm.write(protos[i], f"id{i}")
        if (i + 1) in checks:
            xs.append(i + 1)
            # cùng dữ liệu, đọc ra theo 3 kiểu để so accuracy:
            assoc.append(eval_acc(cpm, protos, i + 1, mode="assoc"))    # ma trận liên kết (NL)
            proto.append(eval_acc(cpm, protos, i + 1, mode="proto"))    # prototype trung bình (NCM)
            hybrid.append(eval_acc(cpm, protos, i + 1, mode="hybrid"))  # trộn cả hai
    ax.plot(xs, assoc, marker="o", ms=3, label="assoc (ma trận NL)")
    ax.plot(xs, proto, marker="s", ms=3, label="proto (xương sống)")
    ax.plot(xs, hybrid, marker="^", ms=3, label="hybrid")
    ax.axvline(dim, ls="--", color="gray", alpha=0.6)
    ax.text(dim + 2, 0.15, "N = dim", fontsize=8)
    ax.set_title(f"(3) assoc vs proto (dim={dim})")
    ax.set_xlabel("Số danh tính")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()


# ------------------------------------------ 4) robustness theo độ giống danh tính (KEY)
def exp_correlation(ax):
    """Phát hiện then chốt: khi các danh tính GIỐNG NHAU (người thân/nhìn giống), proto-backbone
    (= NCM) bền, còn ma trận liên kết & fine-tune SỤP -> lý do CPM v2 dùng proto làm xương sống."""
    # PANEL QUAN TRỌNG NHẤT của ablation: quét "độ giống nhau" giữa các danh tính (shared).
    # shared càng cao = các danh tính nhìn càng giống nhau (ví dụ anh em, người nhà) -> càng khó phân biệt.
    dim, N = 128, 40
    shareds = [0.0, 0.2, 0.4, 0.6, 0.8]

    # Hàm phụ: dựng 1 phương pháp (builder), dạy N danh tính rồi đo accuracy trên chúng.
    # Viết chung để dùng cho cả CPM (có .write/.recall) lẫn baseline (có .teach/.predict).
    def acc_of(builder, mode, protos):
        obj = builder()
        for i in range(N):
            for j in range(6):
                # CPM dùng .write, còn adapter baseline dùng .teach -> chọn hàm theo đối tượng có gì.
                obj.write(sample_with_cos(protos[i], 0.75, seed=1000 * i + j), f"id{i}") if hasattr(obj, "write") \
                    else obj.teach(sample_with_cos(protos[i], 0.75, seed=1000 * i + j), f"id{i}")
        # Tương tự khi đoán: CPM dùng .recall(mode), baseline dùng .predict.
        pred = (lambda q: obj.recall(q, mode=mode)[0]["label"]) if hasattr(obj, "recall") else (lambda q: obj.predict(q))
        c = t = 0
        for i in range(N):
            for j in range(5):
                c += pred(sample_with_cos(protos[i], 0.7, seed=9000 * i + j)) == f"id{i}"
                t += 1
        return c / t

    # 4 phương pháp đem so: CPM-proto (mặc định), CPM-ma trận, NCM, fine-tune.
    series = {
        "CPM proto (mặc định)": (lambda: ContinualPersonalizationMemory(dim, config=CPMConfig(dim)), "proto"),
        "CPM assoc-matrix": (lambda: make_cpm(dim, "3-tier"), "assoc"),
        "NCM": (lambda: NCMAdapter(dim), None),
        "fine-tune": (lambda: FineTuneAdapter(dim), None),
    }
    for name, (builder, mode) in series.items():
        # make_correlated_prototypes: tạo N danh tính có độ giống nhau = s -> quét qua các mức shared.
        ys = [acc_of(builder, mode, make_correlated_prototypes(N, dim, s, seed=4)) for s in shareds]
        ax.plot(shareds, ys, marker="o", ms=3, label=name)
    ax.set_title("(4) Bền theo ĐỘ GIỐNG giữa các danh tính")
    ax.set_xlabel("shared (cao = người/đồ càng giống nhau)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# ----------------------------------------------------------------- 5) openset
def exp_openset(ax) -> dict:
    dim, N = 128, 30
    protos = make_prototypes(2 * N, dim, seed=5)  # 0..N-1 quen, N..2N-1 lạ
    cpm = make_cpm(dim, "3-tier")
    for i in range(N):
        for j in range(5):
            cpm.write(sample_with_cos(protos[i], 0.75, seed=200 * i + j), f"id{i}")

    genuine, impostor = [], []
    for i in range(N):  # quen -> điểm cao
        for j in range(5):
            q = sample_with_cos(protos[i], 0.7, seed=99000 + 10 * i + j)
            genuine.append(cpm.recall(q)[0]["confidence"])
    for i in range(N, 2 * N):  # lạ -> điểm thấp
        for j in range(5):
            q = sample_with_cos(protos[i], 0.7, seed=88000 + 10 * i + j)
            impostor.append(cpm.recall(q)[0]["confidence"])

    genuine, impostor = np.array(genuine), np.array(impostor)
    far, tar, _ = roc_points(genuine, impostor)
    summ = open_set_summary(genuine, impostor)

    ax.plot(far, tar, color="#1b9e77")
    # điểm ứng với ngưỡng đang dùng 0.35
    thr = 0.35
    far035 = (impostor >= thr).mean()
    tar035 = (genuine >= thr).mean()
    ax.scatter([far035], [tar035], color="red", zorder=5)
    ax.annotate(f"ngưỡng 0.35\nTAR={tar035:.2f}, FAR={far035:.2f}", (far035, tar035),
                textcoords="offset points", xytext=(10, -20), fontsize=8)
    ax.plot([0, 1], [0, 1], ls=":", color="gray", alpha=0.5)
    ax.set_title(f"(5) Open-set ROC (AUC={summ['auc']:.3f}, EER={summ['eer']:.2f})")
    ax.set_xlabel("FAR (nhận nhầm người lạ)")
    ax.set_ylabel("TAR (nhận đúng người quen)")
    ax.grid(alpha=0.3)
    summ["far@0.35"] = float(far035)
    summ["tar@0.35"] = float(tar035)
    return summ


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    exp_capacity(axes[0, 0])
    exp_tiers(axes[0, 1])
    exp_recall_mode(axes[0, 2])
    exp_correlation(axes[1, 0])
    summ = exp_openset(axes[1, 1])
    axes[1, 2].axis("off")
    # bảng open-set vào ô trống
    txt = "Open-set summary\n" + "\n".join(
        f"{k:>18}: {v:.3f}" if isinstance(v, float) else f"{k:>18}: {v}" for k, v in summ.items()
    )
    axes[1, 2].text(0.02, 0.95, txt, va="top", family="monospace", fontsize=9)

    fig.suptitle("CPM — Bộ thí nghiệm nghiên cứu (ablation / robustness / open-set)", fontsize=13)
    fig.tight_layout()
    png = os.path.join(RESULTS_DIR, "ablation.png")
    fig.savefig(png, dpi=120)

    print("=== OPEN-SET SUMMARY ===")
    for k, v in summ.items():
        print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    print(f"\nĐã lưu: {png}")


if __name__ == "__main__":
    main()
