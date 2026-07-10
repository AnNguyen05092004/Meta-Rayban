"""
Thí nghiệm retention trên ẢNH THẬT (ArcFace/CLIP) — số liệu chính danh cho báo cáo.

Nạp ảnh từ thư mục (mỗi thư mục con = 1 nhãn), tính embedding thật, chạy so sánh
CPM vs kNN vs Fine-tune giống experiments/retention.py nhưng bằng dữ liệu thật.

Cấu trúc thư mục:
    data/faces/<Ten>/*.jpg     (modality=face)
    data/objects/<Ten>/*.jpg   (modality=object)

Chạy (trên M4, sau khi đã cài stack perception):
    python -m experiments.real_data --data_dir data/faces   --modality face
    python -m experiments.real_data --data_dir data/objects --modality object
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.baselines import CPMAdapter, FineTuneAdapter, KNNAdapter
from experiments.retention import COLORS, run_forgetting

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def embed_dir(emb, data_dir: str, modality: str):
    """Nạp mọi ảnh data_dir/<label>/* -> ({label: [emb, ...]}, dim). Ảnh lỗi bị bỏ qua.

    Tái dùng cho cả retention (real_data) lẫn calibrate ngưỡng (scripts/calibrate_threshold).
    `emb` = một RealEmbedder (truyền vào để dùng chung 1 instance/model).
    """
    from perception.capture import load_image_bgr, load_image_rgb

    persons = sorted(d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d))
    if not persons:
        raise SystemExit(f"Không thấy thư mục con nào trong '{data_dir}'. Xem docs/HUONG_DAN_M4.md mục 4.")

    by_label: dict[str, list] = {}
    dim = None
    for pdir in persons:
        label = os.path.basename(pdir)
        files = sorted(f for f in glob.glob(os.path.join(pdir, "*")) if f.lower().endswith(IMG_EXT))
        vecs = []
        for f in files:
            try:
                v = emb.embed_face(load_image_bgr(f)) if modality == "face" else emb.embed_object(load_image_rgb(f))
                vecs.append(v)
                dim = len(v)
            except Exception as e:
                print(f"  bỏ qua {os.path.basename(f)}: {e}")
        by_label[label] = vecs
    return by_label, dim


def load_dataset(data_dir: str, modality: str, test_ratio: float = 0.4):
    from perception.embed import RealEmbedder

    by_label, dim = embed_dir(RealEmbedder(), data_dir, modality)

    data = []
    for label, vecs in by_label.items():
        if len(vecs) < 2:
            print(f"  ⚠️ {label}: chỉ {len(vecs)} ảnh hợp lệ (<2) -> bỏ nhãn này")
            continue
        n_test = max(1, round(len(vecs) * test_ratio))
        data.append({"label": label, "train": vecs[n_test:], "test": vecs[:n_test]})
        print(f"  {label}: {len(vecs)} ảnh -> {len(vecs) - n_test} train / {n_test} test")

    if len(data) < 2:
        raise SystemExit("Cần ≥ 2 nhãn hợp lệ (mỗi nhãn ≥ 2 ảnh).")
    return data, dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/faces")
    p.add_argument("--modality", default="face", choices=["face", "object"])
    args = p.parse_args()

    print(f"Nạp dữ liệu từ '{args.data_dir}' (modality={args.modality}) ...")
    data, dim = load_dataset(args.data_dir, args.modality)
    x = list(range(1, len(data) + 1))

    adapters = [CPMAdapter(dim), KNNAdapter(dim), FineTuneAdapter(dim)]
    res = {a.name: run_forgetting(a, data) for a in adapters}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, r in res.items():
        ax1.plot(x, r["overall"], marker="o", ms=3, label=name, color=COLORS.get(name))
    ax1.set_title("Accuracy trên các danh tính đã học")
    ax1.set_xlabel("Số danh tính")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend()
    for name, r in res.items():
        ax2.plot(x, r["first_id"], marker="o", ms=3, label=name, color=COLORS.get(name))
    ax2.set_title("Accuracy trên danh tính ĐẦU (chống quên)")
    ax2.set_xlabel("Số danh tính")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle(f"Retention trên ẢNH THẬT ({args.modality}, {len(data)} nhãn, dim={dim})")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "results", f"retention_real_{args.modality}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)

    print("\n=== KẾT QUẢ (ảnh thật) ===")
    print(f"{'Method':<20}{'Acc(đã học)':>14}{'Acc(id đầu)':>14}")
    for name, r in res.items():
        print(f"{name:<20}{r['overall'][-1]:>14.2f}{r['first_id'][-1]:>14.2f}")
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
