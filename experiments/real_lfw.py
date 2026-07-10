"""
Thí nghiệm trên DỮ LIỆU THẬT: LFW (Labeled Faces in the Wild) + facenet embedding.

- Tải LFW qua scikit-learn (bộ chuẩn công khai về nhận diện khuôn mặt).
- Embed bằng facenet-pytorch (InceptionResnetV1, pretrained VGGFace2) -> 512-d.
- So sánh CPM v2 / NCM / kNN / EWC / fine-tune: recognition accuracy + retention.
- Open-set: người QUEN (đã dạy) vs người LẠ (không dạy) -> ROC/AUC/EER.

Chạy:  python -m experiments.real_lfw
Kết quả: experiments/results/real_lfw.png  (+ in số ra màn hình)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

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
from experiments.calibration import calibrate_threshold, collect_open_set_scores
from experiments.metrics import roc_points
from experiments.retention import COLORS, run_forgetting

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_KNOWN = 12       # số người "quen" (dạy)
N_IMPOSTOR = 6     # số người "lạ" (open-set)
IMGS_PER = 14      # ảnh mỗi người
TEST_PER = 5       # ảnh test mỗi người (còn lại train)


def _embedder():
    import torch
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").eval()

    def embed(images: np.ndarray) -> np.ndarray:
        arr = np.asarray(images, dtype=np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)          # (n,3,h,w)
        x = (x - 127.5) / 128.0                                 # whitening facenet
        x = torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        with torch.no_grad():
            e = model(x).numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    return embed


def load_data():
    from sklearn.datasets import fetch_lfw_people

    print("Tải LFW (lần đầu ~200MB, cache lại sau)...")
    lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=0.5)
    images, targets, names = lfw.images, lfw.target, lfw.target_names

    by_person = defaultdict(list)
    for i, t in enumerate(targets):
        by_person[int(t)].append(i)
    eligible = [t for t, ids in by_person.items() if len(ids) >= IMGS_PER]
    eligible = eligible[: N_KNOWN + N_IMPOSTOR]
    if len(eligible) < N_KNOWN + N_IMPOSTOR:
        raise SystemExit("Không đủ người trong LFW với ngưỡng đặt ra.")

    embed = _embedder()
    print(f"Embed {len(eligible)} người × {IMGS_PER} ảnh bằng facenet (CPU, chờ chút)...")

    people = []
    for t in eligible:
        ids = by_person[t][:IMGS_PER]
        emb = embed(np.stack([images[i] for i in ids]))
        people.append({"name": names[t].replace(" ", "_"), "emb": emb})

    known = people[:N_KNOWN]
    impostors = people[N_KNOWN:]

    # dataset dạng {label, train, test} cho retention
    data = []
    for p in known:
        data.append({"label": p["name"], "train": list(p["emb"][TEST_PER:]), "test": list(p["emb"][:TEST_PER])})

    # sanity: cos cùng người vs khác người
    intra = np.mean([known[0]["emb"][a] @ known[0]["emb"][b] for a in range(5) for b in range(5) if a != b])
    inter = np.mean([known[0]["emb"][a] @ known[1]["emb"][b] for a in range(5) for b in range(5)])
    print(f"  sanity facenet: cos cùng-người≈{intra:.2f}, khác-người≈{inter:.2f}")
    return data, impostors


def main():
    data, impostors = load_data()
    dim = len(data[0]["train"][0])
    x = list(range(1, len(data) + 1))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    adapters = [CPMAdapter(dim), NCMAdapter(dim), KNNAdapter(dim), EWCAdapter(dim), FineTuneAdapter(dim)]
    res = {a.name: run_forgetting(a, data) for a in adapters}

    # recognition accuracy cuối + retention id đầu
    print("\n=== NHẬN DIỆN TRÊN LFW THẬT ===")
    print(f"{'Method':<22}{'Acc (đã học)':>14}{'Acc (id đầu)':>14}")
    for name, r in res.items():
        print(f"{name:<22}{r['overall'][-1]:>14.2f}{r['first_id'][-1]:>14.2f}")

    # open-set: CPM proto confidence trên người quen (test) vs người lạ
    cpm = CPMAdapter(dim)
    cpm.reset()
    for it in data:
        for e in it["train"]:
            cpm.teach(e, it["label"])
    known_probes = [(it["label"], it["test"]) for it in data]
    impostor_embs = [e for p in impostors for e in p["emb"]]
    genuine, impostor = collect_open_set_scores(cpm.cpm, known_probes, impostor_embs)

    # Ngưỡng ĐÚNG = calibrate từ ROC thật (policy far1, an toàn) — KHÔNG dùng 0.35 synthetic.
    calib = calibrate_threshold(genuine, impostor, policy="far1")
    thr = calib["threshold"]
    old = 0.35
    print("\n=== OPEN-SET (quen vs lạ, LFW thật) ===")
    print(f"  AUC={calib['auc']:.3f}  EER={calib['eer']:.3f}")
    print(f"  Ngưỡng calibrate (far1): {thr:.3f}  ->  FAR={calib['far']:.3f}, TAR={calib['tar']:.3f}")
    print(f"  (đối chứng) ngưỡng 0.35 cũ ->  FAR={(impostor >= old).mean():.3f}, "
          f"TAR={(genuine >= old).mean():.3f}   <-- minh hoạ vì sao 0.35 SAI cho facenet")
    print("  Lưu ý: số này cho embedder FACENET (thí nghiệm). Deploy dùng RealEmbedder(ArcFace):")
    print("         chạy `python -m scripts.calibrate_threshold` để sinh ngưỡng đúng cho model deploy.")

    # ---- vẽ ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, r in res.items():
        ax1.plot(x, r["overall"], marker="o", ms=3, label=name, color=COLORS.get(name))
    ax1.set_title(f"Recognition accuracy trên LFW ({len(data)} người)")
    ax1.set_xlabel("Số người đã dạy (tuần tự)")
    ax1.set_ylabel("Accuracy (mọi người đã học)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    far, tar, _ = roc_points(genuine, impostor)
    ax2.plot(far, tar, color="#1b9e77", label="ROC")
    # điểm ngưỡng ĐÃ calibrate (đúng) vs 0.35 cũ (sai)
    ax2.scatter([calib["far"]], [calib["tar"]], color="#1b9e77", zorder=5, s=50,
                label=f"calibrate far1 (thr={thr:.2f})")
    ax2.scatter([(impostor >= old).mean()], [(genuine >= old).mean()], color="red", zorder=5, s=55,
                marker="x", label="0.35 cũ (FAR cao)")
    ax2.plot([0, 1], [0, 1], ls=":", color="gray", alpha=0.5)
    ax2.set_title(f"Open-set ROC (AUC={calib['auc']:.3f}, EER={calib['eer']:.2f})")
    ax2.set_xlabel("FAR (nhận nhầm người lạ)")
    ax2.set_ylabel("TAR (nhận đúng người quen)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("CPM trên DỮ LIỆU THẬT — LFW + facenet embedding", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "real_lfw.png")
    fig.savefig(out, dpi=130)
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
