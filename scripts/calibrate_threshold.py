"""
Sinh ngưỡng quen/lạ đã CALIBRATE cho embedder THẬT -> lưu configs/thresholds.json.

Vì sao cần: ngưỡng mặc định 0.35 (suy từ synthetic) cho FAR rất cao trên facenet/ArcFace
(nhận nhầm người lạ). Script đo phân bố điểm genuine (người/đồ ĐÃ dạy) vs impostor (LẠ) trên
chính embedder thật, rồi chọn ngưỡng theo policy (mặc định far1 = an toàn cho trợ thị).

Nguồn impostor (BẮT BUỘC chọn 1):
  --impostor_dir <dir>   thư mục ảnh người/đồ LẠ (folders-per-label HOẶC ảnh phẳng)
  --impostor_split N     giữ N nhãn CUỐI trong --data_dir làm impostor (không dạy)

Ví dụ:
  python -m scripts.calibrate_threshold --data_dir data/faces --modality face --impostor_dir data/impostors
  python -m scripts.calibrate_threshold --data_dir data/faces --modality face --impostor_split 3
  python -m scripts.calibrate_threshold --data_dir data/objects --modality object --impostor_split 2

Kết quả nạp tự động khi tạo app.cli.Assistant / orchestrator với embedder thật.
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

from cpm import (
    CPMConfig,
    ContinualPersonalizationMemory,
    DEFAULT_THRESHOLDS_PATH,
    save_thresholds,
    threshold_key,
)
from experiments.calibration import calibrate_threshold, collect_open_set_scores
from experiments.real_data import IMG_EXT, embed_dir


def _impostor_embeds(emb, path: str, modality: str) -> list:
    """impostor_dir: có thư mục con -> gộp mọi nhãn; ảnh phẳng -> đọc trực tiếp."""
    if any(os.path.isdir(d) for d in glob.glob(os.path.join(path, "*"))):
        by_label, _ = embed_dir(emb, path, modality)
        return [v for vecs in by_label.values() for v in vecs]

    from perception.capture import load_image_bgr, load_image_rgb

    vecs = []
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        if not f.lower().endswith(IMG_EXT):
            continue
        try:
            vecs.append(
                emb.embed_face(load_image_bgr(f)) if modality == "face"
                else emb.embed_object(load_image_rgb(f))
            )
        except Exception as e:
            print(f"  bỏ qua {os.path.basename(f)}: {e}")
    return vecs


def main() -> int:
    p = argparse.ArgumentParser(description="Calibrate ngưỡng quen/lạ cho embedder thật.")
    p.add_argument("--data_dir", default="data/faces")
    p.add_argument("--modality", default="face", choices=["face", "object"])
    p.add_argument("--impostor_dir", default=None, help="thư mục ảnh người/đồ LẠ")
    p.add_argument("--impostor_split", type=int, default=0,
                   help="giữ N nhãn cuối trong data_dir làm impostor (nếu không có impostor_dir)")
    p.add_argument("--policy", default="far1", choices=["far1", "far10", "eer"])
    p.add_argument("--test_ratio", type=float, default=0.4,
                   help="tỉ lệ ảnh mỗi người QUEN để làm probe genuine")
    p.add_argument("--embedder_name", default=None, help="ghi đè khoá embedder (mặc định emb.name)")
    args = p.parse_args()

    from perception.embed import RealEmbedder

    emb = RealEmbedder()
    ename = args.embedder_name or getattr(emb, "name", "real")

    print(f"Nạp danh tính QUEN từ '{args.data_dir}' (modality={args.modality})...")
    by_label, dim = embed_dir(emb, args.data_dir, args.modality)
    labels = [l for l, v in by_label.items() if len(v) >= 2]
    if not labels:
        raise SystemExit("Cần ≥ 1 nhãn QUEN có ≥ 2 ảnh (chừa 1 ảnh làm probe).")

    # --- tách impostor ---
    if args.impostor_dir:
        print(f"Nạp người/đồ LẠ từ '{args.impostor_dir}'...")
        impostor_embs = _impostor_embeds(emb, args.impostor_dir, args.modality)
    elif args.impostor_split > 0:
        if args.impostor_split >= len(labels):
            raise SystemExit("impostor_split phải nhỏ hơn số nhãn quen.")
        imp_labels = labels[-args.impostor_split:]
        labels = labels[: -args.impostor_split]
        impostor_embs = [v for l in imp_labels for v in by_label[l]]
        print(f"Giữ {imp_labels} làm impostor; dạy {labels}.")
    else:
        raise SystemExit("Cần --impostor_dir HOẶC --impostor_split N (mẫu LẠ để đo FAR).")

    if not impostor_embs:
        raise SystemExit("Không có embedding impostor hợp lệ.")

    # --- dạy known (train) + giữ probe (genuine) ---
    cpm = ContinualPersonalizationMemory(dim=dim, modality=args.modality, config=CPMConfig(dim=dim))
    known_probes = []
    for label in labels:
        vecs = by_label[label]
        n_test = max(1, round(len(vecs) * args.test_ratio))
        train, probe = vecs[n_test:], vecs[:n_test]
        if not train:              # quá ít ảnh -> dồn hết vào train
            train, probe = vecs, []
        for v in train:
            cpm.write(v, label)
        if probe:
            known_probes.append((label, probe))

    genuine, impostor = collect_open_set_scores(cpm, known_probes, impostor_embs)
    if genuine.size == 0:
        raise SystemExit("Không có probe genuine (mỗi người quen cần ≥ 2 ảnh).")

    result = calibrate_threshold(genuine, impostor, policy=args.policy)
    key = threshold_key(ename, args.modality)
    save_thresholds({
        key: {
            "threshold": result["threshold"],
            "policy": args.policy,
            "far": result["far"],
            "frr": result["frr"],
            "n_genuine": result["n_genuine"],
            "n_impostor": result["n_impostor"],
        }
    })

    print("\n=== NGƯỠNG CALIBRATE ===")
    print(f"  khoá             : {key}")
    print(f"  policy           : {args.policy}")
    print(f"  threshold        : {result['threshold']:.4f}   (mặc định cũ = 0.35)")
    print(f"  FAR / FRR / TAR  : {result['far']:.3f} / {result['frr']:.3f} / {result['tar']:.3f}")
    print(f"  AUC / EER        : {result['auc']:.3f} / {result['eer']:.3f}")
    print(f"  genuine/impostor : {result['n_genuine']} / {result['n_impostor']} mẫu")
    print(f"\nĐã lưu: {DEFAULT_THRESHOLDS_PATH}")
    print("Assistant/orchestrator với embedder thật sẽ tự nạp ngưỡng này.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
