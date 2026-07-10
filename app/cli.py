"""
CLI orchestrator — nối Perception -> CPM -> phản hồi tiếng Việt.

Chạy thử KHÔNG cần model/camera (embedding synthetic):
    python -m app.cli --smoke

Chạy tương tác (mặc định synthetic; đổi --embedder real trên M4):
    python -m app.cli
Lệnh trong phiên tương tác:
    teach  <face|obj> <ảnh|danh_tính> <nhãn>
    ask    <face|obj> <ảnh|danh_tính>
    fix    <face|obj> <ảnh|danh_tính> <nhãn>
    stats
    quit
"""

from __future__ import annotations

import argparse
import sys

# Đảm bảo in được tiếng Việt trên mọi console (Windows cp1252 -> UTF-8)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from cpm import (
    CPMConfig,
    ContinualPersonalizationMemory,
    load_thresholds,
    resolve_threshold,
    threshold_key,
)
from perception import get_embedder


class Assistant:
    """Điều phối 2 modality (face/object), mỗi cái một CPM cô lập."""

    def __init__(self, embedder_kind: str = "synthetic", user_id: str = "demo"):
        self.embedder = get_embedder(embedder_kind)
        self.cpm = {
            "face": ContinualPersonalizationMemory(
                dim=512, user_id=user_id, modality="face", config=CPMConfig(dim=512)
            ),
            "object": ContinualPersonalizationMemory(
                dim=512, user_id=user_id, modality="object", config=CPMConfig(dim=512)
            ),
        }
        self._apply_calibrated_thresholds()

    def _apply_calibrated_thresholds(self) -> None:
        """Nạp ngưỡng quen/lạ đã calibrate theo (embedder, modality); thiếu -> giữ mặc định.

        Ngưỡng mặc định 0.35 hợp cho embedder synthetic nhưng SAI cho embedder thật
        (facenet/ArcFace: FAR cao). Chạy scripts/calibrate_threshold.py để sinh ngưỡng đúng.
        """
        mapping = load_thresholds()
        ename = getattr(self.embedder, "name", "unknown")
        for modality, cpm in self.cpm.items():
            default = cpm.cfg.recall_threshold
            cpm.cfg.recall_threshold = resolve_threshold(mapping, ename, modality, default)
            missing = threshold_key(ename, modality) not in mapping
            if missing and ename != "synthetic":
                print(
                    f"[cpm] ⚠️ Chưa có ngưỡng calibrate cho ({ename}, {modality}); "
                    f"dùng mặc định {default:.2f} (dễ nhận nhầm người lạ). "
                    "Chạy: python -m scripts.calibrate_threshold ...",
                    file=sys.stderr,
                )

    def _embed(self, modality: str, image):
        if modality == "face":
            return self.embedder.embed_face(image)
        return self.embedder.embed_object(image)

    def teach(self, modality: str, image, label: str) -> str:
        self.cpm[modality].write(self._embed(modality, image), label)
        return f"Đã ghi nhớ ({modality}): {label}"

    def ask(self, modality: str, image) -> str:
        res = self.cpm[modality].recall(self._embed(modality, image))[0]
        if not res["known"]:
            return f"Chưa biết ({modality}) — độ tương đồng {res['confidence']:.2f}"
        return (
            f"Đây là {res['label']} "
            f"(tin cậy {res['confidence']:.2f}, tầng {res['tier']})"
        )

    def fix(self, modality: str, image, label: str) -> str:
        self.cpm[modality].correct(self._embed(modality, image), label)
        return f"Đã sửa ({modality}): đây là {label}"

    def stats(self) -> str:
        return " | ".join(
            f"{m}: {c.stats()['n_labels']} nhãn" for m, c in self.cpm.items()
        )


def smoke() -> int:
    """Kịch bản kiểm tra wiring end-to-end (synthetic, không cần model)."""
    a = Assistant(embedder_kind="synthetic")
    print(">>> SMOKE TEST (synthetic embedder)\n")

    print(a.teach("face", "person_lan", "Lan"))
    print(a.ask("face", "person_lan"))          # -> Lan
    print(a.teach("face", "person_huy", "Huy"))
    print(a.ask("face", "person_huy"))          # -> Huy
    print(a.ask("face", "person_stranger"))     # -> chưa biết
    print(a.fix("face", "person_stranger", "Nam"))
    print(a.ask("face", "person_stranger"))     # -> Nam (sau khi sửa)
    print(a.ask("face", "person_lan"))          # -> vẫn Lan (không quên)

    print(a.teach("object", "my_wallet", "ví của tôi"))
    print(a.ask("object", "my_wallet"))         # -> ví của tôi
    print("\nStats:", a.stats())

    # kiểm tra kỳ vọng cơ bản
    assert a.ask("face", "person_lan").startswith("Đây là Lan")
    assert a.ask("face", "person_stranger").startswith("Đây là Nam")
    print("\n✅ SMOKE PASSED")
    return 0


def interactive(embedder_kind: str) -> int:
    a = Assistant(embedder_kind=embedder_kind)
    print("Trợ lý CPM (gõ 'quit' để thoát). Lệnh: teach/ask/fix/stats")
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        try:
            if cmd == "quit":
                break
            elif cmd == "stats":
                print(a.stats())
            elif cmd == "teach" and len(parts) >= 4:
                print(a.teach(_mod(parts[1]), parts[2], " ".join(parts[3:])))
            elif cmd == "ask" and len(parts) >= 3:
                print(a.ask(_mod(parts[1]), parts[2]))
            elif cmd == "fix" and len(parts) >= 4:
                print(a.fix(_mod(parts[1]), parts[2], " ".join(parts[3:])))
            else:
                print("Lệnh không hợp lệ.")
        except Exception as e:
            print(f"Lỗi: {e}")
    return 0


def _mod(s: str) -> str:
    return "face" if s.lower().startswith("f") else "object"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="chạy kịch bản kiểm tra wiring")
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    args = p.parse_args()
    return smoke() if args.smoke else interactive(args.embedder)


if __name__ == "__main__":
    raise SystemExit(main())
