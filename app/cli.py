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
from pathlib import Path

# Đảm bảo in được tiếng Việt trên mọi console (Windows cp1252 -> UTF-8)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Nạp các "bộ phận" của bộ nhớ học liên tục (CPM = trái tim đồ án):
#   - CPMConfig: bảng cấu hình (số chiều, ngưỡng quen-lạ...)
#   - ContinualPersonalizationMemory: chính bộ nhớ (dạy/hỏi/sửa)
#   - load_thresholds / resolve_threshold / threshold_key: đọc & tra ngưỡng quen-lạ đã hiệu chuẩn
from cpm import (
    CPMConfig,
    ContinualPersonalizationMemory,
    LocalMemoryStore,
    load_thresholds,
    resolve_threshold,
    threshold_key,
)
# get_embedder: lấy "đôi mắt" biến ảnh -> dấu vân tay số (embedding).
# Chọn được bản giả (synthetic, chạy ngay) hoặc bản thật (real, cần model).
from perception import get_embedder


# Lớp Assistant = "bộ khung" nối đôi mắt (Perception) với trí nhớ (CPM).
# Nó lo phần dạy/hỏi/sửa cho cả 2 loại đối tượng: mặt và đồ vật.
# (Phần hiểu câu tiếng Việt để đoán ý định nằm ở orchestrator.py, không phải ở đây.)
class Assistant:
    """Điều phối 2 modality (face/object), mỗi cái một CPM cô lập."""

    # Hàm khởi tạo: dựng sẵn "đôi mắt" và "trí nhớ" khi tạo trợ lý.
    #   - embedder_kind: chọn đôi mắt giả ("synthetic") hay thật ("real"/"auto")
    #   - user_id: tên người dùng, để mỗi người có bộ nhớ riêng
    def __init__(
        self,
        embedder_kind: str = "synthetic",
        user_id: str = "demo",
        memory_dir: str | Path | None = None,
    ):
        # Đôi mắt: biến ảnh -> dấu vân tay số.
        self.embedder = get_embedder(embedder_kind)
        # Hai bộ nhớ TÁCH RIÊNG cho 2 loại đối tượng (modality):
        # face (mặt) và object (đồ vật). Vì mặt dùng model khác đồ vật nên
        # "dấu vân tay" của chúng khác ngôn ngữ, KHÔNG được trộn chung một trí nhớ.
        # dim=512 = mỗi dấu vân tay là dãy 512 con số.
        self.cpm = {
            "face": ContinualPersonalizationMemory(
                dim=512, user_id=user_id, modality="face", config=CPMConfig(dim=512)
            ),
            "object": ContinualPersonalizationMemory(
                dim=512, user_id=user_id, modality="object", config=CPMConfig(dim=512)
            ),
        }
        self.memory_store = LocalMemoryStore(memory_dir) if memory_dir is not None else None
        # Nạp ngưỡng quen-lạ đã hiệu chuẩn (nếu có) cho từng bộ nhớ.
        self._apply_calibrated_thresholds()
        self.restore_memory()

    def _apply_calibrated_thresholds(self) -> None:
        """Nạp ngưỡng quen/lạ đã calibrate theo (embedder, modality); thiếu -> giữ mặc định.

        Ngưỡng mặc định 0.35 hợp cho embedder synthetic nhưng SAI cho embedder thật
        (facenet/ArcFace: FAR cao). Chạy scripts/calibrate_threshold.py để sinh ngưỡng đúng.
        """
        # Đọc bảng ngưỡng đã lưu (từ file cấu hình) và tên của đôi mắt đang dùng.
        mapping = load_thresholds()
        ename = getattr(self.embedder, "name", "unknown")
        # Với TỪNG loại đối tượng, tra ngưỡng đúng cho cặp (đôi mắt, modality).
        for modality, cpm in self.cpm.items():
            # default = ngưỡng mặc định đang cài sẵn trong cấu hình.
            default = cpm.cfg.recall_threshold
            # Nếu có ngưỡng đã hiệu chuẩn thì dùng nó, không thì giữ default.
            cpm.cfg.recall_threshold = resolve_threshold(mapping, ename, modality, default)
            # missing = chưa từng hiệu chuẩn cho cặp (đôi mắt, modality) này.
            missing = threshold_key(ename, modality) not in mapping
            # Cảnh báo: đôi mắt THẬT mà chưa hiệu chuẩn thì rất dễ nhận nhầm người lạ.
            # (Riêng đôi mắt giả "synthetic" thì ngưỡng mặc định vẫn ổn nên không cảnh báo.)
            if missing and ename != "synthetic":
                print(
                    f"[cpm] ⚠️ Chưa có ngưỡng calibrate cho ({ename}, {modality}); "
                    f"dùng mặc định {default:.2f} (dễ nhận nhầm người lạ). "
                    "Chạy: python -m scripts.calibrate_threshold ...",
                    file=sys.stderr,
                )

    def restore_memory(self) -> None:
        """Khôi phục CPM local nếu đã có, rồi áp lại ngưỡng hiện hành."""
        if self.memory_store is None:
            return
        for modality in tuple(self.cpm):
            loaded = self.memory_store.load(self.cpm[modality].user_id, modality)
            if loaded is not None:
                self.cpm[modality] = loaded
        self._apply_calibrated_thresholds()

    def persist_memory(self) -> None:
        """Ghi cả face/object vào local store nếu persistence đang bật."""
        if self.memory_store is None:
            return
        for memory in self.cpm.values():
            self.memory_store.save(memory)

    # _embed: biến ảnh -> dấu vân tay số, chọn đúng cách theo loại đối tượng.
    #   - mặt (face) dùng model nhận mặt; đồ vật (object) dùng model nhận ảnh.
    def _embed(self, modality: str, image):
        if modality == "face":
            return self.embedder.embed_face(image)
        return self.embedder.embed_object(image)

    # teach = DẠY: đưa ảnh + tên, bộ nhớ ghi lại "dấu vân tay này tên là ...".
    # Trả về câu xác nhận đã nhớ.
    def teach(self, modality: str, image, label: str) -> str:
        # write() = dạy CPM: gắn dấu vân tay của ảnh với nhãn (tên).
        self.cpm[modality].write(self._embed(modality, image), label)
        self.persist_memory()
        return f"Đã ghi nhớ ({modality}): {label}"

    # ask = HỎI: đưa ảnh, bộ nhớ so với những gì đã học rồi đoán là ai/cái gì.
    def ask(self, modality: str, image) -> str:
        # recall() trả danh sách phỏng đoán; [0] = phỏng đoán khớp nhất.
        res = self.cpm[modality].recall(self._embed(modality, image))[0]
        # known=False nghĩa là độ giống chưa đủ ngưỡng -> coi là người/vật LẠ.
        if not res["known"]:
            return f"Chưa biết ({modality}) — độ tương đồng {res['confidence']:.2f}"
        # known=True: đủ giống -> báo tên, kèm độ tin cậy và tầng bộ nhớ trả lời.
        return (
            f"Đây là {res['label']} "
            f"(tin cậy {res['confidence']:.2f}, tầng {res['tier']})"
        )

    # fix = SỬA: khi trợ lý đoán sai, đưa ảnh + tên ĐÚNG để nó ghi mạnh lại.
    def fix(self, modality: str, image, label: str) -> str:
        # correct() = sửa CPM: nhấn mạnh dấu vân tay này thuộc về nhãn đúng.
        self.cpm[modality].correct(self._embed(modality, image), label)
        self.persist_memory()
        return f"Đã sửa ({modality}): đây là {label}"

    # stats: tóm tắt đang nhớ bao nhiêu nhãn cho mỗi loại (để xem nhanh).
    def stats(self) -> str:
        return " | ".join(
            f"{m}: {c.stats()['n_labels']} nhãn" for m, c in self.cpm.items()
        )


# smoke = "chạy thử cho có khói": kiểm tra cả mạch dạy->hỏi->sửa có thông không.
# Không cần model/camera nên chạy được ở bất kỳ máy nào.
def smoke() -> int:
    """Kịch bản kiểm tra wiring end-to-end (synthetic, không cần model)."""
    # Dùng đôi mắt GIẢ để chạy được ngay, không cần model hay camera.
    a = Assistant(embedder_kind="synthetic")
    print(">>> SMOKE TEST (synthetic embedder)\n")

    # Kịch bản: dạy Lan -> hỏi lại -> dạy Huy -> hỏi lại...
    print(a.teach("face", "person_lan", "Lan"))
    print(a.ask("face", "person_lan"))          # -> Lan
    print(a.teach("face", "person_huy", "Huy"))
    print(a.ask("face", "person_huy"))          # -> Huy
    # Người chưa dạy bao giờ -> trợ lý phải trả lời "chưa biết".
    print(a.ask("face", "person_stranger"))     # -> chưa biết
    # Sửa: dạy rằng người "lạ" đó thật ra tên Nam.
    print(a.fix("face", "person_stranger", "Nam"))
    print(a.ask("face", "person_stranger"))     # -> Nam (sau khi sửa)
    # Điểm mấu chốt của đồ án: học người mới mà KHÔNG QUÊN Lan cũ.
    print(a.ask("face", "person_lan"))          # -> vẫn Lan (không quên)

    # Làm tương tự với đồ vật để chứng minh 2 loại chạy độc lập.
    print(a.teach("object", "my_wallet", "ví của tôi"))
    print(a.ask("object", "my_wallet"))         # -> ví của tôi
    print("\nStats:", a.stats())

    # kiểm tra kỳ vọng cơ bản
    # assert = "chốt chặn": nếu kết quả sai thì dừng ngay báo lỗi.
    assert a.ask("face", "person_lan").startswith("Đây là Lan")
    assert a.ask("face", "person_stranger").startswith("Đây là Nam")
    print("\n✅ SMOKE PASSED")
    return 0


# interactive: chế độ gõ lệnh tay trong cửa sổ dòng lệnh.
# Người dùng gõ teach/ask/fix/stats, chương trình đọc từng dòng và làm theo.
def interactive(embedder_kind: str) -> int:
    a = Assistant(embedder_kind=embedder_kind)
    print("Trợ lý CPM (gõ 'quit' để thoát). Lệnh: teach/ask/fix/stats")
    # Đọc từng dòng người dùng gõ vào cho tới khi họ gõ 'quit'.
    for line in sys.stdin:
        # Tách dòng thành các từ; từ đầu tiên là tên lệnh.
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        # try/except: nếu lệnh lỗi thì báo lỗi chứ không làm sập cả trợ lý.
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


# _mod: đổi chữ người dùng gõ thành tên loại chuẩn.
# Gõ bắt đầu bằng "f" (face) -> "face", còn lại -> "object".
def _mod(s: str) -> str:
    return "face" if s.lower().startswith("f") else "object"


# main: điểm vào khi chạy "python -m app.cli".
# Đọc tuỳ chọn dòng lệnh rồi quyết định chạy smoke hay chế độ gõ tay.
def main() -> int:
    p = argparse.ArgumentParser()
    # --smoke: chỉ chạy kịch bản kiểm tra nhanh rồi thoát.
    p.add_argument("--smoke", action="store_true", help="chạy kịch bản kiểm tra wiring")
    # --embedder: chọn đôi mắt giả hay thật.
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    args = p.parse_args()
    # Có --smoke thì chạy kiểm tra; không thì vào chế độ gõ lệnh tay.
    return smoke() if args.smoke else interactive(args.embedder)


if __name__ == "__main__":
    raise SystemExit(main())
