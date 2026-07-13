"""
Multi-skill vision assistant orchestrator.

Routes Vietnamese text intents to:
  - CPM personalization and recognition
  - scene/VQA
  - OCR
  - obstacle safety monitor

Smoke:
  python -m app.orchestrator --smoke

Real optional skills:
  python -m app.orchestrator --skills-mode real
"""

from __future__ import annotations

import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.cli import Assistant
from skills import RealOCR, RealScene, StubOCR, StubObstacle, StubScene


def _demojibake(text: str) -> str:
    """Sửa tận gốc lỗi UTF-8-bị-đọc-như-CP1252 (mojibake) ngay tại biên nhập liệu.

    Ví dụ 'ghi nho' bị mojibake -> phục hồi 'ghi nhớ'. Chỉ đổi khi chuỗi TOÀN ký tự thuộc CP1252
    (đặc trưng mojibake) và giải mã lại UTF-8 hợp lệ; chuỗi tiếng Việt ĐÚNG chứa ký tự ngoài CP1252
    nên .encode('cp1252') ném lỗi -> giữ nguyên. An toàn, không phá input hợp lệ.
    """
    if not text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class VisionAssistant:
    """Combine CPM personalization, perception skills, and safety override."""

    TEACH_KW = ("ghi nhớ", "nhớ đây", "đây là", "nhớ là", "hãy nhớ")
    CORRECT_KW = ("sửa", "không phải", "nhầm", "sai rồi")
    RECOG_KW = ("là ai", "ai đây", "ai đang", "người này", "cái gì đây", "vật gì", "đồ gì", "của tôi")
    OBJECT_HINT = ("cái", "đồ", "vật", "ví", "chìa", "túi", "cốc", "điện thoại")

    def __init__(self, embedder_kind: str = "synthetic", skills_mode: str = "stub"):
        self.core = Assistant(embedder_kind=embedder_kind)
        self.scene, self.ocr = self._make_skills(skills_mode)
        self.obstacle = StubObstacle()
        self._query_skills = [self.obstacle, self.ocr, self.scene]

    @staticmethod
    def _make_skills(skills_mode: str):
        if skills_mode != "real":
            return StubScene(), StubOCR()

        scene = StubScene()
        ocr = StubOCR()

        # --- VLM cho scene/VQA: OpenAI -> Gemini -> Stub ---
        vlm = VisionAssistant._select_vlm()
        if vlm is not None:
            scene = RealScene(vlm=vlm[0])
            print(f"[skills] Scene/VQA: dùng VLM {vlm[1]}.", file=sys.stderr)
        else:
            print("[skills] Không có VLM thật (thiếu key hoặc chưa cài SDK) → dùng StubScene.", file=sys.stderr)

        # --- OCR: OpenAI vision -> EasyOCR -> Stub ---
        ocr_fn = VisionAssistant._select_ocr()
        if ocr_fn is not None:
            ocr = RealOCR(ocr_fn=ocr_fn[0])
            print(f"[skills] OCR: dùng {ocr_fn[1]}.", file=sys.stderr)
        else:
            print("[skills] Không có OCR thật (OpenAI/EasyOCR) → dùng StubOCR.", file=sys.stderr)

        return scene, ocr

    @staticmethod
    def _select_vlm():
        """Chọn VLM theo thứ tự ưu tiên: OpenAI -> Gemini. Trả (callable, tên) hoặc None."""
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from skills.providers import openai_vlm

                return openai_vlm(), "OpenAI (gpt-4o-mini)"
            except Exception as exc:
                print(f"[skills] OpenAI VLM lỗi: {exc}", file=sys.stderr)
        if os.environ.get("GEMINI_API_KEY"):
            try:
                from skills.providers import gemini_vlm

                return gemini_vlm(), "Gemini (gemini-1.5-flash)"
            except Exception as exc:
                print(f"[skills] Gemini VLM lỗi: {exc}", file=sys.stderr)
        return None

    @staticmethod
    def _select_ocr():
        """Chọn OCR theo thứ tự: OpenAI vision -> EasyOCR. Trả (callable, tên) hoặc None."""
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from skills.providers import openai_ocr

                return openai_ocr(), "OpenAI vision (gpt-4o-mini)"
            except Exception as exc:
                print(f"[skills] OpenAI OCR lỗi: {exc}", file=sys.stderr)
        try:
            from skills.providers import easyocr_fn

            return easyocr_fn(), "EasyOCR"
        except Exception as exc:
            print(f"[skills] EasyOCR lỗi: {exc}", file=sys.stderr)
        return None

    def _modality(self, query: str) -> str:
        low = query.lower()
        return "object" if any(w in low for w in self.OBJECT_HINT) else "face"

    @staticmethod
    def _parse_label(query: str) -> str | None:
        low = query.lower()
        markers = (" là ", "là ")
        for marker in markers:
            if marker in low:
                return query[low.index(marker) + len(marker) :].strip(" .?!")
        return None

    def _embed(self, frame, modality: str):
        return self.core._embed(modality, frame)

    def handle(self, query: str, frame=None, label: str | None = None, modality: str | None = None) -> str:
        query = _demojibake(query)
        low = query.lower()
        modality = modality or self._modality(query)

        if any(k in low for k in self.TEACH_KW):
            lab = label or self._parse_label(query)
            if not lab:
                return "Bạn muốn tôi ghi nhớ với tên gì?"
            self.core.cpm[modality].write(self._embed(frame, modality), lab)
            return f"Đã ghi nhớ ({modality}): {lab}."

        if any(k in low for k in self.CORRECT_KW):
            lab = label or self._parse_label(query)
            if not lab:
                return "Tên đúng là gì để tôi sửa lại?"
            self.core.cpm[modality].correct(self._embed(frame, modality), lab)
            return f"Đã sửa ({modality}): đây là {lab}."

        safety = self.obstacle.check(frame)
        prefix = (safety["text"] + " ") if safety["danger"] else ""

        if any(k in low for k in self.RECOG_KW):
            res = self.core.cpm[modality].recall(self._embed(frame, modality))[0]
            ans = (
                f"Đây là {res['label']} (tin cậy {res['confidence']:.2f})."
                if res["known"]
                else "Tôi chưa nhận ra người/vật này. Bạn có muốn dạy tôi không?"
            )
            return prefix + ans

        for sk in self._query_skills:
            if sk.can_handle(low):
                result = sk.run(frame, query)
                return result.text if sk is self.obstacle else prefix + result.text

        return prefix + self.scene.run(frame, query).text


def smoke(skills_mode: str = "stub") -> int:
    a = VisionAssistant(embedder_kind="synthetic", skills_mode=skills_mode)
    print(">>> SMOKE ORCHESTRATOR (stub skills)\n")

    def ask(q, frame=None):
        print(f"[Người dùng] {q}")
        print(f"[Trợ lý]     {a.handle(q, frame=frame)}\n")

    ask("Hãy nhớ đây là Lan", frame="person_lan")
    ask("Ai đây?", frame="person_lan")
    ask("Hãy nhớ đây là ví của tôi", frame="my_wallet")
    ask("Cái gì đây?", frame="my_wallet")
    ask("Trước mặt có gì?")
    ask("Đọc chữ trên biển giúp tôi")

    print("--- Dựng tình huống có vật cản gần (0.8m) ---\n")
    a.obstacle.set_distance(0.8)
    ask("Trước mặt có gì?")
    ask("Đường có an toàn không?")
    ask("Ai đây?", frame="person_lan")

    print("--- Đường thông thoáng trở lại ---\n")
    a.obstacle.set_distance(None)
    ask("Có vật cản gì không?")

    assert a.handle("Ai đây?", frame="person_lan").endswith("(tin cậy 1.00).")
    print("SMOKE PASSED")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    p.add_argument("--skills-mode", default="stub", choices=["stub", "real"])
    args = p.parse_args()
    if args.smoke:
        return smoke(args.skills_mode)
    _ = VisionAssistant(embedder_kind=args.embedder, skills_mode=args.skills_mode)
    print("Đã khởi tạo VisionAssistant. Dùng --smoke để chạy demo, hoặc nhúng VisionAssistant vào UI/CLI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
