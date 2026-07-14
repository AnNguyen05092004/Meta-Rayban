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

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Assistant: bộ khung dạy/hỏi/sửa (nối đôi mắt + trí nhớ CPM) từ file cli.py.
from app.cli import Assistant
from perception.embed import SyntheticEmbedder
# Các kỹ năng phụ. Mỗi kỹ năng có 2 bản:
#   - Stub = bản mô phỏng (câu mẫu, chạy ngay không cần model/mạng)
#   - Real = bản thật (gọi model/API, cần key). Thiếu key thì tự lùi về Stub.
# Scene = mô tả cảnh, OCR = đọc chữ, Obstacle = cảnh báo vật cản (an toàn).
from skills import RealObstacle, RealOCR, RealScene, StubOCR, StubObstacle, StubScene


# _demojibake: sửa lỗi phông chữ tiếng Việt bị hỏng (mojibake) ở đầu vào.
# Mojibake = chữ tiếng Việt hiện thành ký tự loằng ngoằng do đọc sai bảng mã.
# Vào: câu (có thể bị hỏng) -> Ra: câu đã phục hồi (hoặc giữ nguyên nếu vốn đã đúng).
def _demojibake(text: str) -> str:
    """Sửa tận gốc lỗi UTF-8-bị-đọc-như-CP1252 (mojibake) ngay tại biên nhập liệu.

    Ví dụ 'ghi nho' bị mojibake -> phục hồi 'ghi nhớ'. Chỉ đổi khi chuỗi TOÀN ký tự thuộc CP1252
    (đặc trưng mojibake) và giải mã lại UTF-8 hợp lệ; chuỗi tiếng Việt ĐÚNG chứa ký tự ngoài CP1252
    nên .encode('cp1252') ném lỗi -> giữ nguyên. An toàn, không phá input hợp lệ.
    """
    if not text:
        return text
    # Thử "giải mã ngược": nếu chuỗi đúng là mojibake thì bước này cho ra chữ đúng.
    try:
        return text.encode("cp1252").decode("utf-8")
    # Chuỗi tiếng Việt ĐÚNG sẽ ném lỗi ở bước trên -> ta giữ nguyên, không phá.
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# VisionAssistant = bộ điều phối (nhạc trưởng): đọc câu tiếng Việt -> đoán ý định
# (dạy/sửa/hỏi/đọc chữ/mô tả cảnh) -> gọi đúng bộ phận -> trả lời.
# Cách đoán ý định: dò xem câu chứa TỪ KHOÁ nào trong các danh sách bên dưới.
class VisionAssistant:
    """Combine CPM personalization, perception skills, and safety override."""

    # Các bộ từ khoá tiếng Việt để nhận ra ý định người dùng:
    TEACH_KW = ("ghi nhớ", "nhớ đây", "đây là", "nhớ là", "hãy nhớ")     # ý định DẠY
    CORRECT_KW = ("sửa", "không phải", "nhầm", "sai rồi")               # ý định SỬA
    RECOG_KW = ("là ai", "ai đây", "ai đang", "người này", "cái gì đây", "vật gì", "đồ gì", "của tôi")  # ý định HỎI
    # OBJECT_HINT: nếu câu chứa các từ này -> nhiều khả năng đang nói về ĐỒ VẬT, không phải mặt người.
    OBJECT_HINT = ("cái", "đồ", "vật", "ví", "chìa", "túi", "cốc", "điện thoại")

    # Khởi tạo nhạc trưởng: dựng sẵn trí nhớ + các kỹ năng phụ.
    #   - embedder_kind: đôi mắt giả ("synthetic") hay thật ("real")
    #   - skills_mode: kỹ năng mô phỏng ("stub") hay thật ("real")
    def __init__(
        self,
        embedder_kind: str = "synthetic",
        skills_mode: str = "stub",
        *,
        core: Assistant | None = None,
        obstacle_mode: str = "stub",
        obstacle=None,
    ):
        # core = bộ khung dạy/hỏi/sửa (đã chứa trí nhớ CPM cho mặt & đồ vật).
        self.core = core or Assistant(embedder_kind=embedder_kind)
        # Chọn kỹ năng mô tả cảnh và đọc chữ (thật hay mô phỏng).
        self.scene, self.ocr = self._make_skills(skills_mode)
        if obstacle is not None:
            self.obstacle = obstacle
        elif obstacle_mode == "real":
            self.obstacle = RealObstacle()
        elif obstacle_mode == "stub":
            self.obstacle = StubObstacle()
        else:
            raise ValueError("obstacle_mode phải là 'stub' hoặc 'real'.")
        self._safety_provider = None
        # Danh sách kỹ năng sẽ lần lượt được thử khi câu hỏi không phải dạy/sửa/hỏi.
        self._query_skills = [self.obstacle, self.ocr, self.scene]

    # _make_skills: chọn bản THẬT hay MÔ PHỎNG cho 2 kỹ năng scene và OCR.
    #   - skills_mode != "real" -> dùng luôn bản mô phỏng (nhanh, không cần key).
    #   - skills_mode == "real" -> thử lấy model thật, thiếu thì tự lùi về mô phỏng.
    @staticmethod
    def _make_skills(skills_mode: str):
        if skills_mode != "real":
            return StubScene(), StubOCR()

        # Mặc định bắt đầu bằng bản mô phỏng, rồi cố nâng cấp lên bản thật nếu được.
        scene = StubScene()
        ocr = StubOCR()

        # --- VLM cho scene/VQA: OpenAI -> Gemini -> Stub ---
        # VLM = model "nhìn ảnh và nói bằng chữ" (mô tả cảnh, hỏi-đáp về ảnh).
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

    # _select_vlm: tìm model mô tả cảnh THẬT theo thứ tự ưu tiên OpenAI -> Gemini.
    # Đọc "chìa khoá" (API key) từ biến môi trường; có key nào dùng model đó.
    # Không có key hoặc lỗi -> trả None để bên gọi lùi về bản mô phỏng.
    @staticmethod
    def _select_vlm():
        """Chọn VLM theo thứ tự ưu tiên: OpenAI -> Gemini. Trả (callable, tên) hoặc None."""
        # Ưu tiên 1: OpenAI (nếu đã đặt biến môi trường OPENAI_API_KEY).
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from skills.providers import openai_vlm

                return openai_vlm(), "OpenAI (gpt-4o-mini)"
            except Exception as exc:
                print(f"[skills] OpenAI VLM lỗi: {exc}", file=sys.stderr)
        # Ưu tiên 2: Gemini (nếu có GEMINI_API_KEY).
        if os.environ.get("GEMINI_API_KEY"):
            try:
                from skills.providers import gemini_vlm

                return gemini_vlm(), "Gemini (gemini-1.5-flash)"
            except Exception as exc:
                print(f"[skills] Gemini VLM lỗi: {exc}", file=sys.stderr)
        # Không có key nào chạy được -> None (bên gọi sẽ dùng StubScene).
        return None

    # _select_ocr: tìm công cụ đọc chữ THẬT theo thứ tự OpenAI vision -> EasyOCR.
    # EasyOCR chạy ngoại tuyến (không cần key) nên luôn được thử cuối cùng.
    @staticmethod
    def _select_ocr():
        """Chọn OCR theo thứ tự: OpenAI vision -> EasyOCR. Trả (callable, tên) hoặc None."""
        # Ưu tiên 1: OpenAI vision (cần OPENAI_API_KEY).
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from skills.providers import openai_ocr

                return openai_ocr(), "OpenAI vision (gpt-4o-mini)"
            except Exception as exc:
                print(f"[skills] OpenAI OCR lỗi: {exc}", file=sys.stderr)
        # EasyOCR có thể tự tải checkpoint lớn ngay lúc khởi tạo. Chỉ bật khi
        # người dùng chủ động chọn fallback offline, không để UI treo bất ngờ.
        if os.environ.get("ENABLE_EASYOCR") == "1":
            try:
                from skills.providers import easyocr_fn

                return easyocr_fn(), "EasyOCR"
            except Exception as exc:
                print(f"[skills] EasyOCR lỗi: {exc}", file=sys.stderr)
        # Không có công cụ nào -> None (bên gọi sẽ dùng StubOCR).
        return None

    # _modality: đoán câu hỏi đang nói về đồ vật hay khuôn mặt.
    # Có từ gợi ý đồ vật (OBJECT_HINT) -> "object"; còn lại mặc định "face".
    def _modality(self, query: str) -> str:
        low = query.lower()
        return "object" if any(w in low for w in self.OBJECT_HINT) else "face"

    # _parse_label: tách "tên" ra khỏi câu dạy/sửa.
    # Cách làm đơn giản: lấy phần chữ đứng SAU chữ "là".
    # VD "đây là Lan" -> "Lan". Không thấy "là" -> trả None (chưa rõ tên).
    @staticmethod
    def _parse_label(query: str) -> str | None:
        low = query.lower()
        markers = (" là ", "là ")
        for marker in markers:
            if marker in low:
                return query[low.index(marker) + len(marker) :].strip(" .?!")
        return None

    # _embed: biến ảnh -> dấu vân tay số, gọi lại đúng hàm của bộ khung core.
    def _embed(self, frame, modality: str):
        """Embedding từ frame RGB của UI/camera, giữ synthetic cho test CLI.

        InsightFace nhận BGR còn OpenCLIP nhận PIL RGB. Chuẩn hoá ở ranh giới
        orchestrator giúp voice loop và UI không vô tình đưa sai định dạng model.
        """
        if isinstance(self.core.embedder, SyntheticEmbedder):
            return self.core._embed(modality, frame)
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Ảnh đầu vào không hợp lệ. Hãy thử lại với khung hình mới.")
        arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8, copy=False))
        if modality == "face":
            import cv2

            return self.core._embed("face", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        from PIL import Image

        return self.core._embed("object", Image.fromarray(arr))

    def set_safety_provider(self, provider) -> None:
        """Dùng kết quả SafetyMonitor mới nhất để tránh infer vật cản hai lần."""
        self._safety_provider = provider

    def check_safety(self, frame=None) -> dict:
        if self._safety_provider is not None:
            assessment = self._safety_provider()
            if assessment is not None:
                return assessment
        return self.obstacle.check(frame)

    # handle: TRÁI TIM điều phối. Nhận 1 câu tiếng Việt (+ ảnh) và trả lời.
    #   - query: câu người dùng nói/gõ
    #   - frame: khung ảnh hiện tại (từ webcam/upload), có thể None
    #   - label/modality: có thể chỉ định sẵn tên hoặc loại; không thì tự đoán
    # Thứ tự xử lý: sửa font -> SAFETY override -> đoán loại -> dạy/sửa/hỏi
    # -> kỹ năng đọc chữ/mô tả cảnh. Khi đang nguy hiểm, tuyệt đối không gửi
    # frame lên VLM/OCR và cũng không làm thay đổi CPM.
    def _handle(self, query: str, frame=None, label: str | None = None, modality: str | None = None) -> str:
        # Bước 0: sửa lỗi phông tiếng Việt (nếu câu bị mojibake).
        query = _demojibake(query)
        low = query.lower()

        # Bước 1: safety là cổng chặn thực sự, không chỉ là tiền tố câu trả lời.
        # Nhờ vậy cảnh báo không phải chờ VLM/OCR, không ghi nhầm khi người dùng
        # đang di chuyển, và không gửi frame nguy hiểm lên provider cloud.
        safety = self.check_safety(frame)
        if safety.get("danger"):
            return safety["text"]

        # Bước 2: xác định loại đối tượng (mặt hay đồ vật) nếu chưa được chỉ định.
        modality = modality or self._modality(query)

        # Bước 3a: câu có từ khoá DẠY -> ghi vào trí nhớ CPM.
        if any(k in low for k in self.TEACH_KW):
            # Lấy tên: ưu tiên tên truyền sẵn, không thì tách từ trong câu.
            lab = label or self._parse_label(query)
            if not lab:
                # Không rõ tên thì hỏi lại chứ không ghi bừa.
                return "Bạn muốn tôi ghi nhớ với tên gì?"
            # write() = dạy CPM gắn dấu vân tay của ảnh với tên này.
            self.core.cpm[modality].write(self._embed(frame, modality), lab)
            self.core.persist_memory()
            return f"Đã ghi nhớ ({modality}): {lab}."

        # Bước 3b: câu có từ khoá SỬA -> sửa lại nhãn trong trí nhớ.
        if any(k in low for k in self.CORRECT_KW):
            lab = label or self._parse_label(query)
            if not lab:
                return "Tên đúng là gì để tôi sửa lại?"
            # correct() = nhấn mạnh dấu vân tay này thuộc về tên đúng.
            self.core.cpm[modality].correct(self._embed(frame, modality), lab)
            self.core.persist_memory()
            return f"Đã sửa ({modality}): đây là {lab}."

        # Bước 4: câu có từ khoá HỎI -> nhận diện xem là ai/cái gì.
        if any(k in low for k in self.RECOG_KW):
            # recall() so ảnh với những gì đã học; [0] = phỏng đoán khớp nhất.
            res = self.core.cpm[modality].recall(self._embed(frame, modality))[0]
            # known=True: đủ giống -> báo tên; ngược lại -> mời người dùng dạy.
            ans = (
                f"Đây là {res['label']} (tin cậy {res['confidence']:.2f})."
                if res["known"]
                else "Tôi chưa nhận ra người/vật này. Bạn có muốn dạy tôi không?"
            )
            return ans

        # Bước 5: không phải dạy/sửa/hỏi -> thử lần lượt các kỹ năng phụ,
        # xem kỹ năng nào "nhận" câu này (đọc chữ, mô tả cảnh, vật cản...).
        for sk in self._query_skills:
            if sk.can_handle(low):
                result = sk.run(frame, query)
                return result.text

        # Bước 6: không kỹ năng nào nhận -> mặc định nhờ VLM mô tả cảnh.
        return self.scene.run(frame, query).text

    def handle(self, query: str, frame=None, label: str | None = None, modality: str | None = None) -> str:
        """Điểm vào chịu lỗi cho UI/voice loop, không làm sập phiên vì một frame xấu."""
        try:
            return self._handle(query, frame=frame, label=label, modality=modality)
        except ValueError as exc:
            message = str(exc).lower()
            if "không phát hiện" in message or "khong phat hien" in message:
                return "Tôi chưa thấy rõ khuôn mặt. Bạn nhìn thẳng camera, đủ sáng và thử lại nhé."
            if "ảnh" in message or "anh" in message:
                return "Tôi chưa nhận được khung hình hợp lệ. Hãy giữ camera ổn định rồi thử lại nhé."
            return f"Dữ liệu chưa phù hợp để xử lý: {exc}"
        except RuntimeError as exc:
            # Provider/model/camera lỗi được trả về nhẹ nhàng, không lộ traceback.
            return f"Tôi chưa xử lý được lúc này: {exc}"


# smoke: chạy thử nhanh toàn bộ điều phối bằng dữ liệu giả (không cần model/camera).
# Diễn một đoạn hội thoại mẫu để thấy dạy/hỏi/vật cản hoạt động ra sao.
def smoke(skills_mode: str = "stub") -> int:
    a = VisionAssistant(embedder_kind="synthetic", skills_mode=skills_mode)
    print(">>> SMOKE ORCHESTRATOR (stub skills)\n")

    # Hàm nhỏ tiện dụng: in câu người dùng và câu trả lời của trợ lý cho gọn.
    def ask(q, frame=None):
        print(f"[Người dùng] {q}")
        print(f"[Trợ lý]     {a.handle(q, frame=frame)}\n")

    # Dạy rồi hỏi lại mặt người và đồ vật.
    ask("Hãy nhớ đây là Lan", frame="person_lan")
    ask("Ai đây?", frame="person_lan")
    ask("Hãy nhớ đây là ví của tôi", frame="my_wallet")
    ask("Cái gì đây?", frame="my_wallet")
    ask("Trước mặt có gì?")
    ask("Đọc chữ trên biển giúp tôi")

    # Giả lập có vật cản cách 0.8m để kiểm tra tính năng cảnh báo chen ngang.
    print("--- Dựng tình huống có vật cản gần (0.8m) ---\n")
    a.obstacle.set_distance(0.8)
    ask("Trước mặt có gì?")
    ask("Đường có an toàn không?")
    # Ngay cả khi chỉ hỏi "Ai đây?", cảnh báo vật cản vẫn phải xuất hiện ở đầu câu.
    ask("Ai đây?", frame="person_lan")

    # Bỏ vật cản đi (None = không còn nguy hiểm) và kiểm tra lại.
    print("--- Đường thông thoáng trở lại ---\n")
    a.obstacle.set_distance(None)
    ask("Có vật cản gì không?")

    # Chốt chặn: với đôi mắt giả, hỏi lại đúng ảnh vừa dạy phải cho độ tin cậy 1.00.
    assert a.handle("Ai đây?", frame="person_lan").endswith("(tin cậy 1.00).")
    print("SMOKE PASSED")
    return 0


# main: điểm vào khi chạy "python -m app.orchestrator".
# Đọc tuỳ chọn dòng lệnh rồi chạy demo (--smoke) hoặc chỉ khởi tạo trợ lý.
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--embedder", default="synthetic", choices=["synthetic", "real", "auto"])
    # --skills-mode: dùng kỹ năng mô phỏng ("stub") hay thật ("real").
    p.add_argument("--skills-mode", default="stub", choices=["stub", "real"])
    p.add_argument("--obstacle-mode", default="stub", choices=["stub", "real"])
    args = p.parse_args()
    if args.smoke:
        return smoke(args.skills_mode)
    # Không có --smoke: chỉ dựng trợ lý (để nơi khác nhúng vào UI/CLI dùng lại).
    _ = VisionAssistant(
        embedder_kind=args.embedder, skills_mode=args.skills_mode, obstacle_mode=args.obstacle_mode
    )
    print("Đã khởi tạo VisionAssistant. Dùng --smoke để chạy demo, hoặc nhúng VisionAssistant vào UI/CLI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
