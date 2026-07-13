"""
CPM configuration.

Continual Personalization Memory (CPM) — lõi Nested-Learning cho đồ án trợ thị.
Một CPM phục vụ MỘT không gian embedding (một modality). Face (ArcFace) và
object (CLIP) nằm ở hai không gian khác nhau -> tạo 2 instance riêng.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Cấu hình của MỘT tầng trí nhớ.
# tiers fast/medium/slow = 3 tầng trí nhớ cập nhật ở tốc độ khác nhau (Continuum Memory System
# của Nested Learning). Chỉ dùng khi bật ma trận liên kết (thành phần tuỳ chọn).
@dataclass
class TierConfig:
    """Cấu hình một tầng bộ nhớ (Continuum Memory System)."""

    name: str
    alpha: float   # cổng giữ (retention) nhân vào toàn ma trận mỗi lần ghi; 1.0 = không phân rã
    eta: float     # tốc độ học của delta-rule
    weight: float  # trọng số khi tổng hợp recall giữa các tầng


# Đây là bảng "núm vặn" của CPM: gom mọi tham số điều chỉnh vào một chỗ.
# modality = loại đối tượng (face/object) — mỗi loại tạo một CPMConfig riêng.
@dataclass
class CPMConfig:
    """Cấu hình CPM cho một modality."""

    # dim = số chiều của dấu vân tay số (embedding) — dãy 512 số đặc trưng cho ảnh.
    dim: int = 512

    # 3 tầng theo tần số: fast (thích nghi nhanh, quên nhanh) -> slow (bền, chống quên)
    tiers: list[TierConfig] = field(
        default_factory=lambda: [
            TierConfig("fast", alpha=0.90, eta=1.0, weight=0.3),
            TierConfig("medium", alpha=0.99, eta=0.7, weight=0.6),
            TierConfig("slow", alpha=1.00, eta=0.5, weight=1.0),
        ]
    )

    recall_threshold: float = 0.35    # cos(query, prototype) tối thiểu để coi là "người/đồ quen"
                                      # (không-gian embedding, giống ngưỡng verification của face rec)
    correct_eta_boost: float = 2.0    # nhân eta khi sửa (correct) để ghi mạnh vào ma trận (nếu bật)
    consolidate_hit_count: int = 3    # số lần xác nhận trước khi bền hoá vào slow (chỉ khi bật ma trận)
    anchor_seed: int = 1234           # seed sinh anchor tất định theo nhãn (chỉ khi bật ma trận)

    # XƯƠNG SỐNG NHẬN DIỆN = prototype trung-bình-thật (bền như NCM, bộ nhớ bị chặn).
    # Ma trận liên kết (associative memory theo HOPE) TẮT mặc định: thí nghiệm cho thấy nó
    # giới hạn sức chứa ~dim và yếu khi các danh tính giống nhau (xem docs/KET_QUA_THI_NGHIEM.md).
    # Bật lại (use_associative_matrix=True) để phân tích/ablation.
    use_associative_matrix: bool = False
    default_recall_mode: str = "proto"    # "proto" (mặc định) | "assoc" | "hybrid" (cần bật ma trận)
    hybrid_w_assoc: float = 0.5
    hybrid_w_proto: float = 1.0

    # Tầng prototype nhanh bằng EMA trong embedding-space. Tắt mặc định để giữ đúng hành vi NCM cũ;
    # bật trong thí nghiệm drift/deploy khi cần bám ngoại hình gần đây.
    use_ema: bool = False
    ema_alpha: float = 0.35       # retention của EMA; thấp hơn = bám mẫu mới nhanh hơn
    ema_weight: float = 0.65      # score = (1-w)*proto_slow + w*proto_ema

    # Tìm cấu hình của một tầng theo tên ("fast"/"medium"/"slow").
    # Vào: tên tầng. Ra: TierConfig tương ứng; không tìm thấy thì báo lỗi KeyError.
    def tier(self, name: str) -> TierConfig:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(name)
