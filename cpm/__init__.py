"""CPM — Continual Personalization Memory (lõi Nested-Learning)."""

from .config import CPMConfig, TierConfig
from .memory import ContinualPersonalizationMemory, LabelRegistry, TierMemory
from .thresholds import (
    DEFAULT_THRESHOLDS_PATH,
    load_thresholds,
    resolve_threshold,
    save_thresholds,
    threshold_key,
)

# __all__ = danh sách tên được xuất ra khi "from cpm import *" — tức "cửa vào" công khai của gói cpm.
__all__ = [
    "CPMConfig",
    "TierConfig",
    "ContinualPersonalizationMemory",
    "LabelRegistry",
    "TierMemory",
    "DEFAULT_THRESHOLDS_PATH",
    "load_thresholds",
    "save_thresholds",
    "resolve_threshold",
    "threshold_key",
]
