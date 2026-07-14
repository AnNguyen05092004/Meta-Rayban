"""JSONL log tối thiểu cho nghiệm thu, không lưu ảnh/audio/API key."""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path


class InteractionLogger:
    def __init__(self, root: str | Path = ".local_logs", *, user_id: str = "demo"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
            raise ValueError("user_id log chỉ gồm chữ, số, dấu gạch dưới hoặc gạch nối (tối đa 64 ký tự).")
        self.path = Path(root) / f"voice_{user_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **record) -> None:
        payload = {"timestamp": datetime.now(UTC).isoformat(), **record}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
