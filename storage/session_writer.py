from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionWriter:
    """Persist runtime telemetry to disk in newline-delimited JSON."""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self._enabled = bool(cfg.get("app.save_session_metrics", True))
        session_dir = Path(cfg.get("app.session_dir", "logs/sessions"))
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = session_dir / f"session_{timestamp}.jsonl"
        self._handle = None

        if self._enabled:
            self._handle = self._path.open("a", encoding="utf-8")
            logger.info(f"SessionWriter writing to {self._path}")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record_type: str, payload: dict[str, Any]) -> None:
        if not self._enabled or self._handle is None:
            return

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": record_type,
            "payload": payload,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
