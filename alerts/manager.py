from __future__ import annotations

import platform
import time

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import winsound
except ImportError:  # pragma: no cover - only unavailable off Windows
    winsound = None


class AlertManager:
    """Handle user-facing alerts such as sound cues and alert bookkeeping."""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self._enabled = bool(cfg.get("alerting.enabled", True))
        self._sound_enabled = bool(cfg.get("alerting.sound_enabled", True))
        self._sound_frequency_hz = int(cfg.get("alerting.sound_frequency_hz", 1200))
        self._sound_duration_ms = int(cfg.get("alerting.sound_duration_ms", 350))
        self._speech_enabled = bool(cfg.get("alerting.speech_enabled", False))
        self._overlay_enabled = bool(cfg.get("alerting.overlay_enabled", True))
        self._last_trigger_at = 0.0
        self._last_level = ""

    @property
    def overlay_enabled(self) -> bool:
        return self._overlay_enabled

    def trigger(self, level: str, state: str) -> None:
        if not self._enabled:
            return

        self._last_trigger_at = time.monotonic()
        self._last_level = level
        logger.warning(f"AlertManager trigger level={level} state={state}")
        self._play_sound(level)

        if self._speech_enabled:
            logger.info("speech_enabled=true but speech backend is not implemented yet")

    def _play_sound(self, level: str) -> None:
        if not self._sound_enabled:
            return

        duration_ms = self._sound_duration_ms
        frequency_hz = self._sound_frequency_hz
        if level == "ALERT":
            duration_ms += 150
        elif level == "CRITICAL":
            duration_ms += 300
            frequency_hz += 250

        if winsound is not None and platform.system() == "Windows":
            winsound.Beep(frequency_hz, duration_ms)
            return

        print("\a", end="", flush=True)
