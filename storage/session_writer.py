from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionWriter:
    """
    บันทึกข้อมูล session ลงไฟล์ JSON เดียว

    โครงสร้าง:
    sessions/
    └── session_20260522_135000/
        ├── session.json   ← timeseries + events + summary ครบในไฟล์เดียว
        └── summary.json   ← สรุปอย่างเดียว (สำหรับ dashboard โหลดเร็ว)

    session.json มี 3 section:
        timeseries : เก็บทุก 1 วินาที — ใช้วาดกราฟ
        events     : เฉพาะ state_change + alert — ใช้แสดง timeline
        summary    : สรุปภาพรวม — ใช้แสดง card บน dashboard
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        self._enabled: bool             = bool(cfg.get("app.save_session_metrics", True))
        self._timeseries_interval: float = float(cfg.get("app.timeseries_interval_sec", 1.0))

        # สร้าง session directory
        session_root              = Path(cfg.get("app.session_dir", "sessions"))
        self._session_id          = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir         = session_root / f"session_{self._session_id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._session_path: Path  = self._session_dir / "session.json"
        self._summary_path: Path  = self._session_dir / "summary.json"

        # in-memory data structure
        self._data: dict[str, Any] = {
            "session_id":       self._session_id,
            "driver_id":        cfg.get("app.driver_id", "unknown"),
            "started_at":       datetime.now().isoformat(timespec="seconds"),
            "ended_at":         None,
            "duration_minutes": 0.0,
            "timeseries":       [],
            "events":           [],
            "summary": {
                "total_frames":    0,
                "peak_score":      0.0,
                "avg_score":       0.0,
                "alerts_count":    0,
                "yawn_count":      0,
                "manual_resets":   0,
                "state_breakdown": {
                    "NO_FACE":  0.0,
                    "ACTIVE":   0.0,
                    "MILD":     0.0,
                    "WARNING":  0.0,
                    "ALERT":    0.0,
                    "CRITICAL": 0.0,
                },
            },
        }

        # tracking
        self._session_start:     float        = time.monotonic()
        self._last_timeseries_t: float        = self._session_start
        self._score_accumulator: list[float]  = []
        self._state_time_start:  float        = self._session_start
        self._current_state:     str          = "ACTIVE"
        self._state_durations:   dict[str, float] = {
            s: 0.0 for s in
            ("NO_FACE", "ACTIVE", "MILD", "WARNING", "ALERT", "CRITICAL")
        }

        if self._enabled:
            # เขียนไฟล์เปล่าก่อน ป้องกันข้อมูลหายถ้าโปรแกรม crash
            self._flush()
            logger.debug(
                f"SessionWriter ready — "
                f"session_id={self._session_id} "
                f"path={self._session_dir}"
            )
        else:
            logger.debug("SessionWriter disabled (app.save_session_metrics=false)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_frame(
        self,
        state:       str,
        score:       float,
        eye_result:  dict,
        mar_result:  dict,
        head_result: dict,
    ) -> None:
        """
        เรียกทุก frame จาก _process_frame()
        บันทึก timeseries จริงเฉพาะทุก interval วินาที
        แต่อัปเดต summary ทุก frame เสมอ
        """
        if not self._enabled:
            return

        now     = time.monotonic()
        elapsed = now - self._session_start

        # อัปเดต summary ทุก frame
        self._data["summary"]["total_frames"] += 1
        self._score_accumulator.append(score)
        self._data["summary"]["peak_score"] = max(
            self._data["summary"]["peak_score"], score
        )

        yawn = mar_result.get("yawn_count", 0)
        if yawn > self._data["summary"]["yawn_count"]:
            self._data["summary"]["yawn_count"] = yawn

        # อัปเดต state duration
        if state != self._current_state:
            self._state_durations[self._current_state] += now - self._state_time_start
            self._current_state    = state
            self._state_time_start = now

        # บันทึก timeseries ทุก interval วินาที
        if now - self._last_timeseries_t >= self._timeseries_interval:
            self._data["timeseries"].append({
                "t":          round(elapsed, 1),
                "score":      round(score, 1),
                "ear":        round(eye_result.get("ear", 0.0), 3),
                "perclos":    round(eye_result.get("perclos", 0.0), 1),
                "mar":        round(mar_result.get("mar", 0.0), 3),
                "yawn_count": mar_result.get("yawn_count", 0),
                "pitch":      round(head_result.get("pitch", 0.0), 1),
                "yaw":        round(head_result.get("yaw", 0.0), 1),
                "roll":       round(head_result.get("roll", 0.0), 1),
                "state":      state,
            })
            self._last_timeseries_t = now

            # flush ทุก interval ป้องกันข้อมูลหายถ้า crash
            self._flush()

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """
        บันทึก event สำคัญ — state_change, alert, manual_reset
        เรียกเฉพาะเมื่อเกิดเหตุการณ์ ไม่ต้องเรียกทุก frame
        """
        if not self._enabled:
            return

        now     = time.monotonic()
        elapsed = now - self._session_start

        event = {
            "t":    round(elapsed, 1),
            "type": event_type,
            **payload,
        }
        self._data["events"].append(event)

        # อัปเดต summary ตาม event type
        if event_type == "alert":
            self._data["summary"]["alerts_count"] += 1
        elif event_type == "manual_reset":
            self._data["summary"]["manual_resets"] += 1

        # flush ทันทีเมื่อมี event สำคัญ
        self._flush()
        logger.debug(f"Event recorded: {event_type} t={elapsed:.1f}s")

    def finalize(self, final_payload: dict[str, Any]) -> None:
        """
        เรียกตอนปิดโปรแกรม
        คำนวณ summary สุดท้ายแล้ว flush ทุกอย่างลงดิสก์
        """
        if not self._enabled:
            return

        now = time.monotonic()

        # อัปเดต duration ของ state สุดท้าย
        self._state_durations[self._current_state] += now - self._state_time_start

        # คำนวณ avg score
        if self._score_accumulator:
            self._data["summary"]["avg_score"] = round(
                sum(self._score_accumulator) / len(self._score_accumulator), 1
            )

        # คำนวณ state breakdown เป็น %
        total_time = sum(self._state_durations.values()) or 1.0
        self._data["summary"]["state_breakdown"] = {
            state: round(dur / total_time * 100, 1)
            for state, dur in self._state_durations.items()
        }

        # อัปเดต metadata
        duration_min                       = (now - self._session_start) / 60.0
        self._data["ended_at"]             = datetime.now().isoformat(timespec="seconds")
        self._data["duration_minutes"]     = round(duration_min, 2)
        self._data["summary"].update(final_payload)

        # เขียน session.json ครบ
        self._flush()

        # เขียน summary.json แยก (dashboard โหลดเร็ว ไม่ต้องโหลด timeseries)
        summary_out = {
            "session_id":       self._data["session_id"],
            "driver_id":        self._data["driver_id"],
            "started_at":       self._data["started_at"],
            "ended_at":         self._data["ended_at"],
            "duration_minutes": self._data["duration_minutes"],
            **self._data["summary"],
        }
        self._summary_path.write_text(
            json.dumps(summary_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"Session finalized — "
            f"duration={duration_min:.1f}min "
            f"frames={self._data['summary']['total_frames']} "
            f"peak={self._data['summary']['peak_score']:.1f} "
            f"alerts={self._data['summary']['alerts_count']}"
        )
        logger.info(f"Session saved  → {self._session_path}")
        logger.info(f"Summary saved  → {self._summary_path}")

    @property
    def path(self) -> Path:
        return self._session_path

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    @property
    def session_id(self) -> str:
        return self._session_id

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """เขียน in-memory data ลงไฟล์ทันที"""
        try:
            self._session_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"SessionWriter flush failed: {e}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.finalize({})
