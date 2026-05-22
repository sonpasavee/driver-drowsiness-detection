from __future__ import annotations

import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class DriverState(Enum):
    """
    สถานะของคนขับ เรียงจากปกติ → อันตราย

    NO_FACE   : ไม่พบใบหน้า (คนขับออกจากกล้อง หรือกล้องถูกบัง)
    ACTIVE    : ปกติ ตื่นตัวดี
    MILD      : เริ่มมีสัญญาณ ยังไม่เตือน
    WARNING   : ง่วงแล้ว เตือนเบาๆ
    ALERT     : ง่วงมาก เตือนดัง
    CRITICAL  : อันตราย ต้องหยุดรถ
    """
    NO_FACE  = auto()
    ACTIVE   = auto()
    MILD     = auto()
    WARNING  = auto()
    ALERT    = auto()
    CRITICAL = auto()


class StateMachine:
    """
    จัดการ state transition ของคนขับ

    ออกแบบมาเพื่อ production:
    - Hysteresis — ต้องอยู่เกิน threshold นานพอถึงจะ escalate
      (ป้องกัน state กระโดดไปมาจาก score ที่ขึ้นลงชั่วคราว)
    - Cooldown — เตือนแล้วต้องรอก่อนถึงจะเตือนซ้ำ
    - No-face timeout — ถ้าไม่เห็นหน้านานเกินไปถือว่าผิดปกติ
    - Event log — บันทึกทุก state transition พร้อม timestamp
    - Re-alert — ถ้า state ยังสูงอยู่หลัง cooldown จะเตือนซ้ำอัตโนมัติ
    """

    # ต้องอยู่ใน state ใหม่นานแค่ไหน (วินาที) ถึงจะ confirm transition
    _DEFAULT_HYSTERESIS: dict[DriverState, float] = {
        DriverState.MILD:     3.0,
        DriverState.WARNING:  2.0,
        DriverState.ALERT:    1.0,
        DriverState.CRITICAL: 0.5,
    }

    # cooldown หลังเตือนแต่ละ state (วินาที)
    _DEFAULT_COOLDOWN: dict[DriverState, float] = {
        DriverState.MILD:     30.0,
        DriverState.WARNING:  10.0,
        DriverState.ALERT:    5.0,
        DriverState.CRITICAL: 2.0,
    }

    # นานแค่ไหนที่ไม่เห็นหน้า ถึงจะ → NO_FACE (วินาที)
    _DEFAULT_NO_FACE_TIMEOUT: float = 3.0

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        # อ่าน threshold จาก config (override ได้)
        self._mild_thr: float = cfg.get("state_machine.mild_threshold", 40.0)
        self._warn_thr:     float = cfg.state_machine.warn_threshold
        self._alert_thr:    float = cfg.state_machine.alert_threshold
        self._critical_thr: float = cfg.state_machine.critical_threshold
        self._cooldown_sec: float = cfg.state_machine.cooldown_sec
        self._no_face_timeout: float = cfg.get(
            "state_machine.no_face_timeout_sec",
            self._DEFAULT_NO_FACE_TIMEOUT,
        )
        self._hysteresis: dict[DriverState, float] = {
            DriverState.MILD: float(
                cfg.get(
                    "state_machine.hysteresis_mild_sec",
                    self._DEFAULT_HYSTERESIS[DriverState.MILD],
                )
            ),
            DriverState.WARNING: float(
                cfg.get(
                    "state_machine.hysteresis_warning_sec",
                    self._DEFAULT_HYSTERESIS[DriverState.WARNING],
                )
            ),
            DriverState.ALERT: float(
                cfg.get(
                    "state_machine.hysteresis_alert_sec",
                    self._DEFAULT_HYSTERESIS[DriverState.ALERT],
                )
            ),
            DriverState.CRITICAL: float(
                cfg.get(
                    "state_machine.hysteresis_critical_sec",
                    self._DEFAULT_HYSTERESIS[DriverState.CRITICAL],
                )
            ),
        }
        self._cooldowns: dict[DriverState, float] = {
            DriverState.MILD: float(
                cfg.get(
                    "state_machine.mild_cooldown_sec",
                    self._DEFAULT_COOLDOWN[DriverState.MILD],
                )
            ),
            DriverState.WARNING: float(
                cfg.get(
                    "state_machine.warning_cooldown_sec",
                    self._DEFAULT_COOLDOWN[DriverState.WARNING],
                )
            ),
            DriverState.ALERT: float(
                cfg.get(
                    "state_machine.alert_cooldown_sec",
                    self._DEFAULT_COOLDOWN[DriverState.ALERT],
                )
            ),
            DriverState.CRITICAL: float(
                cfg.get(
                    "state_machine.critical_cooldown_sec",
                    self._DEFAULT_COOLDOWN[DriverState.CRITICAL],
                )
            ),
        }

        # state ปัจจุบัน
        self._state: DriverState         = DriverState.ACTIVE
        self._prev_state: DriverState    = DriverState.ACTIVE

        # hysteresis tracking
        self._candidate_state: DriverState  = DriverState.ACTIVE
        self._candidate_since: float        = time.monotonic()

        # no-face tracking
        self._last_face_time: float = time.monotonic()

        # cooldown tracking (per state)
        self._last_alert_time: dict[DriverState, float] = {}

        # event log
        self._events: list[dict] = []

        # สถิติ
        self._state_durations: dict[DriverState, float] = {
            s: 0.0 for s in DriverState
        }
        self._last_state_start: float = time.monotonic()
        self._session_start: float    = time.monotonic()
        self._total_frames: int       = 0

        logger.info(
            f"StateMachine ready — "
            f"warn={self._warn_thr} "
            f"alert={self._alert_thr} "
            f"critical={self._critical_thr}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        score_result: dict,
        face_detected: bool,
    ) -> dict:
        """
        อัปเดต state จาก score และสถานะการเจอหน้า

        Parameters
        ----------
        score_result  : dict จาก ScoreEngine.update()
        face_detected : bool — LandmarkDetector เจอหน้าไหม

        Returns dict:
            state           : str   — ชื่อ state ปัจจุบัน
            prev_state      : str   — state ก่อนหน้า
            changed         : bool  — state เพิ่งเปลี่ยนไหม
            should_alert    : bool  — ควรเล่นเสียง/ไฟเตือนไหม
            alert_level     : str   — ระดับการเตือน (MILD/WARNING/ALERT/CRITICAL)
            score           : float — score ที่รับเข้ามา
            state_duration  : float — อยู่ใน state นี้นานแค่ไหนแล้ว (วินาที)
            no_face_duration: float — ไม่เห็นหน้ามากี่วินาที
            session_minutes : float — session นี้นานแค่ไหนแล้ว
        """
        now = time.monotonic()
        self._total_frames += 1

        # ----------------------------------------------------------
        # 1. ตรวจ no-face
        # ----------------------------------------------------------
        if face_detected:
            self._last_face_time = now
        no_face_duration = now - self._last_face_time

        state_before_update = self._state

        if no_face_duration >= self._no_face_timeout:
            self._transition_to(DriverState.NO_FACE, now, reason="no_face_timeout")
        else:
            # ----------------------------------------------------------
            # 2. คำนวณ target state จาก score
            # ----------------------------------------------------------
            if score_result.get("calibrating"):
                target = DriverState.ACTIVE
            else:
                score  = score_result.get("score", 0.0)
                target = self._score_to_state(score)

            # ----------------------------------------------------------
            # 3. Hysteresis — ต้องอยู่กับ target นานพอ
            # ----------------------------------------------------------
            self._update_hysteresis(target, now)

        # ----------------------------------------------------------
        # 4. ตรวจว่าควรเตือนไหม
        # ----------------------------------------------------------
        should_alert, alert_level = self._check_should_alert(now)

        # ----------------------------------------------------------
        # 5. อัปเดต duration สถิติ
        # ----------------------------------------------------------
        self._state_durations[self._state] += now - self._last_state_start
        self._last_state_start = now

        changed = self._state != state_before_update
        previous_state = state_before_update
        self._prev_state = previous_state

        state_duration  = now - (
            self._events[-1]["timestamp"]
            if self._events and self._events[-1]["to"] == self._state.name
            else self._session_start
        )
        session_minutes = (now - self._session_start) / 60.0

        return {
            "state":            self._state.name,
            "prev_state":       previous_state.name,
            "changed":          changed,
            "should_alert":     should_alert,
            "alert_level":      alert_level,
            "score":            score_result.get("score", 0.0),
            "state_duration":   round(state_duration, 1),
            "no_face_duration": round(no_face_duration, 1),
            "session_minutes":  round(session_minutes, 1),
        }

    def reset(self) -> None:
        """Reset state machine — เรียกเมื่อเปลี่ยนคนขับ"""
        now = time.monotonic()
        self._state              = DriverState.ACTIVE
        self._prev_state         = DriverState.ACTIVE
        self._candidate_state    = DriverState.ACTIVE
        self._candidate_since    = now
        self._last_face_time     = now
        self._last_alert_time    = {}
        self._events             = []
        self._state_durations    = {s: 0.0 for s in DriverState}
        self._last_state_start   = now
        self._session_start      = now
        self._total_frames       = 0
        logger.info("StateMachine reset")

    @property
    def current_state(self) -> DriverState:
        return self._state

    @property
    def events(self) -> list[dict]:
        """รายการ state transition ทั้งหมดใน session"""
        return list(self._events)

    @property
    def state_summary(self) -> dict[str, float]:
        """สัดส่วนเวลาที่อยู่ใน state ต่างๆ (%)"""
        total = sum(self._state_durations.values()) or 1.0
        return {
            s.name: round(d / total * 100, 1)
            for s, d in self._state_durations.items()
        }

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_hysteresis(self, target: DriverState, now: float) -> None:
        """
        ยืนยัน state ใหม่ก็ต่อเมื่ออยู่กับ target นานพอ
        ป้องกัน state กระโดดจาก score spike ชั่วคราว
        """
        if target != self._candidate_state:
            # target เปลี่ยน → reset timer
            self._candidate_state = target
            self._candidate_since = now
            return

        # target เดิม — เช็คว่าอยู่นานพอไหม
        required = self._hysteresis.get(target, 0.0)
        elapsed  = now - self._candidate_since

        if elapsed >= required:
            if target != self._state:
                self._transition_to(target, now, reason="score_threshold")

    def _transition_to(
        self,
        new_state: DriverState,
        now: float,
        reason: str = "",
    ) -> None:
        """บันทึก state transition และ log"""
        if new_state == self._state:
            return

        old_state  = self._state
        self._state = new_state

        event = {
            "timestamp": now,
            "from":      old_state.name,
            "to":        new_state.name,
            "reason":    reason,
        }
        self._events.append(event)

        # log level ตาม severity
        if new_state in (DriverState.ALERT, DriverState.CRITICAL):
            logger.warning(
                f"State: {old_state.name} → {new_state.name} "
                f"[{reason}]"
            )
        else:
            logger.info(
                f"State: {old_state.name} → {new_state.name} "
                f"[{reason}]"
            )

    def _check_should_alert(self, now: float) -> tuple[bool, str]:
        """
        ตรวจว่าควรเตือนไหม โดยคำนึง cooldown

        Returns (should_alert, alert_level)
        """
        alertable = {
            DriverState.WARNING:  "WARNING",
            DriverState.ALERT:    "ALERT",
            DriverState.CRITICAL: "CRITICAL",
            DriverState.NO_FACE:  "WARNING",
        }

        if self._state not in alertable:
            return False, ""

        level    = alertable[self._state]
        cooldown = self._cooldowns.get(self._state, self._cooldown_sec)
        last     = self._last_alert_time.get(self._state, 0.0)

        if now - last >= cooldown:
            self._last_alert_time[self._state] = now
            return True, level

        return False, ""

    def _score_to_state(self, score: float) -> DriverState:
        thresholds = [
            (self._critical_thr, DriverState.CRITICAL),
            (self._alert_thr, DriverState.ALERT),
            (self._warn_thr, DriverState.WARNING),
            (self._mild_thr, DriverState.MILD),
            (0.0, DriverState.ACTIVE),
        ]
        for threshold, state in thresholds:
            if score >= threshold:
                return state
        return DriverState.ACTIVE


# ----------------------------------------------------------------------
# Visualize helpers
# ----------------------------------------------------------------------

_STATE_COLORS = {
    "NO_FACE":  (100, 100, 100),
    "ACTIVE":   (0, 220, 0),
    "MILD":     (0, 220, 180),
    "WARNING":  (0, 165, 255),
    "ALERT":    (0, 60, 255),
    "CRITICAL": (0, 0, 255),
}

_STATE_ICONS = {
    "NO_FACE":  "👤",
    "ACTIVE":   "✅",
    "MILD":     "😐",
    "WARNING":  "⚠️",
    "ALERT":    "🚨",
    "CRITICAL": "☠️",
}


def draw_state(frame: np.ndarray, result: dict) -> np.ndarray:
    """วาด state badge + alert indicator ลงบน frame"""
    h, w = frame.shape[:2]
    state = result["state"]
    color = _STATE_COLORS.get(state, (200, 200, 200))

    # --- State badge ซ้ายบน ---
    badge = f"[ {state} ]"
    cv2.putText(frame, badge, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # --- ถ้ากำลังเตือน → กรอบแดงกระพริบ ---
    if result["should_alert"]:
        flash = int(time.monotonic() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(
            frame,
            f"! {result['alert_level']} !",
            (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
        )

    # --- Info row ล่างขวา ---
    info_lines = [
        f"Duration : {result['state_duration']:.1f}s",
        f"Session  : {result['session_minutes']:.1f}min",
        f"No face  : {result['no_face_duration']:.1f}s",
    ]
    for i, line in enumerate(reversed(info_lines)):
        cv2.putText(
            frame, line,
            (w - 220, h - 15 - i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
        )

    return frame


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from capture.camera import Camera
    from detectors.landmark import LandmarkDetector
    from analyzers.eye_analyzer import EyeAnalyzer
    from analyzers.mar_analyzer import MARAnalyzer
    from analyzers.head_pose import HeadPoseAnalyzer
    from analyzers.score_engine import ScoreEngine, draw_score

    log = get_logger("state_machine_test")

    with Camera() as cam:
        detector  = LandmarkDetector()
        eye_ana   = EyeAnalyzer()
        mar_ana   = MARAnalyzer()
        head_ana  = HeadPoseAnalyzer()
        engine    = ScoreEngine()
        sm        = StateMachine()

        log.info(
            "กด Q ออก | หลับตา/หาว/ก้มหัว → state เพิ่ม | กด R reset ทุกตัว"
        )

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks     = detector.detect(frame)
            face_detected = landmarks is not None

            if face_detected:
                eye_result   = eye_ana.update(landmarks)
                mar_result   = mar_ana.update(landmarks)
                head_result  = head_ana.update(landmarks, frame.shape[:2])
                score_result = engine.update(eye_result, mar_result, head_result)
            else:
                score_result = {"score": 0.0, "calibrating": False,
                                "level": "ACTIVE", "components": {},
                                "trend": "STABLE", "peak": 0.0,
                                "session_minutes": 0.0}

            sm_result = sm.update(score_result, face_detected)

            # วาด
            frame = draw_score(frame, score_result)
            frame = draw_state(frame, sm_result)

            if sm_result["changed"]:
                log.info(
                    f"State changed: {sm_result['prev_state']} "
                    f"→ {sm_result['state']} "
                    f"(score={sm_result['score']:.1f})"
                )

            if sm_result["should_alert"]:
                log.warning(
                    f"ALERT triggered: level={sm_result['alert_level']} "
                    f"state={sm_result['state']}"
                )

            cv2.imshow("State Machine - Production Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                eye_ana.reset()
                mar_ana.reset()
                head_ana.reset()
                engine.reset()
                sm.reset()
                log.info("All reset")

        cv2.destroyAllWindows()

        log.info("=== Session Summary ===")
        for state_name, pct in sm.state_summary.items():
            if pct > 0:
                log.info(f"  {state_name}: {pct:.1f}%")
        log.info(f"  Events: {len(sm.events)} transitions")
        log.info(f"  Peak score: {engine.peak_score:.1f}")
