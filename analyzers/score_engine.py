from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class ScoreEngine:
    """
    รวมสัญญาณจาก EyeAnalyzer / MARAnalyzer / HeadPoseAnalyzer
    เป็น Drowsiness Score เดียว (0–100)

    สูตร (weighted sum):
        score = (PERCLOS   × 0.40)   ← ตัวชี้วัดหลัก
              + (EAR_score × 0.25)   ← ตาหรี่ลงเรื่อยๆ
              + (MAR_score × 0.20)   ← หาวบ่อย
              + (Head_score× 0.15)   ← หัวก้ม/หัน

    น้ำหนักอ้างอิงจากงานวิจัย NHTSA และ ISO 15007

    Score interpretation:
        0  – 40  : ACTIVE    — ตื่นตัวดี
        40 – 60  : MILD      — เริ่มมีสัญญาณ
        60 – 80  : WARNING   — ควรเตือน
        80 – 90  : ALERT     — เตือนดัง
        90 – 100 : CRITICAL  — อันตรายมาก
    """

    # น้ำหนักแต่ละ signal (รวมกันต้องได้ 1.0)
    _W_PERCLOS = 0.40
    _W_EAR = 0.25
    _W_MAR = 0.20
    _W_HEAD = 0.15
    _DEFAULT_SMOOTH_WINDOW = 10
    _DEFAULT_HISTORY_WINDOW = 300

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        # โหลด weight จาก config ถ้ามี (override default)
        self._w_perclos: float = cfg.get("scoring.weight_perclos") or self._W_PERCLOS
        self._w_ear:     float = cfg.get("scoring.weight_ear")     or self._W_EAR
        self._w_mar:     float = cfg.get("scoring.weight_mar")     or self._W_MAR
        self._w_head:    float = cfg.get("scoring.weight_head")    or self._W_HEAD
        self._smooth_window: int = int(
            cfg.get("scoring.smooth_window", self._DEFAULT_SMOOTH_WINDOW)
        )
        self._history_window: int = int(
            cfg.get("scoring.history_window_frames", self._DEFAULT_HISTORY_WINDOW)
        )
        self._eye_consec_score_frames: int = int(
            cfg.get("scoring.eye_consec_score_frames", 20)
        )
        self._head_consec_score_frames: int = int(
            cfg.get("scoring.head_consec_score_frames", 20)
        )
        self._perclos_max_percent: float = float(
            cfg.get("scoring.perclos_max_percent", 80.0)
        )
        self._yawn_rate_max_per_min: float = float(
            cfg.get("scoring.yawn_rate_max_per_min", 6.0)
        )
        self._eye_drowsy_min_score: float = float(
            cfg.get("scoring.eye_drowsy_min_score", 80.0)
        )
        self._yawning_min_score: float = float(
            cfg.get("scoring.yawning_min_score", 60.0)
        )
        self._distracted_min_score: float = float(
            cfg.get("scoring.distracted_min_score", 70.0)
        )

        # normalize ให้รวมกันได้ 1.0 เสมอ (กันกรณีแก้ config ผิด)
        total = self._w_perclos + self._w_ear + self._w_mar + self._w_head
        self._w_perclos /= total
        self._w_ear     /= total
        self._w_mar     /= total
        self._w_head    /= total

        # smoothing buffer
        self._score_buf: deque[float] = deque(maxlen=self._smooth_window)

        # history สำหรับ trend
        self._score_history: deque[tuple[float, float]] = deque(
            maxlen=self._history_window
        )  # (timestamp, score)

        # สถิติ session
        self._total_frames:   int   = 0
        self._peak_score:     float = 0.0
        self._session_start:  float = time.monotonic()

        logger.info(
            f"ScoreEngine ready — weights: "
            f"PERCLOS={self._w_perclos:.2f} "
            f"EAR={self._w_ear:.2f} "
            f"MAR={self._w_mar:.2f} "
            f"HEAD={self._w_head:.2f}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        eye_result:  dict,
        mar_result:  dict,
        head_result: dict,
    ) -> dict:
        """
        รับผลจาก analyzer ทั้ง 3 ตัว แล้วคืน score รวม

        Parameters
        ----------
        eye_result  : dict จาก EyeAnalyzer.update()
        mar_result  : dict จาก MARAnalyzer.update()
        head_result : dict จาก HeadPoseAnalyzer.update()

        Returns dict:
            score           : float  — 0–100 smooth score
            score_raw       : float  — 0–100 ก่อน smooth
            level           : str    — ACTIVE / MILD / WARNING / ALERT / CRITICAL
            components      : dict   — breakdown แต่ละ signal (0–100 ต่อตัว)
            trend           : str    — RISING / FALLING / STABLE
            peak            : float  — score สูงสุดใน session นี้
            calibrating     : bool   — true ถ้ายัง calibrate อยู่สักตัว
            session_minutes : float  — เวลาขับรถใน session นี้ (นาที)
        """
        now = time.monotonic()
        self._total_frames += 1

        # ถ้ายัง calibrate อยู่ คืน score 0 ก่อน
        calibrating = (
            eye_result.get("calibrating", False)
            or mar_result.get("calibrating", False)
            or head_result.get("calibrating", False)
        )

        if calibrating:
            return self._calibrating_result(now)

        # ----------------------------------------------------------
        # คำนวณ component score แต่ละตัว (0–100)
        # ----------------------------------------------------------

        # 1. PERCLOS score — linear map 0–80% → 0–100
        perclos       = eye_result.get("perclos", 0.0)
        perclos_score = min(perclos / max(self._perclos_max_percent, 0.1) * 100.0, 100.0)

        # 2. EAR score — ยิ่ง EAR ต่ำ ยิ่ง score สูง
        #    คำนวณจาก consec frames เทียบกับ threshold
        consec_eye    = eye_result.get("consec", 0)
        consec_thr = max(self._eye_consec_score_frames, 1)
        ear_score     = min(consec_eye / consec_thr * 100.0, 100.0)

        # boost ถ้า drowsy flag ขึ้น
        if eye_result.get("drowsy", False):
            ear_score = max(ear_score, self._eye_drowsy_min_score)

        # 3. MAR score — yawn_rate + yawning flag
        yawn_rate  = mar_result.get("yawn_rate", 0.0)   # ครั้ง/นาที
        mar_score  = min(yawn_rate / max(self._yawn_rate_max_per_min, 0.1) * 100.0, 100.0)
        if mar_result.get("yawning", False):
            mar_score = max(mar_score, self._yawning_min_score)

        # 4. HEAD score — distraction
        consec_pitch = head_result.get("consec_pitch", 0)
        consec_yaw   = head_result.get("consec_yaw",   0)
        consec_roll  = head_result.get("consec_roll",  0)
        max_consec   = max(consec_pitch, consec_yaw, consec_roll)
        head_score = min(max_consec / max(self._head_consec_score_frames, 1) * 100.0, 100.0)
        if head_result.get("distracted", False):
            head_score = max(head_score, self._distracted_min_score)

        # ----------------------------------------------------------
        # Weighted sum
        # ----------------------------------------------------------
        score_raw = (
            perclos_score * self._w_perclos
            + ear_score   * self._w_ear
            + mar_score   * self._w_mar
            + head_score  * self._w_head
        )
        score_raw = float(np.clip(score_raw, 0.0, 100.0))

        # Smooth
        self._score_buf.append(score_raw)
        score = float(np.mean(self._score_buf))

        # อัปเดต history
        self._score_history.append((now, score))
        self._peak_score = max(self._peak_score, score)

        # level
        level = self._score_to_level(score)

        # trend
        trend = self._compute_trend()

        session_minutes = (now - self._session_start) / 60.0

        if self._total_frames % 30 == 0:  # log ทุก ~1 วินาที
            logger.debug(
                f"Score={score:.1f} ({level}) | "
                f"PERCLOS={perclos_score:.0f} "
                f"EAR={ear_score:.0f} "
                f"MAR={mar_score:.0f} "
                f"HEAD={head_score:.0f} | "
                f"trend={trend}"
            )

        return {
            "score":     round(score, 1),
            "score_raw": round(score_raw, 1),
            "level":     level,
            "components": {
                "perclos_score": round(perclos_score, 1),
                "ear_score":     round(ear_score, 1),
                "mar_score":     round(mar_score, 1),
                "head_score":    round(head_score, 1),
            },
            "trend":           trend,
            "peak":            round(self._peak_score, 1),
            "calibrating":     False,
            "session_minutes": round(session_minutes, 1),
        }

    def reset(self) -> None:
        """Reset session ใหม่ เช่น เมื่อเปลี่ยนคนขับ"""
        self._score_buf.clear()
        self._score_history.clear()
        self._total_frames  = 0
        self._peak_score    = 0.0
        self._session_start = time.monotonic()
        logger.info("ScoreEngine reset")

    @property
    def peak_score(self) -> float:
        return self._peak_score

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score < 40:
            return "ACTIVE"
        elif score < 60:
            return "MILD"
        elif score < 80:
            return "WARNING"
        elif score < 90:
            return "ALERT"
        else:
            return "CRITICAL"

    def _compute_trend(self) -> str:
        """
        เปรียบเทียบ score เฉลี่ย 3 วินาทีแรก vs 3 วินาทีหลัง
        ใน history เพื่อบอกทิศทาง
        """
        if len(self._score_history) < 60:
            return "STABLE"

        scores = [s for _, s in self._score_history]
        first_half = np.mean(scores[:len(scores) // 2])
        last_half  = np.mean(scores[len(scores) // 2:])
        diff = last_half - first_half

        if diff > 5:
            return "RISING"
        elif diff < -5:
            return "FALLING"
        return "STABLE"

    def _calibrating_result(self, now: float) -> dict:
        session_minutes = (now - self._session_start) / 60.0
        return {
            "score": 0.0, "score_raw": 0.0,
            "level": "ACTIVE",
            "components": {
                "perclos_score": 0.0,
                "ear_score":     0.0,
                "mar_score":     0.0,
                "head_score":    0.0,
            },
            "trend": "STABLE",
            "peak":  0.0,
            "calibrating": True,
            "session_minutes": round(session_minutes, 1),
        }


# ----------------------------------------------------------------------
# Visualize helpers
# ----------------------------------------------------------------------

# สีประจำ level
_LEVEL_COLORS = {
    "ACTIVE":   (0, 220, 0),
    "MILD":     (0, 220, 180),
    "WARNING":  (0, 165, 255),
    "ALERT":    (0, 60, 255),
    "CRITICAL": (0, 0, 255),
}

_TREND_SYMBOLS = {
    "RISING":  "▲ RISING",
    "FALLING": "▼ FALLING",
    "STABLE":  "● STABLE",
}


def draw_score(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    วาด Drowsiness Score dashboard ลงบน frame
    แสดง: score gauge, level, components breakdown, trend
    """
    h, w = frame.shape[:2]

    if result.get("calibrating"):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (40, 40, 80), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "Calibrating all analyzers...",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        return frame

    score = result["score"]
    level = result["level"]
    color = _LEVEL_COLORS.get(level, (200, 200, 200))

    # --- Score gauge (แถบนอน) ---
    gx, gy, gw, gh = 10, h - 60, w - 20, 18
    # พื้นหลัง gradient สี
    for i, (seg_color) in enumerate([
        (0, 220, 0),      # 0–40   ACTIVE
        (0, 220, 180),    # 40–60  MILD
        (0, 165, 255),    # 60–80  WARNING
        (0, 60, 255),     # 80–90  ALERT
        (0, 0, 255),      # 90–100 CRITICAL
    ]):
        seg_starts = [0, 0.40, 0.60, 0.80, 0.90]
        seg_ends   = [0.40, 0.60, 0.80, 0.90, 1.00]
        sx = gx + int(gw * seg_starts[i])
        ex = gx + int(gw * seg_ends[i])
        cv2.rectangle(frame, (sx, gy), (ex, gy + gh), seg_color, -1)

    # overlay สีเทาส่วนที่ยังไม่ถึง
    filled = gx + int(gw * score / 100.0)
    cv2.rectangle(frame, (filled, gy), (gx + gw, gy + gh), (30, 30, 30), -1)

    # เส้นชี้ตำแหน่ง score
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (80, 80, 80), 1)
    cv2.line(frame, (filled, gy - 4), (filled, gy + gh + 4), (255, 255, 255), 2)

    # label score
    cv2.putText(frame, f"DROWSINESS: {score:.0f}/100",
                (gx, gy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Level badge ---
    badge_text = f"  {level}  "
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    bx, by = w - bw - 20, 15
    cv2.rectangle(frame, (bx - 6, by - 4), (bx + bw + 6, by + bh + 4),
                  color, -1)
    cv2.putText(frame, badge_text, (bx, by + bh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # --- Components breakdown ---
    comps = result["components"]
    comp_items = [
        ("PERCLOS", comps["perclos_score"], _W_PERCLOS_LABEL := "40%"),
        ("EAR",     comps["ear_score"],     "25%"),
        ("YAWN",    comps["mar_score"],     "20%"),
        ("HEAD",    comps["head_score"],    "15%"),
    ]
    for i, (name, val, weight) in enumerate(comp_items):
        bx2 = 10
        by2 = 10 + i * 22
        bar_len = int(120 * val / 100.0)
        bar_col = (0, 0, 255) if val > 70 else (0, 180, 255) if val > 40 else (0, 200, 100)
        cv2.rectangle(frame, (bx2 + 60, by2 + 2),
                      (bx2 + 60 + 120, by2 + 16), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx2 + 60, by2 + 2),
                      (bx2 + 60 + bar_len, by2 + 16), bar_col, -1)
        cv2.putText(frame, f"{name}({weight})", (bx2, by2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(frame, f"{val:.0f}", (bx2 + 185, by2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # --- Trend + Peak + Session ---
    trend_sym = _TREND_SYMBOLS.get(result["trend"], "●")
    trend_col = (0, 80, 255) if result["trend"] == "RISING" else \
                (0, 200, 100) if result["trend"] == "FALLING" else \
                (180, 180, 180)
    cv2.putText(frame, trend_sym, (10, h - 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, trend_col, 1)
    cv2.putText(frame,
                f"Peak={result['peak']:.0f}  Session={result['session_minutes']:.1f}min",
                (120, h - 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

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

    log = get_logger("score_engine_test")

    with Camera() as cam:
        detector  = LandmarkDetector()
        eye_ana   = EyeAnalyzer()
        mar_ana   = MARAnalyzer()
        head_ana  = HeadPoseAnalyzer()
        engine    = ScoreEngine()

        log.info(
            "กด Q เพื่อออก | "
            "หลับตา/หาว/ก้มหัว → score ขึ้น | "
            "กด R เพื่อ reset ทุกตัว"
        )

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                eye_result  = eye_ana.update(landmarks)
                mar_result  = mar_ana.update(landmarks)
                head_result = head_ana.update(landmarks, frame.shape[:2])
                score_result = engine.update(eye_result, mar_result, head_result)

                frame = draw_score(frame, score_result)

                # แสดง sub-status มุมขวาล่าง
                sub = []
                if eye_result.get("drowsy"):
                    sub.append("DROWSY_EYE")
                if mar_result.get("yawning"):
                    sub.append("YAWNING")
                if head_result.get("distracted"):
                    sub.append("HEAD_DIST")
                if sub:
                    cv2.putText(frame, " | ".join(sub),
                                (10, frame.shape[0] - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            else:
                cv2.putText(frame, "No face detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Score Engine - Production Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                eye_ana.reset()
                mar_ana.reset()
                head_ana.reset()
                engine.reset()
                log.info("All analyzers reset")

        cv2.destroyAllWindows()
        log.info(
            f"สรุป: frames={engine.total_frames} "
            f"peak_score={engine.peak_score:.1f}"
        )
