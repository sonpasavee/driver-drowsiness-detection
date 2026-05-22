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

# MediaPipe landmark indices สำหรับปาก
# outer lip: มุมซ้าย, มุมขวา, บนกลาง, ล่างกลาง
# inner lip: บนใน, ล่างใน
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_TOP_L  = 82
MOUTH_TOP_R  = 312
MOUTH_BOT_L  = 87
MOUTH_BOT_R  = 317

# จุดทั้งหมดที่ใช้วาด
MOUTH_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
]


def _compute_mar(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio (MAR)

    สูตร: MAR = (|top-bottom| ซ้าย + |top-bottom| กลาง + |top-bottom| ขวา)
                ─────────────────────────────────────────────────────────────
                                  2 × |left-right|

    ค่าปกติปากปิด  : 0.0 – 0.3
    ค่าปากอ้า      : 0.3 – 0.5
    ค่าหาว         : > 0.5
    """
    left  = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top   = landmarks[MOUTH_TOP]
    bot   = landmarks[MOUTH_BOTTOM]
    tl    = landmarks[MOUTH_TOP_L]
    bl    = landmarks[MOUTH_BOT_L]
    tr    = landmarks[MOUTH_TOP_R]
    br    = landmarks[MOUTH_BOT_R]

    # vertical distances 3 จุด
    A = np.linalg.norm(tl - bl)
    B = np.linalg.norm(top - bot)
    C = np.linalg.norm(tr - br)

    # horizontal distance
    D = np.linalg.norm(left - right)

    if D < 1e-6:
        return 0.0

    return float((A + B + C) / (2.0 * D))


class MARAnalyzer:
    """
    ตรวจจับการหาว (yawning) จาก Mouth Aspect Ratio

    มี 3 phase เหมือน EyeAnalyzer:
    1. CALIBRATING — 5 วินาทีแรก เก็บ baseline MAR ปากปิดปกติ
    2. ACTIVE      — ปากปิดปกติ
    3. YAWNING     — ปากอ้ากว้างเกิน threshold ติดต่อกัน

    นับจำนวนครั้งที่หาวสะสมด้วย (yawn_count)
    """

    _DEFAULT_CALIB_DURATION: float = 5.0
    _DEFAULT_CALIB_MIN_SAMPLES: int = 30
    _DEFAULT_CALIB_RATIO: float = 1.8

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        self._fallback_threshold: float = cfg.analyzer.mar_threshold
        self._consec_frames: int        = cfg.analyzer.consec_drowsy_frames
        self._calib_duration: float = cfg.get(
            "analyzer.mar_calibration_duration_sec",
            self._DEFAULT_CALIB_DURATION,
        )
        self._calib_min_samples: int = int(
            cfg.get(
                "analyzer.mar_calibration_min_samples",
                self._DEFAULT_CALIB_MIN_SAMPLES,
            )
        )
        self._calib_ratio: float = cfg.get(
            "analyzer.mar_calibration_ratio",
            self._DEFAULT_CALIB_RATIO,
        )
        self._yawn_window: float = cfg.get("analyzer.yawn_window_sec", 30.0)

        # --- Calibration ---
        self._calibrating: bool          = True
        self._calib_samples: list[float] = []
        self._calib_start: float         = time.monotonic()
        self._calib_remaining: float     = self._calib_duration
        self._adaptive_threshold: Optional[float] = None

        # --- Runtime state ---
        self._consec_count: int = 0
        self._yawning: bool     = False
        self._yawn_count: int   = 0          # นับครั้งที่หาวสะสม
        self._yawn_start: Optional[float] = None  # timestamp เริ่มหาว

        # history ย้อนหลัง 30 วินาที สำหรับ yawn rate
        self._yawn_history: deque[float] = deque()
        self._total_frames: int = 0

        logger.info(
            f"MARAnalyzer ready — calibrating {self._calib_duration}s | "
            f"fallback_threshold={self._fallback_threshold} | "
            f"consec_frames={self._consec_frames}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks: np.ndarray) -> dict:
        """
        รับ landmark array (468, 2) แล้วคืนผลการวิเคราะห์

        Returns dict:
            mar              : float — MAR ค่าปัจจุบัน
            mouth_open       : bool  — ปากอ้าอยู่ไหม
            yawning          : bool  — กำลังหาวอยู่ไหม
            consec           : int   — frame ที่ปากอ้าติดต่อกัน
            yawn_count       : int   — จำนวนครั้งที่หาวสะสม
            yawn_rate        : float — ครั้ง/นาที ใน 30 วิที่ผ่านมา
            calibrating      : bool
            calib_remaining  : float
            threshold        : float — MAR threshold ที่ใช้จริง
        """
        now = time.monotonic()
        self._total_frames += 1

        mar = _compute_mar(landmarks)

        self._update_calibration(mar, now)

        mouth_open = mar > self.threshold

        # อัปเดต consecutive
        if mouth_open:
            self._consec_count += 1
        else:
            # ปากปิดแล้ว — ถ้าเพิ่งจบการหาว บันทึกไว้
            if self._yawning:
                self._record_yawn(now)
            self._consec_count = 0
            self._yawning = False
            self._yawn_start = None

        yawning = (
            not self._calibrating
            and self._consec_count >= self._consec_frames
        )

        # เริ่มหาวครั้งใหม่
        if yawning and not self._yawning:
            self._yawning = True
            self._yawn_start = now
            logger.debug(f"Yawn started (count will be={self._yawn_count + 1})")

        self._trim_yawn_history(now)
        yawn_rate = self._compute_yawn_rate()

        return {
            "mar":             round(mar, 4),
            "mouth_open":      mouth_open,
            "yawning":         yawning,
            "consec":          self._consec_count,
            "yawn_count":      self._yawn_count,
            "yawn_rate":       round(yawn_rate, 2),
            "calibrating":     self._calibrating,
            "calib_remaining": round(self._calib_remaining, 1),
            "calib_duration":  round(self._calib_duration, 1),
            "threshold":       round(self.threshold, 3),
        }

    def reset(self) -> None:
        """Reset ทุกอย่าง recalibrate ใหม่ เรียกเมื่อเปลี่ยนคนขับ"""
        self._calibrating        = True
        self._calib_samples      = []
        self._calib_start        = time.monotonic()
        self._calib_remaining    = self._calib_duration
        self._adaptive_threshold = None
        self._consec_count       = 0
        self._yawning            = False
        self._yawn_count         = 0
        self._yawn_start         = None
        self._yawn_history.clear()
        self._total_frames       = 0
        logger.info("MARAnalyzer reset — recalibrating...")

    @property
    def threshold(self) -> float:
        return self._adaptive_threshold if self._adaptive_threshold else self._fallback_threshold

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def yawn_count(self) -> int:
        return self._yawn_count

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_calibration(self, mar: float, now: float) -> None:
        """
        เก็บ MAR sample ระหว่างปากปิดปกติ 5 วินาทีแรก
        threshold = percentile 80 × 1.8
        ทำให้ต้องอ้าปากกว้างกว่าปกติมากๆ ถึงจะนับว่าหาว
        """
        if not self._calibrating:
            return

        elapsed = now - self._calib_start
        self._calib_remaining = max(0.0, self._calib_duration - elapsed)

        if elapsed < self._calib_duration:
            self._calib_samples.append(mar)
            return

        if len(self._calib_samples) >= self._calib_min_samples:
            baseline = float(np.percentile(self._calib_samples, 80))
            self._adaptive_threshold = baseline * self._calib_ratio
            logger.info(
                f"MAR Calibration complete — "
                f"samples={len(self._calib_samples)} "
                f"baseline={baseline:.3f} "
                f"threshold={self._adaptive_threshold:.3f}"
            )
        else:
            logger.warning(
                f"MAR Calibration ได้ sample แค่ {len(self._calib_samples)} "
                f"— ใช้ fallback={self._fallback_threshold}"
            )

        self._calibrating = False

    def _record_yawn(self, now: float) -> None:
        """บันทึก yawn event พร้อม timestamp"""
        self._yawn_count += 1
        self._yawn_history.append(now)
        duration = (now - self._yawn_start) if self._yawn_start else 0.0
        logger.info(
            f"Yawn recorded — count={self._yawn_count} "
            f"duration={duration:.1f}s"
        )

    def _trim_yawn_history(self, now: float) -> None:
        cutoff = now - self._yawn_window
        while self._yawn_history and self._yawn_history[0] < cutoff:
            self._yawn_history.popleft()

    def _compute_yawn_rate(self) -> float:
        """ครั้งที่หาวต่อนาที คำนวณจาก window 30 วินาที"""
        if not self._yawn_history:
            return 0.0
        return len(self._yawn_history) * (60.0 / self._yawn_window)


# ----------------------------------------------------------------------
# Visualize helpers
# ----------------------------------------------------------------------

def draw_mar_status(
    frame: np.ndarray,
    result: dict,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- Calibration phase ---
    if result["calibrating"]:
        remaining = result["calib_remaining"]
        duration = max(float(result.get("calib_duration", MARAnalyzer._DEFAULT_CALIB_DURATION)), 0.1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 140, 140), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(
            frame,
            f"Calibrating mouth... keep mouth closed ({remaining:.1f}s)",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2,
        )
        bar_w = int(w * (1.0 - remaining / duration))
        cv2.rectangle(frame, (0, 55), (bar_w, 60), (0, 180, 180), -1)
        return frame

    # --- วาดขอบปาก ---
    if landmarks is not None:
        for idx in MOUTH_OUTLINE:
            x, y = landmarks[idx]
            color = (0, 0, 255) if result["yawning"] else \
                    (0, 165, 255) if result["mouth_open"] else \
                    (0, 255, 0)
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

    # --- Status ---
    if result["yawning"]:
        status_text  = "YAWNING!"
        status_color = (0, 0, 255)
    elif result["mouth_open"]:
        status_text  = "MOUTH OPEN"
        status_color = (0, 165, 255)
    else:
        status_text  = "ACTIVE"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # --- Metrics ---
    metrics = [
        f"MAR        : {result['mar']:.3f}",
        f"Threshold  : {result['threshold']:.3f}  (adaptive)",
        f"Consec     : {result['consec']} frames",
        f"Yawn count : {result['yawn_count']} ครั้ง",
        f"Yawn rate  : {result['yawn_rate']:.1f} /min",
    ]
    for i, text in enumerate(metrics):
        cv2.putText(frame, text, (10, 68 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # --- MAR bar ---
    bar_x, bar_y, bar_w, bar_h = 10, h - 30, 220, 12
    filled    = int(bar_w * min(result["mar"] / max(result["threshold"] * 2, 0.01), 1.0))
    bar_color = (0, 0, 255) if result["yawning"] else \
                (0, 165, 255) if result["mouth_open"] else \
                (0, 200, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_color, -1)
    # เส้น threshold
    thr_x = int(bar_w * 0.5)
    cv2.line(frame, (bar_x + thr_x, bar_y - 2),
             (bar_x + thr_x, bar_y + bar_h + 2), (255, 255, 0), 1)
    cv2.putText(frame, f"MAR {result['mar']:.2f}",
                (bar_x + bar_w + 8, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from capture.camera import Camera
    from detectors.landmark import LandmarkDetector

    log = get_logger("mar_analyzer_test")

    with Camera() as cam:
        detector = LandmarkDetector()
        analyzer = MARAnalyzer()
        log.info("กด Q เพื่อออก | อ้าปากกว้างๆ ดู YAWNING | กด R เพื่อ recalibrate")

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                result = analyzer.update(landmarks)
                frame  = draw_mar_status(frame, result, landmarks)

                if not result["calibrating"]:
                    log.debug(
                        f"MAR={result['mar']:.3f} "
                        f"thr={result['threshold']:.3f} "
                        f"yawning={result['yawning']} "
                        f"count={result['yawn_count']} "
                        f"rate={result['yawn_rate']:.1f}/min"
                    )
            else:
                cv2.putText(frame, "No face detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("MAR Analyzer - Production Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                analyzer.reset()
                log.info("Manual recalibrate triggered")

        cv2.destroyAllWindows()
        log.info(
            f"สรุป: total={analyzer.total_frames} "
            f"yawn_count={analyzer.yawn_count} "
            f"threshold={analyzer.threshold:.3f}"
        )
