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

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _compute_ear(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """
    Eye Aspect Ratio (EAR)
    สูตร: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    ค่าปกติตาเปิด  : ~0.25–0.35
    ค่าตาปิด       : < 0.20
    """
    p = landmarks[eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


class EyeAnalyzer:
    """
    วิเคราะห์สถานะตาจาก facial landmarks

    มี 3 phase:
    1. CALIBRATING — 5 วินาทีแรก เก็บค่า EAR baseline ของคนขับ
    2. ACTIVE      — ปกติ ตาเปิด
    3. DROWSY      — ตาปิดติดต่อกันเกิน threshold
    """

    # จำนวนวินาทีที่ calibrate
    _CALIB_DURATION: float = 5.0
    # ต้องได้ sample อย่างน้อยกี่ตัวถึงจะ calibrate สำเร็จ
    _CALIB_MIN_SAMPLES: int = 30
    # threshold = baseline × ratio นี้
    _CALIB_RATIO: float = 0.75

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        # ค่า fallback จาก config ถ้า calibrate ไม่สำเร็จ
        self._fallback_threshold: float = cfg.analyzer.ear_threshold
        self._perclos_window: float     = cfg.analyzer.perclos_window_sec
        self._consec_frames: int        = cfg.analyzer.consec_drowsy_frames

        # --- Calibration state ---
        self._calibrating: bool           = True
        self._calib_samples: list[float]  = []
        self._calib_start: float          = time.monotonic()
        self._calib_remaining: float      = self._CALIB_DURATION
        self._adaptive_threshold: Optional[float] = None

        # --- Runtime state ---
        self._perclos_buffer: deque[tuple[float, bool]] = deque()
        self._consec_count: int  = 0
        self._total_frames: int  = 0
        self._closed_frames: int = 0

        logger.info(
            f"EyeAnalyzer ready — calibrating {self._CALIB_DURATION}s | "
            f"fallback_threshold={self._fallback_threshold} | "
            f"perclos_window={self._perclos_window}s | "
            f"consec_frames={self._consec_frames}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks: np.ndarray) -> dict:
        """
        รับ landmark array (468, 2) แล้วคืนผลการวิเคราะห์

        Returns dict:
            ear          : float  — EAR เฉลี่ยสองตา
            ear_left     : float  — EAR ตาซ้าย
            ear_right    : float  — EAR ตาขวา
            eye_closed   : bool   — ตาปิดอยู่ตอนนี้ไหม
            consec       : int    — frame ปิดติดต่อกัน
            perclos      : float  — % (0–100)
            drowsy       : bool   — ง่วงหรือไม่
            calibrating  : bool   — ยังอยู่ใน calibration phase ไหม
            calib_remaining : float — วินาทีที่เหลือของ calibration
            threshold    : float  — threshold ที่ใช้จริงตอนนี้
        """
        now = time.monotonic()
        self._total_frames += 1

        ear_left  = _compute_ear(landmarks, LEFT_EYE)
        ear_right = _compute_ear(landmarks, RIGHT_EYE)
        ear       = (ear_left + ear_right) / 2.0

        # อัปเดต calibration ก่อนเสมอ
        self._update_calibration(ear, now)

        eye_closed = ear < self.threshold

        # อัปเดต consecutive counter
        if eye_closed:
            self._consec_count += 1
            self._closed_frames += 1
        else:
            self._consec_count = 0

        # อัปเดต PERCLOS buffer
        self._perclos_buffer.append((now, eye_closed))
        self._trim_perclos_buffer(now)
        perclos = self._compute_perclos()

        # ยังไม่ตัดสิน drowsy ระหว่าง calibrate
        drowsy = (
            not self._calibrating
            and self._consec_count >= self._consec_frames
        )

        return {
            "ear":              round(ear, 4),
            "ear_left":         round(ear_left, 4),
            "ear_right":        round(ear_right, 4),
            "eye_closed":       eye_closed,
            "consec":           self._consec_count,
            "perclos":          round(perclos, 2),
            "drowsy":           drowsy,
            "calibrating":      self._calibrating,
            "calib_remaining":  round(self._calib_remaining, 1),
            "threshold":        round(self.threshold, 3),
        }

    def reset(self) -> None:
        """
        Reset ทุกอย่างกลับไปเริ่ม calibration ใหม่
        เรียกเมื่อเปลี่ยนคนขับ
        """
        self._calibrating        = True
        self._calib_samples      = []
        self._calib_start        = time.monotonic()
        self._calib_remaining    = self._CALIB_DURATION
        self._adaptive_threshold = None
        self._perclos_buffer.clear()
        self._consec_count  = 0
        self._total_frames  = 0
        self._closed_frames = 0
        logger.info("EyeAnalyzer reset — recalibrating...")

    @property
    def threshold(self) -> float:
        """threshold ที่ใช้จริง — adaptive ถ้า calibrate สำเร็จ / fallback ถ้าไม่สำเร็จ"""
        return self._adaptive_threshold if self._adaptive_threshold else self._fallback_threshold

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def closed_ratio(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._closed_frames / self._total_frames

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_calibration(self, ear: float, now: float) -> None:
        """
        เก็บ EAR sample ระหว่าง calibration phase

        ขั้นตอน:
        1. เก็บค่า EAR ทุก frame ใน 5 วินาทีแรก
        2. พอครบเวลา คำนวณ baseline = percentile 80 (กัน outlier จากตาปิดช่วงสั้นๆ)
        3. threshold = baseline × 0.75
        4. ถ้าได้ sample น้อยเกินไป ใช้ fallback threshold แทน
        """
        if not self._calibrating:
            return

        elapsed = now - self._calib_start
        self._calib_remaining = max(0.0, self._CALIB_DURATION - elapsed)

        if elapsed < self._CALIB_DURATION:
            self._calib_samples.append(ear)
            return

        # ครบเวลาแล้ว — คำนวณ threshold
        if len(self._calib_samples) >= self._CALIB_MIN_SAMPLES:
            # percentile 80 = ค่าที่ EAR ส่วนใหญ่ตอนตาเปิดอยู่ใต้นี้
            # ทำให้ outlier จากตากระพริบหรือหันหน้าไม่กระทบ
            baseline = float(np.percentile(self._calib_samples, 80))
            self._adaptive_threshold = baseline * self._CALIB_RATIO
            logger.info(
                f"Calibration complete — "
                f"samples={len(self._calib_samples)} "
                f"baseline={baseline:.3f} "
                f"threshold={self._adaptive_threshold:.3f}"
            )
        else:
            logger.warning(
                f"Calibration ได้ sample แค่ {len(self._calib_samples)} "
                f"(ต้องการ {self._CALIB_MIN_SAMPLES}+) — ใช้ fallback={self._fallback_threshold}"
            )

        self._calibrating = False

    def _trim_perclos_buffer(self, now: float) -> None:
        cutoff = now - self._perclos_window
        while self._perclos_buffer and self._perclos_buffer[0][0] < cutoff:
            self._perclos_buffer.popleft()

    def _compute_perclos(self) -> float:
        if not self._perclos_buffer:
            return 0.0
        closed = sum(1 for _, is_closed in self._perclos_buffer if is_closed)
        return (closed / len(self._perclos_buffer)) * 100.0


# ----------------------------------------------------------------------
# Visualize helpers
# ----------------------------------------------------------------------

def draw_eye_status(
    frame: np.ndarray,
    result: dict,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- Calibration phase — แสดง countdown แล้วออกเลย ---
    if result["calibrating"]:
        remaining = result["calib_remaining"]

        # overlay สีเหลืองโปร่งแสง
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 180, 180), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(
            frame,
            f"Calibrating... look at camera normally ({remaining:.1f}s)",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
        )

        # countdown bar
        bar_w = int(w * (1.0 - remaining / EyeAnalyzer._CALIB_DURATION))
        cv2.rectangle(frame, (0, 55), (bar_w, 60), (0, 200, 200), -1)
        return frame

    # --- วาดจุดตา ---
    if landmarks is not None:
        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = landmarks[idx]
            color = (0, 0, 255) if result["eye_closed"] else (0, 255, 0)
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

    # --- Status text ---
    if result["drowsy"]:
        status_color = (0, 0, 255)
        status_text  = "DROWSY!"
    elif result["eye_closed"]:
        status_color = (0, 165, 255)
        status_text  = "EYES CLOSED"
    else:
        status_color = (0, 255, 0)
        status_text  = "ACTIVE"

    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # --- Metrics ---
    metrics = [
        f"EAR       : {result['ear']:.3f}  (L={result['ear_left']:.3f} R={result['ear_right']:.3f})",
        f"Threshold : {result['threshold']:.3f}  (adaptive)",
        f"PERCLOS   : {result['perclos']:.1f}%",
        f"Consec    : {result['consec']} frames",
    ]
    for i, text in enumerate(metrics):
        cv2.putText(frame, text, (10, 68 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # --- PERCLOS bar ---
    bar_x, bar_y, bar_w, bar_h = 10, h - 30, 220, 12
    filled    = int(bar_w * min(result["perclos"], 100) / 100.0)
    bar_color = (0, 0, 255) if result["perclos"] > 30 else (0, 200, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_color, -1)
    cv2.putText(frame, f"PERCLOS {result['perclos']:.0f}%",
                (bar_x + bar_w + 8, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from capture.camera import Camera
    from detectors.landmark import LandmarkDetector

    log = get_logger("eye_analyzer_test")

    with Camera() as cam:
        detector = LandmarkDetector()
        analyzer = EyeAnalyzer()
        log.info("กด Q เพื่อออก | หลับตาค้าง 1 วิ ดู DROWSY | กด R เพื่อ recalibrate")

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                result = analyzer.update(landmarks)
                frame  = draw_eye_status(frame, result, landmarks)

                if not result["calibrating"]:
                    log.debug(
                        f"EAR={result['ear']:.3f} "
                        f"thr={result['threshold']:.3f} "
                        f"PERCLOS={result['perclos']:.1f}% "
                        f"consec={result['consec']} "
                        f"drowsy={result['drowsy']}"
                    )
            else:
                cv2.putText(frame, "No face detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Eye Analyzer - Production Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                analyzer.reset()
                log.info("Manual recalibrate triggered")

        cv2.destroyAllWindows()
        log.info(
            f"สรุป: total={analyzer.total_frames} "
            f"closed_ratio={analyzer.closed_ratio:.2%} "
            f"threshold={analyzer.threshold:.3f}"
        )