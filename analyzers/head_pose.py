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

# ---------------------------------------------------------------------------
# Reference points สำหรับคำนวณมุมหัว (6 จุด standard สำหรับ solvePnP)
# จุดพวกนี้เลือกเพราะอยู่บนโครงกระดูก ไม่ขยับตาม expression
# ---------------------------------------------------------------------------
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_C  = 263
RIGHT_EYE_C = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57

# 3D model points ของใบหน้ามาตรฐาน (หน่วย mm สมมติ)
# ใช้ค่ามาตรฐานจาก OpenCV face landmark paper
_MODEL_POINTS_3D = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,   -63.6,  -12.5 ],   # Chin
    [-43.3,  32.7,  -26.0 ],   # Left eye corner
    [43.3,   32.7,  -26.0 ],   # Right eye corner
    [-28.9, -28.9,  -24.1 ],   # Left mouth corner
    [28.9,  -28.9,  -24.1 ],   # Right mouth corner
], dtype=np.float64)

LANDMARK_INDICES = [
    NOSE_TIP, CHIN,
    LEFT_EYE_C, RIGHT_EYE_C,
    LEFT_MOUTH, RIGHT_MOUTH,
]


def _build_camera_matrix(w: int, h: int) -> np.ndarray:
    """
    สร้าง camera matrix แบบ approximate
    ใช้เมื่อไม่มี calibration จริง
    focal length ≈ ความกว้างภาพ (rule of thumb)
    """
    focal = w
    cx, cy = w / 2.0, h / 2.0
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ],
    ], dtype=np.float64)


class HeadPoseAnalyzer:
    """
    คำนวณมุมหัว Pitch / Yaw / Roll จาก facial landmarks
    โดยใช้ OpenCV solvePnP (Perspective-n-Point)

    - ไม่ต้องใช้โมเดลพิเศษ คำนวณจาก geometry ล้วนๆ
    - Adaptive baseline — เก็บมุมหัว "ตรง" ของแต่ละคนตอน calibrate
    - ตรวจ distraction 3 แบบ:
        NODDING  = pitch ก้มเกินไป (หลับใน)
        LOOKING_AWAY = yaw หันหน้าออก
        HEAD_TILT = roll เอียงมากผิดปกติ
    """

    _DEFAULT_CALIB_DURATION: float = 5.0
    _DEFAULT_CALIB_MIN_SAMPLES: int = 30
    _DEFAULT_SMOOTH_WINDOW: int = 5

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        # threshold จาก config (องศา)
        self._pitch_threshold: float = cfg.analyzer.head_pitch_threshold
        self._yaw_threshold: float = cfg.analyzer.head_yaw_threshold
        self._roll_threshold: float = cfg.get("analyzer.head_roll_threshold", 20.0)
        self._calib_duration: float = cfg.get(
            "analyzer.head_calibration_duration_sec",
            self._DEFAULT_CALIB_DURATION,
        )
        self._calib_min_samples: int = int(
            cfg.get(
                "analyzer.head_calibration_min_samples",
                self._DEFAULT_CALIB_MIN_SAMPLES,
            )
        )
        self._smooth_window: int = int(
            cfg.get("analyzer.head_smoothing_window", self._DEFAULT_SMOOTH_WINDOW)
        )

        # camera matrix — จะสร้างจริงเมื่อได้ frame แรก
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # --- Calibration ---
        self._calibrating: bool             = True
        self._calib_pitch: list[float]      = []
        self._calib_yaw: list[float]        = []
        self._calib_roll: list[float]       = []
        self._calib_start: float            = time.monotonic()
        self._calib_remaining: float        = self._calib_duration
        self._baseline_pitch: float         = 0.0
        self._baseline_yaw: float           = 0.0
        self._baseline_roll: float          = 0.0

        # --- Smoothing buffer ---
        self._pitch_buf: deque[float] = deque(maxlen=self._smooth_window)
        self._yaw_buf: deque[float] = deque(maxlen=self._smooth_window)
        self._roll_buf: deque[float] = deque(maxlen=self._smooth_window)

        # --- Runtime state ---
        self._consec_pitch: int = 0
        self._consec_yaw:   int = 0
        self._consec_roll:  int = 0
        self._consec_frames: int = cfg.analyzer.consec_drowsy_frames
        self._total_frames: int  = 0

        logger.info(
            f"HeadPoseAnalyzer ready — "
            f"pitch_thr={self._pitch_threshold}° "
            f"yaw_thr={self._yaw_threshold}° "
            f"roll_thr={self._roll_threshold}°"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        landmarks: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> dict:
        """
        คำนวณมุมหัวจาก landmarks

        Parameters
        ----------
        landmarks    : np.ndarray (468, 2) จาก LandmarkDetector
        frame_shape  : (height, width) ของ frame

        Returns dict:
            pitch           : float — องศาก้ม(+)/เงย(-) จาก baseline
            yaw             : float — องศาหันขวา(+)/ซ้าย(-) จาก baseline
            roll            : float — องศาเอียงขวา(+)/ซ้าย(-) จาก baseline
            nodding         : bool  — หัวก้มเกินไป (หลับใน)
            looking_away    : bool  — หันหน้าออกจากกล้อง
            head_tilt       : bool  — หัวเอียงมากผิดปกติ
            distracted      : bool  — มีอย่างใดอย่างหนึ่งข้างบน
            consec_pitch    : int
            consec_yaw      : int
            consec_roll     : int
            calibrating     : bool
            calib_remaining : float
        """
        h, w = frame_shape
        self._total_frames += 1

        # สร้าง camera matrix ครั้งแรก
        if self._camera_matrix is None:
            self._camera_matrix = _build_camera_matrix(w, h)
            logger.debug(f"Camera matrix built for {w}x{h}")

        # เลือก 6 จุด reference
        image_points = np.array(
            [landmarks[i] for i in LANDMARK_INDICES],
            dtype=np.float64,
        )

        # solvePnP — คำนวณ rotation vector
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.warning("solvePnP ล้มเหลว")
            return self._empty_result()

        # แปลง rotation vector → Euler angles (องศา)
        rmat, _ = cv2.Rodrigues(rvec)
        pitch_raw, yaw_raw, roll_raw = self._rotation_matrix_to_euler(rmat)

        # Smooth ด้วย moving average
        self._pitch_buf.append(pitch_raw)
        self._yaw_buf.append(yaw_raw)
        self._roll_buf.append(roll_raw)

        pitch_smooth = float(np.mean(self._pitch_buf))
        yaw_smooth   = float(np.mean(self._yaw_buf))
        roll_smooth  = float(np.mean(self._roll_buf))

        # Calibration
        now = time.monotonic()
        self._update_calibration(pitch_smooth, yaw_smooth, roll_smooth, now)

        # มุมสัมพัทธ์จาก baseline (หัวตรงของคนนั้น)
        pitch = pitch_smooth - self._baseline_pitch
        yaw   = yaw_smooth   - self._baseline_yaw
        roll  = roll_smooth  - self._baseline_roll

        # ตรวจสถานะ
        nodding      = abs(pitch) > self._pitch_threshold
        looking_away = abs(yaw)   > self._yaw_threshold
        head_tilt    = abs(roll)  > self._roll_threshold

        # consecutive counters
        self._consec_pitch = self._consec_pitch + 1 if nodding      else 0
        self._consec_yaw   = self._consec_yaw   + 1 if looking_away else 0
        self._consec_roll  = self._consec_roll  + 1 if head_tilt    else 0

        distracted = (
            not self._calibrating
            and (
                self._consec_pitch >= self._consec_frames
                or self._consec_yaw >= self._consec_frames
                or self._consec_roll >= self._consec_frames
            )
        )

        return {
            "pitch":           round(pitch, 2),
            "yaw":             round(yaw, 2),
            "roll":            round(roll, 2),
            "nodding":         nodding,
            "looking_away":    looking_away,
            "head_tilt":       head_tilt,
            "distracted":      distracted,
            "consec_pitch":    self._consec_pitch,
            "consec_yaw":      self._consec_yaw,
            "consec_roll":     self._consec_roll,
            "calibrating":     self._calibrating,
            "calib_remaining": round(self._calib_remaining, 1),
            "calib_duration":  round(self._calib_duration, 1),
            # สำหรับ draw arrow
            "_rvec":           rvec,
            "_tvec":           tvec,
        }

    def reset(self) -> None:
        """Reset และ recalibrate ใหม่"""
        self._calibrating     = True
        self._calib_pitch     = []
        self._calib_yaw       = []
        self._calib_roll      = []
        self._calib_start     = time.monotonic()
        self._calib_remaining = self._calib_duration
        self._baseline_pitch  = 0.0
        self._baseline_yaw    = 0.0
        self._baseline_roll   = 0.0
        self._pitch_buf.clear()
        self._yaw_buf.clear()
        self._roll_buf.clear()
        self._consec_pitch    = 0
        self._consec_yaw      = 0
        self._consec_roll     = 0
        self._total_frames    = 0
        logger.info("HeadPoseAnalyzer reset — recalibrating...")

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_calibration(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        now: float,
    ) -> None:
        """
        เก็บมุมหัว "ตรง" ของคนขับคนนั้น
        baseline = median ของ sample ทั้งหมด (robust กว่า mean)
        """
        if not self._calibrating:
            return

        elapsed = now - self._calib_start
        self._calib_remaining = max(0.0, self._calib_duration - elapsed)

        if elapsed < self._calib_duration:
            self._calib_pitch.append(pitch)
            self._calib_yaw.append(yaw)
            self._calib_roll.append(roll)
            return

        if len(self._calib_pitch) >= self._calib_min_samples:
            self._baseline_pitch = float(np.median(self._calib_pitch))
            self._baseline_yaw   = float(np.median(self._calib_yaw))
            self._baseline_roll  = float(np.median(self._calib_roll))
            logger.info(
                f"HeadPose Calibration complete — "
                f"baseline pitch={self._baseline_pitch:.1f}° "
                f"yaw={self._baseline_yaw:.1f}° "
                f"roll={self._baseline_roll:.1f}°"
            )
        else:
            logger.warning(
                f"HeadPose Calibration sample น้อย "
                f"({len(self._calib_pitch)}) — ใช้ baseline=0"
            )

        self._calibrating = False

    @staticmethod
    def _rotation_matrix_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
        """
        แปลง 3×3 rotation matrix → Euler angles (องศา)
        คืน (pitch, yaw, roll)
        """
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = 0.0

        return float(pitch), float(yaw), float(roll)

    def _empty_result(self) -> dict:
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "nodding": False, "looking_away": False, "head_tilt": False,
            "distracted": False,
            "consec_pitch": 0, "consec_yaw": 0, "consec_roll": 0,
            "calibrating": self._calibrating,
            "calib_remaining": self._calib_remaining,
            "calib_duration": self._calib_duration,
            "_rvec": None, "_tvec": None,
        }


# ----------------------------------------------------------------------
# Visualize helpers
# ----------------------------------------------------------------------

def draw_head_pose(
    frame: np.ndarray,
    result: dict,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- Calibration phase ---
    if result["calibrating"]:
        remaining = result["calib_remaining"]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (100, 80, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(
            frame,
            f"Calibrating head... look straight ahead ({remaining:.1f}s)",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2,
        )
        duration = max(float(result.get("calib_duration", HeadPoseAnalyzer._DEFAULT_CALIB_DURATION)), 0.1)
        bar_w = int(w * (1.0 - remaining / duration))
        cv2.rectangle(frame, (0, 55), (bar_w, 60), (180, 140, 0), -1)
        return frame

    # --- วาดแกนหัว (pose axis) ---
    if (
        landmarks is not None
        and result["_rvec"] is not None
        and result["_tvec"] is not None
    ):
        _draw_pose_axis(frame, result["_rvec"], result["_tvec"],
                        _build_camera_matrix(w, h))

    # --- Status ---
    if result["distracted"]:
        status_text  = "DISTRACTED!"
        status_color = (0, 0, 255)
    elif result["nodding"] or result["looking_away"] or result["head_tilt"]:
        status_text  = "HEAD WARNING"
        status_color = (0, 165, 255)
    else:
        status_text  = "HEAD OK"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # --- Metrics ---
    def _angle_color(val: float, thr: float) -> tuple:
        return (0, 0, 255) if abs(val) > thr else (220, 220, 220)

    metrics = [
        (f"Pitch : {result['pitch']:+.1f}°  {'NODDING' if result['nodding'] else ''}",
         _angle_color(result["pitch"], 20.0)),
        (f"Yaw   : {result['yaw']:+.1f}°  {'AWAY' if result['looking_away'] else ''}",
         _angle_color(result["yaw"], 35.0)),
        (f"Roll  : {result['roll']:+.1f}°  {'TILT' if result['head_tilt'] else ''}",
         _angle_color(result["roll"], 20.0)),
    ]
    for i, (text, color) in enumerate(metrics):
        cv2.putText(frame, text, (10, 68 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1)

    # --- Gauge วงกลม yaw ---
    cx_g, cy_g, r_g = w - 55, 55, 40
    cv2.circle(frame, (cx_g, cy_g), r_g, (60, 60, 60), 2)
    yaw_clamped = max(-90, min(90, result["yaw"]))
    indicator_x = int(cx_g + r_g * np.sin(np.radians(yaw_clamped)))
    indicator_y = int(cy_g - r_g * np.cos(np.radians(yaw_clamped)) * 0.3)
    cv2.line(frame, (cx_g, cy_g), (indicator_x, indicator_y),
             (0, 200, 255), 2)
    cv2.putText(frame, "YAW", (cx_g - 12, cy_g + r_g + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return frame


def _draw_pose_axis(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    length: float = 50.0,
) -> None:
    """วาดแกน X(แดง) Y(เขียว) Z(น้ำเงิน) จากจมูก"""
    dist = np.zeros((4, 1))
    axis = np.float32([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, -length],
    ])
    origin_3d = np.zeros((1, 3), dtype=np.float32)

    img_pts, _   = cv2.projectPoints(axis,      rvec, tvec, camera_matrix, dist)
    origin_pt, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist)

    o = tuple(origin_pt[0].ravel().astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red Y=green Z=blue

    for pt, color in zip(img_pts, colors):
        p = tuple(pt.ravel().astype(int))
        cv2.arrowedLine(frame, o, p, color, 2, tipLength=0.2)


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from capture.camera import Camera
    from detectors.landmark import LandmarkDetector

    log = get_logger("head_pose_test")

    with Camera() as cam:
        detector = LandmarkDetector()
        analyzer = HeadPoseAnalyzer()
        log.info(
            "กด Q เพื่อออก | "
            "ก้มหัว → NODDING | หันหน้า → LOOKING AWAY | กด R recalibrate"
        )

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                result = analyzer.update(landmarks, frame.shape[:2])
                frame  = draw_head_pose(frame, result, landmarks)

                if not result["calibrating"]:
                    log.debug(
                        f"pitch={result['pitch']:+.1f}° "
                        f"yaw={result['yaw']:+.1f}° "
                        f"roll={result['roll']:+.1f}° "
                        f"distracted={result['distracted']}"
                    )
            else:
                cv2.putText(frame, "No face detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Head Pose - Production Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                analyzer.reset()
                log.info("Manual recalibrate triggered")

        cv2.destroyAllWindows()
        log.info(f"สรุป: total={analyzer.total_frames}")
