from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from alerts import AlertManager
from analyzers.eye_analyzer import EyeAnalyzer
from analyzers.head_pose import HeadPoseAnalyzer
from analyzers.mar_analyzer import MARAnalyzer
from analyzers.score_engine import ScoreEngine
from capture.camera import Camera
from core.state_machine import StateMachine
from detectors.landmark import LandmarkDetector
from storage import SessionWriter
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 13, 14, 17, 0]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]
NOSE_INDEX = 1
FOREHEAD_INDEX = 10
CHIN_INDEX = 152
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454

STATE_COLORS = {
    "NO_FACE": (110, 110, 110),
    "ACTIVE": (0, 220, 0),
    "MILD": (0, 210, 180),
    "WARNING": (0, 165, 255),
    "ALERT": (0, 80, 255),
    "CRITICAL": (0, 0, 255),
}


@dataclass
class RuntimeStats:
    last_frame_ts: float = 0.0
    fps: float = 0.0
    frames: int = 0

    def __post_init__(self) -> None:
        self.last_frame_ts = time.monotonic()

    def update(self) -> float:
        now = time.monotonic()
        delta = max(now - self.last_frame_ts, 1e-6)
        self.fps = 1.0 / delta
        self.last_frame_ts = now
        self.frames += 1
        return self.fps


class DriverMonitoringApp:
    """Production-style application loop for driver drowsiness detection."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self.logger = get_logger(self.__class__.__name__)

        self.window_name = self.cfg.get("app.window_name", "Driver Drowsiness Detection")
        self.show_landmarks = bool(self.cfg.get("app.show_landmarks", True))
        self.show_fps = bool(self.cfg.get("app.show_fps", True))
        self.frame_log_enabled = bool(self.cfg.get("app.frame_log_enabled", False))
        self.frame_log_interval_frames = max(
            1, int(self.cfg.get("app.frame_log_interval_frames", 30))
        )

        self._running = True
        self._stats = RuntimeStats()

        # components
        self.camera = Camera(self.cfg)
        self.detector = LandmarkDetector(self.cfg)
        self.eye_analyzer = EyeAnalyzer(self.cfg)
        self.mar_analyzer = MARAnalyzer(self.cfg)
        self.head_pose_analyzer = HeadPoseAnalyzer(self.cfg)
        self.score_engine = ScoreEngine(self.cfg)
        self.state_machine = StateMachine(self.cfg)
        self.alert_manager = AlertManager(self.cfg)
        self.session_writer = SessionWriter(self.cfg)

        self._register_signal_handlers()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        self.logger.info("Starting driver monitoring application")

        try:
            while self._running:
                frame = self.camera.read()
                if frame is None:
                    continue

                fps = self._stats.update()
                processed_frame = self._process_frame(frame, fps)
                cv2.imshow(self.window_name, processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.logger.info("Quit requested by user")
                    break
                if key == ord("r"):
                    self.logger.info("Manual reset requested")
                    self._reset_runtime()

        except KeyboardInterrupt:
            self.logger.info("Interrupted by keyboard")
        finally:
            self.close()

        return 0

    def close(self) -> None:
        self._running = False

        self.session_writer.finalize(
            {
                "frames": self._stats.frames,
                "peak_score": round(self.score_engine.peak_score, 1),
                "final_state": self.state_machine.current_state.name,
            }
        )

        self.camera.release()
        cv2.destroyAllWindows()

        self.logger.info(
            "Application stopped | "
            "frames=%s peak_score=%.1f final_state=%s "
            "session_file=%s summary_file=%s",
            self._stats.frames,
            self.score_engine.peak_score,
            self.state_machine.current_state.name,
            self.session_writer.path,
            self.session_writer.summary_path,
        )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        landmarks = self.detector.detect(frame)
        face_detected = landmarks is not None

        if face_detected:
            eye_result = self.eye_analyzer.update(landmarks)
            mar_result = self.mar_analyzer.update(landmarks)
            head_result = self.head_pose_analyzer.update(landmarks, frame.shape[:2])
            score_result = self.score_engine.update(eye_result, mar_result, head_result)
        else:
            eye_result = self._empty_eye_result()
            mar_result = self._empty_mar_result()
            head_result = self.head_pose_analyzer._empty_result()
            score_result = self._empty_score_result()

        state_result = self.state_machine.update(score_result, face_detected)

        if state_result["should_alert"]:
            self.alert_manager.trigger(
                state_result["alert_level"],
                state_result["state"],
            )
            self.session_writer.write_event(
                "alert",
                {
                    "level": state_result["alert_level"],
                    "score": state_result["score"],
                    "state": state_result["state"],
                },
            )

        if self.frame_log_enabled:
            self.session_writer.write_frame(
                state=state_result["state"],
                score=score_result["score"],
                eye_result=eye_result,
                mar_result=mar_result,
                head_result=head_result,
            )

        if state_result["changed"]:
            self.session_writer.write_event(
                "state_change",
                {
                    "from": state_result["prev_state"],
                    "to": state_result["state"],
                    "score": state_result["score"],
                },
            )
            self.logger.info(
                "State changed %s -> %s | score=%.1f",
                state_result["prev_state"],
                state_result["state"],
                state_result["score"],
            )

        output = frame.copy()
        if self.show_landmarks and face_detected:
            self._draw_minimal_landmarks(output, landmarks, eye_result, mar_result)

        self._draw_compact_hud(
            output,
            fps=fps,
            face_detected=face_detected,
            eye_result=eye_result,
            mar_result=mar_result,
            head_result=head_result,
            score_result=score_result,
            state_result=state_result,
        )

        return output

    # ------------------------------------------------------------------
    # Empty result helpers
    # ------------------------------------------------------------------

    def _empty_eye_result(self) -> dict:
        return {
            "calibrating": False,
            "eye_closed": False,
            "drowsy": False,
            "perclos": 0.0,
            "ear": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "consec": 0,
            "calib_remaining": 0.0,
            "threshold": self.eye_analyzer.threshold,
        }

    def _empty_mar_result(self) -> dict:
        return {
            "calibrating": False,
            "mouth_open": False,
            "yawning": False,
            "mar": 0.0,
            "consec": 0,
            "yawn_count": self.mar_analyzer.yawn_count,
            "yawn_rate": 0.0,
            "calib_remaining": 0.0,
            "threshold": self.mar_analyzer.threshold,
        }

    def _empty_score_result(self) -> dict:
        return {
            "score": 0.0,
            "score_raw": 0.0,
            "level": "ACTIVE",
            "components": {
                "perclos_score": 0.0,
                "ear_score": 0.0,
                "mar_score": 0.0,
                "head_score": 0.0,
            },
            "trend": "STABLE",
            "peak": self.score_engine.peak_score,
            "calibrating": False,
            "session_minutes": 0.0,
        }

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_minimal_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        eye_result: dict,
        mar_result: dict,
    ) -> None:
        eye_color = (0, 0, 255) if eye_result["eye_closed"] else (0, 255, 0)
        mouth_color = (0, 0, 255) if mar_result["yawning"] else (0, 200, 255)

        for eye_idx in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
            pts = np.array(
                [[int(landmarks[i][0]), int(landmarks[i][1])] for i in eye_idx],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], isClosed=True, color=eye_color, thickness=1)

        mouth_pts = np.array(
            [[int(landmarks[i][0]), int(landmarks[i][1])] for i in MOUTH_INDICES],
            dtype=np.int32,
        )
        cv2.polylines(frame, [mouth_pts], isClosed=True, color=mouth_color, thickness=1)

        # Sparse anchor points keep the face readable without cluttering the frame.
        anchor_points = {
            NOSE_INDEX: (255, 255, 0),
            FOREHEAD_INDEX: (255, 200, 0),
            CHIN_INDEX: (255, 200, 0),
            LEFT_CHEEK_INDEX: (200, 200, 200),
            RIGHT_CHEEK_INDEX: (200, 200, 200),
        }
        for idx, color in anchor_points.items():
            x, y = landmarks[idx]
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

        for iris_indices in (LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES):
            iris_points = landmarks[iris_indices]
            iris_center = iris_points.mean(axis=0)
            cv2.circle(
                frame,
                (int(iris_center[0]), int(iris_center[1])),
                2,
                (0, 220, 255),
                -1,
            )

    def _draw_compact_hud(
        self,
        frame: np.ndarray,
        fps: float,
        face_detected: bool,
        eye_result: dict,
        mar_result: dict,
        head_result: dict,
        score_result: dict,
        state_result: dict,
    ) -> None:
        h, w = frame.shape[:2]
        state = state_result["state"]
        state_color = STATE_COLORS.get(state, (220, 220, 220))
        score = float(score_result.get("score", 0.0))
        calibrating = bool(score_result.get("calibrating"))

        panel_w = 260
        panel_h = 74 if calibrating else 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (60, 60, 60), 1)

        if calibrating:
            remaining = max(
                eye_result.get("calib_remaining", 0.0),
                mar_result.get("calib_remaining", 0.0),
                head_result.get("calib_remaining", 0.0),
            )
            cv2.putText(
                frame,
                f"Calibrating  {remaining:.1f}s",
                (24, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 200, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                state,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                state_color,
                2,
            )

            metrics = [
                (
                    f"EAR {eye_result.get('ear', 0.0):.3f}  "
                    f"PERCLOS {eye_result.get('perclos', 0.0):.0f}%  "
                    f"Yawns {mar_result.get('yawn_count', 0)}"
                ),
                (
                    f"P {head_result.get('pitch', 0.0):+.0f} deg  "
                    f"Y {head_result.get('yaw', 0.0):+.0f} deg  "
                    f"R {head_result.get('roll', 0.0):+.0f} deg  "
                    f"Score {score:.0f}"
                ),
            ]
            for i, line in enumerate(metrics):
                cv2.putText(
                    frame,
                    line,
                    (24, 68 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )

        badge_w = 160
        badge_h = 44
        badge_x1 = w - badge_w - 14
        badge_y1 = 14
        cv2.rectangle(
            frame,
            (badge_x1, badge_y1),
            (badge_x1 + badge_w, badge_y1 + badge_h),
            state_color,
            -1,
        )
        cv2.putText(
            frame,
            "CALIB" if calibrating else state,
            (badge_x1 + 12, badge_y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
        )

        if not face_detected:
            self._draw_banner(frame, "No face detected", (0, 0, 200))

        bar_x1 = 14
        bar_x2 = w - 14
        bar_y1 = h - 38
        bar_y2 = h - 18
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 30), -1)
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * max(0.0, min(score, 100.0)) / 100.0)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x, bar_y2), state_color, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), 1)

        footer_parts = []
        if self.show_fps:
            footer_parts.append(f"FPS {fps:.0f}")
        footer_parts.append(f"Score {score:.0f}/100")
        footer_parts.append(f"{score_result.get('trend', 'STABLE')}")
        footer_parts.append(f"{score_result.get('session_minutes', 0.0):.1f}min")
        cv2.putText(
            frame,
            "   ".join(footer_parts),
            (14, h - 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
        )

    def _draw_banner(self, frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        top = max(h - 90, 0)
        cv2.rectangle(overlay, (14, top), (w - 14, h - 48), color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (14, top), (w - 14, h - 48), color, 1)
        cv2.putText(
            frame,
            text,
            (28, h - 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )

    # ------------------------------------------------------------------
    # Reset + signal
    # ------------------------------------------------------------------

    def _reset_runtime(self) -> None:
        self.eye_analyzer.reset()
        self.mar_analyzer.reset()
        self.head_pose_analyzer.reset()
        self.score_engine.reset()
        self.state_machine.reset()
        self.session_writer.write_event(
            "manual_reset",
            {
                "frame": self._stats.frames,
            },
        )

    def _register_signal_handlers(self) -> None:
        def _handle_signal(signum, _frame) -> None:
            self.logger.info("Received signal=%s, stopping", signum)
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except ValueError:
                pass


def main() -> int:
    app = DriverMonitoringApp()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
