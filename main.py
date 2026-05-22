from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from alerts import AlertManager
from analyzers.eye_analyzer import EyeAnalyzer, draw_eye_status
from analyzers.head_pose import HeadPoseAnalyzer, draw_head_pose
from analyzers.mar_analyzer import MARAnalyzer, draw_mar_status
from analyzers.score_engine import ScoreEngine, draw_score
from capture.camera import Camera
from core.state_machine import StateMachine, draw_state
from detectors.landmark import LandmarkDetector
from storage import SessionWriter
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RuntimeStats:
    last_frame_ts: float = time.monotonic()
    fps: float = 0.0
    frames: int = 0

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
        self._running = True
        self._stats = RuntimeStats()

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
        self.session_writer.close()
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info(
            "Application stopped | frames=%s peak_score=%.1f final_state=%s session_file=%s",
            self._stats.frames,
            self.score_engine.peak_score,
            self.state_machine.current_state.name,
            self.session_writer.path,
        )

    def _process_frame(self, frame, fps: float):
        landmarks = self.detector.detect(frame)
        face_detected = landmarks is not None

        if face_detected:
            eye_result = self.eye_analyzer.update(landmarks)
            mar_result = self.mar_analyzer.update(landmarks)
            head_result = self.head_pose_analyzer.update(landmarks, frame.shape[:2])
            score_result = self.score_engine.update(eye_result, mar_result, head_result)
        else:
            eye_result = {
                "calibrating": False,
                "eye_closed": False,
                "drowsy": False,
                "perclos": 0.0,
                "ear": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "consec": 0,
                "threshold": self.eye_analyzer.threshold,
            }
            mar_result = {
                "calibrating": False,
                "mouth_open": False,
                "yawning": False,
                "mar": 0.0,
                "consec": 0,
                "yawn_count": self.mar_analyzer.yawn_count,
                "yawn_rate": 0.0,
                "threshold": self.mar_analyzer.threshold,
            }
            head_result = self.head_pose_analyzer._empty_result()
            score_result = {
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

        state_result = self.state_machine.update(score_result, face_detected)
        if state_result["should_alert"]:
            self.alert_manager.trigger(state_result["alert_level"], state_result["state"])

        output = frame.copy()
        if self.show_landmarks and face_detected:
            output = self.detector.draw(output, landmarks)
        if face_detected:
            output = draw_eye_status(output, eye_result, landmarks)
            output = draw_mar_status(output, mar_result, landmarks)
            output = draw_head_pose(output, head_result, landmarks)
        else:
            cv2.putText(
                output,
                "No face detected",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        output = draw_score(output, score_result)
        if self.alert_manager.overlay_enabled:
            output = draw_state(output, state_result)

        if self.show_fps:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, output.shape[0] - 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

        self.session_writer.write(
            "frame",
            {
                "face_detected": face_detected,
                "fps": round(fps, 2),
                "state": state_result["state"],
                "score": score_result["score"],
                "score_level": score_result["level"],
                "eye": {
                    "ear": eye_result["ear"],
                    "perclos": eye_result["perclos"],
                    "drowsy": eye_result["drowsy"],
                },
                "mouth": {
                    "mar": mar_result["mar"],
                    "yawning": mar_result["yawning"],
                    "yawn_count": mar_result["yawn_count"],
                },
                "head": {
                    "pitch": head_result["pitch"],
                    "yaw": head_result["yaw"],
                    "roll": head_result["roll"],
                    "distracted": head_result["distracted"],
                },
            },
        )

        if state_result["changed"]:
            self.session_writer.write("state_change", state_result)
            self.logger.info(
                "State changed %s -> %s | score=%.1f",
                state_result["prev_state"],
                state_result["state"],
                state_result["score"],
            )

        return output

    def _reset_runtime(self) -> None:
        self.eye_analyzer.reset()
        self.mar_analyzer.reset()
        self.head_pose_analyzer.reset()
        self.score_engine.reset()
        self.state_machine.reset()
        self.session_writer.write("manual_reset", {"frame": self._stats.frames})

    def _register_signal_handlers(self) -> None:
        def _handle_signal(signum, _frame) -> None:
            self.logger.info("Received signal=%s, stopping application", signum)
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except ValueError:
                # Can happen in non-main threads.
                pass


def main() -> int:
    app = DriverMonitoringApp()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
