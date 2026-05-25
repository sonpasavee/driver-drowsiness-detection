from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np

# Support running this file directly via `python detectors/landmark.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe landmark indices that the rest of the project can reuse.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOUTH = [61, 291, 13, 14, 17, 0]
NOSE_TIP = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
FOREHEAD = 10


class LandmarkDetector:
    """
    Detect facial landmarks and return pixel coordinates as `(N, 2)`.

    The detector supports two MediaPipe layouts:
    - legacy `mp.solutions.face_mesh.FaceMesh`
    - newer `mp.tasks.vision.FaceLandmarker`
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()

        self._detection_confidence: float = cfg.detector.detection_confidence
        self._tracking_confidence: float = cfg.detector.tracking_confidence
        self._max_num_faces: int = cfg.detector.max_num_faces
        self._model_asset_path = Path(
            cfg.detector.get("model_asset_path", "assets/face_landmarker_v2.task")
        )

        self._backend_name = "unknown"
        self._face_mesh: Any = self._create_backend()

        self._detection_count: int = 0
        self._no_face_count: int = 0

        self._warmup()
        logger.debug(
            "LandmarkDetector ready - "
            f"backend={self._backend_name} "
            f"confidence={self._detection_confidence} "
            f"max_faces={self._max_num_faces}"
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks from a BGR frame and return pixel coordinates."""
        if not self._validate_frame(frame_bgr):
            return None

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self._backend_name == "solutions":
            result = self._face_mesh.process(frame_rgb)
            if not result.multi_face_landmarks:
                self._no_face_count += 1
                return None

            self._detection_count += 1
            lm = result.multi_face_landmarks[0].landmark
            return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._face_mesh.detect(mp_image)
        if not result.face_landmarks:
            self._no_face_count += 1
            return None

        self._detection_count += 1
        lm = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def draw(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[np.ndarray],
        draw_iris: bool = True,
    ) -> np.ndarray:
        """Draw landmarks on top of a copy of the input frame."""
        frame = frame_bgr.copy()

        if landmarks is None:
            cv2.putText(
                frame,
                "No face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = landmarks[idx]
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

        if draw_iris:
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x, y = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, (255, 150, 0), -1)

        for idx in MOUTH:
            x, y = landmarks[idx]
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 255), -1)

        return frame

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def no_face_count(self) -> int:
        return self._no_face_count

    def _warmup(self) -> None:
        """Send one blank frame so MediaPipe initializes before first use."""
        logger.debug(f"Warmup MediaPipe backend={self._backend_name}...")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        if self._backend_name == "solutions":
            self._face_mesh.process(blank)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
            self._face_mesh.detect(mp_image)

        logger.debug("Warmup complete")

    def _create_backend(self) -> Any:
        solutions = getattr(mp, "solutions", None)
        face_mesh_module = getattr(solutions, "face_mesh", None)
        if face_mesh_module is not None:
            self._backend_name = "solutions"
            return face_mesh_module.FaceMesh(
                max_num_faces=self._max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self._detection_confidence,
                min_tracking_confidence=self._tracking_confidence,
            )

        if not self._model_asset_path.exists():
            raise FileNotFoundError(
                "MediaPipe Tasks backend detected but model file is missing: "
                f"{self._model_asset_path.resolve()}\n"
                "Download `face_landmarker_v2.task` into `assets/` or set "
                "`detector.model_asset_path` in `configs/system.yaml`."
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_asset_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=self._max_num_faces,
            min_face_detection_confidence=self._detection_confidence,
            min_face_presence_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        self._backend_name = "tasks"
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> bool:
        """Validate that the frame looks like a color image."""
        if frame is None:
            logger.warning("Received frame=None")
            return False
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return False
        return True


if __name__ == "__main__":
    from capture.camera import Camera

    log = get_logger("landmark_test")

    with Camera() as cam:
        detector = LandmarkDetector()
        log.info("Press Q to exit")

        while True:
            frame = cam.read()
            if frame is None:
                continue

            landmarks = detector.detect(frame)
            frame = detector.draw(frame, landmarks)

            status = (
                f"detected={detector.detection_count} "
                f"no_face={detector.no_face_count}"
            )
            cv2.putText(
                frame,
                status,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.imshow("Landmark - Production Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        log.info(
            f"Summary: detect={detector.detection_count} "
            f"no_face={detector.no_face_count}"
        )
