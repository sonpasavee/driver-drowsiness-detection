from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Support running this file directly via `python capture/camera.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class Camera:
    """
    Camera wrapper for production-style usage.
    - reconnects if camera access drops
    - validates frames to skip nearly-black images
    - supports context manager usage
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()
        self._index: int = cfg.camera.index
        self._width: int = cfg.camera.width
        self._height: int = cfg.camera.height
        self._fps: int = cfg.camera.fps
        self._reconnect_attempts: int = cfg.camera.reconnect_attempts
        self._reconnect_delay: float = cfg.camera.reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._fail_count: int = 0

        self._open()

    def read(self) -> Optional[np.ndarray]:
        """Read one frame and reconnect automatically when needed."""
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Camera is not ready, reconnecting...")
            if not self._reconnect():
                return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            self._fail_count += 1
            logger.warning(f"Failed to read frame (attempt {self._fail_count})")
            if self._fail_count >= 3:
                logger.error("Failed to read frame 3 times in a row, reconnecting...")
                self._reconnect()
                self._fail_count = 0
            return None

        if not self._is_valid_frame(frame):
            logger.debug("Received an invalid or too-dark frame, skipping")
            return None

        self._fail_count = 0
        self._frame_count += 1
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug(f"Camera released after {self._frame_count} frames")

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def _open(self) -> bool:
        """Open camera and configure resolution / fps."""
        logger.debug(f"Opening camera index={self._index}")
        cap = cv2.VideoCapture(self._index)

        if not cap.isOpened():
            logger.error(f"Cannot open camera index={self._index}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.debug(
            f"Camera ready - resolution={actual_w}x{actual_h} fps={actual_fps:.1f}"
        )

        self._cap = cap
        return True

    def _reconnect(self) -> bool:
        """Reconnect camera based on configured retry settings."""
        self.release()
        for attempt in range(1, self._reconnect_attempts + 1):
            logger.warning(
                f"Reconnect attempt {attempt}/{self._reconnect_attempts}..."
            )
            time.sleep(self._reconnect_delay)
            if self._open():
                logger.warning("Reconnect successful")
                return True
        logger.error("Reconnect failed")
        return False

    @staticmethod
    def _is_valid_frame(frame: np.ndarray) -> bool:
        """Reject nearly black frames."""
        return frame.mean() > 5.0


if __name__ == "__main__":
    log = get_logger("camera_test")

    try:
        with Camera() as cam:
            log.info("Press Q to exit")

            while True:
                frame = cam.read()
                if frame is None:
                    continue

                info = (
                    f"frame={cam.frame_count} "
                    f"size={frame.shape[1]}x{frame.shape[0]}"
                )
                cv2.putText(
                    frame,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        log.info("stop")
    finally:
        cv2.destroyAllWindows()
        log.info("Camera closed")
