import cv2
import mediapipe as mp
import numpy as np


class LandmarkDetector:
    def __init__(
        self,
        max_num_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def detect(self, frame_bgr):
        """
        รับ frame BGR จาก OpenCV
        คืน numpy array shape (468, 2) เป็น pixel coords
        ถ้าไม่เจอหน้าคืน None
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        return points

    def draw(self, frame_bgr, landmarks):
        """
        วาด landmark ลงบน frame เพื่อ debug
        รับ frame BGR และ numpy array จาก detect()
        """
        if landmarks is None:
            return frame_bgr

        frame = frame_bgr.copy()
        h, w = frame.shape[:2]

        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        return frame


if __name__ == "__main__":
    from capture.camera import Camera

    cam = Camera(index=0)
    detector = LandmarkDetector()
    print("กำลังทดสอบ landmark detection กด Q เพื่อออก")

    while True:
        frame = cam.read()
        if frame is None:
            break

        landmarks = detector.detect(frame)

        if landmarks is not None:
            frame = detector.draw(frame, landmarks)
            cv2.putText(
                frame,
                f"Face detected — {len(landmarks)} points",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Landmark Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()