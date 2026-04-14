"""
DeepShield AI — Face Extractor
Detects and crops faces from images using OpenCV Haar Cascades
and MediaPipe Face Detection for robust, multi-method extraction."""

import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceExtractor:
    """Extracts faces from images using OpenCV and MediaPipe."""

    def __init__(self):
        # OpenCV Haar Cascade (always available)
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # MediaPipe Face Detection
        self.mp_face_detection = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range model (better for varied distances)
                min_detection_confidence=0.5,
            )

        # MediaPipe Face Mesh (for landmark-level analysis)
        self.mp_face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def detect_faces_haar(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascades. Returns list of (x, y, w, h)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def detect_faces_mediapipe(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe. Returns list of (x, y, w, h)."""
        if not self.mp_face_detection:
            return []

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                fw = min(int(bbox.width * w), w - x)
                fh = min(int(bbox.height * h), h - y)
                faces.append((x, y, fw, fh))
        return faces

    def detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using the best available method.
        Tries MediaPipe first, falls back to Haar Cascade.
        Returns list of (x, y, w, h) bounding boxes.
        """
        faces = self.detect_faces_mediapipe(img)
        if not faces:
            faces = self.detect_faces_haar(img)
        return faces

    def crop_face(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2
    ) -> np.ndarray:
        """Crop a face region with optional padding around it."""
        x, y, w, h = bbox
        img_h, img_w = img.shape[:2]

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        return img[y1:y2, x1:x2]

    def get_face_landmarks(self, img: np.ndarray) -> Optional[dict]:
        """
        Get detailed face landmarks using MediaPipe Face Mesh.
        Returns dict with landmark positions grouped by facial feature,
        or None if no face is detected.
        """
        if not self.mp_face_mesh:
            return None

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # Convert to pixel coordinates
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append({
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z,
            })

        # Key landmark indices (MediaPipe Face Mesh standard)
        landmark_groups = {
            "left_eye": [33, 133, 160, 159, 158, 144, 145, 153],
            "right_eye": [362, 263, 387, 386, 385, 373, 374, 380],
            "left_iris": [468, 469, 470, 471, 472],
            "right_iris": [473, 474, 475, 476, 477],
            "nose": [1, 2, 5, 4, 6, 19, 94, 370],
            "mouth": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
                      95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
                      312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
            "jaw": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            "forehead_center": [10, 151, 9, 8],
        }

        result = {"all_landmarks": landmarks}
        for group_name, indices in landmark_groups.items():
            valid = [landmarks[i] for i in indices if i < len(landmarks)]
            result[group_name] = valid

        return result

    def extract_primary_face(
        self, img: np.ndarray, padding: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect and crop the largest/primary face in the image.
        Returns (cropped_face, bbox) or (None, None) if no face found.
        """
        faces = self.detect_faces(img)
        if not faces:
            return None, None

        # Pick the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        cropped = self.crop_face(img, largest, padding)
        return cropped, largest

    def draw_face_boxes(
        self, img: np.ndarray, faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 200), thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes around detected faces."""
        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result

    def close(self):
        """Release MediaPipe resources."""
        if self.mp_face_detection:
            self.mp_face_detection.close()
        if self.mp_face_mesh:
            self.mp_face_mesh.close()
