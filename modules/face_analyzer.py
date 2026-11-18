
import math


class FaceAnalyzer:

    # MediaPipe Face Mesh landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Eye classification thresholds
    # CLOSED below 0.19
    # OPEN above 0.23
    # Between them, retain the previous classification.

    CLOSED_THRESHOLD = 0.19
    OPEN_THRESHOLD = 0.23

    def __init__(self):

        self.eye_status = "OPEN"

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""

        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def _calculate_ear(self, eye_points, landmarks):
        """Calculate Eye Aspect Ratio."""

        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]

        vertical_1 = self._distance(p2, p6)
        vertical_2 = self._distance(p3, p5)

        horizontal = self._distance(p1, p4)

        if horizontal < 1e-6:
            return None

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        if not math.isfinite(ear):
            return None

        return ear

    def _classify_eyes(self, average_ear):
        """
        Hysteresis-based eye classification.

        Prevents rapid OPEN/CLOSED switching
        around a single threshold.
        """

        if average_ear < self.CLOSED_THRESHOLD:

            self.eye_status = "CLOSED"

        elif average_ear > self.OPEN_THRESHOLD:

            self.eye_status = "OPEN"

        # Between thresholds:
        # Retain the previous state.

        return self.eye_status

    def analyze(self, face_landmarks):
        """Analyze facial landmarks and return eye measurements."""

        if face_landmarks is None:
            return None

        landmarks = face_landmarks.landmark

        if len(landmarks) <= max(
            self.LEFT_EYE + self.RIGHT_EYE
        ):
            return None

        left_ear = self._calculate_ear(
            self.LEFT_EYE,
            landmarks
        )

        right_ear = self._calculate_ear(
            self.RIGHT_EYE,
            landmarks
        )

        if left_ear is None or right_ear is None:
            return None

        average_ear = (left_ear + right_ear) / 2.0

        eye_status = self._classify_eyes(
            average_ear
        )

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "average_ear": average_ear,
            "eye_status": eye_status
        }

    def reset(self):

        self.eye_status = "OPEN"
