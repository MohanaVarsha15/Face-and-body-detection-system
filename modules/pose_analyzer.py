import math


class PoseAnalyzer:

    def __init__(self):
        pass

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def analyze(self, pose_landmarks):

        if pose_landmarks is None:
            return None

        landmarks = pose_landmarks.landmark

        # -----------------------------
        # MediaPipe Landmark IDs
        # -----------------------------
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        LEFT_HIP = 23
        RIGHT_HIP = 24

        LEFT_KNEE = 25
        RIGHT_KNEE = 26

        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        # -----------------------------
        # Midpoint Y Coordinates
        # -----------------------------
        shoulder_y = (
            landmarks[LEFT_SHOULDER].y +
            landmarks[RIGHT_SHOULDER].y
        ) / 2

        hip_y = (
            landmarks[LEFT_HIP].y +
            landmarks[RIGHT_HIP].y
        ) / 2

        knee_y = (
            landmarks[LEFT_KNEE].y +
            landmarks[RIGHT_KNEE].y
        ) / 2

        ankle_y = (
            landmarks[LEFT_ANKLE].y +
            landmarks[RIGHT_ANKLE].y
        ) / 2

        # -----------------------------
        # Vertical Body Segment Lengths
        # -----------------------------
        shoulder_hip = abs(hip_y - shoulder_y)
        hip_knee = abs(knee_y - hip_y)
        knee_ankle = abs(ankle_y - knee_y)

        # -----------------------------
        # Determine Body Posture
        # -----------------------------
        posture = "UNKNOWN"

        if shoulder_hip > 0.18 and hip_knee > 0.15 and knee_ankle > 0.12:
            posture = "STANDING"

        elif shoulder_hip > 0.18 and hip_knee <= 0.15:
            posture = "SITTING"

        elif shoulder_hip <= 0.10:
            posture = "LYING"

        # -----------------------------
        # Return Analysis
        # -----------------------------
        return {
            "posture": posture,
            "shoulder_hip": shoulder_hip,
            "hip_knee": hip_knee,
            "knee_ankle": knee_ankle
        }