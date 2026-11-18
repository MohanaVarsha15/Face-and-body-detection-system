import mediapipe as mp


class PoseDetector:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, rgb_frame):

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            return results.pose_landmarks

        return None

    def close(self):
        self.pose.close()