
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


class FaceDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(

            static_image_mode=False,

            max_num_faces=1,

            refine_landmarks=True,

            min_detection_confidence=0.65,

            min_tracking_confidence=0.75

        )

        # Landmark stabilization
        self.previous_landmarks = None

        self.alpha = 0.40

    def _smooth_landmarks(self, current_landmarks):

        """
        Smooth facial landmark coordinates using EMA.

        This reduces small frame-to-frame movements
        produced by MediaPipe tracking.
        """

        if self.previous_landmarks is None:

            self.previous_landmarks = current_landmarks

            return current_landmarks

        smoothed = landmark_pb2.NormalizedLandmarkList()

        for i, current in enumerate(
            current_landmarks.landmark
        ):

            previous = self.previous_landmarks.landmark[i]

            landmark = smoothed.landmark.add()

            landmark.x = (
                self.alpha * current.x
                + (1 - self.alpha) * previous.x
            )

            landmark.y = (
                self.alpha * current.y
                + (1 - self.alpha) * previous.y
            )

            landmark.z = (
                self.alpha * current.z
                + (1 - self.alpha) * previous.z
            )

            # Preserve optional landmark attributes
            if current.HasField("visibility"):
                landmark.visibility = current.visibility

            if current.HasField("presence"):
                landmark.presence = current.presence

        self.previous_landmarks = smoothed

        return smoothed

    def detect(self, rgb_frame):

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            raw_landmarks = results.multi_face_landmarks[0]

            return self._smooth_landmarks(raw_landmarks)

        # Reset when face disappears
        self.previous_landmarks = None

        return None

    def close(self):

        self.face_mesh.close()

        self.previous_landmarks = None
