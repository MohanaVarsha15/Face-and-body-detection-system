"""
---------------------------------------------------------
Face Detector Module
---------------------------------------------------------

Purpose:
    Detects the patient's face using MediaPipe Face Mesh.

Responsibilities:
    - Initialize the Face Mesh model
    - Detect facial landmarks
    - Return the detected landmarks

This module DOES NOT:
    - Detect eye blinking
    - Detect pain
    - Calculate EAR
    - Analyze facial expressions

Those tasks belong to face_analyzer.py.
---------------------------------------------------------
"""

import mediapipe as mp


class FaceDetector:
    """
    FaceDetector uses MediaPipe Face Mesh
    to detect a patient's facial landmarks.
    """

    def __init__(self):
        """
        Initialize the MediaPipe Face Mesh model.
        """

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, rgb_frame):
        """
        Detect facial landmarks from an RGB frame.

        Parameters
        ----------
        rgb_frame : numpy.ndarray
            RGB image captured from OpenCV.

        Returns
        -------
        face_landmarks :
            MediaPipe face landmarks if detected.

        None :
            If no face is detected.
        """

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]

        return None

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.face_mesh.close()