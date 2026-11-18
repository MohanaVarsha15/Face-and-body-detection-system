
import cv2
import time
from collections import deque

from modules.face_detector import FaceDetector
from modules.face_analyzer import FaceAnalyzer
from modules.pose_detector import PoseDetector
from modules.event_logger import EventLogger


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PERSON_MISSING_LIMIT = 15
CAMERA_WARMUP_FRAMES = 10

WINDOW_NAME = "Face and Body Detection System"


# ==========================================================
# 2. STABLE CLASSIFIER
# ==========================================================

class StableClassifier:

    def __init__(self, window_size=7, min_votes=5):

        self.history = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.current_label = "ANALYZING"

    def update(self, new_label):

        if new_label is None:
            return self.current_label

        self.history.append(new_label)

        counts = {}

        for label in self.history:
            counts[label] = counts.get(label, 0) + 1

        best_label = max(counts, key=counts.get)

        if counts[best_label] >= self.min_votes:
            self.current_label = best_label

        return self.current_label

    def reset(self):

        self.history.clear()
        self.current_label = "ANALYZING"


# ==========================================================
# 3. DISPLAY HELPER
# ==========================================================

def draw_text(frame, text, position, color):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


# ==========================================================
# 4. WEBCAM INITIALIZATION
# ==========================================================

def open_webcam():

    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DEFAULT", cv2.CAP_ANY),
        ("DIRECTSHOW", cv2.CAP_DSHOW)
    ]

    camera_indices = [CAMERA_INDEX, 1, 2]

    for camera_index in camera_indices:

        for backend_name, backend in backends:

            print(
                f"Trying camera {camera_index} "
                f"using {backend_name}..."
            )

            camera = cv2.VideoCapture(
                camera_index,
                backend
            )

            if not camera.isOpened():

                camera.release()
                continue

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                FRAME_WIDTH
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                FRAME_HEIGHT
            )

            camera.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            time.sleep(1.5)

            successful_reads = 0

            for _ in range(CAMERA_WARMUP_FRAMES):

                success, frame = camera.read()

                if (
                    success
                    and frame is not None
                    and frame.ndim == 3
                    and frame.shape[2] == 3
                    and frame.size > 0
                ):

                    successful_reads += 1

                time.sleep(0.03)

            if successful_reads >= 5:

                print("--------------------------------")
                print("WEBCAM CONNECTED SUCCESSFULLY")
                print(f"Camera index : {camera_index}")
                print(f"Backend      : {backend_name}")
                print("--------------------------------")

                return camera

            print(
                f"Camera {camera_index} with {backend_name} "
                "failed the frame test."
            )

            camera.release()

    return None


# ==========================================================
# 5. INITIALIZE APPLICATION COMPONENTS
# ==========================================================

face_detector = None
face_analyzer = None
pose_detector = None

event_logger = None
cap = None


try:

    print("Initializing detection modules...")

    # ------------------------------------------------------
    # Initialize detection modules
    # ------------------------------------------------------

    face_detector = FaceDetector()

    face_analyzer = FaceAnalyzer()

    pose_detector = PoseDetector()

    # ------------------------------------------------------
    # Initialize eye classifier
    # ------------------------------------------------------

    eye_classifier = StableClassifier(
        window_size=7,
        min_votes=5
    )

    # ------------------------------------------------------
    # Initialize SQLite event logger
    # ------------------------------------------------------

    event_logger = EventLogger()

    # ------------------------------------------------------
    # Open webcam
    # ------------------------------------------------------

    print("\nInitializing webcam...")

    cap = open_webcam()

    if cap is None:

        raise RuntimeError(
            "Unable to obtain stable webcam frames. "
            "Check camera permissions and backend."
        )

    # ======================================================
    # 6. INITIAL STATUS VARIABLES
    # ======================================================

    person_missing_frames = 0

    person_status = "DETECTING..."
    eye_status = "ANALYZING"

    consecutive_read_failures = 0

    # ======================================================
    # 7. APPLICATION STARTUP
    # ======================================================

    print("\n==========================================")
    print("       FACE AND BODY DETECTION SYSTEM")
    print("==========================================")
    print("Person detection      : ENABLED")
    print("Eye status detection  : ENABLED")
    print("SQLite event logging  : ENABLED")
    print("Face mesh display     : DISABLED")
    print("Body skeleton display : DISABLED")
    print("------------------------------------------")
    print("Press Q or ESC to quit.")
    print("==========================================")

    # ======================================================
    # 8. MAIN DETECTION LOOP
    # ======================================================

    while True:

        # --------------------------------------------------
        # READ WEBCAM FRAME
        # --------------------------------------------------

        success, frame = cap.read()

        if not success or frame is None:

            consecutive_read_failures += 1

            print(
                f"Webcam read failure: "
                f"{consecutive_read_failures}"
            )

            if consecutive_read_failures >= 10:

                print("Webcam connection lost.")
                break

            continue

        consecutive_read_failures = 0

        # Validate frame format
        if frame.ndim != 3 or frame.shape[2] != 3:

            print("Invalid webcam frame format.")
            continue

        # Resize frame
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ==================================================
        # 9. PERSON DETECTION
        # ==================================================

        pose_landmarks = pose_detector.detect(rgb_frame)

        if pose_landmarks is not None:

            person_missing_frames = 0

            person_status = "PERSON DETECTED"

        else:

            person_missing_frames += 1

            if person_missing_frames >= PERSON_MISSING_LIMIT:

                person_status = "NO PERSON DETECTED"

        # ==================================================
        # 10. FACE AND EYE DETECTION
        # ==================================================

        face_landmarks = face_detector.detect(rgb_frame)

        if face_landmarks is not None:

            analysis = face_analyzer.analyze(
                face_landmarks
            )

            if analysis is not None:

                raw_eye_status = analysis.get(
                    "eye_status"
                )

                eye_status = eye_classifier.update(
                    raw_eye_status
                )

            else:

                eye_status = "ANALYZING"

        else:

            eye_status = "FACE NOT VISIBLE"

            eye_classifier.reset()

            face_analyzer.reset()

        # ==================================================
        # 11. SAVE STATUS CHANGES TO SQLITE
        # ==================================================

        # Record person status changes
        event_logger.record_status_change(
            event_type="person",
            new_status=person_status
        )

        # Record eye status changes
        event_logger.record_status_change(
            event_type="eyes",
            new_status=eye_status
        )

        # ==================================================
        # 12. STATUS COLORS
        # ==================================================

        if person_status == "PERSON DETECTED":

            person_color = (0, 255, 0)

        elif person_status == "NO PERSON DETECTED":

            person_color = (0, 0, 255)

        else:

            person_color = (0, 255, 255)

        if eye_status == "OPEN":

            eye_color = (0, 255, 0)

        elif eye_status == "CLOSED":

            eye_color = (0, 0, 255)

        else:

            eye_color = (0, 255, 255)

        # ==================================================
        # 13. DISPLAY ONLY TWO STATUS VALUES
        # ==================================================

        draw_text(
            frame,
            "FACE AND BODY DETECTION SYSTEM",
            (20, 40),
            (255, 0, 255)
        )

        draw_text(
            frame,
            f"Person: {person_status}",
            (20, 90),
            person_color
        )

        draw_text(
            frame,
            f"Eyes: {eye_status}",
            (20, 135),
            eye_color
        )

        # ==================================================
        # 14. DISPLAY WEBCAM
        # ==================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break


except KeyboardInterrupt:

    print("\nApplication interrupted by user.")


except Exception as error:

    print("\nAPPLICATION ERROR:")
    print(error)


finally:

    # ======================================================
    # 15. RELEASE RESOURCES
    # ======================================================

    if cap is not None:
        cap.release()

    if face_detector is not None:
        face_detector.close()

    if pose_detector is not None:
        pose_detector.close()

    if event_logger is not None:
        event_logger.close()

    cv2.destroyAllWindows()

    print("\n==========================================")
    print("Detection system stopped.")
    print("Webcam released successfully.")
    print("==========================================")
