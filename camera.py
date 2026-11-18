
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera opened. Testing frame capture...")

for i in range(100):
    ret, frame = cap.read()

    if ret and frame is not None:
        print(f"Frame {i + 1}: SUCCESS")
        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(f"Frame {i + 1}: FAILED")
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
