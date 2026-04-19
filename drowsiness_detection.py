import cv2
import numpy as np
import winsound
import threading
import csv
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ─────────────────────────────────────────
# FUNCTION 1: Calculate Euclidean Distance
# ─────────────────────────────────────────
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# ─────────────────────────────────────────
# FUNCTION 2: Calculate EAR for one eye
# ─────────────────────────────────────────
def calculate_EAR(eye_points):
    vertical1 = calculate_distance(eye_points[1], eye_points[5])
    vertical2 = calculate_distance(eye_points[2], eye_points[4])
    horizontal = calculate_distance(eye_points[0], eye_points[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# ─────────────────────────────────────────
# FUNCTION 3: Continuous alarm
# ─────────────────────────────────────────
def continuous_alarm(stop_event):
    while not stop_event.is_set():
        winsound.Beep(1000, 500)  # beep every 0.5 seconds

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.4,  # lowered for better detection
    min_tracking_confidence=0.4         # lowered for better tracking
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ─────────────────────────────────────────
# EYE LANDMARK INDICES
# ─────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
frame_counter = 0
no_face_counter = 0          # counts frames where no face detected
NO_FACE_THRESHOLD = 30       # 30 frames no face = possible head down

alarm_playing = False
stop_alarm_event = threading.Event()

# ─────────────────────────────────────────
# CSV LOG SETUP
# ─────────────────────────────────────────
log_file = open('drowsiness_log.csv', 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['Date', 'Time', 'EAR Value', 'Alert Status'])
print("Logging drowsiness events to drowsiness_log.csv")
print("Starting Drowsiness Detection... Press 'Q' to quit")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access webcam!")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if results.face_landmarks:
        # Face detected — reset no face counter
        no_face_counter = 0

        face_landmarks = results.face_landmarks[0]

        landmarks = {}
        for idx, lm in enumerate(face_landmarks):
            landmarks[idx] = (int(lm.x * w), int(lm.y * h))

        left_eye_pts  = [landmarks[i] for i in LEFT_EYE]
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE]

        left_EAR  = calculate_EAR(left_eye_pts)
        right_EAR = calculate_EAR(right_eye_pts)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        for point in left_eye_pts + right_eye_pts:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if frame_counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)

                if not alarm_playing:
                    alarm_playing = True
                    stop_alarm_event.clear()

                    now = datetime.now()
                    log_writer.writerow([
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        f"{avg_EAR:.2f}",
                        "DROWSY ALERT"
                    ])
                    log_file.flush()

                    t = threading.Thread(target=continuous_alarm,
                                         args=(stop_alarm_event,))
                    t.daemon = True
                    t.start()
        else:
            # Eyes open — stop alarm, reset counter
            frame_counter = 0
            if alarm_playing:
                stop_alarm_event.set()  # stops continuous alarm
                alarm_playing = False
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Frame Count: {frame_counter}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        # No face detected
        no_face_counter += 1

        if no_face_counter >= NO_FACE_THRESHOLD:
            # Person likely bent head down — treat as drowsy
            cv2.putText(frame, "HEAD DOWN DETECTED!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)

            if not alarm_playing:
                alarm_playing = True
                stop_alarm_event.clear()

                now = datetime.now()
                log_writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "N/A",
                    "HEAD DOWN ALERT"
                ])
                log_file.flush()

                t = threading.Thread(target=continuous_alarm,
                                     args=(stop_alarm_event,))
                t.daemon = True
                t.start()
        else:
            cv2.putText(frame, "No Face Detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.putText(frame, "Press Q to Quit", (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────
stop_alarm_event.set()
cap.release()
cv2.destroyAllWindows()
log_file.close()
print("Program ended. Check drowsiness_log.csv for logs!")