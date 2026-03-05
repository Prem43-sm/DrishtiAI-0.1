import time

import cv2
import face_recognition
import face_recognition_models
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from tensorflow.keras.models import load_model

from face_memory import FaceMemory
from features.engine.timetable_engine import get_current_session
from features.tracking.live_tracker import update_location
from gui.attendance_manager import AttendanceManager
from gui.utils import resource_path

# Landmark predictor fix for packaged/runtime usage.
predictor_path = resource_path(
    "face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
)
face_recognition.api.pose_predictor_model_location = predictor_path

EMOTION_MODEL_PATH = "best_emotion_model.h5"
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]


class CameraWorker(QThread):
    frame_ready = Signal(QImage, str, str, float, int, str)

    def __init__(self, camera_id=0, camera_name="Camera"):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.running = True
        self.process_this_frame = True

        self.memory = FaceMemory.get_instance()
        self.attendance = AttendanceManager()

        self.today_attendance = set()
        self.last_emotion_log_second = {}
        self.active_class = None
        self.active_period = None
        self.active_subject = None

        self.model = load_model(EMOTION_MODEL_PATH, compile=False)
        print("WORKER INIT:", camera_id, camera_name)

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()

        while self.running:
            popup_msg = ""
            name = "Unknown"
            emotion = "---"

            class_name, period, subject = get_current_session()
            if (
                class_name != self.active_class
                or period != self.active_period
                or subject != self.active_subject
            ):
                self.active_class = class_name
                self.active_period = period
                self.active_subject = subject
                self.today_attendance.clear()

                if class_name and period:
                    self.attendance.set_active_session(class_name, period, subject)
                    print(f"ACTIVE -> {class_name} | P{period} | {subject}")
                else:
                    self.attendance.stop_session()
                    print("NO ACTIVE SESSION")

            ret, frame = cap.read()
            if not ret:
                continue

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if self.process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = self.memory.get_name(face_encoding)

                    if name != "Unknown" and self.active_class:
                        update_location(name, self.camera_name, self.active_class)

                    # Rescale to original frame.
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    face = frame[top:bottom, left:right]
                    if face.size != 0:
                        face_resized = cv2.resize(face, (96, 96)) / 255.0
                        face_resized = np.reshape(face_resized, (1, 96, 96, 3))
                        preds = self.model.predict(face_resized, verbose=0)
                        emotion = CLASS_NAMES[int(np.argmax(preds))]

                    # Per-second live emotion logging for analytics.
                    if name != "Unknown" and emotion != "---":
                        now_sec = int(time.time())
                        last_sec = self.last_emotion_log_second.get(name, -1)
                        if now_sec != last_sec:
                            self.attendance.log_emotion_sample(name, emotion)
                            self.last_emotion_log_second[name] = now_sec

                    # Mark attendance after emotion prediction so emotion gets stored.
                    if (
                        name != "Unknown"
                        and self.active_period
                        and name not in self.today_attendance
                    ):
                        marked = self.attendance.mark(name, emotion)
                        if marked:
                            self.today_attendance.add(name)
                            popup_msg = f"{name} -> P{self.active_period} OK"

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{name} - {emotion}",
                        (left, max(top - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            self.process_this_frame = not self.process_this_frame

            count = self.attendance.today_count()
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            cv2.putText(
                frame,
                f"Camera: {self.camera_name}",
                (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            session_text = (
                f"{self.active_class} | {self.active_subject} | P{self.active_period}"
                if self.active_class
                else "No Active Session"
            )
            cv2.putText(
                frame,
                session_text,
                (28, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

            self.frame_ready.emit(img, name, emotion, fps, count, popup_msg)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()
