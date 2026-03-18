import time
from threading import Lock

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
from gui.emotion_model_runtime import get_preferred_model_path, infer_class_names
from gui.settings_manager import SettingsManager
from gui.utils import resource_path

# Landmark predictor fix for packaged/runtime usage.
predictor_path = resource_path(
    "face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
)
face_recognition.api.pose_predictor_model_location = predictor_path

class CameraWorker(QThread):
    frame_ready = Signal(QImage, str, str, float, int, str)
    _shared_model = None
    _shared_model_path = None
    _shared_class_names = None
    _model_lock = Lock()
    _predict_lock = Lock()
    _session_refresh_seconds = 2.0
    _emotion_size = (96, 96)
    _inv_255 = np.float32(1.0 / 255.0)

    def __init__(self, camera_id=0, camera_name="Camera", process_every_n_frames=2):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.running = True
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._frame_counter = 0
        self._last_session_check_ts = 0.0
        self._last_name = "Unknown"
        self._last_emotion = "---"

        self.memory = FaceMemory.get_instance()
        self.attendance = AttendanceManager()

        self.today_attendance = set()
        self.last_emotion_log_second = {}
        self.active_class = None
        self.active_period = None
        self.active_subject = None

        self.model, self.class_names, self.model_path = self._get_shared_model_bundle()
        print("WORKER INIT:", camera_id, camera_name)

    @classmethod
    def _resolve_model_path(cls):
        try:
            configured_path = SettingsManager().load().get("model_path", "")
        except Exception:
            configured_path = ""
        return get_preferred_model_path(configured_path)

    @classmethod
    def _get_shared_model_bundle(cls):
        with cls._model_lock:
            model_path = cls._resolve_model_path()
            if cls._shared_model is None or cls._shared_model_path != model_path:
                cls._shared_model = load_model(model_path, compile=False)
                cls._shared_model_path = model_path
                cls._shared_class_names = infer_class_names(
                    model=cls._shared_model,
                    model_path=model_path,
                )
        return cls._shared_model, list(cls._shared_class_names or []), cls._shared_model_path

    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_time = time.time()

        while self.running:
            popup_msg = ""
            name = self._last_name
            emotion = self._last_emotion

            loop_ts = time.time()
            if (loop_ts - self._last_session_check_ts) >= self._session_refresh_seconds:
                self._last_session_check_ts = loop_ts
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
                time.sleep(0.01)
                continue

            should_process = (self._frame_counter % self.process_every_n_frames) == 0
            if should_process:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                detections = []
                emotion_inputs = []
                emotion_indices = []

                for idx, ((top, right, bottom, left), face_encoding) in enumerate(
                    zip(face_locations, face_encodings)
                ):
                    detected_name = self.memory.get_name(face_encoding)

                    if detected_name != "Unknown" and self.active_class:
                        update_location(detected_name, self.camera_name, self.active_class)

                    # Rescale to original frame.
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    detections.append(
                        {
                            "box": (top, right, bottom, left),
                            "name": detected_name,
                            "emotion": "---",
                        }
                    )

                    face = frame[top:bottom, left:right]
                    if face.size == 0:
                        continue

                    face_resized = cv2.resize(
                        face,
                        self._emotion_size,
                        interpolation=cv2.INTER_AREA,
                    )
                    face_float = face_resized.astype(np.float32, copy=False)
                    face_float *= self._inv_255
                    emotion_inputs.append(face_float)
                    emotion_indices.append(idx)

                if emotion_inputs:
                    emotion_batch = np.asarray(emotion_inputs, dtype=np.float32)
                    with self._predict_lock:
                        predictions = self.model.predict(emotion_batch, verbose=0)
                    for det_idx, pred in zip(emotion_indices, predictions):
                        pred_idx = int(np.argmax(pred))
                        if 0 <= pred_idx < len(self.class_names):
                            label = self.class_names[pred_idx]
                        else:
                            label = f"Class {pred_idx + 1}"
                        detections[det_idx]["emotion"] = label

                if detections:
                    now_sec = int(loop_ts)
                    for det in detections:
                        top, right, bottom, left = det["box"]
                        det_name = det["name"]
                        det_emotion = det["emotion"]

                        # Per-second live emotion logging for analytics.
                        if det_name != "Unknown" and det_emotion != "---":
                            last_sec = self.last_emotion_log_second.get(det_name, -1)
                            if now_sec != last_sec:
                                self.attendance.log_emotion_sample(det_name, det_emotion)
                                self.last_emotion_log_second[det_name] = now_sec

                        # Mark attendance after emotion prediction so emotion gets stored.
                        if (
                            det_name != "Unknown"
                            and self.active_period
                            and det_name not in self.today_attendance
                        ):
                            marked = self.attendance.mark(det_name, det_emotion)
                            if marked:
                                self.today_attendance.add(det_name)
                                popup_msg = f"{det_name} -> P{self.active_period} OK"

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{det_name} - {det_emotion}",
                            (left, max(top - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    name = detections[-1]["name"]
                    emotion = detections[-1]["emotion"]
                else:
                    name = "Unknown"
                    emotion = "---"

                self._last_name = name
                self._last_emotion = emotion

            self._frame_counter += 1

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
            # Copy detaches image memory from NumPy buffer to avoid cross-thread invalid access.
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

            self.frame_ready.emit(img, name, emotion, fps, count, popup_msg)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(3000)
