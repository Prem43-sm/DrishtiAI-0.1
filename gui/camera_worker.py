import time
from threading import Lock

import cv2
import face_recognition
import face_recognition_models
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from tensorflow.keras.models import load_model

from features.engine.timetable_engine import get_current_session
from features.tracking.live_tracker import update_location
from gui.attendance_manager import AttendanceManager
from gui.camera_backend import open_camera_capture
from gui.emotion_model_runtime import get_preferred_model_path, infer_class_names
from gui.face_memory import FaceMemory
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
    _track_match_floor_px = 60.0
    _allowed_fps_values = (25, 30, 60)

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
        self._analysis_buffer = []

        self.memory = FaceMemory.get_instance()
        runtime_settings = self._load_runtime_settings()
        self.target_fps = runtime_settings["fps"]
        self.recognition_frames = runtime_settings["recognition_frames"]
        self.emotion_frames = runtime_settings["emotion_frames"]
        self.analysis_frames = max(self.recognition_frames, self.emotion_frames)
        self._frame_interval_seconds = 1.0 / float(self.target_fps)
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
    def reset_shared_model(cls):
        with cls._model_lock:
            cls._shared_model = None
            cls._shared_model_path = None
            cls._shared_class_names = None

    @classmethod
    def reload_shared_model(cls):
        cls.reset_shared_model()
        return cls._get_shared_model_bundle()

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

    @staticmethod
    def _clamp_recognition_frames(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 1
        return max(1, min(50, value))

    @staticmethod
    def _clamp_emotion_frames(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 10
        return max(1, min(50, value))

    @classmethod
    def _clamp_target_fps(cls, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 30
        return value if value in cls._allowed_fps_values else 30

    @classmethod
    def _load_runtime_settings(cls):
        try:
            settings = SettingsManager().load()
        except Exception:
            settings = {}
        return {
            "fps": cls._clamp_target_fps(settings.get("fps", 30)),
            "recognition_frames": cls._clamp_recognition_frames(settings.get("recognition_frames", 1)),
            "emotion_frames": cls._clamp_emotion_frames(settings.get("emotion_frames", 10)),
        }

    @staticmethod
    def _unknown_identity_match():
        return {
            "name": "Unknown",
            "similarity": 0.0,
            "confidence": 0.0,
            "threshold": 0.0,
        }

    @staticmethod
    def _frame_to_qimage(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def _capture_frames(self, cap, first_frame, num_frames):
        yielded = 0

        if first_frame is not None:
            yielded += 1
            yield first_frame

        while self.running and yielded < num_frames:
            ret, next_frame = cap.read()
            if not ret:
                break
            yielded += 1
            yield next_frame

    def _analyze_frame(self, frame, include_identity=True):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        if include_identity and face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        else:
            face_encodings = [None] * len(face_locations)
        detections = []
        emotion_inputs = []
        emotion_indices = []

        for idx, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_encodings[idx] if idx < len(face_encodings) else None
            if include_identity and face_encoding is not None:
                match = self.memory.match_face(face_encoding)
            else:
                match = self._unknown_identity_match()

            # Rescale to original frame size for drawing and attendance flow.
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            detections.append(
                {
                    "box": (top, right, bottom, left),
                    "name": match["name"],
                    "emotion": "---",
                    "emotion_score": 0.0,
                    "similarity_score": float(match["similarity"]),
                    "confidence": float(match["confidence"]),
                    "threshold": float(match["threshold"]),
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
                pred_score = float(pred[pred_idx]) if len(pred) else 0.0
                if 0 <= pred_idx < len(self.class_names):
                    label = self.class_names[pred_idx]
                else:
                    label = f"Class {pred_idx + 1}"
                detections[det_idx]["emotion"] = label
                detections[det_idx]["emotion_score"] = pred_score

        return detections

    @staticmethod
    def _box_center(box):
        top, right, bottom, left = box
        return ((left + right) / 2.0, (top + bottom) / 2.0)

    @staticmethod
    def _box_size(box):
        top, right, bottom, left = box
        return max(1.0, float(right - left)), max(1.0, float(bottom - top))

    def _find_track_index(self, tracks, detection, used_track_indexes):
        det_center_x, det_center_y = self._box_center(detection["box"])
        det_width, det_height = self._box_size(detection["box"])
        best_index = None
        best_distance = float("inf")

        for index, track in enumerate(tracks):
            if index in used_track_indexes:
                continue

            track_center_x, track_center_y = self._box_center(track["box"])
            track_width, track_height = self._box_size(track["box"])
            distance = ((det_center_x - track_center_x) ** 2 + (det_center_y - track_center_y) ** 2) ** 0.5
            distance_limit = max(
                self._track_match_floor_px,
                0.75 * max(det_width, det_height, track_width, track_height),
            )

            if distance <= distance_limit and distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    @staticmethod
    def _is_confident_detection(detection):
        return (
            detection.get("name") != "Unknown"
            and float(detection.get("similarity_score", 0.0))
            >= float(detection.get("threshold", 0.0))
        )

    def _merge_detections_into_tracks(
        self,
        tracks,
        detections,
        collect_identity=True,
        collect_emotion=True,
    ):
        used_track_indexes = set()

        for detection in detections:
            track_index = self._find_track_index(tracks, detection, used_track_indexes)
            if track_index is None:
                tracks.append(
                    {
                        "box": detection["box"],
                        "emotion": detection["emotion"],
                        "emotion_score": float(detection.get("emotion_score", 0.0)),
                        "identity_samples": [],
                        "emotion_samples": [],
                    }
                )
                track_index = len(tracks) - 1

            track = tracks[track_index]
            track["box"] = detection["box"]
            track["emotion"] = detection["emotion"]
            track["emotion_score"] = float(detection.get("emotion_score", 0.0))

            if collect_identity and self._is_confident_detection(detection):
                track["identity_samples"].append(
                    (
                        detection["name"],
                        float(detection["similarity_score"]),
                        float(detection["confidence"]),
                    )
                )

            if collect_emotion and detection.get("emotion") != "---":
                track["emotion_samples"].append(
                    (
                        detection["emotion"],
                        float(detection.get("emotion_score", 0.0)),
                    )
                )

            used_track_indexes.add(track_index)

    @staticmethod
    def _aggregate_identity(samples):
        if not samples:
            return "Unknown", 0.0, 0.0

        grouped = {}
        for predicted_name, similarity_score, confidence in samples:
            grouped.setdefault(
                predicted_name,
                {
                    "similarities": [],
                    "confidences": [],
                },
            )
            grouped[predicted_name]["similarities"].append(similarity_score)
            grouped[predicted_name]["confidences"].append(confidence)

        ranked = []
        for predicted_name, data in grouped.items():
            vote_count = len(data["similarities"])
            avg_similarity = float(sum(data["similarities"]) / vote_count)
            avg_confidence = float(sum(data["confidences"]) / vote_count)
            ranked.append((predicted_name, vote_count, avg_similarity, avg_confidence))

        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        best_name, best_votes, best_avg_similarity, best_avg_confidence = ranked[0]

        if len(ranked) > 1:
            _, second_votes, second_avg_similarity, _ = ranked[1]
            if best_votes == second_votes and abs(best_avg_similarity - second_avg_similarity) <= 1e-6:
                return "Unknown", 0.0, 0.0

        return best_name, best_avg_similarity, best_avg_confidence

    @staticmethod
    def _aggregate_emotion(samples, fallback_emotion="---", fallback_score=0.0):
        if not samples:
            return fallback_emotion or "---", float(fallback_score)

        grouped = {}
        for emotion_label, emotion_score in samples:
            grouped.setdefault(
                emotion_label,
                {
                    "votes": 0,
                    "score_total": 0.0,
                    "best_score": 0.0,
                },
            )
            grouped[emotion_label]["votes"] += 1
            grouped[emotion_label]["score_total"] += float(emotion_score)
            grouped[emotion_label]["best_score"] = max(
                grouped[emotion_label]["best_score"],
                float(emotion_score),
            )

        ranked = []
        for emotion_label, data in grouped.items():
            vote_count = int(data["votes"])
            avg_score = float(data["score_total"] / max(1, vote_count))
            ranked.append((emotion_label, vote_count, avg_score, float(data["best_score"])))

        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        best_emotion, _, best_avg_score, _ = ranked[0]
        return best_emotion, best_avg_score

    def _finalize_multi_frame_detections(self, tracks):
        finalized = []

        for track in tracks:
            name, similarity_score, confidence = self._aggregate_identity(
                track["identity_samples"]
            )
            emotion, emotion_score = self._aggregate_emotion(
                track["emotion_samples"],
                fallback_emotion=track.get("emotion", "---"),
                fallback_score=track.get("emotion_score", 0.0),
            )
            finalized.append(
                {
                    "box": track["box"],
                    "name": name,
                    "emotion": emotion,
                    "emotion_score": emotion_score,
                    "similarity_score": similarity_score,
                    "confidence": confidence,
                }
            )

        finalized.sort(key=lambda det: (det["box"][0], det["box"][3]))
        return finalized

    def _emit_detecting_status(self, frame):
        preview = frame.copy()
        cv2.putText(
            preview,
            "Detecting...",
            (28, max(preview.shape[0] - 24, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        self.frame_ready.emit(
            self._frame_to_qimage(preview),
            "Detecting...",
            "---",
            0.0,
            self.attendance.today_count(),
            "",
        )

    def recognize_face_multi_frame(
        self,
        frame_generator,
        num_frames,
        identity_frames=None,
        emotion_frames=None,
    ):
        tracks = []
        final_frame = None
        processed_frames = 0
        started_at = time.time()
        status_emitted = False
        identity_limit = self._clamp_recognition_frames(
            num_frames if identity_frames is None else identity_frames
        )
        emotion_limit = self._clamp_emotion_frames(
            num_frames if emotion_frames is None else emotion_frames
        )

        for frame in frame_generator:
            if frame is None:
                continue

            final_frame = frame
            processed_frames += 1
            include_identity = processed_frames <= identity_limit
            include_emotion = processed_frames <= emotion_limit
            detections = self._analyze_frame(frame, include_identity=include_identity)
            self._merge_detections_into_tracks(
                tracks,
                detections,
                collect_identity=include_identity,
                collect_emotion=include_emotion,
            )

            if (
                not status_emitted
                and (time.time() - started_at) >= 1.0
            ):
                self._emit_detecting_status(frame)
                status_emitted = True

            if processed_frames >= num_frames:
                break

        if final_frame is None:
            return None, [], 0

        return final_frame, self._finalize_multi_frame_detections(tracks), processed_frames

    def _enforce_target_fps(self, loop_started_at):
        remaining = self._frame_interval_seconds - (time.perf_counter() - loop_started_at)
        if remaining > 0:
            time.sleep(remaining)

    def _apply_detections(self, frame, detections, loop_ts):
        popup_msg = ""

        if detections:
            now_sec = int(loop_ts)
            for det in detections:
                top, right, bottom, left = det["box"]
                det_name = det["name"]
                det_emotion = det["emotion"]

                if det_name != "Unknown" and self.active_class:
                    update_location(det_name, self.camera_name, self.active_class)

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

            return detections[-1]["name"], detections[-1]["emotion"], popup_msg

        return "Unknown", "---", popup_msg

    def run(self):
        cap, backend_name, _ = open_camera_capture(self.camera_id)
        if cap is None:
            return

        print(f"CAMERA OPENED: {self.camera_id} via {backend_name}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_emit_time = time.perf_counter()

        while self.running:
            loop_started_at = time.perf_counter()
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
                self._analysis_buffer.append(frame.copy())

                if self.analysis_frames <= 1:
                    detections = self._analyze_frame(frame, include_identity=True)
                    name, emotion, popup_msg = self._apply_detections(frame, detections, loop_ts)
                    self._last_name = name
                    self._last_emotion = emotion
                    self._analysis_buffer.clear()
                elif len(self._analysis_buffer) >= self.analysis_frames:
                    batch_frames = self._analysis_buffer[:self.analysis_frames]
                    self._analysis_buffer.clear()
                    analyzed_frame, detections, _ = self.recognize_face_multi_frame(
                        iter(batch_frames),
                        len(batch_frames),
                        identity_frames=min(self.recognition_frames, len(batch_frames)),
                        emotion_frames=min(self.emotion_frames, len(batch_frames)),
                    )
                    if analyzed_frame is not None:
                        frame = analyzed_frame
                        name, emotion, popup_msg = self._apply_detections(frame, detections, loop_ts)
                        self._last_name = name
                        self._last_emotion = emotion

            self._frame_counter += 1

            count = self.attendance.today_count()

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

            img = self._frame_to_qimage(frame)
            emit_time = time.perf_counter()
            fps = 1 / (emit_time - prev_emit_time) if emit_time > prev_emit_time else 0.0
            prev_emit_time = emit_time

            self.frame_ready.emit(img, name, emotion, fps, count, popup_msg)
            self._enforce_target_fps(loop_started_at)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(3000)
