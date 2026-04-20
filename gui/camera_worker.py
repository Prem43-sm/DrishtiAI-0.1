import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import face_recognition
import face_recognition_models
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from tensorflow.keras.models import load_model

from core.project_paths import ensure_runtime_layout
from features.engine.timetable_engine import get_current_session
from features.tracking.live_tracker import update_location
from gui.attendance_manager import AttendanceManager
from gui.camera_backend import open_camera_capture
from gui.emotion_model_runtime import (
    get_preferred_model_path,
    infer_class_names,
    infer_model_image_size,
    model_uses_embedded_preprocessing,
    prepare_emotion_image_input,
)
from gui.face_memory import FaceMemory
from gui.settings_manager import SettingsManager
from gui.utils import resource_path

# Landmark predictor fix for packaged/runtime usage.
predictor_path = resource_path(
    "models/face_recognition/models/shape_predictor_68_face_landmarks.dat"
)
face_recognition.api.pose_predictor_model_location = predictor_path

class CameraWorker(QThread):
    frame_ready = Signal(QImage, str, str, float, int, str)
    _shared_model = None
    _shared_model_path = None
    _shared_class_names = None
    _shared_input_size = None
    _shared_uses_embedded_preprocessing = False
    _model_lock = Lock()
    _predict_lock = Lock()
    _session_refresh_seconds = 2.0
    _emotion_size = (96, 96)
    _track_match_floor_px = 60.0
    _allowed_fps_values = (25, 30, 60)
    _emotion_crop_padding = 0.18

    def __init__(
        self,
        camera_id=0,
        camera_name="Camera",
        process_every_n_frames=2,
        display_fps_limit=None,
    ):
        super().__init__()
        ensure_runtime_layout()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.running = True
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._frame_counter = 0
        self._last_session_check_ts = 0.0
        self._last_name = "Unknown"
        self._last_emotion = "---"
        self._last_emit_time = 0.0

        self.memory = FaceMemory.get_instance()
        runtime_settings = self._load_runtime_settings()
        self.target_fps = runtime_settings["fps"]
        self.display_fps_limit = self._resolve_display_fps_limit(
            display_fps_limit,
            self.target_fps,
        )
        self.recognition_frames = runtime_settings["recognition_frames"]
        self.emotion_frames = runtime_settings["emotion_frames"]
        self.analysis_frames = max(self.recognition_frames, self.emotion_frames)
        self._frame_interval_seconds = 1.0 / float(self.target_fps)
        self._display_frame_interval_seconds = (
            1.0 / float(self.display_fps_limit)
            if self.display_fps_limit > 0
            else 0.0
        )
        self._analysis_buffer = deque(maxlen=self.analysis_frames)
        self._analysis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"camera-analysis-{camera_id}",
        )
        self._analysis_future = None
        self._latest_detections = []
        self.attendance = AttendanceManager()

        self.today_attendance = set()
        self.last_emotion_log_second = {}
        self.active_class = None
        self.active_period = None
        self.active_subject = None

        self.model, self.class_names, self.model_path, self._emotion_size = self._get_shared_model_bundle()
        self._emotion_uses_embedded_preprocessing = bool(
            self.__class__._shared_uses_embedded_preprocessing
        )
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
            cls._shared_input_size = None
            cls._shared_uses_embedded_preprocessing = False

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
                cls._shared_input_size = infer_model_image_size(
                    model=cls._shared_model,
                    model_path=model_path,
                )
                cls._shared_uses_embedded_preprocessing = model_uses_embedded_preprocessing(
                    cls._shared_model
                )
        return (
            cls._shared_model,
            list(cls._shared_class_names or []),
            cls._shared_model_path,
            tuple(cls._shared_input_size or cls._emotion_size),
        )

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

    @classmethod
    def _resolve_display_fps_limit(cls, value, fallback_fps):
        if value is None:
            return max(1, int(fallback_fps))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(fallback_fps))

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

    @classmethod
    def _extract_square_face_crop(cls, frame, box):
        frame_height, frame_width = frame.shape[:2]
        top, right, bottom, left = [int(value) for value in box]

        face_width = max(1, right - left)
        face_height = max(1, bottom - top)
        side = int(round(max(face_width, face_height) * (1.0 + (cls._emotion_crop_padding * 2.0))))
        side = max(1, side)

        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        crop_left = int(round(center_x - (side / 2.0)))
        crop_top = int(round(center_y - (side / 2.0)))
        crop_right = crop_left + side
        crop_bottom = crop_top + side

        pad_left = max(0, -crop_left)
        pad_top = max(0, -crop_top)
        pad_right = max(0, crop_right - frame_width)
        pad_bottom = max(0, crop_bottom - frame_height)

        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(frame_width, crop_right)
        crop_bottom = min(frame_height, crop_bottom)

        crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        if crop.size == 0:
            return crop

        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )

        return crop

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

            face = self._extract_square_face_crop(frame, (top, right, bottom, left))
            if face.size == 0:
                continue

            face_resized = cv2.resize(
                face,
                self._emotion_size,
                interpolation=cv2.INTER_AREA if face.shape[0] >= self._emotion_size[0] else cv2.INTER_LINEAR,
            )
            # Match the training pipeline, keeping RGB ordering and the model's
            # expected pixel range (raw 0..255 for EfficientNetV2 exports with
            # built-in rescaling, /255.0 for legacy models without it).
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_float = prepare_emotion_image_input(
                face_rgb,
                model=self.model,
            )
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
        identity_limit = self._clamp_recognition_frames(
            num_frames if identity_frames is None else identity_frames
        )
        emotion_limit = self._clamp_emotion_frames(
            num_frames if emotion_frames is None else emotion_frames
        )
        identity_start = max(1, num_frames - identity_limit + 1)
        emotion_start = max(1, num_frames - emotion_limit + 1)

        for frame in frame_generator:
            if frame is None:
                continue

            final_frame = frame
            processed_frames += 1
            include_identity = processed_frames >= identity_start
            include_emotion = processed_frames >= emotion_start
            detections = self._analyze_frame(frame, include_identity=include_identity)
            self._merge_detections_into_tracks(
                tracks,
                detections,
                collect_identity=include_identity,
                collect_emotion=include_emotion,
            )

            if processed_frames >= num_frames:
                break

        if final_frame is None:
            return None, [], 0

        return final_frame, self._finalize_multi_frame_detections(tracks), processed_frames

    def _run_analysis_batch(self, batch_items):
        batch_frames = [frame for frame, _ in batch_items]
        if not batch_frames:
            return {"detections": [], "analysis_ts": time.time()}

        _, detections, _ = self.recognize_face_multi_frame(
            iter(batch_frames),
            len(batch_frames),
            identity_frames=min(self.recognition_frames, len(batch_frames)),
            emotion_frames=min(self.emotion_frames, len(batch_frames)),
        )
        return {
            "detections": detections,
            "analysis_ts": float(batch_items[-1][1]),
        }

    def _submit_analysis_if_ready(self):
        if self._analysis_future is not None:
            return
        if len(self._analysis_buffer) < self.analysis_frames:
            return

        batch_items = list(self._analysis_buffer)
        self._analysis_buffer.clear()
        self._analysis_future = self._analysis_executor.submit(
            self._run_analysis_batch,
            batch_items,
        )

    def _collect_analysis_result(self):
        if self._analysis_future is None or not self._analysis_future.done():
            return ""

        try:
            result = self._analysis_future.result()
        except Exception as exc:
            print(f"ANALYSIS ERROR ({self.camera_name}): {exc}")
            self._analysis_future = None
            return ""

        self._analysis_future = None
        detections = list(result.get("detections", []))
        analysis_ts = float(result.get("analysis_ts", time.time()))
        self._latest_detections = detections

        name, emotion, popup_msg = self._process_detections(detections, analysis_ts)
        self._last_name = name
        self._last_emotion = emotion
        return popup_msg

    def _enforce_target_fps(self, loop_started_at):
        remaining = self._frame_interval_seconds - (time.perf_counter() - loop_started_at)
        if remaining > 0:
            time.sleep(remaining)

    def _shutdown_analysis_executor(self):
        self._analysis_buffer.clear()
        executor = self._analysis_executor
        self._analysis_executor = None
        self._analysis_future = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _should_emit_preview_frame(self, emit_time):
        if self._display_frame_interval_seconds <= 0:
            return True
        if self._last_emit_time <= 0:
            return True
        return (emit_time - self._last_emit_time) >= self._display_frame_interval_seconds

    @staticmethod
    def _draw_detections(frame, detections):
        for det in detections:
            top, right, bottom, left = det["box"]
            det_name = det["name"]
            det_emotion = det["emotion"]

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

    def _process_detections(self, detections, loop_ts):
        popup_msg = ""

        if detections:
            now_sec = int(loop_ts)
            for det in detections:
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

            return detections[-1]["name"], detections[-1]["emotion"], popup_msg

        return "Unknown", "---", popup_msg

    def run(self):
        cap, backend_name, _ = open_camera_capture(self.camera_id)
        if cap is None:
            self._shutdown_analysis_executor()
            return

        print(f"CAMERA OPENED: {self.camera_id} via {backend_name}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_emit_time = time.perf_counter()

        while self.running:
            loop_started_at = time.perf_counter()

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

            popup_msg = self._collect_analysis_result()
            should_process = (self._frame_counter % self.process_every_n_frames) == 0
            if should_process:
                self._analysis_buffer.append((frame.copy(), loop_ts))
            self._submit_analysis_if_ready()

            self._frame_counter += 1
            name = self._last_name
            emotion = self._last_emotion
            count = self.attendance.today_count()
            self._draw_detections(frame, self._latest_detections)

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

            emit_time = time.perf_counter()
            if self._should_emit_preview_frame(emit_time):
                img = self._frame_to_qimage(frame)
                fps = 1 / (emit_time - prev_emit_time) if emit_time > prev_emit_time else 0.0
                prev_emit_time = emit_time
                self._last_emit_time = emit_time
                self.frame_ready.emit(img, name, emotion, fps, count, popup_msg)
            self._enforce_target_fps(loop_started_at)

        cap.release()
        self._shutdown_analysis_executor()

    def stop(self):
        self.running = False
        self.wait(3000)
