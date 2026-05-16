from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import cv2
import numpy as np


try:
    import mediapipe as mp
except Exception:
    mp = None


def classify_focus(score: float) -> str:
    score = float(score)
    if score >= 80:
        return "Highly Focused"
    if score >= 60:
        return "Moderately Focused"
    if score >= 40:
        return "Distracted"
    return "Low Attention"


@dataclass
class FocusSessionStats:
    student_id: str = "Class Session"
    started_at: float = field(default_factory=time.time)
    active_seconds: float = 0.0
    inactive_seconds: float = 0.0
    movement_count: int = 0
    sleep_detection_count: int = 0
    looking_away_count: int = 0
    processed_frames: int = 0

    def to_report(self) -> dict[str, Any]:
        total = max(1.0, self.active_seconds + self.inactive_seconds)
        attention = self.active_seconds / total * 100.0
        distraction = 100.0 - attention
        movement_penalty = min(15.0, self.movement_count * 0.7)
        sleep_penalty = min(20.0, self.sleep_detection_count * 2.0)
        focus_score = max(0.0, min(100.0, attention - movement_penalty - sleep_penalty))
        return {
            "student_id": self.student_id,
            "date": date.today().isoformat(),
            "focus_score": round(focus_score, 2),
            "attention_percentage": round(attention, 2),
            "distraction_percentage": round(distraction, 2),
            "active_time": round(self.active_seconds, 2),
            "inactive_time": round(self.inactive_seconds, 2),
            "movement_count": int(self.movement_count),
            "sleep_detection_count": int(self.sleep_detection_count),
            "looking_away_count": int(self.looking_away_count),
            "status": classify_focus(focus_score),
        }


class FocusTrackingEngine:
    """MediaPipe Face Mesh focus estimator with OpenCV fallback overlays."""

    def __init__(self):
        self.available = mp is not None
        self._last_ts = None
        self._last_nose = None
        self.stats = FocusSessionStats()
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.available:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._face_mesh = None

    def reset(self, student_id: str) -> None:
        self.stats = FocusSessionStats(student_id=student_id or "Class Session")
        self._last_ts = None
        self._last_nose = None

    def analyze_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        now = time.time()
        delta = 0.0 if self._last_ts is None else max(0.0, now - self._last_ts)
        self._last_ts = now
        self.stats.processed_frames += 1

        if not self.available:
            return self._analyze_frame_with_opencv(frame, delta)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            self.stats.inactive_seconds += delta
            state = {
                "status": "Face not detected",
                "focused": False,
                "sleeping": False,
                "looking_away": False,
                "movement": False,
                "face_detected": False,
            }
            return self._draw_status(frame, state), state

        landmarks = result.multi_face_landmarks[0].landmark
        height, width = frame.shape[:2]
        ear = self._average_ear(landmarks, width, height)
        nose = self._point(landmarks[1], width, height)
        face_center_x = self._face_center_x(landmarks, width)
        looking_away = abs(nose[0] - face_center_x) > width * 0.075
        sleeping = ear < 0.18
        movement = self._register_movement(nose)
        focused = not sleeping and not looking_away

        if focused:
            self.stats.active_seconds += delta
        else:
            self.stats.inactive_seconds += delta
        if sleeping:
            self.stats.sleep_detection_count += 1
        if looking_away:
            self.stats.looking_away_count += 1
        if movement:
            self.stats.movement_count += 1

        state = {
            "status": "Focused" if focused else "Distracted",
            "focused": focused,
            "sleeping": sleeping,
            "looking_away": looking_away,
            "movement": movement,
            "face_detected": True,
            "ear": round(ear, 3),
        }
        return self._draw_status(frame, state), state

    def _analyze_frame_with_opencv(self, frame: np.ndarray, delta: float) -> tuple[np.ndarray, dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            self.stats.inactive_seconds += delta
            state = {
                "status": "Face not detected",
                "focused": False,
                "sleeping": False,
                "looking_away": False,
                "movement": False,
                "face_detected": False,
                "mode": "OpenCV fallback",
            }
            return self._draw_status(frame, state), state

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center = (int(x + w / 2), int(y + h / 2))
        movement = self._register_movement(center)
        self.stats.active_seconds += delta
        if movement:
            self.stats.movement_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (32, 210, 92), 2)
        state = {
            "status": "Focused (OpenCV fallback)",
            "focused": True,
            "sleeping": False,
            "looking_away": False,
            "movement": movement,
            "face_detected": True,
            "mode": "OpenCV fallback",
        }
        return self._draw_status(frame, state), state

    def _register_movement(self, nose: tuple[int, int]) -> bool:
        if self._last_nose is None:
            self._last_nose = nose
            return False
        distance = math.dist(nose, self._last_nose)
        self._last_nose = nose
        return distance > 22.0

    def _average_ear(self, landmarks, width: int, height: int) -> float:
        left = self._eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144], width, height)
        right = self._eye_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380], width, height)
        return (left + right) / 2.0

    def _eye_aspect_ratio(self, landmarks, indexes, width: int, height: int) -> float:
        points = [self._point(landmarks[i], width, height) for i in indexes]
        vertical_1 = math.dist(points[1], points[5])
        vertical_2 = math.dist(points[2], points[4])
        horizontal = max(1.0, math.dist(points[0], points[3]))
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    @staticmethod
    def _point(landmark, width: int, height: int) -> tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _face_center_x(landmarks, width: int) -> float:
        xs = [landmarks[i].x * width for i in (234, 454, 10, 152)]
        return sum(xs) / len(xs)

    @staticmethod
    def _draw_status(frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        color = (32, 210, 92) if state.get("focused") else (48, 132, 255)
        cv2.putText(frame, state.get("status", "---"), (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        details = []
        if state.get("sleeping"):
            details.append("sleep")
        if state.get("looking_away"):
            details.append("away")
        if state.get("movement"):
            details.append("movement")
        if details:
            cv2.putText(frame, ", ".join(details), (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        return frame
