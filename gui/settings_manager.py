import json
import os
from copy import deepcopy

from core.project_paths import SETTINGS_FILE, ensure_runtime_layout
from gui.emotion_model_runtime import DEFAULT_MODEL_PATH


ALLOWED_FPS_VALUES = (25, 30, 60)


default_settings = {
    "camera_index": 0,
    "resolution": "640x480",
    "fps": 30,
    "process_frame": 2,
    "recognition_frames": 1,
    "emotion_frames": 10,
    "face_tolerance": 0.5,
    "auto_attendance": True,
    "model_path": DEFAULT_MODEL_PATH,
    "auto_load_model": True,
    "attendance_path": "storage/attendance",
    "course_name": "MSc-IT",
    "snapshot_path": "storage/snapshots",
    "auto_snapshot": False,
    "theme": "dark",
    "cameras": [
        {"id": 0, "name": "Main Camera", "class": "MSc-IT"}
    ],
}


class SettingsManager:
    def __init__(self):
        ensure_runtime_layout()
        if not os.path.exists(SETTINGS_FILE):
            self.save(default_settings)

    def _normalize(self, data):
        merged = deepcopy(default_settings)
        if isinstance(data, dict):
            merged.update(data)

        if not isinstance(merged.get("cameras"), list):
            merged["cameras"] = deepcopy(default_settings["cameras"])

        if merged.get("theme") not in ("dark", "light"):
            merged["theme"] = default_settings["theme"]

        try:
            fps = int(merged.get("fps", default_settings["fps"]))
        except (TypeError, ValueError):
            fps = default_settings["fps"]
        merged["fps"] = fps if fps in ALLOWED_FPS_VALUES else default_settings["fps"]

        try:
            process_frame = int(merged.get("process_frame", default_settings["process_frame"]))
        except (TypeError, ValueError):
            process_frame = default_settings["process_frame"]
        merged["process_frame"] = max(1, min(20, process_frame))

        try:
            recognition_frames = int(merged.get("recognition_frames", 1))
        except (TypeError, ValueError):
            recognition_frames = default_settings["recognition_frames"]
        merged["recognition_frames"] = max(1, min(50, recognition_frames))

        try:
            emotion_frames = int(merged.get("emotion_frames", default_settings["emotion_frames"]))
        except (TypeError, ValueError):
            emotion_frames = default_settings["emotion_frames"]
        merged["emotion_frames"] = max(1, min(50, emotion_frames))

        return merged

    def load(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = deepcopy(default_settings)

        normalized = self._normalize(data)
        if normalized != data:
            self.save(normalized)

        return normalized

    def save(self, data):
        os.makedirs(os.path.dirname(SETTINGS_FILE) or ".", exist_ok=True)
        normalized = self._normalize(data)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=4)

    def reset(self):
        current = self.load()
        fresh = deepcopy(default_settings)
        fresh["cameras"] = current.get("cameras", deepcopy(default_settings["cameras"]))
        self.save(fresh)
