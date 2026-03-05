import json
import os
from copy import deepcopy

from utils import resource_path


SETTINGS_FILE = resource_path("settings.json")


default_settings = {
    "camera_index": 0,
    "resolution": "640x480",
    "fps": 30,
    "process_frame": 2,
    "face_tolerance": 0.5,
    "auto_attendance": True,
    "model_path": "best_emotion_model.h5",
    "auto_load_model": True,
    "attendance_path": "attendance",
    "course_name": "MSc-IT",
    "snapshot_path": "snapshots",
    "auto_snapshot": False,
    "theme": "dark",
    "cameras": [
        {"id": 0, "name": "Main Camera", "class": "MSc-IT"}
    ],
}


class SettingsManager:
    def __init__(self):
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
