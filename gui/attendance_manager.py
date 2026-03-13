import os
import csv
from datetime import datetime

import pandas as pd


class AttendanceManager:
    ATTENDANCE_COLUMNS = ["Name", "Class", "Subject", "Period", "Time", "Emotion"]
    EMOTION_COLUMNS = ["Date", "Name", "Class", "Subject", "Period", "Time", "Emotion"]

    def __init__(self):
        self.base_dir = "attendance"
        os.makedirs(self.base_dir, exist_ok=True)

        self.active_class = None
        self.active_period = None
        self.active_subject = None

        self.file = None
        self._df_cache = None
        self._marked_names = set()

        # Central emotion analytics file used by Emotion Analytics page.
        self.emotion_file = os.path.join(self.base_dir, "emotion_data.csv")
        self._ensure_emotion_file()

    # ============================================================
    # FILE SETUP
    # ============================================================
    def _ensure_emotion_file(self):
        if os.path.exists(self.emotion_file) and os.path.getsize(self.emotion_file) > 0:
            return
        with open(self.emotion_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.EMOTION_COLUMNS)

    # ============================================================
    # SET ACTIVE SESSION  (called by CameraWorker)
    # ============================================================
    def set_active_session(self, class_name, period, subject=None):
        if (
            self.active_class == class_name
            and self.active_period == period
            and self.active_subject == subject
        ):
            return

        self.active_class = class_name
        self.active_period = period
        self.active_subject = subject

        today = datetime.now().strftime("%Y-%m-%d")
        class_dir = os.path.join(self.base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        self.file = os.path.join(class_dir, f"{today}_P{period}.csv")

        if not os.path.exists(self.file):
            pd.DataFrame(columns=self.ATTENDANCE_COLUMNS).to_csv(self.file, index=False)

        self._df_cache = pd.read_csv(self.file)
        for column in self.ATTENDANCE_COLUMNS:
            if column not in self._df_cache.columns:
                self._df_cache[column] = ""
        self._df_cache = self._df_cache[self.ATTENDANCE_COLUMNS]
        self._marked_names = set(
            self._df_cache["Name"].dropna().astype(str).tolist()
        )

        print(f"SESSION -> {class_name} | P{period} | {subject}")

    # ============================================================
    def stop_session(self):
        self.active_class = None
        self.active_period = None
        self.active_subject = None
        self.file = None
        self._df_cache = None
        self._marked_names.clear()
        print("SESSION STOPPED")

    # ============================================================
    # MARK ATTENDANCE
    # ============================================================
    def mark(self, name, emotion=""):
        if not self.file or self._df_cache is None:
            return False

        # already marked in this period
        if name in self._marked_names:
            return False

        time_now = datetime.now().strftime("%H:%M:%S")
        emotion_text = str(emotion or "")

        self._df_cache.loc[len(self._df_cache)] = [
            name,
            self.active_class,
            self.active_subject,
            self.active_period,
            time_now,
            emotion_text,
        ]
        self._marked_names.add(name)
        self._df_cache.to_csv(self.file, index=False)
        self._append_emotion_row(name, time_now, emotion_text)
        return True

    def _append_emotion_row(self, name, time_now, emotion):
        self._ensure_emotion_file()
        with open(self.emotion_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d"),
                    name,
                    self.active_class,
                    self.active_subject,
                    self.active_period,
                    time_now,
                    emotion,
                ]
            )

    # ============================================================
    # LIVE EMOTION SAMPLE (once per second per student from worker)
    # ============================================================
    def log_emotion_sample(self, name, emotion):
        if not name or name == "Unknown":
            return False
        emotion_text = str(emotion or "").strip()
        if not emotion_text or emotion_text == "---":
            return False

        time_now = datetime.now().strftime("%H:%M:%S")
        self._append_emotion_row(name, time_now, emotion_text)
        return True

    # ============================================================
    # COUNT
    # ============================================================
    def today_count(self):
        if self._df_cache is None:
            return 0
        return len(self._df_cache)

    # ============================================================
    # SESSION INFO
    # ============================================================
    def get_session_info(self):
        return {
            "class": self.active_class,
            "period": self.active_period,
            "subject": self.active_subject,
            "file": self.file,
        }
