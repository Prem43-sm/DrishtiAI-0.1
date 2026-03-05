import json
import os
from datetime import datetime, timedelta


TIMETABLE_DIR = "timetable"


class TimeTableEngine:
    def __init__(self):
        os.makedirs(TIMETABLE_DIR, exist_ok=True)
        self.active_class = None
        self.active_period = None
        self.active_subject = None

    def _reset_active(self):
        self.active_class = None
        self.active_period = None
        self.active_subject = None

    def _parse_hhmm(self, text):
        try:
            return datetime.strptime(text, "%H:%M").time()
        except Exception:
            return None

    def _is_time_in_slot(self, now_time, start_time, end_time):
        # Normal slot: 08:00 -> 20:00
        if start_time <= end_time:
            return start_time <= now_time <= end_time

        # Overnight slot: 23:00 -> 07:00
        return now_time >= start_time or now_time <= end_time

    def _load_class_timetable(self, file_name):
        class_name = file_name.replace(".json", "")
        path = os.path.join(TIMETABLE_DIR, file_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return class_name, {}
        return class_name, data.get("days", {})

    def check_current_slot(self):
        now = datetime.now()
        now_time = now.time()
        today_name = now.strftime("%A")
        prev_day_name = (now - timedelta(days=1)).strftime("%A")

        self._reset_active()

        all_periods_cache = []

        for file_name in os.listdir(TIMETABLE_DIR):
            if not file_name.endswith(".json"):
                continue

            class_name, days = self._load_class_timetable(file_name)
            all_periods_cache.append((class_name, days))

            # 1) Check today's periods (includes normal and overnight start time segment).
            for p in days.get(today_name, []):
                start = self._parse_hhmm(str(p.get("start", "")))
                end = self._parse_hhmm(str(p.get("end", "")))
                if start is None or end is None:
                    continue

                if self._is_time_in_slot(now_time, start, end):
                    self.active_class = class_name
                    self.active_period = p.get("period")
                    self.active_subject = p.get("subject")
                    return True

            # 2) Check previous day's overnight slots for after-midnight continuation.
            for p in days.get(prev_day_name, []):
                start = self._parse_hhmm(str(p.get("start", "")))
                end = self._parse_hhmm(str(p.get("end", "")))
                if start is None or end is None:
                    continue

                if start <= end:
                    continue

                # If overnight, after-midnight active part is now <= end.
                if now_time <= end:
                    self.active_class = class_name
                    self.active_period = p.get("period")
                    self.active_subject = p.get("subject")
                    return True

        # Fallback:
        # If current weekday data is missing/misconfigured, still try to match by time
        # across all configured timetable slots.
        for class_name, days in all_periods_cache:
            for _, day_periods in days.items():
                for p in day_periods:
                    start = self._parse_hhmm(str(p.get("start", "")))
                    end = self._parse_hhmm(str(p.get("end", "")))
                    if start is None or end is None:
                        continue

                    if self._is_time_in_slot(now_time, start, end):
                        self.active_class = class_name
                        self.active_period = p.get("period")
                        self.active_subject = p.get("subject")
                        return True

        return False

    def get_active_class(self):
        return self.active_class

    def get_active_period(self):
        return self.active_period

    def get_active_subject(self):
        return self.active_subject


_engine = TimeTableEngine()


def get_current_session():
    if _engine.check_current_slot():
        return (
            _engine.get_active_class(),
            _engine.get_active_period(),
            _engine.get_active_subject(),
        )
    return None, None, None
