from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.project_paths import ANALYTICS_REPORTS_DIR, ATTENDANCE_DIR, ensure_runtime_layout
from features.analytics.analytics_database import analytics_connection
from features.analytics.emotion_performance_engine import EmotionPerformanceEngine
from features.analytics.student_data import discover_students


OVERALL_WEIGHTS = {
    "attendance_score": 0.25,
    "emotion_score": 0.25,
    "performance_score": 0.30,
    "focus_score": 0.20,
}


def classify_overall(score: float) -> str:
    score = float(score)
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Improving"
    return "Needs Attention"


@dataclass
class OverallStudentReportEngine:
    """Combines attendance, emotion, performance, and focus into one student rank."""

    emotion_engine: EmotionPerformanceEngine | None = None

    def __post_init__(self) -> None:
        if self.emotion_engine is None:
            self.emotion_engine = EmotionPerformanceEngine()

    def build_class_report(
        self,
        emotion_samples: pd.DataFrame,
        class_name: str,
        month: int,
        year: int,
    ) -> list[dict[str, Any]]:
        ensure_runtime_layout()
        students = discover_students(class_name)
        students = sorted(set(students) | set(self._students_from_attendance(class_name, month, year)))
        rows = [
            self.build_student_report(emotion_samples, class_name, student, month, year)
            for student in students
            if str(student).strip()
        ]
        rows.sort(key=lambda row: row["overall_points"], reverse=True)
        previous_score = None
        previous_rank = 0
        for index, row in enumerate(rows, start=1):
            if previous_score is None or row["overall_points"] != previous_score:
                previous_rank = index
                previous_score = row["overall_points"]
            row["rank"] = previous_rank
        return rows

    def build_student_report(
        self,
        emotion_samples: pd.DataFrame,
        class_name: str,
        student_id: str,
        month: int,
        year: int,
    ) -> dict[str, Any]:
        attendance = self._attendance_summary(class_name, student_id, month, year)
        emotion = self.emotion_engine.build_monthly_report(
            emotion_samples,
            student_id,
            month,
            year,
        )
        focus = self._focus_summary(student_id, month, year)
        overall_points = self._overall_points(
            attendance["attendance_score"],
            emotion["emotion_score"],
            emotion["performance_score"],
            focus["focus_score"],
            focus["has_focus_data"],
        )
        return {
            "student_id": student_id,
            "class": class_name,
            "month": int(month),
            "year": int(year),
            "attendance_score": attendance["attendance_score"],
            "present_sessions": attendance["present_sessions"],
            "total_sessions": attendance["total_sessions"],
            "emotion_score": emotion["emotion_score"],
            "engagement_score": emotion["engagement_score"],
            "stability_score": emotion["stability_score"],
            "performance_score": emotion["performance_score"],
            "performance_status": emotion["performance_status"],
            "focus_score": focus["focus_score"],
            "has_focus_data": focus["has_focus_data"],
            "focus_sessions": focus["focus_sessions"],
            "focus_status": focus["focus_status"],
            "overall_points": overall_points,
            "overall_status": classify_overall(overall_points),
            "rank": None,
            "summary": self._summary(attendance, emotion, focus, overall_points),
            "recommendations": self._recommendations(attendance, emotion, focus, overall_points),
        }

    def _overall_points(
        self,
        attendance_score: float,
        emotion_score: float,
        performance_score: float,
        focus_score: float | None,
        has_focus_data: bool,
    ) -> float:
        components = [
            (attendance_score, OVERALL_WEIGHTS["attendance_score"]),
            (emotion_score, OVERALL_WEIGHTS["emotion_score"]),
            (performance_score, OVERALL_WEIGHTS["performance_score"]),
        ]
        if has_focus_data and focus_score is not None:
            components.append((focus_score, OVERALL_WEIGHTS["focus_score"]))

        total_weight = sum(weight for _, weight in components) or 1.0
        score = sum(value * weight for value, weight in components) / total_weight
        return round(max(0.0, min(100.0, score)), 2)

    def _students_from_attendance(self, class_name: str, month: int, year: int) -> list[str]:
        class_dir = Path(ATTENDANCE_DIR) / class_name
        if not class_dir.exists():
            return []
        names = set()
        for csv_file in self._attendance_files(class_dir, month, year):
            try:
                data = pd.read_csv(csv_file, usecols=["Name"])
            except Exception:
                continue
            names.update(data["Name"].dropna().astype(str).str.strip().tolist())
        return sorted(name for name in names if name)

    def _attendance_summary(self, class_name: str, student_id: str, month: int, year: int) -> dict[str, Any]:
        class_dir = Path(ATTENDANCE_DIR) / class_name
        files = self._attendance_files(class_dir, month, year)
        present_sessions = 0
        for csv_file in files:
            try:
                data = pd.read_csv(csv_file, usecols=["Name"])
            except Exception:
                continue
            names = set(data["Name"].dropna().astype(str).str.strip())
            if str(student_id).strip() in names:
                present_sessions += 1
        total_sessions = len(files)
        attendance_score = (present_sessions / total_sessions * 100.0) if total_sessions else 0.0
        return {
            "attendance_score": round(attendance_score, 2),
            "present_sessions": int(present_sessions),
            "total_sessions": int(total_sessions),
        }

    def _attendance_files(self, class_dir: Path, month: int, year: int) -> list[Path]:
        if not class_dir.exists():
            return []
        prefix = f"{int(year):04d}-{int(month):02d}-"
        days_in_month = calendar.monthrange(int(year), int(month))[1]
        valid_dates = {f"{prefix}{day:02d}" for day in range(1, days_in_month + 1)}
        files = []
        for csv_file in sorted(class_dir.glob("*.csv")):
            if "_P" not in csv_file.name:
                continue
            date_part = csv_file.name.split("_P", 1)[0]
            if date_part in valid_dates:
                files.append(csv_file)
        return files

    def _focus_summary(self, student_id: str, month: int, year: int) -> dict[str, Any]:
        start = f"{int(year):04d}-{int(month):02d}-01"
        days_in_month = calendar.monthrange(int(year), int(month))[1]
        end = f"{int(year):04d}-{int(month):02d}-{days_in_month:02d}"
        with analytics_connection() as conn:
            rows = conn.execute(
                """
                SELECT focus_score, status
                FROM focus_reports
                WHERE student_id = ? AND date BETWEEN ? AND ?
                ORDER BY date DESC, created_at DESC
                """,
                (str(student_id), start, end),
            ).fetchall()
        focus_rows = [
            {
                "focus_score": float(row["focus_score"] or 0.0),
                "status": row["status"] or "Saved",
                "date": "",
                "active_time": "",
                "inactive_time": "",
            }
            for row in rows
        ]
        focus_rows.extend(self._focus_exports(student_id, month, year))
        focus_rows = self._dedupe_focus_rows(focus_rows)

        if not focus_rows:
            return {
                "focus_score": None,
                "has_focus_data": False,
                "focus_sessions": 0,
                "focus_status": "No Data",
            }
        scores = [float(row["focus_score"] or 0.0) for row in focus_rows]
        average = round(sum(scores) / len(scores), 2)
        return {
            "focus_score": average,
            "has_focus_data": True,
            "focus_sessions": len(focus_rows),
            "focus_status": focus_rows[0]["status"] or "Saved",
        }

    def _focus_exports(self, student_id: str, month: int, year: int) -> list[dict[str, Any]]:
        reports_dir = Path(ANALYTICS_REPORTS_DIR)
        if not reports_dir.exists():
            return []

        rows = []
        safe_student = self._safe_export_student(student_id)
        patterns = [f"focus_{safe_student}_*.csv", f"focus_{safe_student}_*.xlsx"]
        for pattern in patterns:
            for report_file in sorted(reports_dir.glob(pattern)):
                data = self._read_focus_export(report_file)
                if data.empty:
                    continue
                for _, row in data.iterrows():
                    if str(row.get("student_id", "")).strip() != str(student_id).strip():
                        continue
                    report_date = pd.to_datetime(row.get("date"), errors="coerce")
                    if pd.isna(report_date) or report_date.month != int(month) or report_date.year != int(year):
                        continue
                    rows.append(
                        {
                            "focus_score": float(row.get("focus_score", 0.0) or 0.0),
                            "status": str(row.get("status", "Exported Report") or "Exported Report"),
                            "date": str(row.get("date", "")),
                            "active_time": str(row.get("active_time", "")),
                            "inactive_time": str(row.get("inactive_time", "")),
                        }
                    )
        return rows

    def _dedupe_focus_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = []
        seen = set()
        for row in rows:
            key = (
                round(float(row.get("focus_score", 0.0) or 0.0), 2),
                row.get("date", ""),
                row.get("active_time", ""),
                row.get("inactive_time", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    def _read_focus_export(self, report_file: Path) -> pd.DataFrame:
        try:
            if report_file.suffix.lower() == ".csv":
                return pd.read_csv(report_file)
            if report_file.suffix.lower() == ".xlsx":
                return pd.read_excel(report_file)
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def _safe_export_student(self, student_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(student_id))

    def _summary(
        self,
        attendance: dict[str, Any],
        emotion: dict[str, Any],
        focus: dict[str, Any],
        overall_points: float,
    ) -> str:
        return (
            f"Overall score is {overall_points:.1f}/100 ({classify_overall(overall_points)}). "
            f"Attendance is {attendance['attendance_score']:.1f}% across "
            f"{attendance['total_sessions']} sessions. Emotion score is "
            f"{emotion['emotion_score']:.1f}, performance is {emotion['performance_score']:.1f}, "
            f"and average focus is {self._format_score(focus['focus_score'])}."
        )

    def _recommendations(
        self,
        attendance: dict[str, Any],
        emotion: dict[str, Any],
        focus: dict[str, Any],
        overall_points: float,
    ) -> list[str]:
        notes = []
        if attendance["attendance_score"] < 75:
            notes.append("Improve attendance consistency before academic gaps widen.")
        if emotion["engagement_score"] < 60:
            notes.append("Use interactive checks and short prompts to improve classroom engagement.")
        if not focus["has_focus_data"]:
            notes.append("Save a focus monitoring session for this student to include focus in the report.")
        elif focus["focus_score"] < 60:
            notes.append("Add focus support through seating, shorter checkpoints, or teacher follow-up.")
        if emotion["performance_score"] < 60:
            notes.append("Plan remedial practice based on current performance trend.")
        if overall_points >= 85:
            notes.append("Maintain current routine and offer enrichment work.")
        if not notes:
            notes.append("Continue regular monitoring and review the next monthly report.")
        return notes

    def _format_score(self, value: float | None) -> str:
        if value is None:
            return "No Data"
        return f"{value:.1f}"
