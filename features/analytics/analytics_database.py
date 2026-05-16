from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from core.project_paths import ANALYTICS_DB_FILE, ensure_runtime_layout


def initialize_analytics_database() -> None:
    """Create analytics tables without touching legacy CSV storage."""
    ensure_runtime_layout()
    with sqlite3.connect(str(ANALYTICS_DB_FILE)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                month INTEGER NOT NULL,
                year INTEGER NOT NULL,
                happy_percentage REAL NOT NULL DEFAULT 0,
                sad_percentage REAL NOT NULL DEFAULT 0,
                angry_percentage REAL NOT NULL DEFAULT 0,
                neutral_percentage REAL NOT NULL DEFAULT 0,
                fear_percentage REAL NOT NULL DEFAULT 0,
                surprise_percentage REAL NOT NULL DEFAULT 0,
                engagement_score REAL NOT NULL DEFAULT 0,
                stability_score REAL NOT NULL DEFAULT 0,
                performance_score REAL NOT NULL DEFAULT 0,
                performance_status TEXT NOT NULL,
                report_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS focus_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                focus_score REAL NOT NULL DEFAULT 0,
                attention_percentage REAL NOT NULL DEFAULT 0,
                distraction_percentage REAL NOT NULL DEFAULT 0,
                active_time REAL NOT NULL DEFAULT 0,
                inactive_time REAL NOT NULL DEFAULT 0,
                movement_count INTEGER NOT NULL DEFAULT 0,
                sleep_detection_count INTEGER NOT NULL DEFAULT 0,
                looking_away_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                report_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emotion_reports_student_period "
            "ON emotion_reports(student_id, year, month)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_focus_reports_student_date "
            "ON focus_reports(student_id, date)"
        )


@contextmanager
def analytics_connection():
    initialize_analytics_database()
    conn = sqlite3.connect(str(ANALYTICS_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_emotion_report(report: dict[str, Any]) -> int:
    payload = {
        "student_id": report.get("student_id", ""),
        "month": int(report.get("month", 0)),
        "year": int(report.get("year", 0)),
        "happy_percentage": float(report.get("happy_percentage", 0.0)),
        "sad_percentage": float(report.get("sad_percentage", 0.0)),
        "angry_percentage": float(report.get("angry_percentage", 0.0)),
        "neutral_percentage": float(report.get("neutral_percentage", 0.0)),
        "fear_percentage": float(report.get("fear_percentage", 0.0)),
        "surprise_percentage": float(report.get("surprise_percentage", 0.0)),
        "engagement_score": float(report.get("engagement_score", 0.0)),
        "stability_score": float(report.get("stability_score", 0.0)),
        "performance_score": float(report.get("performance_score", 0.0)),
        "performance_status": report.get("performance_status", "Average"),
        "report_path": report.get("report_path", ""),
        "created_at": report.get("created_at") or datetime.now().isoformat(timespec="seconds"),
    }
    with analytics_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO emotion_reports (
                student_id, month, year,
                happy_percentage, sad_percentage, angry_percentage,
                neutral_percentage, fear_percentage, surprise_percentage,
                engagement_score, stability_score, performance_score,
                performance_status, report_path, created_at
            )
            VALUES (
                :student_id, :month, :year,
                :happy_percentage, :sad_percentage, :angry_percentage,
                :neutral_percentage, :fear_percentage, :surprise_percentage,
                :engagement_score, :stability_score, :performance_score,
                :performance_status, :report_path, :created_at
            )
            """,
            payload,
        )
        return int(cur.lastrowid)


def save_focus_report(report: dict[str, Any]) -> int:
    payload = {
        "student_id": report.get("student_id", ""),
        "date": str(report.get("date", "")),
        "focus_score": float(report.get("focus_score", 0.0)),
        "attention_percentage": float(report.get("attention_percentage", 0.0)),
        "distraction_percentage": float(report.get("distraction_percentage", 0.0)),
        "active_time": float(report.get("active_time", 0.0)),
        "inactive_time": float(report.get("inactive_time", 0.0)),
        "movement_count": int(report.get("movement_count", 0)),
        "sleep_detection_count": int(report.get("sleep_detection_count", 0)),
        "looking_away_count": int(report.get("looking_away_count", 0)),
        "status": report.get("status", "Distracted"),
        "report_path": report.get("report_path", ""),
        "created_at": report.get("created_at") or datetime.now().isoformat(timespec="seconds"),
    }
    with analytics_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO focus_reports (
                student_id, date, focus_score, attention_percentage,
                distraction_percentage, active_time, inactive_time,
                movement_count, sleep_detection_count, looking_away_count,
                status, report_path, created_at
            )
            VALUES (
                :student_id, :date, :focus_score, :attention_percentage,
                :distraction_percentage, :active_time, :inactive_time,
                :movement_count, :sleep_detection_count, :looking_away_count,
                :status, :report_path, :created_at
            )
            """,
            payload,
        )
        return int(cur.lastrowid)


def update_report_path(table_name: str, report_id: int, report_path: str | Path) -> None:
    if table_name not in {"emotion_reports", "focus_reports"}:
        raise ValueError("Unsupported analytics table")
    with analytics_connection() as conn:
        conn.execute(
            f"UPDATE {table_name} SET report_path = ? WHERE id = ?",
            (str(report_path), int(report_id)),
        )


def fetch_reports(table_name: str, limit: int = 250) -> list[dict[str, Any]]:
    if table_name not in {"emotion_reports", "focus_reports"}:
        raise ValueError("Unsupported analytics table")
    with analytics_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table_name} ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def analytics_database_path() -> Path:
    initialize_analytics_database()
    return ANALYTICS_DB_FILE

