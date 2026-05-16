from __future__ import annotations

from pathlib import Path
from typing import Any

from features.analytics.analytics_database import (
    fetch_reports,
    save_emotion_report,
    save_focus_report,
    update_report_path,
)
from features.analytics.emotion_performance_engine import EmotionPerformanceEngine
from features.analytics.report_exporter import export_emotion_report, export_focus_report
from features.analytics.student_data import load_emotion_samples


def generate_emotion_report(student_id: str, month: int, year: int) -> dict[str, Any]:
    """Internal service API used by UI routes/windows to create emotion reports."""
    samples = load_emotion_samples()
    report = EmotionPerformanceEngine().build_monthly_report(samples, student_id, month, year)
    report["id"] = save_emotion_report(report)
    return report


def create_focus_report(report: dict[str, Any]) -> dict[str, Any]:
    """Internal service API used by focus tracking workers after a session ends."""
    saved = dict(report)
    saved["id"] = save_focus_report(saved)
    return saved


def export_saved_emotion_report(report: dict[str, Any], format_name: str) -> Path:
    path = export_emotion_report(report, format_name)
    report_id = report.get("id")
    if report_id:
        update_report_path("emotion_reports", int(report_id), path)
    return path


def export_saved_focus_report(report: dict[str, Any], format_name: str) -> Path:
    path = export_focus_report(report, format_name)
    report_id = report.get("id")
    if report_id:
        update_report_path("focus_reports", int(report_id), path)
    return path


def list_emotion_reports(limit: int = 250) -> list[dict[str, Any]]:
    return fetch_reports("emotion_reports", limit=limit)


def list_focus_reports(limit: int = 250) -> list[dict[str, Any]]:
    return fetch_reports("focus_reports", limit=limit)

