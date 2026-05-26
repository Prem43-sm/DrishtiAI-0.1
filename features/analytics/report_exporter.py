from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from core.project_paths import ANALYTICS_REPORTS_DIR, ensure_runtime_layout
from features.analytics.ai_insight_engine import emotion_insights, focus_insights
from features.analytics.student_data import EMOTION_LABELS


def export_emotion_report(report: dict[str, Any], format_name: str) -> Path:
    ensure_runtime_layout()
    target = _target_path("emotion", report.get("student_id", "student"), format_name)
    flat = _flatten_emotion_report(report)
    _export_flat_report(target, flat, report, format_name, "Emotion Performance Analytics")
    return target


def export_focus_report(report: dict[str, Any], format_name: str) -> Path:
    ensure_runtime_layout()
    target = _target_path("focus", report.get("student_id", "student"), format_name)
    flat = _flatten_focus_report(report)
    _export_flat_report(target, flat, report, format_name, "Focus Mode Monitoring")
    return target


def export_overall_student_report(report: dict[str, Any], format_name: str) -> Path:
    ensure_runtime_layout()
    target = _target_path("overall", report.get("student_id", "student"), format_name)
    flat = _flatten_overall_report(report)
    _export_flat_report(target, flat, report, format_name, "Overall Student Report")
    return target


def _target_path(prefix: str, student_id: str, format_name: str) -> Path:
    suffix = {"PDF": ".pdf", "CSV": ".csv", "Excel": ".xlsx"}[format_name]
    safe_student = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(student_id))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ANALYTICS_REPORTS_DIR / f"{prefix}_{safe_student}_{timestamp}{suffix}"


def _export_flat_report(target: Path, flat: dict[str, Any], report: dict[str, Any], format_name: str, title: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "CSV":
        pd.DataFrame([flat]).to_csv(target, index=False)
    elif format_name == "Excel":
        pd.DataFrame([flat]).to_excel(target, index=False)
    elif format_name == "PDF":
        _export_pdf(target, flat, report, title)
    else:
        raise ValueError("Unsupported export format")


def _flatten_emotion_report(report: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "student_id": report.get("student_id"),
        "month": report.get("month"),
        "year": report.get("year"),
        "emotion_score": report.get("emotion_score"),
        "engagement_score": report.get("engagement_score"),
        "stability_score": report.get("stability_score"),
        "performance_score": report.get("performance_score"),
        "performance_status": report.get("performance_status"),
        "positive_negative_ratio": report.get("positive_negative_ratio"),
        "summary": report.get("summary"),
        "recommendations": " | ".join(report.get("recommendations", [])),
        "ai_summary": " | ".join(emotion_insights(report)),
    }
    for label in EMOTION_LABELS:
        flat[f"{label}_percentage"] = report.get(f"{label}_percentage", 0)
    return flat


def _flatten_focus_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": report.get("student_id"),
        "date": report.get("date"),
        "focus_score": report.get("focus_score"),
        "attention_percentage": report.get("attention_percentage"),
        "distraction_percentage": report.get("distraction_percentage"),
        "active_time": report.get("active_time"),
        "inactive_time": report.get("inactive_time"),
        "movement_count": report.get("movement_count"),
        "sleep_detection_count": report.get("sleep_detection_count"),
        "looking_away_count": report.get("looking_away_count"),
        "status": report.get("status"),
        "ai_summary": " | ".join(focus_insights(report)),
    }


def _flatten_overall_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": report.get("student_id"),
        "class": report.get("class"),
        "month": report.get("month"),
        "year": report.get("year"),
        "rank": report.get("rank"),
        "overall_points": report.get("overall_points"),
        "overall_status": report.get("overall_status"),
        "attendance_score": report.get("attendance_score"),
        "present_sessions": report.get("present_sessions"),
        "total_sessions": report.get("total_sessions"),
        "emotion_score": report.get("emotion_score"),
        "engagement_score": report.get("engagement_score"),
        "stability_score": report.get("stability_score"),
        "performance_score": report.get("performance_score"),
        "performance_status": report.get("performance_status"),
        "focus_score": _format_optional_score(report.get("focus_score")),
        "focus_sessions": report.get("focus_sessions"),
        "focus_status": report.get("focus_status"),
        "summary": report.get("summary"),
        "recommendations": " | ".join(report.get("recommendations", [])),
    }


def _export_pdf(target: Path, flat: dict[str, Any], report: dict[str, Any], title: str) -> None:
    with PdfPages(target) as pdf:
        fig = Figure(figsize=(8.27, 11.69), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.axis("off")
        y = 0.97
        ax.text(0.02, y, title, fontsize=17, weight="bold", color="#1f2937")
        y -= 0.05
        for key, value in flat.items():
            line = f"{key.replace('_', ' ').title()}: {value}"
            ax.text(0.02, y, line[:115], fontsize=9, color="#111827")
            y -= 0.027
            if y < 0.08:
                pdf.savefig(fig)
                fig = Figure(figsize=(8.27, 11.69), tight_layout=True)
                ax = fig.add_subplot(111)
                ax.axis("off")
                y = 0.97
        pdf.savefig(fig)

        if "counts" in report:
            fig = Figure(figsize=(8.27, 5.0), tight_layout=True)
            ax = fig.add_subplot(111)
            labels = list(report["counts"].keys())
            values = list(report["counts"].values())
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title("Emotion Distribution")
            pdf.savefig(fig)

        if "overall_points" in report:
            fig = Figure(figsize=(8.27, 4.8), tight_layout=True)
            ax = fig.add_subplot(111)
            labels = ["Attendance", "Emotion", "Performance", "Focus", "Overall"]
            values = [
                report.get("attendance_score", 0),
                report.get("emotion_score", 0),
                report.get("performance_score", 0),
                report.get("focus_score") or 0,
                report.get("overall_points", 0),
            ]
            ax.bar(labels, values, color=["#2563eb", "#22c55e", "#f59e0b", "#38bdf8", "#a855f7"])
            ax.set_ylim(0, 100)
            ax.set_title("Combined Score Breakdown")
            ax.set_ylabel("Points")
            pdf.savefig(fig)


def _format_optional_score(value: Any) -> Any:
    if value is None:
        return "No Data"
    return value
