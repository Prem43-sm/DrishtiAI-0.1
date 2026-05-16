from __future__ import annotations

from typing import Any


def emotion_insights(report: dict[str, Any]) -> list[str]:
    insights = []
    stress = float(report.get("sad_percentage", 0)) + float(report.get("fear_percentage", 0)) + float(report.get("angry_percentage", 0))
    positive = float(report.get("happy_percentage", 0)) + float(report.get("neutral_percentage", 0))
    engagement = float(report.get("engagement_score", 0))
    stability = float(report.get("stability_score", 0))

    if stress >= 35:
        insights.append("Student shows elevated stress signals during the selected month.")
    if engagement < 55:
        insights.append("Student engagement is low and may need targeted classroom interaction.")
    if stability < 60:
        insights.append("Emotional stability is inconsistent; monitor recurring negative emotion days.")
    if positive >= 70:
        insights.append("Positive emotional engagement is strong and supports learning readiness.")
    if not insights:
        insights.append("Emotion pattern is balanced with no strong risk signal in this period.")
    return insights


def focus_insights(report: dict[str, Any]) -> list[str]:
    insights = []
    focus_score = float(report.get("focus_score", 0))
    inactive = float(report.get("inactive_time", 0))
    movement = int(report.get("movement_count", 0))
    sleeping = int(report.get("sleep_detection_count", 0))
    looking_away = int(report.get("looking_away_count", 0))

    if focus_score < 40:
        insights.append("Low focus detected; student needs closer attention support.")
    if inactive >= 1800:
        insights.append("Student attention decreases during longer sessions.")
    if movement >= 20:
        insights.append("Excessive movement suggests restlessness or environmental distraction.")
    if sleeping > 0:
        insights.append("Sleeping or prolonged eye closure was detected during monitoring.")
    if looking_away >= 10:
        insights.append("Looking-away frequency is high and may indicate off-screen distraction.")
    if not insights:
        insights.append("Focus pattern is healthy for the captured session.")
    return insights

