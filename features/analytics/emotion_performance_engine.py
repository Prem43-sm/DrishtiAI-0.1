from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from features.analytics.student_data import EMOTION_LABELS


EMOTION_WEIGHTS = {
    "happy": 2,
    "neutral": 1,
    "surprise": 1,
    "sad": -1,
    "fear": -2,
    "angry": -2,
}


@dataclass
class EmotionPerformanceEngine:
    """Calculates monthly emotion, engagement, stability, and performance scores."""

    def build_monthly_report(
        self,
        samples: pd.DataFrame,
        student_id: str,
        month: int,
        year: int,
    ) -> dict[str, Any]:
        period = samples[
            (samples["student"].astype(str) == str(student_id))
            & (samples["month"] == int(month))
            & (samples["year"] == int(year))
        ].copy()
        total = int(len(period))

        counts = {label: 0 for label in EMOTION_LABELS}
        if total:
            observed = period["emotion"].value_counts().to_dict()
            for label in EMOTION_LABELS:
                counts[label] = int(observed.get(label, 0))

        percentages = {
            f"{label}_percentage": round((counts[label] / total * 100.0) if total else 0.0, 2)
            for label in EMOTION_LABELS
        }
        weighted_total = sum(counts[label] * EMOTION_WEIGHTS[label] for label in EMOTION_LABELS)
        average_weight = weighted_total / total if total else 0.0
        emotion_score = self._clamp((average_weight + 2.0) / 4.0 * 100.0)
        engagement_score = self._clamp(
            percentages["happy_percentage"]
            + percentages["neutral_percentage"] * 0.75
            + percentages["surprise_percentage"] * 0.65
        )
        negative_percentage = (
            percentages["sad_percentage"]
            + percentages["angry_percentage"]
            + percentages["fear_percentage"]
        )
        stability_score = self._clamp(100.0 - (negative_percentage * 0.85) - (percentages["surprise_percentage"] * 0.15))
        performance_score = self._clamp(
            (emotion_score * 0.45) + (engagement_score * 0.35) + (stability_score * 0.20)
        )

        trend = self._daily_trend(period)
        positive = counts["happy"] + counts["neutral"] + counts["surprise"]
        negative = counts["sad"] + counts["fear"] + counts["angry"]

        return {
            "student_id": student_id,
            "month": int(month),
            "year": int(year),
            "total_samples": total,
            "counts": counts,
            **percentages,
            "emotion_score": round(emotion_score, 2),
            "engagement_score": round(engagement_score, 2),
            "stability_score": round(stability_score, 2),
            "performance_score": round(performance_score, 2),
            "performance_status": classify_performance(performance_score),
            "positive_negative_ratio": f"{positive}:{negative}",
            "trend": trend,
            "summary": self._summary(percentages, performance_score),
            "recommendations": self._recommendations(percentages, engagement_score, stability_score),
        }

    def _daily_trend(self, period: pd.DataFrame) -> pd.DataFrame:
        if period.empty or "date" not in period.columns:
            return pd.DataFrame(columns=["date", "score"])
        daily = period.copy()
        daily["weighted"] = daily["emotion"].map(EMOTION_WEIGHTS).fillna(0)
        trend = daily.groupby("date", as_index=False)["weighted"].mean()
        trend["score"] = ((trend["weighted"] + 2.0) / 4.0 * 100.0).clip(0, 100).round(2)
        return trend[["date", "score"]]

    def _summary(self, percentages: dict[str, float], performance_score: float) -> str:
        dominant = max(EMOTION_LABELS, key=lambda label: percentages[f"{label}_percentage"])
        return (
            f"Dominant emotion is {dominant}. Final performance score is "
            f"{performance_score:.1f}, classified as {classify_performance(performance_score)}."
        )

    def _recommendations(
        self,
        percentages: dict[str, float],
        engagement_score: float,
        stability_score: float,
    ) -> list[str]:
        notes = []
        if engagement_score < 60:
            notes.append("Increase interactive checks, short quizzes, and one-to-one prompting.")
        if stability_score < 65:
            notes.append("Schedule a supportive check-in and monitor negative emotion spikes.")
        if percentages["angry_percentage"] + percentages["fear_percentage"] > 25:
            notes.append("Review class context for stress triggers and provide calmer pacing.")
        if not notes:
            notes.append("Maintain current learning rhythm and continue monthly monitoring.")
        return notes

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, float(value)))


def classify_performance(score: float) -> str:
    score = float(score)
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Attention"

