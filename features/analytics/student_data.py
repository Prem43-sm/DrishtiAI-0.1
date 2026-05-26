from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.project_paths import ATTENDANCE_DIR, KNOWN_FACES_DIR, ensure_runtime_layout
from features.student_records_db import list_student_records


EMOTION_LABELS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]


def discover_classes() -> list[str]:
    ensure_runtime_layout()
    classes = set()
    for row in list_student_records():
        class_name = str(row.get("class_name", "")).strip()
        if class_name:
            classes.add(class_name)
    for root in (Path(KNOWN_FACES_DIR), Path(ATTENDANCE_DIR)):
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_dir():
                classes.add(item.name)
    return sorted(classes)


def discover_students(class_name: str = "All Classes") -> list[str]:
    ensure_runtime_layout()
    students = set()
    for row in list_student_records(None if class_name == "All Classes" else class_name):
        if str(row.get("status", "Active")).lower() == "left":
            continue
        name = str(row.get("student_name", "")).strip()
        if name:
            students.add(name)
    known_root = Path(KNOWN_FACES_DIR)
    if known_root.exists():
        class_dirs = (
            [known_root / class_name]
            if class_name and class_name != "All Classes"
            else [p for p in known_root.iterdir() if p.is_dir()]
        )
        for class_dir in class_dirs:
            if not class_dir.exists():
                continue
            for item in class_dir.iterdir():
                if item.suffix.lower() == ".npy":
                    students.add(item.stem)

    emotion_df = load_emotion_samples()
    if not emotion_df.empty:
        filtered = emotion_df
        if class_name and class_name != "All Classes":
            filtered = filtered[filtered["class"].astype(str) == class_name]
        students.update(filtered["student"].dropna().astype(str).tolist())
    return sorted(s for s in students if s.strip())


def load_emotion_samples() -> pd.DataFrame:
    ensure_runtime_layout()
    attendance_root = Path(ATTENDANCE_DIR)
    if not attendance_root.exists():
        return pd.DataFrame()
    preferred = attendance_root / "emotion_data.csv"
    files = [preferred] if preferred.exists() else sorted(attendance_root.rglob("*.csv"))
    frames = []
    for csv_file in files:
        part = _read_emotion_csv(csv_file, attendance_root)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["emotion"] = data["emotion"].map(normalize_emotion)
    data = data[data["emotion"].isin(EMOTION_LABELS)]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    return data.reset_index(drop=True)


def normalize_emotion(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "angry": "angry",
        "anger": "angry",
        "fearful": "fear",
        "fear": "fear",
        "happy": "happy",
        "happiness": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprised": "surprise",
        "surprise": "surprise",
    }
    return aliases.get(text, text)


def _read_emotion_csv(csv_file: Path, attendance_root: Path) -> pd.DataFrame:
    try:
        header = pd.read_csv(csv_file, nrows=0)
    except Exception:
        return pd.DataFrame()

    student_col = _find_column(header.columns, ["name", "student", "student_name", "studentid"])
    emotion_col = _find_column(header.columns, ["emotion", "mood", "expression"])
    date_col = _find_column(header.columns, ["date", "day", "recorded_date"])
    time_col = _find_column(header.columns, ["time", "recorded_time"])
    class_col = _find_column(header.columns, ["class", "classname", "batch"])
    section_col = _find_column(header.columns, ["section", "division"])

    if not student_col or not emotion_col:
        return pd.DataFrame()

    usecols = [student_col, emotion_col]
    for column in (date_col, time_col, class_col, section_col):
        if column and column not in usecols:
            usecols.append(column)
    try:
        raw = pd.read_csv(csv_file, usecols=usecols)
    except Exception:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["student"] = raw[student_col].astype(str).str.strip()
    out["emotion"] = raw[emotion_col].astype(str).str.strip()
    out["class"] = (
        raw[class_col].astype(str).str.strip()
        if class_col
        else _infer_class_name(csv_file, attendance_root)
    )
    out["section"] = raw[section_col].astype(str).str.strip() if section_col else ""

    if date_col:
        date_text = raw[date_col].astype(str).str.strip()
    else:
        inferred_date = _infer_date_from_filename(csv_file.name)
        date_text = pd.Series([inferred_date] * len(raw))
    if time_col:
        out["datetime"] = pd.to_datetime(date_text + " " + raw[time_col].astype(str), errors="coerce")
    else:
        out["datetime"] = pd.to_datetime(date_text, errors="coerce")
    out["date"] = out["datetime"].dt.normalize()

    return out[(out["student"] != "") & (out["emotion"] != "")]


def _find_column(columns, candidates: list[str]):
    lookup = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def _infer_class_name(csv_file: Path, attendance_root: Path) -> str:
    if csv_file.parent == attendance_root:
        return "General"
    return csv_file.parent.name


def _infer_date_from_filename(filename: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    return pd.Timestamp.now().strftime("%Y-%m-%d")
