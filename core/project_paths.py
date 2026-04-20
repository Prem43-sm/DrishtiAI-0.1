from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", app_root()))
    return project_root()


def app_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def resolve_app_path(path_text: str | Path | None) -> Path:
    path = Path(str(path_text or "").strip())
    if not str(path):
        return app_root()
    if path.is_absolute():
        return path
    return app_root() / path


def resolve_resource_path(path_text: str | Path | None) -> Path:
    path = Path(str(path_text or "").strip())
    if not str(path):
        return resource_root()
    if path.is_absolute():
        return path
    return resource_root() / path


SETTINGS_FILE = app_path("settings.json")

STORAGE_DIR = app_path("storage")
ATTENDANCE_DIR = STORAGE_DIR / "attendance"
KNOWN_FACES_DIR = STORAGE_DIR / "known_faces"
REPORTS_DIR = STORAGE_DIR / "reports"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
NOISE_ALERTS_DIR = SNAPSHOTS_DIR / "noise_alerts"
TIMETABLE_DIR = STORAGE_DIR / "timetable"

MODELS_DIR = app_path("models")
EMOTION_MODELS_DIR = MODELS_DIR / "emotion"
LEGACY_EMOTION_MODELS_DIR = EMOTION_MODELS_DIR / "legacy"
FACE_RECOGNITION_MODELS_DIR = MODELS_DIR / "face_recognition"

DATASETS_DIR = app_path("datasets")
EMOTION_PIPELINE_DIR = DATASETS_DIR / "emotion_pipeline"
LEGACY_DATASETS_DIR = DATASETS_DIR / "legacy"
FINAL_DATASET_DIR = LEGACY_DATASETS_DIR / "final_dataset"
RAW_DATA_DIR = LEGACY_DATASETS_DIR / "raw_data"
RAW_SOURCE_DUMP_DIR = LEGACY_DATASETS_DIR / "source_dump"
PROCESSED_DATA_DIR = LEGACY_DATASETS_DIR / "processed_data"

TOOLS_DIR = app_path("tools")
ARCHIVE_DIR = app_path("archive")


def ensure_runtime_layout() -> None:
    for directory in (
        STORAGE_DIR,
        ATTENDANCE_DIR,
        KNOWN_FACES_DIR,
        REPORTS_DIR,
        SNAPSHOTS_DIR,
        NOISE_ALERTS_DIR,
        TIMETABLE_DIR,
        MODELS_DIR,
        EMOTION_MODELS_DIR,
        LEGACY_EMOTION_MODELS_DIR,
        FACE_RECOGNITION_MODELS_DIR,
        DATASETS_DIR,
        TOOLS_DIR,
        ARCHIVE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
