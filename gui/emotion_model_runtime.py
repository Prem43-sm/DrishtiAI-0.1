import json
import os
from pathlib import Path

from gui.utils import resource_path


LEGACY_CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
DEFAULT_CLASS_NAMES = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]
DEFAULT_MODEL_PATH = "New_1_best_emotion_model.h5"
LEGACY_MODEL_PATH = "best_emotion_model.h5"
LEGACY_DATASET_ROOT = "Final_Dataset"
SIX_CLASS_DATASET_ROOTS = [
    "emotion_model_pipeline/combined_dataset_6class_high_accuracy",
    "emotion_model_pipeline/prepared_dataset_6class",
]
DISPLAY_LABEL_ALIASES = {
    "Suprise": "Surprise",
}


def normalize_label(label):
    text = str(label or "").strip()
    return DISPLAY_LABEL_ALIASES.get(text, text)


def normalize_class_names(class_names):
    return [normalize_label(name) for name in class_names if str(name or "").strip()]


def resolve_project_path(path):
    text = str(path or "").strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return text
    return resource_path(text)


def resolve_existing_path(path):
    resolved = resolve_project_path(path)
    if resolved and os.path.exists(resolved):
        return resolved
    return ""


def get_preferred_model_path(configured_path=None):
    for candidate in (configured_path, DEFAULT_MODEL_PATH, LEGACY_MODEL_PATH):
        resolved = resolve_existing_path(candidate)
        if resolved:
            return resolved
    return resolve_project_path(configured_path or DEFAULT_MODEL_PATH)


def _manifest_candidates(model_path=None):
    candidates = []
    resolved_model = resolve_project_path(model_path)
    if resolved_model:
        candidates.append(Path(resolved_model).resolve().with_name("class_names.json"))

    candidates.extend(
        Path(resource_path(rel_path))
        for rel_path in [
            "class_names.json",
            "emotion_model_pipeline/combined_dataset_6class_high_accuracy/class_names.json",
            "emotion_model_pipeline/prepared_dataset_6class/class_names.json",
        ]
    )

    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_class_names_from_manifest(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []

    if isinstance(payload, dict):
        values = payload.get("class_order") or payload.get("class_names") or []
    elif isinstance(payload, list):
        values = payload
    else:
        values = []

    return normalize_class_names(values)


def get_output_units(model):
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list) and output_shape:
        output_shape = output_shape[0]
    if not output_shape:
        return None

    try:
        return int(output_shape[-1])
    except (TypeError, ValueError, IndexError):
        return None


def infer_class_names(model=None, output_units=None, model_path=None):
    if output_units is None and model is not None:
        output_units = get_output_units(model)

    for manifest_path in _manifest_candidates(model_path):
        class_names = _load_class_names_from_manifest(manifest_path)
        if class_names and (output_units is None or len(class_names) == output_units):
            return class_names

    if output_units == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES[:]
    if output_units == len(LEGACY_CLASS_NAMES):
        return LEGACY_CLASS_NAMES[:]
    if output_units and output_units > 0:
        return [f"Class {idx + 1}" for idx in range(output_units)]
    return DEFAULT_CLASS_NAMES[:]


def get_dataset_root(output_units=None, class_names=None):
    if output_units is None and class_names is not None:
        output_units = len(class_names)

    if output_units == len(LEGACY_CLASS_NAMES):
        candidates = [LEGACY_DATASET_ROOT] + SIX_CLASS_DATASET_ROOTS
    else:
        candidates = SIX_CLASS_DATASET_ROOTS + [LEGACY_DATASET_ROOT]

    for candidate in candidates:
        resolved = resolve_existing_path(candidate)
        if resolved:
            return resolved

    return resolve_project_path(candidates[0])


def get_dataset_split_dir(split_name, output_units=None, class_names=None):
    return os.path.join(get_dataset_root(output_units=output_units, class_names=class_names), split_name)


def get_misbehavior_alert_emotions(class_names=None):
    alerts = {"Angry"}
    if class_names:
        names = set(normalize_class_names(class_names))
    else:
        names = set(DEFAULT_CLASS_NAMES) | set(LEGACY_CLASS_NAMES)
    if "Fear" in names:
        alerts.add("Fear")
    if "Ahegao" in names:
        alerts.add("Ahegao")
    return alerts


def get_model_display_name(model_path):
    text = str(model_path or "").strip()
    return os.path.basename(text) if text else DEFAULT_MODEL_PATH
