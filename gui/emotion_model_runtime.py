import json
import os
from pathlib import Path

from core.project_paths import resolve_app_path
from gui.utils import resource_path


LEGACY_CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
DEFAULT_CLASS_NAMES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
DEFAULT_MODEL_PATH = "models/emotion/step11_high_accuracy/final_model.h5"
DEFAULT_TRAINING_MODEL_PATH = "models/emotion/trained_emotion_model.h5"
LEGACY_MODEL_PATHS = [
    "models/emotion/legacy/New_1_best_emotion_model.h5",
    "models/emotion/legacy/best_emotion_model.h5",
]
LEGACY_DATASET_ROOT = "datasets/legacy/final_dataset"
SIX_CLASS_DATASET_ROOTS = [
    "datasets/emotion_pipeline/combined_dataset_6class_high_accuracy",
    "datasets/emotion_pipeline/prepared_dataset_6class",
]
MODEL_METADATA_FILENAMES = [
    "class_names.json",
    "class_indices.json",
    "training_config.json",
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
    app_candidate = resolve_app_path(text)
    if app_candidate.exists():
        return str(app_candidate)
    return resource_path(text)


def resolve_existing_path(path):
    resolved = resolve_project_path(path)
    if resolved and os.path.exists(resolved):
        return resolved
    return ""


def get_preferred_model_path(configured_path=None):
    configured_text = str(configured_path or "").strip()
    if configured_text:
        resolved = resolve_project_path(configured_text)
        if os.path.exists(resolved):
            return resolved
        raise FileNotFoundError(f"Configured model file not found: {resolved}")

    checked_paths = []
    for candidate in [DEFAULT_MODEL_PATH] + LEGACY_MODEL_PATHS:
        resolved = resolve_existing_path(candidate)
        checked_paths.append(resolve_project_path(candidate))
        if resolved:
            return resolved

    raise FileNotFoundError(
        "No emotion model file was found. "
        f"Checked: {', '.join(checked_paths)}"
    )


def _metadata_candidates(model_path=None):
    candidates = []
    resolved_model = resolve_project_path(model_path)
    if resolved_model:
        model_dir = Path(resolved_model).resolve().parent
        search_dirs = [model_dir]

        # Training exports sometimes keep epoch checkpoints in an `epoch_models`
        # subfolder while metadata stays one level above beside `final_model.h5`.
        for parent_dir in model_dir.parents:
            if parent_dir == model_dir.anchor:
                break
            search_dirs.append(parent_dir)
            if len(search_dirs) >= 3:
                break

        for search_dir in search_dirs:
            for filename in MODEL_METADATA_FILENAMES:
                candidates.append(search_dir / filename)

    candidates.extend(
        Path(resource_path(rel_path))
        for rel_path in [
            "class_names.json",
            "class_indices.json",
            "training_config.json",
            "datasets/emotion_pipeline/combined_dataset_6class_high_accuracy/class_names.json",
            "datasets/emotion_pipeline/prepared_dataset_6class/class_names.json",
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


def _load_json_payload(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _ordered_labels_from_class_indices(class_indices):
    if not isinstance(class_indices, dict):
        return []

    ordered_items = []
    for label, index in class_indices.items():
        try:
            ordered_items.append((int(index), label))
        except (TypeError, ValueError):
            return []

    ordered_items.sort(key=lambda item: item[0])
    return [label for _, label in ordered_items]


def _load_class_names_from_metadata(path):
    payload = _load_json_payload(path)
    if payload is None:
        return []

    if isinstance(payload, dict):
        if Path(path).name == "class_indices.json":
            values = _ordered_labels_from_class_indices(payload)
        else:
            values = payload.get("class_order") or payload.get("class_names") or []
            if not values:
                values = _ordered_labels_from_class_indices(payload.get("class_indices"))
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

    for metadata_path in _metadata_candidates(model_path):
        class_names = _load_class_names_from_metadata(metadata_path)
        if class_names and (output_units is None or len(class_names) == output_units):
            return class_names

    if output_units == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES[:]
    if output_units == len(LEGACY_CLASS_NAMES):
        return LEGACY_CLASS_NAMES[:]
    if output_units and output_units > 0:
        return [f"Class {idx + 1}" for idx in range(output_units)]
    return DEFAULT_CLASS_NAMES[:]


def infer_model_image_size(model=None, model_path=None, fallback=(96, 96)):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]

    try:
        height = int(input_shape[1])
        width = int(input_shape[2])
    except (TypeError, ValueError, IndexError):
        height = 0
        width = 0

    if height > 0 and width > 0:
        return (height, width)

    for metadata_path in _metadata_candidates(model_path):
        if Path(metadata_path).name != "training_config.json":
            continue
        payload = _load_json_payload(metadata_path)
        if not isinstance(payload, dict):
            continue
        try:
            size = int(payload.get("img_size", 0))
        except (TypeError, ValueError):
            size = 0
        if size > 0:
            return (size, size)

    return fallback


def _iter_model_layers(layer_or_model):
    for child_layer in getattr(layer_or_model, "layers", []) or []:
        yield child_layer
        yield from _iter_model_layers(child_layer)


def model_uses_embedded_preprocessing(model=None):
    if model is None:
        return False

    for layer in _iter_model_layers(model):
        if layer.__class__.__name__ == "Rescaling":
            return True

    return False


def prepare_emotion_image_input(image_rgb, model=None):
    image = image_rgb.astype("float32", copy=False)
    if model_uses_embedded_preprocessing(model):
        return image
    return image / 255.0


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
