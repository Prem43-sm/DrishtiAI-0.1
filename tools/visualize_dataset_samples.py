from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR
if not (ROOT_DIR / "gui").exists() and (ROOT_DIR.parent / "gui").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from core.project_paths import REPORTS_DIR
from gui.emotion_model_runtime import normalize_class_names, resolve_project_path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT_DIR = str(REPORTS_DIR / "dataset_visualizations")
DATASET_CANDIDATES = [
    "datasets/emotion_pipeline/combined_dataset_6class_face_crops",
    "datasets/emotion_pipeline/combined_dataset_6class_high_accuracy",
    "datasets/emotion_pipeline/prepared_dataset_6class",
    "datasets/legacy/final_dataset",
]
KNOWN_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize DrishtiAI emotion dataset samples. "
            "The script can read a dataset root with train/val/test splits or a single split folder."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Dataset root or split folder. Defaults to the active DrishtiAI emotion dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated sample grids and summary files are saved.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["test"],
        help="Splits to visualize when a dataset root is provided. Example: --splits train val test",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of random samples to show for each class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save images without opening matplotlib windows.",
    )
    return parser.parse_args()


def warn_if_not_face_env() -> None:
    executable_text = str(Path(sys.executable).resolve()).lower()
    if "face_env" not in executable_text:
        print(
            "Warning: this script is not running from face_env. "
            "Recommended command: .\\face_env\\Scripts\\python.exe tools\\visualize_dataset_samples.py"
        )


def resolve_existing_path(path_text: str) -> Path | None:
    resolved = Path(resolve_project_path(path_text)).resolve()
    if resolved.exists():
        return resolved
    return None


def resolve_dataset_path(cli_dataset_path: str) -> Path:
    if cli_dataset_path.strip():
        dataset_path = resolve_existing_path(cli_dataset_path.strip())
        if dataset_path is None:
            raise FileNotFoundError(f"Dataset path not found: {cli_dataset_path}")
        return dataset_path

    for candidate in DATASET_CANDIDATES:
        dataset_path = resolve_existing_path(candidate)
        if dataset_path is not None:
            return dataset_path

    raise FileNotFoundError(
        "No DrishtiAI dataset was found. Checked:\n - " + "\n - ".join(DATASET_CANDIDATES)
    )


def load_manifest_class_order(dataset_root: Path) -> list[str]:
    manifest_path = dataset_root / "class_names.json"
    if not manifest_path.is_file():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []

    if isinstance(payload, dict):
        values = payload.get("class_order") or payload.get("class_names") or []
    elif isinstance(payload, list):
        values = payload
    else:
        values = []

    return normalize_class_names(values)


def get_split_context(dataset_path: Path, requested_splits: list[str]) -> tuple[Path, list[tuple[str, Path]]]:
    split_dirs = {
        child.name.lower(): child
        for child in dataset_path.iterdir()
        if child.is_dir() and child.name.lower() in KNOWN_SPLITS
    }
    if split_dirs:
        dataset_root = dataset_path
        selected_names = [name.lower() for name in requested_splits if name.lower() in split_dirs]
        if not selected_names:
            available = ", ".join(sorted(split_dirs))
            raise ValueError(f"No requested splits were found under {dataset_root}. Available: {available}")
        return dataset_root, [(name, split_dirs[name]) for name in selected_names]

    split_name = dataset_path.name.lower()
    if split_name in KNOWN_SPLITS:
        dataset_root = dataset_path.parent if (dataset_path.parent / "class_names.json").exists() else dataset_path
        return dataset_root, [(split_name, dataset_path)]

    return dataset_path, [(dataset_path.name.lower(), dataset_path)]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def normalize_folder_name(folder_name: str) -> str:
    normalized = normalize_class_names([folder_name])
    if normalized:
        return normalized[0]
    return folder_name.strip()


def resolve_class_pairs(split_dir: Path, manifest_order: list[str]) -> list[tuple[str, str]]:
    folder_names = sorted(child.name for child in split_dir.iterdir() if child.is_dir())
    if not folder_names:
        raise ValueError(f"No class folders found in: {split_dir}")

    normalized_folder_map = {normalize_folder_name(folder_name): folder_name for folder_name in folder_names}

    if manifest_order and all(label in normalized_folder_map for label in manifest_order):
        return [(normalized_folder_map[label], label) for label in manifest_order]

    return [(folder_name, normalize_folder_name(folder_name)) for folder_name in folder_names]


def select_sample_paths(image_paths: list[Path], sample_limit: int, rng: random.Random) -> list[Path]:
    candidates = list(image_paths)
    rng.shuffle(candidates)

    selected: list[Path] = []
    for image_path in candidates:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: skipping unreadable image: {image_path}")
            continue
        selected.append(image_path)
        if len(selected) >= sample_limit:
            break

    return selected


def load_image_rgb(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: failed to read image: {image_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def collect_split_samples(
    split_dir: Path,
    class_pairs: list[tuple[str, str]],
    samples_per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    split_rows: list[dict[str, object]] = []

    for folder_name, display_name in class_pairs:
        class_dir = split_dir / folder_name
        image_paths = [path for path in sorted(class_dir.iterdir()) if is_image_file(path)]
        sample_paths = select_sample_paths(image_paths, samples_per_class, rng)
        split_rows.append(
            {
                "folder_name": folder_name,
                "display_name": display_name,
                "class_dir": str(class_dir),
                "image_count": len(image_paths),
                "sample_paths": [str(path) for path in sample_paths],
            }
        )

    return split_rows


def create_split_grid(
    dataset_name: str,
    split_name: str,
    split_rows: list[dict[str, object]],
    samples_per_class: int,
):
    num_rows = len(split_rows)
    num_cols = max(1, samples_per_class)
    fig_width = max(14, num_cols * 3.2)
    fig_height = max(6, num_rows * 2.8)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes_array = np.asarray(axes, dtype=object).reshape(num_rows, num_cols)

    for col_idx in range(num_cols):
        axes_array[0, col_idx].set_title(f"Sample {col_idx + 1}", fontsize=11, pad=10)

    for row_idx, row_info in enumerate(split_rows):
        sample_paths = [Path(path_text) for path_text in row_info["sample_paths"]]
        class_label = str(row_info["display_name"])
        image_count = int(row_info["image_count"])

        for col_idx in range(num_cols):
            axis = axes_array[row_idx, col_idx]
            axis.axis("off")

            if col_idx == 0:
                axis.set_ylabel(
                    f"{class_label}\n({image_count} images)",
                    rotation=0,
                    fontsize=10,
                    labelpad=56,
                    va="center",
                )

            if col_idx >= len(sample_paths):
                axis.text(0.5, 0.5, "No sample", ha="center", va="center", fontsize=10, color="#6B7280")
                continue

            image = load_image_rgb(sample_paths[col_idx])
            if image is None:
                axis.text(0.5, 0.5, "Unreadable", ha="center", va="center", fontsize=10, color="#991B1B")
                continue

            axis.imshow(image)
            axis.axis("off")

    fig.suptitle(f"{dataset_name} - {split_name.title()} Sample Grid", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    return fig


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    warn_if_not_face_env()

    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be a positive integer.")

    dataset_path = resolve_dataset_path(args.dataset_path)
    dataset_root, split_context = get_split_context(dataset_path, args.splits)
    manifest_order = load_manifest_class_order(dataset_root)
    output_dir = Path(resolve_project_path(args.output_dir)).resolve() / dataset_path.name

    print(f"Dataset path: {dataset_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")

    summary: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "dataset_root": str(dataset_root),
        "samples_per_class": int(args.samples_per_class),
        "seed": int(args.seed),
        "splits": {},
    }

    for split_name, split_dir in split_context:
        class_pairs = resolve_class_pairs(split_dir, manifest_order)
        split_rows = collect_split_samples(
            split_dir=split_dir,
            class_pairs=class_pairs,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )

        print(f"\n{split_name.title()} split")
        print("-" * 60)
        for row in split_rows:
            print(f"{row['display_name']}: {row['image_count']} images")

        figure = create_split_grid(
            dataset_name=dataset_path.name,
            split_name=split_name,
            split_rows=split_rows,
            samples_per_class=args.samples_per_class,
        )
        output_path = output_dir / f"{split_name}_samples_grid.png"
        save_figure(figure, output_path)
        print(f"Saved sample grid: {output_path}")

        if not args.no_display:
            plt.show(block=False)
            plt.pause(0.1)

        plt.close(figure)

        summary["splits"][split_name] = {
            "split_dir": str(split_dir),
            "class_order": [row["display_name"] for row in split_rows],
            "class_folders": [row["folder_name"] for row in split_rows],
            "class_counts": {str(row["display_name"]): int(row["image_count"]) for row in split_rows},
            "output_file": str(output_path),
        }

    summary_path = output_dir / "dataset_sample_summary.json"
    save_summary(summary, summary_path)
    print(f"\nSaved summary: {summary_path}")
    print("Finished generating dataset sample grids.")


if __name__ == "__main__":
    main()
