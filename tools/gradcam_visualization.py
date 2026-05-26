from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from gui.emotion_model_runtime import get_preferred_model_path, infer_class_names
except Exception:
    get_preferred_model_path = None
    infer_class_names = None


DEFAULT_IMAGE_SIZE = (160, 160)
DEFAULT_ALPHA = 0.4
DEFAULT_REPORTS_DIR = ROOT_DIR / "storage" / "reports" / "gradcam"
DEFAULT_REPORT_SUFFIX = "_gradcam_report.png"
CONV_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.Conv2DTranspose,
)


def resolve_default_model_path() -> str:
    if get_preferred_model_path is None:
        return ""

    try:
        return str(get_preferred_model_path())
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    default_model_path = resolve_default_model_path()
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM visualization for a trained TensorFlow/Keras CNN model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path,
        help="Path to the trained .h5/.keras model. Defaults to the project's configured model when available.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to the input image used for Grad-CAM.",
    )
    parser.add_argument(
        "--last-conv-layer",
        type=str,
        default="",
        help="Optional manual Grad-CAM target layer name. If omitted, the script auto-detects the last connected 4D feature-map layer.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help="Optional class index to visualize. Defaults to the top predicted class.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Blend strength for the overlay image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the superimposed Grad-CAM image.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="",
        help=(
            "Optional path to save the full Grad-CAM report image. "
            "If omitted, the script saves a report under storage/reports/gradcam/."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip opening the Matplotlib preview window and only save the image files.",
    )
    parser.add_argument(
        "--list-conv-layers",
        action="store_true",
        help="Print Grad-CAM-safe feature-map layers and convolutional layer names, then exit.",
    )
    return parser.parse_args()


def _iter_layers_recursive(layer_or_model: tf.keras.layers.Layer, seen: set[int] | None = None):
    if seen is None:
        seen = set()

    for layer in getattr(layer_or_model, "layers", []) or []:
        layer_id = id(layer)
        if layer_id in seen:
            continue
        seen.add(layer_id)
        yield layer
        yield from _iter_layers_recursive(layer, seen)


def _first_connected_output(layer: tf.keras.layers.Layer):
    try:
        return layer.get_output_at(0)
    except (AttributeError, ValueError):
        return layer.output


def _first_connected_input(layer: tf.keras.layers.Layer):
    try:
        return layer.get_input_at(0)
    except (AttributeError, ValueError):
        return layer.input


def list_conv_layer_names(model: tf.keras.Model) -> list[str]:
    names: list[str] = []

    for layer in _iter_layers_recursive(model):
        if isinstance(layer, CONV_LAYER_TYPES):
            names.append(layer.name)

    return names


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    conv_layer_names = list_conv_layer_names(model)
    if conv_layer_names:
        return conv_layer_names[-1]

    return find_last_connected_feature_map_layer_name(model)


def list_connected_feature_map_layer_names(model: tf.keras.Model) -> list[str]:
    names: list[str] = []

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, list) and output_shape:
            output_shape = output_shape[0]
        if output_shape is not None and len(output_shape) == 4:
            names.append(layer.name)

    return list(reversed(names))


def find_last_connected_feature_map_layer_name(model: tf.keras.Model) -> str:
    feature_map_layer_names = list_connected_feature_map_layer_names(model)
    if feature_map_layer_names:
        return feature_map_layer_names[-1]

    raise ValueError("Could not find a connected 4D feature-map layer for Grad-CAM.")


def build_feature_extractor_and_head(
    model: tf.keras.Model,
    layer_name: str,
) -> tuple[tf.keras.Model, tf.keras.Model] | None:
    """Split the model into a feature extractor and a prediction head."""

    for layer in model.layers:
        if layer.name == layer_name:
            target_output = _first_connected_output(layer)
            feature_extractor = tf.keras.Model(model.inputs, target_output)
            prediction_head = tf.keras.Model(target_output, model.output)
            return feature_extractor, prediction_head

        if isinstance(layer, tf.keras.Model):
            nested_models = build_feature_extractor_and_head(layer, layer_name)
            if nested_models is None:
                continue

            inner_feature_extractor, inner_prediction_head = nested_models
            connected_activation = inner_feature_extractor(_first_connected_input(layer))
            feature_extractor = tf.keras.Model(model.inputs, connected_activation)

            classifier_input = tf.keras.Input(
                shape=inner_feature_extractor.output_shape[1:],
                dtype=inner_feature_extractor.output.dtype,
            )
            classifier_output = inner_prediction_head(classifier_input)
            outer_tail_model = tf.keras.Model(_first_connected_output(layer), model.output)
            classifier_output = outer_tail_model(classifier_output)
            prediction_head = tf.keras.Model(classifier_input, classifier_output)
            return feature_extractor, prediction_head

    return None


def model_has_embedded_rescaling(model: tf.keras.Model) -> bool:
    for layer in _iter_layers_recursive(model):
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
    return False


def infer_image_size(model: tf.keras.Model, default_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]

    try:
        height = int(input_shape[1])
        width = int(input_shape[2])
    except (IndexError, TypeError, ValueError):
        return default_size

    if height <= 0 or width <= 0:
        return default_size

    return (height, width)


def load_class_names(model: tf.keras.Model, model_path: str) -> list[str]:
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list) and output_shape:
        output_shape = output_shape[0]

    try:
        output_units = int(output_shape[-1])
    except (IndexError, TypeError, ValueError):
        output_units = 0

    if infer_class_names is not None:
        try:
            return infer_class_names(model=model, output_units=output_units, model_path=model_path)
        except Exception:
            pass

    return [f"Class {index}" for index in range(output_units)]


def load_image(image_path: str, target_size: tuple[int, int], keep_pixel_range: bool) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(original_rgb, target_size, interpolation=cv2.INTER_AREA)

    input_array = resized_rgb.astype("float32")
    if not keep_pixel_range:
        input_array /= 255.0

    return original_rgb, np.expand_dims(input_array, axis=0)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute a normalized Grad-CAM heatmap using TensorFlow GradientTape."""

    if not last_conv_layer_name:
        last_conv_layer_name = find_last_connected_feature_map_layer_name(model)

    split_models = build_feature_extractor_and_head(model, last_conv_layer_name)
    if split_models is None:
        raise ValueError(f"Could not resolve layer '{last_conv_layer_name}' in the loaded model.")

    feature_extractor, prediction_head = split_models

    with tf.GradientTape() as tape:
        # This follows the same logic as the reference snippet:
        # extract feature maps first, then run only the remaining head layers.
        feature_maps = feature_extractor(img_array, training=False)
        tape.watch(feature_maps)
        predictions = prediction_head(feature_maps, training=False)

        if pred_index is None:
            pred_index_tensor = tf.argmax(predictions[0])
        else:
            num_classes = predictions.shape[-1]
            if num_classes is not None and not 0 <= pred_index < int(num_classes):
                raise ValueError(f"class_index must be between 0 and {int(num_classes) - 1}.")
            pred_index_tensor = tf.convert_to_tensor(pred_index, dtype=tf.int32)

        # Select the score for the class we want to explain.
        class_channel = predictions[:, pred_index_tensor]

    # Gradient of the selected class score with respect to the feature maps.
    grads = tape.gradient(class_channel, feature_maps)
    if grads is None:
        raise RuntimeError(
            "Gradients could not be computed for the selected layer. "
            "Try selecting a connected 4D layer from --list-conv-layers."
        )

    # Average the gradients across the spatial dimensions to get channel weights.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel in the feature maps by its importance for the class.
    feature_maps = feature_maps[0]
    heatmap = tf.linalg.matvec(feature_maps, pooled_grads)
    heatmap = tf.squeeze(heatmap)

    # Keep only positive influence and normalize to [0, 1].
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))

    return heatmap.numpy().astype("float32"), int(pred_index_tensor.numpy()), predictions[0].numpy()


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize, colorize, and blend a Grad-CAM heatmap onto the original image."""

    height, width = image_rgb.shape[:2]
    resized_heatmap = cv2.resize(heatmap.astype("float32"), (width, height), interpolation=cv2.INTER_CUBIC)
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(255 * resized_heatmap)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    image_uint8 = np.clip(image_rgb, 0, 255).astype("uint8")
    superimposed_rgb = cv2.addWeighted(image_uint8, 1.0, heatmap_color_rgb, alpha, 0)

    return resized_heatmap, heatmap_color_rgb, superimposed_rgb


def build_default_report_path(image_path: str) -> Path:
    image_file = Path(image_path).expanduser()
    image_stem = image_file.stem or "image"
    return (DEFAULT_REPORTS_DIR / f"{image_stem}{DEFAULT_REPORT_SUFFIX}").resolve()


def save_rgb_image(output_path: str | Path, image_rgb: np.ndarray) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_file), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not saved:
        raise OSError(f"Could not save image to: {output_file}")
    return output_file


def create_results_figure(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    superimposed_rgb: np.ndarray,
    title_text: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title_text, fontsize=14, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed_rgb)
    axes[2].set_title("Superimposed Grad-CAM")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def save_report_figure(
    output_path: str | Path,
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    superimposed_rgb: np.ndarray,
    title_text: str,
) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig = create_results_figure(
        original_rgb=original_rgb,
        heatmap=heatmap,
        superimposed_rgb=superimposed_rgb,
        title_text=title_text,
    )
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_file


def display_results(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    superimposed_rgb: np.ndarray,
    title_text: str,
) -> None:
    fig = create_results_figure(
        original_rgb=original_rgb,
        heatmap=heatmap,
        superimposed_rgb=superimposed_rgb,
        title_text=title_text,
    )
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.model:
        raise ValueError("Please pass --model <path-to-model> or configure a default project model first.")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(str(model_path), compile=False)

    if args.list_conv_layers:
        feature_map_layer_names = list_connected_feature_map_layer_names(model)
        conv_layer_names = list_conv_layer_names(model)
        if feature_map_layer_names:
            print("Grad-CAM-safe connected 4D feature-map layers:")
            for layer_name in feature_map_layer_names:
                print(layer_name)
            print(f"\nAuto-detected Grad-CAM layer: {feature_map_layer_names[-1]}\n")

        if conv_layer_names:
            print("Nested convolutional layers in the model:")
            for layer_name in conv_layer_names:
                print(layer_name)
            print(f"\nLast actual convolutional layer name: {conv_layer_names[-1]}")
        elif not feature_map_layer_names:
            print("No convolutional or 4D feature-map layers were found.")
        return

    if not args.image:
        raise ValueError("Please pass --image <path-to-image> to generate a Grad-CAM visualization.")

    target_size = infer_image_size(model, default_size=DEFAULT_IMAGE_SIZE)
    keep_pixel_range = model_has_embedded_rescaling(model)
    class_names = load_class_names(model, str(model_path))
    last_conv_layer_name = args.last_conv_layer.strip() or find_last_connected_feature_map_layer_name(model)

    # If you want to choose the layer manually, run:
    # python tools/gradcam_visualization.py --model <model_path> --list-conv-layers
    original_rgb, input_batch = load_image(
        image_path=args.image,
        target_size=target_size,
        keep_pixel_range=keep_pixel_range,
    )

    heatmap, class_index, predictions = make_gradcam_heatmap(
        img_array=input_batch,
        model=model,
        last_conv_layer_name=last_conv_layer_name,
        pred_index=args.class_index,
    )

    resized_heatmap, _, superimposed_rgb = overlay_heatmap(
        image_rgb=original_rgb,
        heatmap=heatmap,
        alpha=args.alpha,
    )

    if 0 <= class_index < len(class_names):
        class_label = class_names[class_index]
    else:
        class_label = f"Class {class_index}"

    confidence = float(predictions[class_index]) if class_index < len(predictions) else float(np.max(predictions))
    title_text = (
        f"Predicted class: {class_label} | Confidence: {confidence:.4f} | "
        f"Layer: {last_conv_layer_name}"
    )
    report_output_path = Path(args.report_output).expanduser().resolve() if args.report_output else build_default_report_path(args.image)
    overlay_output_path = Path(args.output).expanduser().resolve() if args.output else None

    if overlay_output_path is not None and overlay_output_path == report_output_path:
        raise ValueError("--output and --report-output must point to different files.")

    print(f"Model path: {model_path}")
    print(f"Image path: {Path(args.image).resolve()}")
    print(f"Input size used: {target_size}")
    print(f"Grad-CAM layer: {last_conv_layer_name}")
    print(f"Predicted class: {class_label} (index={class_index}, confidence={confidence:.4f})")

    saved_report_path = save_report_figure(
        output_path=report_output_path,
        original_rgb=original_rgb,
        heatmap=resized_heatmap,
        superimposed_rgb=superimposed_rgb,
        title_text=title_text,
    )
    print(f"Saved Grad-CAM report to: {saved_report_path}")

    if overlay_output_path is not None:
        saved_overlay_path = save_rgb_image(overlay_output_path, superimposed_rgb)
        print(f"Saved Grad-CAM overlay to: {saved_overlay_path}")

    if not args.no_display:
        display_results(
            original_rgb=original_rgb,
            heatmap=resized_heatmap,
            superimposed_rgb=superimposed_rgb,
            title_text=title_text,
        )


if __name__ == "__main__":
    # Example:
    # .\face_env\Scripts\python.exe tools\gradcam_visualization.py --image path\to\image.jpg
    # .\face_env\Scripts\python.exe tools\gradcam_visualization.py --image path\to\image.jpg --no-display
    # .\face_env\Scripts\python.exe tools\gradcam_visualization.py --image path\to\image.jpg --output path\to\overlay.png --report-output path\to\report.png
    # .\face_env\Scripts\python.exe tools\gradcam_visualization.py --list-conv-layers
    # .\face_env\Scripts\python.exe tools\gradcam_visualization.py --image path\to\image.jpg --last-conv-layer efficientnetv2-b1
    main()
