"""Inference utilities for a trained TF Object Detection API model."""

from pathlib import Path
from typing import Optional

import numpy as np


def load_saved_model(export_dir: Path):
    """Load an exported TF SavedModel for inference.

    Args:
        export_dir: Path to the SavedModel export directory (contains saved_model.pb).

    Returns:
        A callable TensorFlow detection function.
    """
    import tensorflow as tf

    model = tf.saved_model.load(str(export_dir))
    return model.signatures["serving_default"]


def run_inference_on_image(
    detect_fn,
    image_path: Path,
    min_score_threshold: float = 0.5,
) -> dict:
    """Run object detection on a single image.

    Args:
        detect_fn: TensorFlow detection function returned by load_saved_model.
        image_path: Path to the input image file.
        min_score_threshold: Detections with scores below this value are filtered out.

    Returns:
        Dictionary with keys: boxes, classes, scores, num_detections,
        each filtered to results above the threshold.
    """
    import tensorflow as tf

    raw = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    input_tensor = tf.expand_dims(tf.cast(image, tf.uint8), 0)

    detections = detect_fn(input_tensor=input_tensor)

    num = int(detections["num_detections"].numpy()[0])
    scores = detections["detection_scores"].numpy()[0][:num]
    boxes = detections["detection_boxes"].numpy()[0][:num]
    classes = detections["detection_classes"].numpy()[0][:num].astype(int)

    mask = scores >= min_score_threshold
    return {
        "boxes": boxes[mask].tolist(),
        "classes": classes[mask].tolist(),
        "scores": scores[mask].tolist(),
        "num_detections": int(mask.sum()),
    }


def load_label_map(label_map_path: Path) -> dict[int, str]:
    """Parse a TF OD API .pbtxt label map into an id-to-name mapping.

    Args:
        label_map_path: Path to the .pbtxt label map file.

    Returns:
        Dictionary mapping integer class ids to class name strings.
    """
    id_to_name: dict[int, str] = {}
    current_id: Optional[int] = None

    for line in label_map_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("id:"):
            current_id = int(line.split(":", 1)[1].strip())
        elif line.startswith("name:") and current_id is not None:
            name = line.split(":", 1)[1].strip().strip("'\"")
            id_to_name[current_id] = name
            current_id = None

    return id_to_name
