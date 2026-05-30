"""Model configuration utilities for TF Object Detection API."""

import re
from pathlib import Path


def patch_pipeline_config(
    template_path: Path,
    output_path: Path,
    substitutions: dict[str, str],
) -> None:
    """Apply substitutions to a pipeline.config template and write the result.

    Placeholders in the template use the format {{KEY}}.

    Args:
        template_path: Path to the pipeline.config template file.
        output_path: Path where the patched config will be written.
        substitutions: Mapping of placeholder keys to replacement values.
    """
    content = template_path.read_text()
    for key, value in substitutions.items():
        content = content.replace(f"{{{{{key}}}}}", value)

    undefined = re.findall(r"\{\{[A-Z_]+\}\}", content)
    if undefined:
        raise ValueError(f"Unresolved placeholders in pipeline config: {undefined}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)


def build_pipeline_substitutions(
    num_classes: int,
    label_map_path: Path,
    train_record_path: Path,
    val_record_path: Path,
    checkpoint_path: Path,
    batch_size: int = 8,
    num_steps: int = 50000,
) -> dict[str, str]:
    """Build the substitution dictionary for a standard pipeline config.

    Args:
        num_classes: Number of object classes in the dataset.
        label_map_path: Path to the .pbtxt label map file.
        train_record_path: Path to the training TFRecord file.
        val_record_path: Path to the validation TFRecord file.
        checkpoint_path: Path to the pretrained model checkpoint.
        batch_size: Training batch size.
        num_steps: Total training steps.

    Returns:
        Dictionary mapping template placeholder keys to string values.
    """
    return {
        "NUM_CLASSES": str(num_classes),
        "LABEL_MAP_PATH": str(label_map_path),
        "TRAIN_RECORD_PATH": str(train_record_path),
        "VAL_RECORD_PATH": str(val_record_path),
        "CHECKPOINT_PATH": str(checkpoint_path),
        "BATCH_SIZE": str(batch_size),
        "NUM_STEPS": str(num_steps),
    }
