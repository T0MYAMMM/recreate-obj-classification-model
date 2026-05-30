"""Training pipeline for TF Object Detection API."""

import subprocess
import sys
from pathlib import Path


def run_training(
    pipeline_config_path: Path,
    model_dir: Path,
    tf_models_dir: Path,
    num_train_steps: int = 50000,
    checkpoint_every_n: int = 1000,
) -> None:
    """Launch the TF Object Detection API training loop as a subprocess.

    Args:
        pipeline_config_path: Path to the populated pipeline.config file.
        model_dir: Output directory for checkpoints and summaries.
        tf_models_dir: Root directory of the tensorflow/models repository.
        num_train_steps: Number of training steps to run.
        checkpoint_every_n: Save a checkpoint every N steps.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    script = tf_models_dir / "research" / "object_detection" / "model_main_tf2.py"

    if not script.exists():
        raise FileNotFoundError(
            f"TF OD API training script not found at {script}. "
            "Ensure the tensorflow/models submodule is initialised and up to date."
        )

    cmd = [
        sys.executable,
        str(script),
        f"--pipeline_config_path={pipeline_config_path}",
        f"--model_dir={model_dir}",
        f"--num_train_steps={num_train_steps}",
        f"--checkpoint_every_n={checkpoint_every_n}",
        "--alsologtostderr",
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training exited with non-zero status: {result.returncode}")


def run_evaluation(
    pipeline_config_path: Path,
    model_dir: Path,
    tf_models_dir: Path,
    checkpoint_dir: Path,
) -> None:
    """Launch the TF Object Detection API evaluation loop as a subprocess.

    Args:
        pipeline_config_path: Path to the populated pipeline.config file.
        model_dir: Directory where evaluation results will be written.
        tf_models_dir: Root directory of the tensorflow/models repository.
        checkpoint_dir: Directory containing the model checkpoints to evaluate.
    """
    script = tf_models_dir / "research" / "object_detection" / "model_main_tf2.py"

    cmd = [
        sys.executable,
        str(script),
        f"--pipeline_config_path={pipeline_config_path}",
        f"--model_dir={model_dir}",
        f"--checkpoint_dir={checkpoint_dir}",
        "--alsologtostderr",
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation exited with non-zero status: {result.returncode}")
