"""Entry point for the ReCreate object classification pipeline."""

import argparse
import logging
from pathlib import Path

from src.utils.helpers import configure_logging, require_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReCreate ML — object classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- download --
    dl = subparsers.add_parser("download", help="Download dataset from Roboflow")
    dl.add_argument("--workspace", default="recreate", help="Roboflow workspace")
    dl.add_argument("--project", required=True, help="Roboflow project name")
    dl.add_argument("--version", type=int, required=True, help="Dataset version")
    dl.add_argument("--format", default="tfrecord", help="Export format")
    dl.add_argument("--output-dir", default="data/raw", help="Save directory")

    # -- train --
    tr = subparsers.add_parser("train", help="Run the training pipeline")
    tr.add_argument("--pipeline-config", required=True, help="Path to pipeline.config")
    tr.add_argument("--model-dir", required=True, help="Output checkpoint directory")
    tr.add_argument("--tf-models-dir", default="models", help="tensorflow/models dir")
    tr.add_argument("--steps", type=int, default=50000, help="Training steps")

    # -- predict --
    pr = subparsers.add_parser("predict", help="Run inference on an image")
    pr.add_argument("--export-dir", required=True, help="Exported SavedModel directory")
    pr.add_argument("--label-map", required=True, help="Path to label_map.pbtxt")
    pr.add_argument("--image", required=True, help="Path to input image")
    pr.add_argument("--threshold", type=float, default=0.5, help="Score threshold")

    return parser.parse_args()


def cmd_download(args: argparse.Namespace) -> None:
    from src.data.download import download_dataset

    api_key = require_env("ROBOFLOW_API_KEY")
    dataset_path = download_dataset(
        api_key=api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        format=args.format,
        output_dir=args.output_dir,
    )
    logging.info("Dataset downloaded to: %s", dataset_path)


def cmd_train(args: argparse.Namespace) -> None:
    from src.training.train import run_training

    run_training(
        pipeline_config_path=Path(args.pipeline_config),
        model_dir=Path(args.model_dir),
        tf_models_dir=Path(args.tf_models_dir),
        num_train_steps=args.steps,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    import json
    from src.inference.predict import load_saved_model, load_label_map, run_inference_on_image

    detect_fn = load_saved_model(Path(args.export_dir))
    label_map = load_label_map(Path(args.label_map))
    result = run_inference_on_image(
        detect_fn=detect_fn,
        image_path=Path(args.image),
        min_score_threshold=args.threshold,
    )

    for box, cls_id, score in zip(result["boxes"], result["classes"], result["scores"]):
        label = label_map.get(cls_id, str(cls_id))
        print(f"{label}: {score:.2f}  box={box}")


def main() -> None:
    configure_logging()
    args = parse_args()

    dispatch = {
        "download": cmd_download,
        "train": cmd_train,
        "predict": cmd_predict,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
