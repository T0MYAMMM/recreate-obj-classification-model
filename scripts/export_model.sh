#!/usr/bin/env bash
# Exports a trained TF OD API checkpoint to a SavedModel for inference.
#
# Usage:
#   ./scripts/export_model.sh \
#     --pipeline-config configs/pipeline.config \
#     --trained-checkpoint-dir output/model \
#     --output-dir output/exported_model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TF_MODELS_DIR="$REPO_ROOT/models"
EXPORT_SCRIPT="$TF_MODELS_DIR/research/object_detection/exporter_main_v2.py"

PIPELINE_CONFIG=""
CHECKPOINT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline-config) PIPELINE_CONFIG="$2"; shift 2 ;;
        --trained-checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$PIPELINE_CONFIG" || -z "$CHECKPOINT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --pipeline-config <path> --trained-checkpoint-dir <path> --output-dir <path>" >&2
    exit 1
fi

if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "ERROR: exporter_main_v2.py not found at $EXPORT_SCRIPT" >&2
    echo "Run scripts/setup_environment.sh first." >&2
    exit 1
fi

echo "==> Exporting model..."
python "$EXPORT_SCRIPT" \
    --input_type image_tensor \
    --pipeline_config_path "$PIPELINE_CONFIG" \
    --trained_checkpoint_dir "$CHECKPOINT_DIR" \
    --output_directory "$OUTPUT_DIR"

echo "==> Model exported to: $OUTPUT_DIR"
