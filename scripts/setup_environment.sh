#!/usr/bin/env bash
# Installs project dependencies and compiles the TF OD API protobuf definitions.
# Run once after cloning, from the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "==> Initialising git submodules..."
git -C "$REPO_ROOT" submodule update --init --recursive

TF_MODELS_DIR="$REPO_ROOT/models"
OD_DIR="$TF_MODELS_DIR/research/object_detection"

if [ ! -d "$OD_DIR" ]; then
    echo "ERROR: TF OD API directory not found at $OD_DIR" >&2
    exit 1
fi

echo "==> Compiling protobuf definitions..."
cd "$TF_MODELS_DIR/research"
protoc object_detection/protos/*.proto --python_out=.

echo "==> Installing TF OD API and SLIM..."
cp object_detection/packages/tf2/setup.py .
pip install .

echo "==> Environment setup complete."
