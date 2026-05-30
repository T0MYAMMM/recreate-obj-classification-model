# ReCreate Object Classification Model

A Python pipeline for training and deploying a custom object detection model using the
TensorFlow Object Detection API. The project was developed as a Bangkit 2023
Product-based Capstone and uses datasets managed through Roboflow under the
"recreate" workspace.

## Architecture / Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | TensorFlow 2.13 |
| Detection API | TF Object Detection API (tensorflow/models) |
| Dataset management | Roboflow |
| Model architecture | SSD MobileNet V2 FPNLite 640x640 (configurable) |

## Project Structure

```
recreate-obj-classification-model/
├── src/
│   ├── data/
│   │   ├── download.py          # Roboflow dataset download
│   │   └── preprocessing.py     # Label map generation, VOC parsing
│   ├── models/
│   │   └── config.py            # Pipeline config template patching
│   ├── training/
│   │   └── train.py             # Training and evaluation runners
│   ├── inference/
│   │   └── predict.py           # SavedModel loading and inference
│   └── utils/
│       └── helpers.py           # Logging, env vars, filesystem helpers
├── configs/
│   └── pipeline.config.template # Parameterised SSD MobileNet V2 config
├── scripts/
│   ├── setup_environment.sh     # One-time environment setup
│   └── export_model.sh          # Checkpoint-to-SavedModel export
├── notebooks/                   # Exploratory notebooks (not tracked)
├── models/                      # Git submodule: tensorflow/models
├── main.py                      # Unified CLI entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Local Setup and Installation

### Prerequisites

- Python 3.10 or later
- `protoc` (Protocol Buffers compiler) — install via your system package manager
- Git with submodule support

### Steps

1. Clone the repository and initialise the submodule:

   ```bash
   git clone https://github.com/T0MYAMMM/recreate-obj-classification-model.git
   cd recreate-obj-classification-model
   ```

2. Copy the environment file and add your credentials:

   ```bash
   cp .env.example .env
   # Edit .env and set ROBOFLOW_API_KEY and related values
   ```

3. Run the environment setup script (installs dependencies and compiles protobuf):

   ```bash
   bash scripts/setup_environment.sh
   ```

## Usage

All pipeline stages are accessible through the `main.py` CLI.

### Download dataset from Roboflow

```bash
python main.py download \
    --workspace recreate \
    --project <your-project-name> \
    --version 1 \
    --format tfrecord \
    --output-dir data/raw
```

### Configure and start training

Patch the pipeline config template with your dataset paths:

```python
from pathlib import Path
from src.models.config import patch_pipeline_config, build_pipeline_substitutions

subs = build_pipeline_substitutions(
    num_classes=10,
    label_map_path=Path("data/raw/label_map.pbtxt"),
    train_record_path=Path("data/raw/train/train.tfrecord"),
    val_record_path=Path("data/raw/valid/valid.tfrecord"),
    checkpoint_path=Path("pretrained/ssd_mobilenet_v2_fpnlite_640x640"),
    batch_size=8,
    num_steps=50000,
)
patch_pipeline_config(
    template_path=Path("configs/pipeline.config.template"),
    output_path=Path("configs/pipeline.config"),
    substitutions=subs,
)
```

Then launch training:

```bash
python main.py train \
    --pipeline-config configs/pipeline.config \
    --model-dir output/model \
    --steps 50000
```

### Export trained model

```bash
bash scripts/export_model.sh \
    --pipeline-config configs/pipeline.config \
    --trained-checkpoint-dir output/model \
    --output-dir output/exported_model
```

### Run inference on an image

```bash
python main.py predict \
    --export-dir output/exported_model \
    --label-map data/raw/label_map.pbtxt \
    --image path/to/image.jpg \
    --threshold 0.5
```

## Future Improvements

- Add a `Makefile` to chain download, config-patch, training, export, and predict steps
  into a single reproducible workflow.
- Support additional model architectures from the TF2 Model Zoo via a `--model-type`
  flag in the CLI.
- Integrate TensorBoard log streaming and a metrics summary table printed at the end
  of training.
- Add automated evaluation reporting (mAP per class) after each checkpoint evaluation.
- Package the inference module as a REST API using FastAPI for serving predictions.
- Add a Docker image to eliminate the protobuf compilation and submodule setup steps.
