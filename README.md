# Image-Based Breed Recognition for Indian Cattle and Buffaloes

This project explores image-based identification of cattle and buffalo breeds found in India. It brings together breed classification experiments, YOLO-based animal segmentation, evaluation artifacts, and a Flask web application for running inference on uploaded media.

The repository is a research prototype. Training notebooks and scripts document several experiment versions, while the web application uses locally supplied trained checkpoints.

## What It Does

- **Breed classification:** Runs EfficientNet-B2, ViT-Small, and ConvNeXt-Tiny on an uploaded image and displays each model's top predictions.
- **Segmentation:** Applies a general cattle detector as a gate, then runs the custom YOLO segmentation model to produce an annotated image or video.
- **Segment then classify:** For an image, detects and segments animal regions, crops them, classifies each crop with EfficientNet-B2, and labels predictions as known or unknown using confidence and entropy thresholds.
- **Model evaluation:** Includes classification reports, per-class metrics, training histories, confusion matrices, and YOLO segmentation evaluation outputs.

The segmentation-and-classification path uses EfficientNet-B2 for breed classification; the three classification models are not combined into an ensemble in the current app.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `web_app/` | Flask application, inference services, templates, styles, and app dependencies |
| `prepare_data.py` | Older CSV dataset preparation workflow |
| `dataset_model.py`, `train.py`, `evaluate.py` | Older EfficientNet-B0 setup, training, and evaluation workflow |
| `EfficientNet_*.ipynb`, `VIT_Setup.ipynb`, `ConvNeXT Model/` | Later classification experiments and their artifacts |
| `Yolo_seg/`, `yolo8_training.ipynb`, `yolo_dataset_preparation.ipynb` | Segmentation dataset preparation, training, prediction, and evaluation artifacts |
| `classification_report*.txt`, `per_class_metrics*.csv`, `*_curves*.png` | Classification evaluation outputs |
| `runs/`, `Yolo_seg/evaluation_*/` | YOLO training and segmentation evaluation outputs |

The notebooks cover multiple experiment iterations and may depend on notebook execution order, external datasets, and local paths. The root Python scripts are an older EfficientNet-B0 workflow; they are not a single-command reproduction of the later EfficientNet-B2, ViT-Small, or ConvNeXt-Tiny experiments.

## Classification Results

The checked-in V2 classification reports record the following test-set top-k accuracy:

| Model | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| EfficientNet-B2 | 80.96% | 93.11% | 95.98% |
| ViT-Small | 78.85% | 92.20% | 95.69% |

These figures are transcribed from `classification_report_v2.txt` and `classification_report_vit_small.txt`. They describe the saved experiment reports, not a guarantee of performance on new images. Per-class performance varies; consult the reports and CSV metrics for more detail.

## Run the Web Application

### Prerequisites

- Python 3.11 is recommended; the project notebooks were run with Python 3.11.
- A compatible PyTorch and torchvision installation. CPU inference is supported; a compatible CUDA-enabled PyTorch build can be used when an NVIDIA GPU is available.
- FFmpeg available on `PATH` to encode videos after segmentation. Image workflows do not require video encoding.
- The trained model files listed below. They are ignored by the repository's `.gitignore` and are not normally included when the source is pushed to Git.

### Install dependencies

From the repository root, create and activate a virtual environment, then install the web app requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r web_app\requirements.txt
```

For GPU inference, install a PyTorch/torchvision build compatible with your CUDA version, following the official PyTorch installation instructions. Confirm FFmpeg is available with:

```powershell
ffmpeg -version
```

### Provide model files

Place the trained checkpoints and YOLO weights at the expected locations:

```text
best_model_v2_effnetb2.pth
best_model_v2_vit_small.pth
ConvNeXT Model/best_convnext_tiny.pth
yolov8m.pt
Yolo_seg/runs/segment/breed_segmentation/yolov8n_seg_optimized-2/weights/best.pt
```

The project does not provide automatic checkpoint downloads. Obtain the weights separately and ensure the filenames and locations match the paths used by the inference services.

### Update local paths

Several inference and training files contain absolute paths from the original development machine. Before running the app from a different location, update the project-root and model-path constants in:

- `web_app/inference/classification.py`
- `web_app/inference/segmentation.py`
- `web_app/inference/segmentation_classification.py`

The dataset scripts and some notebooks also contain machine-specific dataset paths. Update those paths before running those workflows.

### Start the server

```powershell
Set-Location web_app
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The application loads its inference models at startup, which may take a while. It selects CUDA when available to PyTorch and otherwise uses the CPU.

The app accepts JPEG, PNG, BMP, and WebP images. Video segmentation supports MP4, AVI, MOV, and MKV uploads. The maximum upload size is 500 MB. Breed classification and segment-then-classify accept images; the segmentation workflow accepts images or videos.

The app uses Flask's development server and is configured for local access. It is not configured as a production deployment.

## Training and Data

The repository includes multiple generations of dataset and model experiments. The later V2 classification reports and checkpoints correspond to the 62-class classification experiment. Dataset CSVs contain image paths; the underlying image datasets are external and are not bundled as a complete dataset in this repository.

`prepare_data.py` is an older script that scans a cattle/buffalo folder hierarchy, filters classes with fewer than 20 images, and writes a stratified 70/15/15 split. Other notebooks use different filters and dataset versions. Review and update their data paths, input files, class lists, and output paths before running them.

The root `train.py` and `evaluate.py` scripts operate on the older EfficientNet-B0 setup. The later EfficientNet-B2 and ViT-Small experiments are primarily documented in notebooks; ConvNeXt-Tiny experiments are under `ConvNeXT Model/`; YOLO segmentation preparation and evaluation are under `Yolo_seg/` and the YOLO notebooks.

## Notes and Limitations

- Model weights, YOLO run outputs, and dataset directories are ignored by `.gitignore`; a clone may not contain the assets needed to launch inference.
- Absolute paths must be updated for another machine, and CSV image paths must resolve to the corresponding local image files for training or evaluation.
- Inference speed depends on the available hardware; CPU inference can be substantially slower than GPU inference.
- Classification confidence and entropy thresholds are heuristic rejection rules, not calibrated probabilities.
- The breed classes and segmentation behavior are limited to the datasets used for these experiments. Performance on other breeds, image conditions, or field settings has not been established by the saved reports.

## License and Data Attribution

No project license or complete dataset attribution document is included in this repository. Confirm the licensing and attribution requirements for the code, model weights, and source datasets before redistributing them.
