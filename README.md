# 🫁 End-to-End AI System for Pneumonia Detection

> **PneumoniaMNIST** (MedMNIST v2) — Binary Classification: Normal vs Pneumonia

A production-quality deep learning system for automated pneumonia detection from chest X-rays, implementing and comparing multiple CNN architectures.

---

## 📋 Project Overview

This project implements a complete machine learning pipeline for pneumonia detection, including:

- **3 model architectures**: UNet (modified for classification), ResNet18, EfficientNet-B0
- **6 experiments**: Each architecture trained both from scratch and with pretrained (ImageNet) weights
- **Full evaluation suite**: Accuracy, Precision, Recall, F1, ROC-AUC with visualizations
- **Automated experiment runner**: Train all models, evaluate, and generate comparison tables

## 🗂 Repository Structure

```
.
├── data/                              # Data loading and preprocessing
│   ├── dataset.py                     # PneumoniaMNIST dataset wrapper
│   ├── transforms.py                  # Augmentations & normalization
│   └── dataloaders.py                 # DataLoader factory
│
├── models/                            # Model architectures
│   ├── unet_classifier.py             # UNet (scratch + pretrained encoder)
│   ├── resnet_classifier.py           # ResNet18 (scratch + pretrained)
│   ├── efficientnet_classifier.py     # EfficientNet-B0 (scratch + pretrained)
│   └── utils.py                       # Model factory & param counting
│
├── task1_classification/              # CNN Classification pipeline
│   ├── config.yaml                    # Training configuration
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Metrics & visualizations
│   └── experiment_runner.py           # Run all 6 experiments
│
├── task2_report_generation/           # VLM Report Generation
│   ├── image_preprocessor.py          # Base64 conversion & upscaling
│   ├── prompts.py                     # 3 prompt strategies
│   ├── sample_selection.py            # Test set sample selection
│   ├── vlm_pipeline.py                # VLM orchestration
│   ├── report_generator.py            # Report generation engine
│   ├── evaluation.py                  # Qualitative evaluation
│   └── run_task2.py                   # CLI entry point
│
├── task3_retrieval/                    # (Placeholder) Image retrieval
│
├── llm_service.py                     # LLM Factory (Ollama, Docker, etc.)
│
├── notebooks/
│   └── colab_demo.ipynb               # Google Colab demo notebook
│
├── reports/
│   ├── task1_classification_report.md # Classification analysis report
│   ├── task2_report_generation.md     # VLM report generation analysis
│   └── generated_reports/             # Individual VLM outputs (JSON)
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd medical-ai-test

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🏋️ Task 1: CNN Classification

### Train a Single Model

```bash
# Train ResNet18 from scratch
python -m task1_classification.train --model resnet --pretrained false

# Train ResNet18 with pretrained weights
python -m task1_classification.train --model resnet --pretrained true

# Train UNet from scratch
python -m task1_classification.train --model unet --pretrained false

# Train EfficientNet-B0 pretrained
python -m task1_classification.train --model efficientnet --pretrained true

# Custom hyperparameters
python -m task1_classification.train --model resnet --pretrained true --epochs 50 --lr 0.0005 --batch_size 128
```

### Run All 6 Experiments

```bash
python -m task1_classification.experiment_runner
```

This will:
1. Train all 6 model variants (3 architectures × 2 variants)
2. Evaluate each on the test set
3. Generate plots (confusion matrix, ROC curve, training curves, failure cases)
4. Save results to `reports/experiment_comparison.csv`

### Evaluate a Single Model

```bash
python -m task1_classification.evaluate --model resnet --pretrained true
```

### Generated Outputs

| Directory       | File                              | Description                    |
| --------------- | --------------------------------- | ------------------------------ |
| `reports/`      | `experiment_comparison.csv`       | Full comparison of all models  |
| `reports/`      | `*_confusion_matrix.png`          | Confusion matrix heatmaps     |
| `reports/`      | `*_roc_curve.png`                 | ROC curves                    |
| `reports/`      | `*_training_curves.png`           | Loss/accuracy/AUC over epochs |
| `reports/`      | `*_failure_cases.png`             | Misclassified examples        |
| `results/`      | `*_results.json`                  | Per-model training history    |
| `results/`      | `*_test_metrics.json`             | Per-model test metrics        |
| `checkpoints/`  | `*_best.pth`                      | Best model weights (by AUC)   |

## 🧪 Experiments

| Model        | From Scratch | Pretrained |
| ------------ | :----------: | :--------: |
| UNet         | ✅           | ✅         |
| ResNet18     | ✅           | ✅         |
| EfficientNet | ✅           | ✅         |

All experiments use identical training configurations for fair comparison:
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Early Stopping**: patience=7 (based on validation AUC)
- **Max Epochs**: 30
- **Batch Size**: 64

---

## 🩺 Task 2: Medical Report Generation with VLMs

### Overview

Uses pretrained VLMs (MedGemma, Qwen-VL) to generate structured radiology reports from chest X-ray images. Implements 3 prompting strategies and qualitative evaluation.

### VLM Setup

#### Option A: Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen2-VL multimodal model
ollama pull qwen2-vl

# Pull MedGemma (if available)
ollama pull medgemma

# Default endpoint: http://localhost:11434
```

#### Option B: Docker Model Runner

```bash
# Pull model image
docker model pull medgemma

# Docker Model Runner runs on localhost:8080 by default
# API endpoint: http://localhost:8080/v1/chat/completions
```

### Running Task 2

```bash
# Generate reports with Ollama + Qwen2-VL
python -m task2_report_generation.run_task2 \
    --provider ollama \
    --model qwen2-vl \
    --num_samples 10

# Generate reports with Docker + MedGemma
python -m task2_report_generation.run_task2 \
    --provider docker \
    --model medgemma \
    --num_samples 10

# Custom settings
python -m task2_report_generation.run_task2 \
    --provider ollama \
    --model qwen2-vl \
    --temperature 0.3 \
    --cnn_model resnet \
    --cnn_pretrained true
```

### Task 2 Outputs

| Directory                   | File                           | Description                     |
| --------------------------- | ------------------------------ | ------------------------------- |
| `reports/`                  | `task2_report_generation.md`   | Full analysis report            |
| `reports/generated_reports/`| `*.json`                       | Individual VLM reports          |
| `results/`                  | `selected_samples.json`        | Selected sample metadata        |
| `results/`                  | `evaluation_results.json`      | VLM vs GT vs CNN comparison     |
| `results/`                  | `strategy_analysis.json`       | Per-prompt strategy analysis    |

---

## 🔍 Task 3: Semantic Image Retrieval with PGVector

### Overview

Content-based image retrieval (CBIR) system using CNN embeddings stored in PostgreSQL with PGVector. Provides a FastAPI REST API for image similarity search.

### PGVector Setup (Docker)

```bash
# Start PGVector container
docker run -p 5432:5432 \
  --env POSTGRES_PASSWORD=postgres \
  --env POSTGRES_USER=postgres \
  --env POSTGRES_DB=postgres \
  -v ~/medical_db:/var/lib/postgresql/data \
  --name medicaldb \
  -d pgvector/pgvector:pg16
```

### Running Task 3

```bash
# Run the full pipeline: build index → evaluate → visualize → report
python -m task3_retrieval.run_task3

# Skip rebuilding index (if already built)
python -m task3_retrieval.run_task3 --skip_build

# Start the FastAPI server
uvicorn task3_retrieval.app.main:app --reload --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/build-index` | POST | Extract embeddings and build index |
| `/search/image` | POST | Image-to-image similarity search |
| `/search/text` | POST | Text-to-image search |
| `/health` | GET | Health check |

### Task 3 Outputs

| Directory | File | Description |
| --------- | ---- | ----------- |
| `reports/` | `task3_retrieval_system.md` | Full analysis report |
| `reports/retrieval_visualizations/` | `retrieval_grid.png` | Query-vs-results grids |
| `results/` | `retrieval_evaluation.json` | Precision@k metrics |

---

## 🔬 Dataset

- **Source**: [MedMNIST v2](https://medmnist.com/) — PneumoniaMNIST
- **Task**: Binary classification (Normal vs Pneumonia)
- **Image size**: 28 × 28 grayscale
- **Splits**: Train / Validation / Test (predefined by MedMNIST)

The dataset is automatically downloaded on first run.

## 🔄 Reproducibility

All experiments are fully reproducible:

```bash
# Set seed (default: 42)
python -m task1_classification.train --model resnet --pretrained true --seed 42
```

Reproducibility measures:
- Fixed random seeds (Python, NumPy, PyTorch, CUDA)
- Deterministic cuDNN operations (`torch.backends.cudnn.deterministic = True`)
- Consistent data splits from MedMNIST
- All hyperparameters saved in `config.yaml` and result JSONs

## 💻 Hardware

- The code automatically detects and uses GPU (CUDA) if available
- All experiments can run on CPU (slower but fully functional)
- Estimated training time per model: ~2-5 min (GPU) / ~15-30 min (CPU)

## 📓 Colab Notebook

A ready-to-run Google Colab notebook is provided at `notebooks/colab_demo.ipynb`:

1. Installs dependencies
2. Downloads the dataset
3. Trains ResNet18 (pretrained) as a demo
4. Evaluates and displays results
5. Shows confusion matrix, ROC curve, and sample predictions

## 📝 Reports

- **Task 1**: [`reports/task1_classification_report.md`](reports/task1_classification_report.md) — Architecture comparison, metrics, failure analysis
- **Task 2**: [`reports/task2_report_generation.md`](reports/task2_report_generation.md) — VLM evaluation, prompting strategies, qualitative analysis
- **Task 3**: [`reports/task3_retrieval_system.md`](reports/task3_retrieval_system.md) — Retrieval evaluation, Precision@k, architecture analysis

---

**License**: MIT

**Author**: Medical AI Research Project
