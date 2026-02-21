# AI System for Pneumonia Detection


## Project Overview

This project implements a complete three-task medical AI pipeline:

- **Task 1**: CNN classification — 3 architectures × 2 variants (scratch / pretrained)
- **Task 2**: VLM report generation — MedGemma, Qwen-VL via Ollama / Docker / HuggingFace
- **Task 3**: Semantic image retrieval — FastAPI + PGVector + CNN embeddings

## 🗂epository Structure

```
.
├── data/                              # Data loading and preprocessing
│   ├── dataset.py
│   ├── transforms.py
│   └── dataloaders.py
│
├── models/                            # Model architectures
│   ├── unet_classifier.py             # UNet (scratch + pretrained encoder)
│   ├── resnet_classifier.py           # ResNet18 (scratch + pretrained)
│   ├── efficientnet_classifier.py     # EfficientNet-B0 (scratch + pretrained)
│   └── utils.py
│
├── task1_classification/              # CNN Classification pipeline
│   ├── config.yaml
│   ├── train.py
│   ├── evaluate.py
│   └── experiment_runner.py
│
├── task2_report_generation/           # VLM Report Generation
│   ├── hf_medgemma_pipeline.py        # HuggingFace MedGemma (local)
│   ├── vlm_pipeline.py                # Ollama / Docker pipeline
│   ├── image_preprocessor.py
│   ├── prompts.py                     # 3 prompt strategies
│   ├── sample_selection.py
│   ├── report_generator.py
│   ├── evaluation.py
│   ├── llm_service.py                 # LLM Factory (Ollama, Docker)
│   ├── run_task2.py                   # Ollama / Docker CLI
│   └── run_task2_hf.py                # HuggingFace CLI
│
├── task3_retrieval/                   # Semantic Image Retrieval
│   ├── app/
│   │   ├── main.py                    # FastAPI app
│   │   ├── database.py                # SQLAlchemy + PGVector
│   │   ├── models.py                  # ORM (MedicalImage + Vector)
│   │   ├── schemas.py                 # Pydantic schemas
│   │   ├── embedding_service.py       # ResNet18 feature extractor
│   │   ├── crud.py                    # DB operations
│   │   ├── retrieval_service.py       # High-level retrieval
│   │   └── config.py
│   ├── scripts/
│   │   ├── build_index.py
│   │   ├── evaluate.py                # Precision@k
│   │   └── visualize_results.py
│   └── run_task3.py
│
├── notebooks/
│   └── colab_demo.ipynb
│
├── reports/
│   ├── task1_classification_report.md
│   ├── task2_report_generation.md
│   ├── task3_retrieval_system.md
│   ├── generated_reports/             # Individual VLM JSON outputs
│   └── retrieval_visualizations/
│
├── requirements.txt
└── README.md
```

---

## Hardware

All experiments were run on a **personal laptop** with the following configuration:

| Component | Specification                                          |
|-----------|--------------------------------------------------------|
| **OS**    | Manjaro Linux                                          |
| **CPU**   | 13th Gen Intel Core i7-13620H (16 threads) @ 4.700 GHz |
| **GPU**   | NVIDIA GeForce RTX 4050 Max-Q / Mobile (6 GB VRAM)     |
| **RAM**   | 16 GB total                                            |

---

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- CUDA 12.x (for GPU acceleration — optional)

### Option A — uv (recommended, fast)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/HamzaGbada/medical-ai-test.git
cd medical-ai-test

# Create venv and install all dependencies in one step
uv init
uv pip install -r requirements.txt

# GPU users — install PyTorch with CUDA 12.1 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Option B — pip (standard)

```bash
# Clone and enter directory
git clone https://github.com/HamzaGbada/medical-ai-test.git
cd medical-ai-test

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
    
# Install dependencies
pip install -r requirements.txt

# GPU users — install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

    
---

## Task 1: CNN Classification

### Train a Single Model

```bash
# Train ResNet18 with pretrained weights
python -m task1_classification.train --model resnet --pretrained true

# Train UNet from scratch (medical baseline)
python -m task1_classification.train --model unet --pretrained false

# Train EfficientNet-B0 pretrained
python -m task1_classification.train --model efficientnet --pretrained true
```

### Run All 6 Experiments

```bash
python -m task1_classification.experiment_runner
```

### Evaluate

```bash
python -m task1_classification.evaluate --model resnet --pretrained true
```

### Task 1 Outputs

| Directory      | File                        | Description                 |
|----------------|-----------------------------|-----------------------------|
| `reports/`     | `experiment_comparison.csv` | Full 6-model comparison     |
| `reports/`     | `*_confusion_matrix.png`    | Confusion matrices          |
| `reports/`     | `*_roc_curve.png`           | ROC curves                  |
| `reports/`     | `*_training_curves.png`     | Loss/AUC over epochs        |
| `reports/`     | `*_failure_cases.png`       | Misclassified examples      |
| `results/`     | `*_test_metrics.json`       | Per-model test metrics      |
| `checkpoints/` | `*_best.pth`                | Best model weights (by AUC) |

---

## Task 2: Medical Report Generation with VLMs

### Option A — Ollama (local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull vision-capable model
ollama pull qwen2.5vl:3b        # or qwen2-vl for larger version
ollama pull MedAIBase/MedGemma1.0:4b   # text-only; uses text fallback

# Run
python -m task2_report_generation.run_task2 \
    --provider ollama \
    --model qwen2.5vl:3b \
    --num_samples 10
```

### Option B — Docker Model Runner

```bash
# Run model
docker model run ai/gemma3-qat:270M-F16

# Generate reports (text-only fallback for vision)
python -m task2_report_generation.run_task2 \
    --provider docker \
    --model ai/gemma3-qat:270M-F16 \
    --num_samples 10
```

### Option C — HuggingFace MedGemma (local GPU inference)

```bash
# Install HF deps
uv pip install transformers>=4.45.0 accelerate>=0.26.0
# or: pip install transformers>=4.45.0 accelerate>=0.26.0

# Accept licence at huggingface.co/google/medgemma-4b-it, then:
huggingface-cli login

python -m task2_report_generation.run_task2_hf \
    --model_id google/medgemma-4b-it \
    --num_samples 10 \
    --image_size 512
```

### Task 2 Outputs

| Directory                    | File                          | Description                 |
|------------------------------|-------------------------------|-----------------------------|
| `reports/generated_reports/` | `*.json`                      | Individual VLM reports      |
| `results/`                   | `evaluation_results.json`     | VLM vs GT vs CNN comparison |
| `results/`                   | `strategy_analysis.json`      | Per-prompt strategy stats   |

---

## Task 3: Semantic Image Retrieval with PGVector

### PGVector Setup (Docker)

```bash
docker run -p 5432:5432 \
  --env POSTGRES_PASSWORD=postgres \
  --env POSTGRES_USER=postgres \
  --env POSTGRES_DB=postgres \
  -v ~/medical_db:/var/lib/postgresql/data \
  --name medicaldb \
  -d pgvector/pgvector:pg16
```

### Run Full Pipeline

```bash
# Build index → evaluate → visualize → report
python -m task3_retrieval.run_task3

# Or use --skip_build if index already built
python -m task3_retrieval.run_task3 --skip_build
```

### Start FastAPI Server

```bash
uvicorn task3_retrieval.app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### API Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/build-index` | POST | Extract embeddings, insert into PGVector, create IVFFLAT index |
| `/search/image` | POST | Image-to-image cosine similarity search |
| `/search/text` | POST | Text query (returns 501 — CNN encoder only) |
| `/health` | GET | Health check with DB status and index count |

### Task 3 Outputs

| Directory | File | Description |
| --------- | ---- | ----------- |
| `reports/` | `task3_retrieval_system.md` | Full analysis report |
| `reports/retrieval_visualizations/` | `retrieval_grid.png` | Query-vs-results grids |
| `results/` | `retrieval_evaluation.json` | Precision@k metrics (624 queries) |

---

## Dataset

- **Source**: [MedMNIST v2](https://medmnist.com/) — PneumoniaMNIST
- **Task**: Binary classification (Normal vs Pneumonia)
- **Splits**: 4,708 train / 524 val / 624 test
- **Image size**: 28 × 28 grayscale (auto-downloaded on first run)
- **Class distribution**: Normal: 37% | Pneumonia: 63%

## Reproducibility

```bash
python -m task1_classification.train --model resnet --pretrained true --seed 42
```

- Fixed seeds: Python, NumPy, PyTorch, CUDA
- Deterministic cuDNN (`torch.backends.cudnn.deterministic = True`)
- All hyperparameters logged in `config.yaml` and result JSONs

## Reports

| Task | Report | Key Content |
| ---- | ------ | ----------- |
| Task 1 | [`task1_classification_report.md`](task1_classification_report.md) | Architecture comparison, metrics, failure analysis |
| Task 2 | [`task2_report_generation.md`](task2_report_generation.md) | VLM qualitative analysis, prompting strategies |
| Task 3 | [`task3_retrieval_system.md`](task3_retrieval_system.md) | Precision@k, retrieval grid analysis |

