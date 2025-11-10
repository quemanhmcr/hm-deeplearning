# Deep Learning Project - HuggingFace Challenge

## Project Structure
```
huggingface_project/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics
│   ├── utils/             # Utility functions
│   └── configs/           # Configuration files
├── data/                  # Data directories
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External datasets
├── scripts/              # Training and utility scripts
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
├── logs/                 # Training logs
├── checkpoints/          # Model checkpoints
└── artifacts/            # Generated artifacts
```

## 🚀 Setup

### Environment Requirements
- **Conda**: Installed at `E:\miniconda3`
- **Environment**: `kaggle-hm` with Python 3.11.13
- **GPU**: CUDA 12.4 support (for PyTorch)

### Quick Setup
```bash
# Activate conda environment
E:\miniconda3\Scripts\activate.bat kaggle-hm

# Install dependencies
pip install -r requirements.txt
```

### 📖 Full Documentation
See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for complete setup instructions and troubleshooting.