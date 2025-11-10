# Environment Setup Guide

## 🚀 Quick Start

### Prerequisites
- Windows OS
- Conda installed at `E:\miniconda3`

### Environment Activation

**IMPORTANT**: All development work **MUST** be done in the `kaggle-hm` conda environment.

```bash
# Method 1: If conda is in PATH
conda activate kaggle-hm

# Method 2: If conda is NOT in PATH (fallback)
E:\miniconda3\Scripts\activate.bat kaggle-hm
```

### Verification
```bash
python --version  # Should show Python 3.11.13
conda list        # Should show packages listed below
```

## 📦 Environment Details

### Conda Installation
- **Location**: `E:\miniconda3`
- **Version**: 25.9.1
- **Environment**: `kaggle-hm`
- **Python**: 3.11.13

### Core Deep Learning Stack
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4 support)
- **TorchVision**: 0.21.0+cu124
- **TorchAudio**: 2.6.0+cu124
- **TensorFlow**: Not installed (PyTorch-only setup)

### Data Science Libraries
- **NumPy**: 2.2.6
- **Pandas**: 2.2.3
- **Polars**: 1.25.0 (High-performance DataFrame)
- **Scikit-learn**: 1.7.2
- **SciPy**: 1.16.3

### Computer Vision
- **OpenCV**: 4.12.0.88
- **Pillow**: 11.3.0
- **ImageIO**: 2.37.2

### Development Tools
- **Testing**: pytest 9.0.0, pytest-cov, pytest-mock, pytest-benchmark
- **Code Quality**:
  - Black 25.9.0 (formatting)
  - Ruff 0.14.4 (linting)
  - MyPy 1.18.2 (type checking)
- **Pre-commit**: 4.4.0
- **Jupyter**: Built-in notebook support

### Monitoring & Logging
- **TensorBoard**: 2.20.0
- **Weights & Biases**: Not installed (can be added)
- **Memory Profiler**: 0.61.0

### Configuration & Utilities
- **OmegaConf**: 2.3.0 (Configuration management)
- **PyYAML**: 6.0.3
- **TQDM**: 4.67.1 (Progress bars)
- **Requests**: 2.32.5

### Additional Tools
- **ML-Dtypes**: 0.5.3 (Custom ML data types)
- **ML-Training**: 0.1.0 (Training utilities)
- **Namex**: 0.1.0 (Naming utilities)

## 🛠️ Development Workflow

### 1. Activate Environment
```bash
# Always activate first!
E:\miniconda3\Scripts\activate.bat kaggle-hm
```

### 2. Install Additional Dependencies (if needed)
```bash
# From requirements.txt
pip install -r requirements.txt

# Individual packages
pip install package_name
```

### 3. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/unit/test_example.py
```

### 4. Code Quality Checks
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## ⚠️ Important Notes

### For New Developers
1. **NEVER** work in the base conda environment
2. **ALWAYS** activate `kaggle-hm` before development
3. Use the full conda path if not in system PATH: `E:\miniconda3\Scripts\`
4. Verify Python version after activation (should be 3.11.13)

### Environment Management
```bash
# Create requirements file (if needed)
pip freeze > requirements_updated.txt

# Update conda
conda update -n base -c defaults conda

# Clean conda
conda clean --all
```

### CUDA Support
- **CUDA Version**: 12.4
- **PyTorch**: Built with CUDA support
- **GPU Training**: Ready for GPU acceleration

## 🐛 Troubleshooting

### Common Issues

**1. Conda command not found**
```bash
# Use full path instead
E:\miniconda3\Scripts\conda.exe activate kaggle-hm
```

**2. Python version mismatch**
```bash
# Ensure you're in correct environment
where python
# Should point to: E:\miniconda3\envs\kaggle-hm\python.exe
```

**3. Package installation fails**
```bash
# Update pip first
python -m pip install --upgrade pip
# Then install package
pip install package_name
```

**4. GPU not available**
```python
# Check PyTorch CUDA availability
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Number of GPUs
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Conda Documentation](https://docs.conda.io/)
- [Project README](../README.md)

---

**Last Updated**: 2025-11-10
**Environment**: kaggle-hm
**Maintainer**: Dev Team