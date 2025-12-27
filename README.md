# ComFormer: Complete and Efficient Graph Transformers for Crystal Material Property Prediction

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202024-blue)](https://openreview.net/forum?id=BnQY9XiRAS)
[![arXiv](https://img.shields.io/badge/arXiv-2403.11857-b31b1b.svg)](https://arxiv.org/pdf/2403.11857)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/divelab/AIRS)

The package version of ComFormer model (ICLR 2024).

![cover](assets/Comformer.png)

---

## 📋 Requirements

### Minimum Requirements

- **Python**: 3.10 - 3.12
- **PyTorch**: 2.6.0+
- **CUDA**: 11.8+ (for GPU support)

### Supported Platforms

ComFormer supports all major platforms including:
- Linux (x86_64, aarch64/ARM64)
- macOS (Intel and Apple Silicon)
- Windows

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ComFormer

# Install PyTorch 2.6+ (example with CUDA 12.6)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install ComFormer
pip install -e .
```

**That's it!** The installation will automatically install all required dependencies.

### Verify Installation

```bash
# Check dependencies
python tests/test_architecture_detection.py
```

---

## 💻 Installation

### Standard Installation

```bash
# Install ComFormer with all dependencies
pip install -e .
```

### PyTorch Installation Options

For CUDA 12.6:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

For CUDA 11.8:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## 📖 Usage

### Training Custom Models

```python
from comformer import train_model
from pymatgen.core import Structure

# Prepare your data
structures = [...]  # List[pymatgen.Structure]
labels = [...]      # List[float]

# Train model
train_model(
    structures=structures,
    labels=labels,
    output_dir="./my_model",
    epochs=100,
    batch_size=32
)
```

### Making Predictions

```python
from comformer import load_predictor

# Load trained model
predictor = load_predictor("./my_model")

# Predict properties for new structures
predictions = predictor.predict(structures)
```

### Using with PyTorch DataLoader

```python
from comformer.data import CrystalDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = CrystalDataset(structures, labels)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train your model
for batch in loader:
    # Your training code
    pass
```

For complete API documentation, see [PREDICTION_GUIDE.md](PREDICTION_GUIDE.md).

---

## 🧪 Testing

### Verify Dependencies

```bash
python tests/test_architecture_detection.py
```

### Run All Tests

```bash
pytest tests/
```

---

## 📊 Benchmarked Results

### The Materials Project Dataset
![cover](assets/MP.png)

### JARVIS Dataset
![cover](assets/JARVIS.png)

### Matbench
![cover](assets/Matbench.png)

---

## 📖 Citation

Please cite our paper if you find the code helpful or if you want to use the benchmark results. Thank you!

```bibtex
@inproceedings{yan2024complete,
  title={Complete and Efficient Graph Transformers for Crystal Material Property Prediction},
  author={Yan, Keqiang and Fu, Cong and Qian, Xiaofeng and Qian, Xiaoning and Ji, Shuiwang},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

---

## 🙏 Acknowledgments

- PyTorch team for excellent deep learning framework
- PyTorch Geometric team for graph neural network library
- Materials Project for providing materials data

---

## 📧 Contact

For questions and feedback:
- **Email**: junwu.chen@epfl.ch