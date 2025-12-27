# ComFormer: Complete and Efficient Graph Transformers for Crystal Material Property Prediction

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202024-blue)](https://openreview.net/forum?id=BnQY9XiRAS)
[![arXiv](https://img.shields.io/badge/arXiv-2403.11857-b31b1b.svg)](https://arxiv.org/pdf/2403.11857)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/divelab/AIRS)

The package version of ComFormer model (ICLR 2024).

![cover](assets/Comformer.png)

---

## 📑 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training from ExtXYZ Files](#from-extxyz-files-ase)
  - [Training from pymatgen](#from-pymatgen-structures)
  - [Large Dataset Optimization](#large-dataset-optimization)
  - [Model Outputs](#model-outputs)
- [Key Features](#-key-features)
- [Benchmarked Results](#-benchmarked-results)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 📋 Requirements

### Minimum Requirements

- **Python**: 3.10 - 3.12
- **PyTorch**: 2.6.0+
- **CUDA**: 11.8+ (for GPU support)

### Supported Platforms

ComFormer supports all major platforms including:
- **Linux**: x86_64, aarch64/ARM64
- **macOS**: Intel and Apple Silicon
- **Windows**: x86_64
- **NVIDIA GH200 Grace Hopper**: Full support with ARM64 + CUDA

For detailed platform compatibility and installation instructions, see [PLATFORM_COMPATIBILITY.md](tutorial/PLATFORM_COMPATIBILITY.md).

---

## 💻 Installation

### Standard Installation

```bash
pip install git+https://github.com/jwchen25/comformer.git
```

### Specific PyTorch versions

For example, use PyTorch 2.8.0 with CUDA 12.9:
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install git+https://github.com/jwchen25/comformer.git
```

---

## 📖 Usage

### Training Custom Models

#### From pymatgen Structures

```python
from comformer.custom_train import train_custom_icomformer
from pymatgen.core import Structure

# Prepare your data
structures = [...]  # List[pymatgen.Structure]
labels = [...]      # List[float]

# Train model
results = train_custom_icomformer(
    strucs=structures,
    labels=labels,
    output_dir="./my_model",
    n_epochs=100,
    batch_size=32
)
```

#### From ExtXYZ Files (ASE)

Train directly from ASE extxyz files:

```python
from comformer.custom_train import train_from_extxyz

# Train from extxyz file
results = train_from_extxyz(
    extxyz_file="structures.xyz",
    target_property="formation_energy",  # Property from atoms.info or atoms.arrays
    output_dir="./my_model",
    n_epochs=100,
    batch_size=32
)
```

**Supported property types:**
- **Global properties** (atoms.info): energy, formation_energy, band_gap, etc.
- **Per-atom properties** (atoms.arrays): forces, charges, etc. (averaged to scalar)

**Advanced options:**
```python
# Read subset of structures
results = train_from_extxyz(
    extxyz_file="large_dataset.xyz",
    target_property="energy",
    index=":1000",  # Only first 1000 structures
    output_dir="./my_model"
)

# Enable graph caching for repeated experiments
results = train_from_extxyz(
    extxyz_file="structures.xyz",
    target_property="energy",
    cache_graphs=True,
    graph_cache_dir="./graph_cache",
    output_dir="./my_model"
)
```

See [EXTXYZ_INTERFACE.md](tutorial/EXTXYZ_INTERFACE.md) for complete guide.

### Large Dataset Optimization

For large datasets (200k+ samples), ComFormer includes several optimizations:

```python
from comformer.custom_train import train_custom_icomformer

# Optimizations are enabled automatically for large datasets
results = train_custom_icomformer(
    strucs=structures,
    labels=labels,
    output_dir="./my_model",
    n_epochs=100,
    batch_size=32,
    cache_graphs=True,          # Cache graphs for speedup
    graph_cache_dir="./cache",  # Cache directory
    num_workers=4               # Parallel data loading
)
```

See [OPTIMIZATION_GUIDE.md](tutorial/OPTIMIZATION_GUIDE.md) for detailed benchmarks and tuning tips.

### Model Outputs

After training, ComFormer automatically generates:

1. **Best model checkpoint**: `output_dir/best_model.pt`
2. **Test predictions CSV**: `output_dir/test_predictions.csv`
   - Columns: `id`, `target`, `prediction`
3. **Correlation plot**: `output_dir/test_predictions_correlation.jpg`
   - Scatter plot of predictions vs. true values
   - Metrics: R², MAE, RMSE
4. **Training history**: `output_dir/history.json`

Example of generated files:
```
my_model/
├── best_model.pt                          # Trained model
├── test_predictions.csv                   # Predictions on test set
├── test_predictions_correlation.jpg       # Visualization
└── history.json                           # Training metrics
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

For complete API documentation, see [PREDICTION_GUIDE.md](tutorial/PREDICTION_GUIDE.md).

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

- **PyTorch team** for the excellent deep learning framework
- **PyTorch Geometric team** for the graph neural network library
- **Materials Project** for providing comprehensive materials data
- **AIRS** for original code of [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer)

---

## 📧 Contact

For questions and feedback:
- **Email**: junwu.chen@epfl.ch