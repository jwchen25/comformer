# ComFormer ExtXYZ Training Interface

## Overview

This guide describes how to train ComFormer models directly from ASE extxyz files. The interface supports large datasets (200k+ structures) with automatic parallelization and graph caching.

---

## Quick Start

### Basic Usage

```python
from comformer.custom_train import train_from_extxyz

# Train from extxyz file
results = train_from_extxyz(
    extxyz_file="structures.xyz",
    target_property="energy",
    output_dir="./my_model"
)
```

### Large Dataset with Caching

```python
# For large datasets (>10k structures), enable caching
results = train_from_extxyz(
    extxyz_file="large_dataset.xyz",  # 200k structures
    target_property="formation_energy",
    batch_size=128,
    cache_graphs=True,  # Cache graphs for faster retraining
    graph_cache_dir="./graph_cache",
    output_dir="./large_model"
)
```

---

## Installation Requirements

### Install ASE

```bash
pip install ase
```

### Verify Installation

```python
import ase
print(f"ASE version: {ase.__version__}")
```

---

## ExtXYZ File Format

### What is ExtXYZ?

Extended XYZ (extxyz) is a file format that stores atomic structures with properties. It's widely used in materials science and molecular dynamics.

### Format Structure

```
<number of atoms>
<comment line with properties>
<atom1> <x> <y> <z> [properties]
<atom2> <x> <y> <z> [properties]
...
```

### Example ExtXYZ File

```xyz
2
Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" Properties=species:S:1:pos:R:3 energy=-4.5 formation_energy=-2.3 pbc="T T T"
Fe 0.0 0.0 0.0
O 2.0 2.0 2.0
2
Lattice="4.1 0.0 0.0 0.0 4.1 0.0 0.0 0.0 4.1" Properties=species:S:1:pos:R:3 energy=-4.6 formation_energy=-2.4 pbc="T T T"
Fe 0.0 0.0 0.0
O 2.05 2.05 2.05
```

---

## Property Types

### Global Properties (atoms.info)

Properties stored once per structure:
- `energy`: Total energy
- `formation_energy`: Formation energy
- `bandgap`: Band gap
- `volume`: Unit cell volume
- Any custom scalar property

**Access in Python:**
```python
from ase.io import read
atoms = read("structure.xyz")
energy = atoms.info['energy']  # Global property
```

### Per-Atom Properties (atoms.arrays)

Properties stored for each atom:
- `forces`: Forces on each atom (3D vectors)
- `charges`: Atomic charges
- `magmoms`: Magnetic moments
- Any custom per-atom property

**Per-atom properties are automatically averaged:**
```python
# If target_property="forces", the interface will:
# 1. Extract forces for all atoms: [[fx1, fy1, fz1], [fx2, fy2, fz2], ...]
# 2. Average to get scalar: mean(sqrt(fx^2 + fy^2 + fz^2))
```

---

## Function Reference

### train_from_extxyz()

```python
def train_from_extxyz(
    extxyz_file: str,
    target_property: str,
    index: Union[int, str, slice] = ":",
    learning_rate: float = 0.001,
    batch_size: int = 64,
    n_epochs: int = 500,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    output_dir: str = "./extxyz_output",
    cache_graphs: bool = False,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extxyz_file` | str | Required | Path to extxyz file |
| `target_property` | str | Required | Property name to predict |
| `index` | str/int/slice | ":" | Which structures to read |
| `learning_rate` | float | 0.001 | Learning rate |
| `batch_size` | int | 64 | Batch size |
| `n_epochs` | int | 500 | Number of epochs |
| `train_ratio` | float | 0.8 | Training data fraction |
| `val_ratio` | float | 0.1 | Validation data fraction |
| `test_ratio` | float | 0.1 | Test data fraction |
| `cutoff` | float | 8.0 | Neighbor cutoff (Å) |
| `max_neighbors` | int | 12 | Max neighbors per atom |
| `output_dir` | str | "./extxyz_output" | Output directory |
| `cache_graphs` | bool | False | Enable graph caching |
| `graph_cache_dir` | str | None | Cache directory |

#### Index Parameter Examples

```python
# Read all structures
index=":"

# Read first structure
index=0

# Read first 1000 structures
index=":1000"

# Read every 10th structure
index="::10"

# Read structures 100-200
index="100:200"

# Read last 100 structures
index="-100:"
```

---

## Usage Examples

### Example 1: Basic Energy Prediction

```python
from comformer.custom_train import train_from_extxyz

results = train_from_extxyz(
    extxyz_file="materials.xyz",
    target_property="energy",
    learning_rate=0.001,
    batch_size=64,
    n_epochs=100,
    output_dir="./energy_model"
)

print(f"Validation MAE: {results['val_mae_best']:.4f}")
```

### Example 2: Formation Energy with Large Dataset

```python
# For 200k+ structures
results = train_from_extxyz(
    extxyz_file="database.xyz",
    target_property="formation_energy",
    batch_size=128,  # Larger batch for big dataset
    n_epochs=200,
    cache_graphs=True,  # Essential for large datasets
    graph_cache_dir="./persistent_cache",
    output_dir="./formation_energy_model"
)
```

### Example 3: Train on Subset

```python
# Train on first 10k structures
results = train_from_extxyz(
    extxyz_file="huge_database.xyz",
    target_property="bandgap",
    index=":10000",  # Only first 10k
    batch_size=64,
    output_dir="./subset_model"
)
```

### Example 4: Per-Atom Property (Forces)

```python
# Average per-atom forces as target
results = train_from_extxyz(
    extxyz_file="md_trajectory.xyz",
    target_property="forces",  # Per-atom property
    batch_size=32,
    output_dir="./force_model"
)
```

### Example 5: Hyperparameter Tuning with Caching

```python
# First run: Build and cache graphs
results1 = train_from_extxyz(
    extxyz_file="data.xyz",
    target_property="energy",
    learning_rate=0.001,
    cache_graphs=True,
    graph_cache_dir="./cache",
    output_dir="./model_lr001"
)

# Subsequent runs: Reuse cached graphs (much faster!)
for lr in [0.01, 0.1]:
    results = train_from_extxyz(
        extxyz_file="data.xyz",
        target_property="energy",
        learning_rate=lr,
        cache_graphs=True,
        graph_cache_dir="./cache",  # Same cache
        output_dir=f"./model_lr{lr}"
    )
```

---

## Helper Functions

### read_extxyz_file()

Read structures without training:

```python
from comformer.custom_train import read_extxyz_file

# Read all structures
structures, labels = read_extxyz_file(
    filename="data.xyz",
    target_property="energy"
)

print(f"Loaded {len(structures)} structures")
print(f"Label range: [{min(labels):.4f}, {max(labels):.4f}]")

# Read subset
structures, labels = read_extxyz_file(
    filename="data.xyz",
    target_property="energy",
    index=":100"  # First 100 only
)
```

### ase_atoms_to_pymatgen()

Convert single ASE Atoms to pymatgen Structure:

```python
from ase.io import read
from comformer.custom_train import ase_atoms_to_pymatgen

# Read ASE atoms
ase_atoms = read("structure.xyz")

# Convert to pymatgen
pmg_structure = ase_atoms_to_pymatgen(ase_atoms)
```

---

## Creating ExtXYZ Files

### From ASE Atoms

```python
from ase import Atoms
from ase.io import write

# Create structures
atoms_list = []
for i in range(100):
    atoms = Atoms(
        symbols=['Fe', 'O'],
        positions=[[0, 0, 0], [2, 2, 2]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        pbc=True
    )

    # Add properties
    atoms.info['energy'] = -5.0 + i * 0.01
    atoms.info['formation_energy'] = -2.0 + i * 0.005

    atoms_list.append(atoms)

# Write to extxyz
write("dataset.xyz", atoms_list)
```

### From Pymatgen Structures

```python
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

# Convert pymatgen to ASE
adaptor = AseAtomsAdaptor()
structures = [...]  # List of pymatgen Structures
labels = [...]  # List of target values

atoms_list = []
for structure, label in zip(structures, labels):
    atoms = adaptor.get_atoms(structure)
    atoms.info['energy'] = label
    atoms_list.append(atoms)

# Write to extxyz
write("dataset.xyz", atoms_list)
```

---

## Performance Tips

### For Large Datasets (200k+)

1. **Enable Caching**
   ```python
   cache_graphs=True,
   graph_cache_dir="./persistent_cache"
   ```

2. **Increase Batch Size**
   ```python
   batch_size=128  # or higher if memory allows
   ```

3. **Use Subset for Testing**
   ```python
   index=":1000"  # Test with 1000 samples first
   ```

4. **Monitor Memory**
   - Graph construction uses ~100-500 MB per 1000 structures
   - Cache files: ~50-100 MB per 1000 structures

### Preprocessing Time Estimates

| Dataset Size | First Run | With Cache |
|--------------|-----------|------------|
| 1k structures | ~2 minutes | ~2 seconds |
| 10k structures | ~20 minutes | ~10 seconds |
| 100k structures | ~3-6 hours | ~30 seconds |
| 200k structures | ~6-12 hours | ~1 minute |

---

## Troubleshooting

### Error: ASE not installed

```bash
pip install ase
```

### Error: Property not found

Check available properties:
```python
from ase.io import read
atoms = read("data.xyz", index=0)
print("Global properties:", list(atoms.info.keys()))
print("Per-atom properties:", list(atoms.arrays.keys()))
```

### Error: Out of memory

Solutions:
- Reduce batch size
- Reduce `max_neighbors`
- Reduce `cutoff`
- Process in chunks using `index` parameter

### Slow performance

- Enable `cache_graphs=True`
- Use larger batch sizes
- Ensure parallel processing is active (check console output)

---

## Complete Working Example

```python
"""
Complete example: Train ComFormer from extxyz file
"""

from comformer.custom_train import train_from_extxyz
import numpy as np

# Step 1: Create test data (skip if you have real data)
from ase import Atoms
from ase.io import write

atoms_list = []
for i in range(500):
    a = 4.0 + np.random.random() * 0.5
    atoms = Atoms(
        symbols=['Fe', 'O'],
        positions=[[0, 0, 0], [a/2, a/2, a/2]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True
    )
    atoms.info['energy'] = -5.0 + np.random.random() * 2.0
    atoms_list.append(atoms)

write("test_data.xyz", atoms_list)
print("Created test_data.xyz with 500 structures")

# Step 2: Train model
print("\nTraining model...")
results = train_from_extxyz(
    extxyz_file="test_data.xyz",
    target_property="energy",
    learning_rate=0.001,
    batch_size=32,
    n_epochs=50,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    cutoff=6.0,
    max_neighbors=12,
    cache_graphs=True,  # Enable caching
    output_dir="./test_model"
)

# Step 3: Check results
print("\n" + "="*60)
print("Training Results")
print("="*60)
print(f"Best Validation MAE: {results['val_mae_best']:.4f}")
print(f"Final Validation MAE: {results['val_mae_final']:.4f}")
print(f"Model saved to: ./test_model")
print("="*60)
```

---

## See Also

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Performance optimizations for large datasets
- [README.md](README.md) - General ComFormer documentation
- ASE Documentation: https://wiki.fysik.dtu.dk/ase/

---

## Support

For issues or questions:
- Check available properties in your extxyz file
- Enable `cache_graphs=True` for large datasets
- Start with a small subset using `index` parameter
- Refer to test script: `tests/test_extxyz_interface.py`
