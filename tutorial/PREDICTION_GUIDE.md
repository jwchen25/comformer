# ComFormer Prediction Interface User Guide

## Overview

This document describes how to use the ComFormer prediction interface to load trained models and predict properties of crystal structures.

## Quick Start

### 1. Basic Usage

```python
from comformer.predict import load_predictor
from pymatgen.core import Structure

# Load trained model
predictor = load_predictor("./test_output")

# Prepare structures to predict (list of pymatgen.Structure objects)
structures = [structure1, structure2, structure3]

# Make predictions
predictions = predictor.predict(structures)

# Output results
for struct, pred in zip(structures, predictions):
    print(f"{struct.composition}: {pred:.4f}")
```

### 2. Complete Example

```python
from comformer.predict import ComformerPredictor
from pymatgen.core import Lattice, Structure

# Create a simple crystal structure
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Fe"], [[0, 0, 0]])

# Method 1: Using convenience function
from comformer.predict import load_predictor
predictor = load_predictor("./test_output")

# Method 2: Using class method
predictor = ComformerPredictor.from_checkpoint_dir(
    checkpoint_dir="./test_output",
    checkpoint_name="best_model",  # Or "latest" or specific filename
    device="cuda"  # Or "cpu", None for auto-select
)

# Batch prediction
predictions = predictor.predict([structure1, structure2, structure3])

# Single prediction
single_prediction = predictor.predict_single(structure)
```

## Detailed Documentation

### Loading Models

#### Load from Checkpoint Directory

```python
predictor = ComformerPredictor.from_checkpoint_dir(
    checkpoint_dir="./test_output",    # Model directory
    checkpoint_name="best_model",      # Checkpoint name
    device=None                        # Device selection
)
```

**Parameter Description:**

- `checkpoint_dir`: Directory containing model checkpoint and `config.json`
- `checkpoint_name`:
  - `"best_model"`: Auto-load best_model_*.pt with lowest MAE
  - `"latest"`: Load latest checkpoint_*.pt
  - Specific filename: e.g., `"checkpoint_100.pt"`
- `device`:
  - `None`: Auto-select (use GPU if available)
  - `"cuda"`: Force GPU usage
  - `"cpu"`: Force CPU usage

#### Convenience Function

```python
from comformer.predict import load_predictor

predictor = load_predictor("./test_output")
```

This is equivalent to:
```python
ComformerPredictor.from_checkpoint_dir(
    checkpoint_dir="./test_output",
    checkpoint_name="best_model",
    device=None
)
```

### Preparing Structures

Input must be **a list of pymatgen.Structure objects**:

```python
from pymatgen.core import Structure, Lattice

# Create structure from scratch
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Read from file
from pymatgen.core import Structure
structure = Structure.from_file("POSCAR")

# From Materials Project
from pymatgen.ext.matproj import MPRester
with MPRester("your_api_key") as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")

# Prepare batch
structures = [struct1, struct2, struct3]
```

### Making Predictions

#### Batch Prediction

```python
# Basic prediction
predictions = predictor.predict(structures)
# Returns: [0.123, 0.456, 0.789]

# With uncertainty (currently returns 0)
predictions, stds = predictor.predict(structures, return_std=True)
# Returns: ([0.123, 0.456, 0.789], [0.0, 0.0, 0.0])
```

**Parameter Description:**
- `structures`: List of pymatgen.Structure objects
- `batch_size`: Batch size (currently only supports 1)
- `return_std`: Whether to return standard deviation (single model returns 0)

#### Single Prediction

```python
prediction = predictor.predict_single(structure)
print(f"Predicted value: {prediction:.4f}")
```

### Result Interpretation

- The units and meaning of predicted values depend on the target property during training
- If trained on formation energy (eV/atom), predictions are formation energies
- If trained on band gap (eV), predictions are band gaps
- Failed predictions return `nan`

## Advanced Usage

### 1. Check Model Configuration

```python
predictor = load_predictor("./test_output")

# View configuration
print("Cutoff:", predictor.cutoff)
print("Max neighbors:", predictor.max_neighbors)
print("Atom features:", predictor.atom_features)
print("Use lattice:", predictor.use_lattice)
print("Use angle:", predictor.use_angle)
```

### 2. Manual Model Initialization

```python
import json
import torch
from comformer.models.comformer import iComformer, iComformerConfig
from comformer.predict import ComformerPredictor

# Load configuration
with open("./test_output/config.json") as f:
    config = json.load(f)

# Create model
model_config = iComformerConfig(**config["model"])
model = iComformer(config=model_config)

# Load weights
checkpoint = torch.load("./test_output/best_model_5.pt")
model.load_state_dict(checkpoint["model"])

# Create predictor
predictor = ComformerPredictor(model=model, config=config, device="cuda")
```

### 3. Batch Processing Large Numbers of Structures

```python
import numpy as np
from tqdm import tqdm

def predict_large_batch(predictor, structures, batch_size=100):
    """Process large numbers of structures (in batches)"""
    all_predictions = []

    for i in tqdm(range(0, len(structures), batch_size)):
        batch = structures[i:i+batch_size]
        preds = predictor.predict(batch)
        all_predictions.extend(preds)

    return all_predictions

# Usage
structures = [...]  # Large number of structures
predictions = predict_large_batch(predictor, structures)
```

### 4. Error Handling

```python
predictions = predictor.predict(structures)

# Check failed predictions
import math
for i, pred in enumerate(predictions):
    if math.isnan(pred):
        print(f"Warning: Structure {i} prediction failed")
    else:
        print(f"Structure {i}: {pred:.4f}")

# Filter valid predictions
valid_predictions = [p for p in predictions if not math.isnan(p)]
```

## File Structure Requirements

Model directory must contain:

```
test_output/
├── config.json                          # Training configuration (required)
├── best_model_5_neg_mae=-0.3343.pt     # Best model checkpoint
├── checkpoint_4.pt                      # Other checkpoints
└── ...
```

**Required fields in config.json:**
```json
{
    "cutoff": 6.0,
    "max_neighbors": 10,
    "neighbor_strategy": "k-nearest",
    "use_lattice": false,
    "use_angle": false,
    "atom_features": "cgcnn",
    "model": {
        "name": "iComformer",
        "atom_input_features": 92,
        ...
    }
}
```

## Complete Example

See `example_predict.py`:

```bash
# Run example
python example_predict.py
```

Example output:
```
============================================================
ComFormer Prediction Example
============================================================

[Step 1] Loading trained model...
Loading checkpoint from: ./test_output/best_model_5_neg_mae=-0.3343.pt
Model loaded successfully: iComformer
Configuration: cutoff=6.0, max_neighbors=10, atom_features=cgcnn
✓ Model loaded successfully!

[Step 2] Preparing crystal structures...
✓ Created 3 example structures
  Structure 1: Fe (1 atoms)
  Structure 2: Al (4 atoms)
  Structure 3: NaCl (2 atoms)

[Step 3] Making predictions...
Predicting properties for 3 structures...
Prediction completed: 3 values

[Step 4] Results:
------------------------------------------------------------
Structure 1 (Fe): 0.1234
Structure 2 (Al): 0.4567
Structure 3 (NaCl): 0.7890
------------------------------------------------------------

[Step 5] Single structure prediction example:
Single prediction: 0.1234

============================================================
Prediction completed successfully!
============================================================
```

## API Reference

### ComformerPredictor Class

#### Methods

##### `from_checkpoint_dir(checkpoint_dir, checkpoint_name="best_model", device=None)`
Load model from checkpoint directory (class method).

**Returns**: ComformerPredictor instance

##### `predict(structures, batch_size=1, return_std=False)`
Predict properties for multiple structures.

**Parameters**:
- `structures` (List[Structure]): List of pymatgen Structure objects
- `batch_size` (int): Batch size (currently only supports 1)
- `return_std` (bool): Whether to return standard deviation

**Returns**: List[float] or (List[float], List[float])

##### `predict_single(structure)`
Predict property for a single structure.

**Parameters**:
- `structure` (Structure): pymatgen Structure object

**Returns**: float

### Convenience Functions

#### `load_predictor(checkpoint_dir, checkpoint_name="best_model", device=None)`
Convenience function to load predictor.

**Returns**: ComformerPredictor instance

## Frequently Asked Questions

### Q1: How to choose the best model?
A: Use `checkpoint_name="best_model"`, which automatically selects the model with lowest MAE.

### Q2: What if prediction is slow?
A:
1. Ensure GPU usage: `device="cuda"`
2. Current implementation predicts sequentially; batch prediction feature is under development

### Q3: Why do some structure predictions fail?
A: Possible reasons:
- Structure is too large or complex
- Structure contains elements not in training set
- Graph construction failed (e.g., abnormal atomic distances)

### Q4: How to get prediction uncertainty?
A: Current single model does not provide uncertainty estimates. To get uncertainty, you need:
- Train multiple models (ensemble)
- Use Bayesian methods
- Use MC Dropout

### Q5: Can I do batch processing?
A: Yes, pass a list of structures:
```python
predictions = predictor.predict([struct1, struct2, ..., struct100])
```

## Performance Considerations

- **GPU vs CPU**: GPU is 10-50x faster
- **Structure Size**: Large structures (>100 atoms) are slower to predict
- **Graph Construction**: cutoff and max_neighbors affect speed

## References

- **Training Interface**: `comformer/custom_train.py`
- **Model Definition**: `comformer/models/comformer.py`
- **Graph Construction**: `comformer/graphs.py`
