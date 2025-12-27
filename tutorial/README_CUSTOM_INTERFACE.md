# iComformer Custom Dataset Training Interface

This is a simple and easy-to-use Python interface for training iComformer prediction models on custom pymatgen Structure datasets.

## Quick Start

### Install Dependencies

**Quick Installation:**
```bash
pip install torch pymatgen jarvis-tools torch-geometric pytorch-ignite pydantic pydantic-settings pandas numpy
```

**Complete Installation (including all features):**
```bash
pip install -r requirements_custom.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)

### Simplest Usage

```python
from pymatgen.core import Structure
from comformer.custom_train import train_custom_icomformer

# Your data
strucs = [structure1, structure2, ...]  # List of pymatgen.Structure
labels = [value1, value2, ...]          # List of corresponding label values

# Train model
results = train_custom_icomformer(
    strucs=strucs,
    labels=labels,
    output_dir="./my_model"
)

print(f"Best validation MAE: {results['val_mae_best']:.4f}")
```

That's it!

## Main Features

✓ **Automatic Data Conversion**: Automatically converts pymatgen.Structure to required format
✓ **Automatic Data Splitting**: Automatically divides train/validation/test sets
✓ **Automatic Normalization**: Automatically standardizes labels
✓ **Flexible Configuration**: Supports rich hyperparameter configuration
✓ **Complete Output**: Saves model, configuration, predictions, and all information

## Interface Function

### `train_custom_icomformer(strucs, labels, **kwargs)`

**Required Parameters:**
- `strucs`: List of pymatgen.Structure objects
- `labels`: List of corresponding label values

**Common Optional Parameters:**
- `learning_rate=0.001`: Learning rate
- `batch_size=64`: Batch size
- `n_epochs=500`: Number of training epochs
- `train_ratio=0.8`: Training set ratio
- `val_ratio=0.1`: Validation set ratio
- `test_ratio=0.1`: Test set ratio
- `output_dir="./custom_output"`: Output directory
- `cutoff=8.0`: Neighbor search distance cutoff (Angstrom)
- `max_neighbors=12`: Maximum number of neighbors
- `use_lattice=False`: Whether to use lattice vector features
- `use_angle=False`: Whether to use bond angle features

**Return Value:**
Dictionary containing training history and best metrics:
```python
{
    'history': {...},              # Complete training history
    'train_mae_final': float,      # Training MAE of last epoch
    'train_mae_best': float,       # Best training MAE
    'val_mae_final': float,        # Validation MAE of last epoch
    'val_mae_best': float,         # Best validation MAE
}
```

## Complete Example

```python
from pymatgen.core import Lattice, Structure
from comformer.custom_train import train_custom_icomformer
import numpy as np

# Create example dataset
np.random.seed(42)
strucs = []
labels = []

for i in range(100):
    # Create structures with different lattice parameters
    a = 3.5 + 1.0 * np.random.random()
    lattice = Lattice.cubic(a)
    struc = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    strucs.append(struc)
    labels.append(-2.5 + 0.1 * a)  # Formation energy example

# Train model
results = train_custom_icomformer(
    strucs=strucs,
    labels=labels,
    learning_rate=0.001,
    batch_size=32,
    n_epochs=100,
    cutoff=8.0,
    max_neighbors=12,
    use_lattice=True,
    output_dir="./formation_energy_model",
    random_seed=42
)

print(f"Training MAE (best): {results['train_mae_best']:.4f}")
print(f"Validation MAE (best): {results['val_mae_best']:.4f}")
```

## Output Files

After training completes, the following files will be generated in `output_dir`:

```
output_dir/
├── best_model.pt              # Best model weights
├── config.json                # Training configuration
├── ids_train_val_test.json    # Dataset splits
├── train_predictions.csv      # Training set predictions
├── val_predictions.csv        # Validation set predictions
└── test_predictions.csv       # Test set predictions
```

## Test Interface

Run test script to verify interface:

```bash
python test_custom_interface.py
```

## Detailed Documentation

See [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) for:
- Detailed parameter descriptions
- More usage examples
- Frequently asked questions
- Advanced usage

## Core Files

- `comformer/custom_train.py` - Main interface implementation
- `comformer/train.py` - Training logic (bug fixed to support custom loaders)
- `test_custom_interface.py` - Test script
- `CUSTOM_DATASET_GUIDE.md` - Complete user guide

## Technical Details

### Data Flow

1. **pymatgen.Structure** → Convert to jarvis.Atoms format
2. **jarvis.Atoms** → Build PyTorch Geometric graph
3. **Graph data** → Create PyTorch Dataset and DataLoader
4. **Training** → Train using iComformer model

### Graph Construction

- Uses k-nearest neighbor strategy to find atomic neighbors
- Supports periodic boundary conditions
- Optional inclusion of lattice vectors and bond angle information

### Model Architecture

- iComformer: 4-layer graph attention + edge updates
- Supports lattice information and angle features
- Global average pooling + FC layer output

## Citation

If you use this interface, please cite the ComFormer paper:

```
Complete and Efficient Graph Transformers for Crystal Material Property Prediction
ICLR 2024
```

## License

Follows the license of the original ComFormer project.

---

**Note**: On first run, jarvis-tools may need to download atomic feature data, please ensure network connection is stable.
