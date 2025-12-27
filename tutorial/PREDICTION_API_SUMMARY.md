# ComFormer Prediction API - Quick Reference

## One-Line Prediction

```python
from comformer.predict import load_predictor

predictor = load_predictor("./test_output")  # Auto-loads best_model.pt
predictions = predictor.predict(structures)  # structures = List[pymatgen.Structure]
```

## Complete API

### 1. Load Model

```python
from comformer.predict import load_predictor

# Auto-load best_model.pt (recommended)
predictor = load_predictor("./test_output")

# Or load other checkpoints
predictor = load_predictor("./test_output", checkpoint_name="checkpoint_100.pt")
predictor = load_predictor("./test_output", checkpoint_name="latest")

# Specify device
predictor = load_predictor("./test_output", device="cuda")  # GPU
predictor = load_predictor("./test_output", device="cpu")   # CPU
```

### 2. Prepare Input

```python
from pymatgen.core import Structure

# Input: List[pymatgen.Structure]
structures = [
    Structure.from_file("POSCAR1"),
    Structure.from_file("POSCAR2"),
    Structure.from_file("POSCAR3"),
]
```

### 3. Make Predictions

```python
# Batch prediction
predictions = predictor.predict(structures)
# Output: [0.123, 0.456, 0.789]

# Single prediction
prediction = predictor.predict_single(structure)
# Output: 0.123
```

## Input/Output

| Item | Type | Description |
|------|------|-------------|
| **Input** | `List[pymatgen.Structure]` | List of pymatgen Structure objects |
| **Output** | `List[float]` | List of corresponding prediction values |

## Complete Example

```python
from comformer.predict import load_predictor
from pymatgen.core import Lattice, Structure

# 1. Load model
predictor = load_predictor("./test_output")

# 2. Create structures
lattice = Lattice.cubic(4.0)
structure1 = Structure(lattice, ["Fe"], [[0, 0, 0]])
structure2 = Structure(lattice, ["Al"], [[0, 0, 0]])

# 3. Predict
predictions = predictor.predict([structure1, structure2])

# 4. Display results
for struct, pred in zip([structure1, structure2], predictions):
    print(f"{struct.composition}: {pred:.4f}")

# Output:
# Fe1: 0.1234
# Al1: 0.5678
```

## Run Example

```bash
python example_predict.py
```

## Documentation

- **Complete Guide**: `PREDICTION_GUIDE.md`
- **Usage Example**: `example_predict.py`
- **Source Code**: `comformer/predict.py`

## Core Functions

### `load_predictor()`
```python
load_predictor(
    checkpoint_dir: str,                  # Model directory
    checkpoint_name: str = "best_model.pt",  # Checkpoint name (default)
    device: str = None                    # Device (cuda/cpu/None)
) -> ComformerPredictor
```

### `ComformerPredictor.predict()`
```python
predictor.predict(
    structures: List[Structure],   # List of structures
    batch_size: int = 1,          # Batch size
    return_std: bool = False      # Return standard deviation
) -> List[float]
```

### `ComformerPredictor.predict_single()`
```python
predictor.predict_single(
    structure: Structure           # Single structure
) -> float
```

## Requirements

### Required Files
- `checkpoint_dir/config.json` - Training configuration
- `checkpoint_dir/best_model.pt` - Best model weights

### Required Packages
```python
pip install pymatgen torch
```

## Tips

### Batch Processing
```python
# Predict multiple structures at once
predictions = predictor.predict([s1, s2, s3, ..., s100])
```

### Error Handling
```python
import math
predictions = predictor.predict(structures)

for i, pred in enumerate(predictions):
    if math.isnan(pred):
        print(f"Failed: structure {i}")
    else:
        print(f"Success: {pred:.4f}")
```

### GPU Acceleration
```python
# Auto-select (recommended)
predictor = load_predictor("./test_output", device=None)

# Force GPU usage
predictor = load_predictor("./test_output", device="cuda")
```

## Performance

| Scenario | Speed |
|----------|-------|
| Single structure (CPU) | ~0.1-1 s |
| Single structure (GPU) | ~0.01-0.1 s |
| Batch 100 (GPU) | ~1-5 s |

*Actual speed depends on structure size and hardware configuration

## Limitations

1. **Batch Size**: Currently only supports `batch_size=1` (sequential prediction)
2. **Uncertainty**: Single model does not provide uncertainty estimates
3. **Element Coverage**: Only supports elements present in training set

## Troubleshooting

### Issue: Model file not found
```python
FileNotFoundError: Checkpoint file not found: ./test_output/best_model.pt
```
**Solution**:
1. Ensure model is trained and `best_model.pt` is generated
2. Check that both `config.json` and `best_model.pt` exist

### Issue: CUDA out of memory
```python
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU or reduce batch size
```python
predictor = load_predictor("./test_output", device="cpu")
```

### Issue: Prediction returns nan
```python
predictions = [nan, 0.123, nan]
```
**Solution**: Check if failed structures have anomalies (unknown elements, too large, etc.)

## More Information

See `PREDICTION_GUIDE.md` for:
- Advanced usage
- Error handling
- Batch processing tips
- Complete API reference
