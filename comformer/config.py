"""Pydantic model for default configuration and validation."""
"""Implementation based on the template of Matformer."""

import subprocess
from typing import Optional, Union, Literal
import os

# Pydantic v2 (required)
from pydantic import model_validator

from comformer.utils import BaseSettings
from comformer.models.comformer import iComformerConfig, eComformerConfig

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception as exp:
    VERSION = "NA"
    pass


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal["custom"] = "custom"  # Only support custom datasets
    target: str = "target"  # Target property name in custom dataset
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    neighbor_strategy: Literal["k-nearest", "voronoi", "pairwise-k-nearest"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none", "step", "polynomial"] = "onecycle"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 2
    cutoff: float = 4.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    matrix_input: bool = False
    pyg_input: bool = False
    use_lattice: bool = False
    use_angle: bool = False

    # model configuration
    model: Union[
        iComformerConfig,
        eComformerConfig,
    ] = iComformerConfig(name="iComformer")

    # Validator to set atom_input_features based on atom_features (Pydantic v2)
    @model_validator(mode='after')
    def set_input_size(self):
        """Automatically configure node feature dimensionality."""
        if hasattr(self, 'model') and hasattr(self, 'atom_features'):
            self.model.atom_input_features = FEATURESET_SIZE[self.atom_features]
        return self
