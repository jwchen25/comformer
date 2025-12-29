"""
Prediction interface for trained ComFormer models.

This module provides a simple interface to load trained models and predict
properties for new crystal structures.
"""

import json
from pathlib import Path
from typing import List, Union, Dict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import numpy as np
import torch
from pymatgen.core import Structure
from jarvis.core.atoms import Atoms
from ignite.handlers import Checkpoint
from huggingface_hub import hf_hub_download, snapshot_download
from torch_geometric.data import Batch

from comformer.models.comformer import iComformer, eComformer, iComformerConfig, eComformerConfig
from comformer.custom_train import pymatgen_to_jarvis
from comformer.graphs import PygGraph, PygStructureDataset


class ComformerPredictor:
    """
    Predictor class for loading trained ComFormer models and making predictions.

    Example:
        >>> from pymatgen.core import Structure
        >>> from comformer.predict import ComformerPredictor
        >>>
        >>> # Load trained model
        >>> predictor = ComformerPredictor.from_checkpoint_dir("./test_output")
        >>>
        >>> # Prepare structures
        >>> structures = [structure1, structure2, structure3]  # List of pymatgen Structures
        >>>
        >>> # Predict properties
        >>> predictions = predictor.predict(structures)
        >>> print(predictions)  # [0.123, 0.456, 0.789]
    """

    def __init__(
        self,
        model: Union[iComformer, eComformer],
        config: Dict,
        device: str = None,
    ):
        """
        Initialize predictor with a loaded model and configuration.

        Args:
            model: Trained ComFormer model (iComformer or eComformer)
            config: Configuration dictionary from training
            device: Device to run predictions on ('cuda' or 'cpu').
                   If None, automatically selects GPU if available.
        """
        self.model = model
        self.config = config

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Extract graph building parameters from config
        self.cutoff = config.get("cutoff", 8.0)
        self.max_neighbors = config.get("max_neighbors", 12)
        self.neighbor_strategy = config.get("neighbor_strategy", "k-nearest")
        self.use_lattice = config.get("use_lattice", False)
        self.use_angle = config.get("use_angle", False)
        self.atom_features = config.get("atom_features", "cgcnn")

        # Load normalization statistics for denormalization
        self.mean_train = config.get("mean_train", None)
        self.std_train = config.get("std_train", None)

        # Get feature lookup
        self.feature_lookup = PygStructureDataset._get_attribute_lookup(self.atom_features)

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Union[str, Path],
        checkpoint_name: str = "best_model.pt",
        device: str = None,
    ) -> "ComformerPredictor":
        """
        Load a trained model from a checkpoint directory.

        Args:
            checkpoint_dir: Directory containing model checkpoint and config.json
            checkpoint_name: Name of checkpoint file to load.
                           Default is "best_model.pt" (recommended).
                           Can also use "latest" or specific filename like "checkpoint_100.pt"
            device: Device to run predictions on ('cuda' or 'cpu')

        Returns:
            ComformerPredictor instance ready for making predictions

        Raises:
            FileNotFoundError: If config.json or checkpoint file not found
            ValueError: If model configuration is invalid
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load configuration
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Find checkpoint file
        if checkpoint_name == "latest":
            # Find latest checkpoint
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint_*.pt files found in {checkpoint_dir}")
            checkpoint_path = sorted(checkpoint_files)[-1]

        else:
            # Use specified checkpoint (default: best_model.pt)
            checkpoint_path = checkpoint_dir / checkpoint_name
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"Make sure you have trained a model and 'best_model.pt' exists in {checkpoint_dir}"
                )

        print(f"Loading checkpoint from: {checkpoint_path}")

        # Initialize model
        model_config = config.get("model", {})
        model_name = model_config.get("name", "iComformer")

        if model_name == "iComformer":
            model_cls = iComformer
            config_cls = iComformerConfig
        elif model_name == "eComformer":
            model_cls = eComformer
            config_cls = eComformerConfig
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Create model config object
        model_config_obj = config_cls(**model_config)
        model = model_cls(config=model_config_obj)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            # New format: {"model": state_dict}
            model.load_state_dict(checkpoint["model"])
        else:
            # Old format from Checkpoint handler (for backward compatibility)
            to_load = {"model": model}
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        print(f"Model loaded successfully: {model_name}")
        print(f"Configuration: cutoff={config.get('cutoff')}, "
              f"max_neighbors={config.get('max_neighbors')}, "
              f"atom_features={config.get('atom_features')}")

        return cls(model=model, config=config, device=device)

    def _structure_to_graph(self, structure: Structure):
        """
        Convert a pymatgen Structure to a PyTorch Geometric graph.

        Args:
            structure: pymatgen Structure object

        Returns:
            PyTorch Geometric Data object representing the crystal graph
        """
        # Convert pymatgen to jarvis format
        atoms_dict = pymatgen_to_jarvis(structure)
        jarvis_atoms = Atoms.from_dict(atoms_dict)

        # Build graph
        graph = PygGraph.atom_dgl_multigraph(
            jarvis_atoms,
            neighbor_strategy=self.neighbor_strategy,
            cutoff=self.cutoff,
            atom_features="atomic_number",
            max_neighbors=self.max_neighbors,
            compute_line_graph=False,
            use_canonize=False,
            use_lattice=self.use_lattice,
            use_angle=self.use_angle,
        )

        # Convert atomic numbers to features
        z = graph.x
        graph.atomic_number = z.clone()
        z = z.type(torch.IntTensor).squeeze()

        # Handle single atom case
        if z.dim() == 0:
            z = z.unsqueeze(0)

        # Look up features
        f = torch.tensor(self.feature_lookup[z.numpy()]).type(torch.FloatTensor)
        if graph.x.size(0) == 1:
            f = f.unsqueeze(0)

        graph.x = f

        # Add batch info (single structure)
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.int64)

        return graph

    def predict(
        self,
        structures: List[Structure],
        batch_size: int = 32,
        return_std: bool = False,
    ) -> Union[List[float], tuple]:
        """
        Predict properties for a list of crystal structures using batched processing.

        Args:
            structures: List of pymatgen Structure objects
            batch_size: Number of structures to process per batch (default: 32).
                       Set to 1 for sequential processing. Larger batches are more efficient
                       but require more memory. Typical range: 1-128.
            return_std: If True, return (predictions, stds) tuple.
                       Currently returns zeros for std as single model doesn't have uncertainty.

        Returns:
            List of predicted property values, or (predictions, stds) if return_std=True

        Example:
            >>> structures = [structure1, structure2, structure3, ...]
            >>> # Predict with default batch_size=32
            >>> predictions = predictor.predict(structures)
            >>> # Predict with custom batch size
            >>> predictions = predictor.predict(structures, batch_size=64)
            >>> # Predict one structure at a time
            >>> predictions = predictor.predict(structures, batch_size=1)
            >>> print(predictions)
            [0.123, 0.456, 0.789]
        """
        if not structures:
            return [] if not return_std else ([], [])

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        predictions = []

        # Log normalization and batch processing information
        # print(f"Predicting properties for {len(structures)} structures with batch_size={batch_size}...")
        # if self.mean_train is not None and self.std_train is not None:
        #     print(f"Applying denormalization: predictions will be scaled by std_train={self.std_train:.6f} and shifted by mean_train={self.mean_train:.6f}")
        # elif self.mean_train is not None or self.std_train is not None:
        #     print(f"Warning: Partial normalization statistics found (mean_train={self.mean_train}, std_train={self.std_train}). Denormalization will not be applied.")

        # Calculate number of batches
        num_batches = (len(structures) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Get batch of structures
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(structures))
                batch_structures = structures[start_idx:end_idx]

                try:
                    # Convert all structures in batch to graphs using parallel threads
                    batch_graphs = []
                    failed_indices = []

                    # Define conversion function for parallel processing
                    def convert_structure_to_graph_with_index(item):
                        """Convert a structure to graph with its index for tracking."""
                        idx, structure = item
                        try:
                            graph = self._structure_to_graph(structure)
                            return (idx, graph, None)
                        except Exception as e:
                            return (idx, None, str(e))

                    # Determine optimal number of worker threads
                    # Automatically scale based on CPU core count, max 8 to avoid excessive GIL contention
                    num_workers = min(cpu_count(), 8)

                    # Run parallel graph conversion
                    if len(batch_structures) > 1:
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            indexed_batch = [(i, s) for i, s in enumerate(batch_structures)]
                            results = list(executor.map(convert_structure_to_graph_with_index, indexed_batch))
                    else:
                        # For single structure, process directly to avoid thread overhead
                        results = [convert_structure_to_graph_with_index((0, batch_structures[0]))]

                    # Process results and maintain original order
                    result_dict = {}
                    for idx, graph, error in results:
                        if error:
                            print(f"Warning: Failed to convert structure {start_idx + idx} to graph: {error}")
                            failed_indices.append(idx)
                        else:
                            result_dict[idx] = graph

                    # Reconstruct batch_graphs in original order
                    for i in range(len(batch_structures)):
                        if i in result_dict:
                            batch_graphs.append(result_dict[i])

                    if not batch_graphs:
                        # All structures in this batch failed
                        predictions.extend([np.nan] * len(batch_structures))
                        continue

                    # Batch the graphs using PyG Batch
                    batched_graph = Batch.from_data_list(batch_graphs)
                    batched_line_graph = batched_graph

                    # Move batch to device
                    batched_graph = batched_graph.to(self.device)
                    batched_line_graph = batched_line_graph.to(self.device)

                    # Run model on batched data
                    # Model expects [data, ldata, lattice]
                    batch_output = self.model([batched_graph, batched_line_graph, batched_line_graph])

                    # Extract predictions for each structure in the batch
                    # batch_output shape: [batch_actual_size] or [batch_actual_size, 1]
                    batch_preds = batch_output.cpu().detach()

                    # Ensure batch_preds is 1D
                    if batch_preds.dim() > 1:
                        batch_preds = batch_preds.squeeze(-1)

                    # Convert to list
                    batch_preds_list = batch_preds.tolist()

                    # Handle different batch_preds types (ensure it's a list)
                    if not isinstance(batch_preds_list, list):
                        batch_preds_list = [batch_preds_list]

                    # Insert NaN for failed structures and apply denormalization
                    batch_pred_idx = 0
                    for i in range(len(batch_structures)):
                        if i in failed_indices:
                            predictions.append(np.nan)
                        else:
                            pred_value = batch_preds_list[batch_pred_idx]

                            # Apply denormalization if normalization statistics are available
                            if self.std_train is not None and self.mean_train is not None:
                                pred_value = pred_value * self.std_train + self.mean_train

                            predictions.append(pred_value)
                            batch_pred_idx += 1

                except Exception as e:
                    print(f"Warning: Failed to predict batch {batch_idx}: {e}")
                    predictions.extend([np.nan] * len(batch_structures))

        if return_std:
            # For single model, uncertainty is not available
            stds = [0.0] * len(predictions)
            return predictions, stds

        return predictions

    def predict_single(self, structure: Structure) -> float:
        """
        Predict property for a single crystal structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            Predicted property value

        Example:
            >>> prediction = predictor.predict_single(structure)
            >>> print(f"Predicted value: {prediction:.4f}")
        """
        predictions = self.predict([structure], batch_size=1)
        return predictions[0]


def load_predictor(
    checkpoint_dir: Union[str, Path],
    checkpoint_name: str = "best_model.pt",
    device: str = None,
) -> ComformerPredictor:
    """
    Convenience function to load a trained ComFormer predictor.

    Args:
        checkpoint_dir: Directory containing model checkpoint and config.json
        checkpoint_name: Name of checkpoint file to load (default: "best_model.pt")
        device: Device to run predictions on ('cuda' or 'cpu')

    Returns:
        ComformerPredictor instance ready for making predictions

    Example:
        >>> from comformer.predict import load_predictor
        >>> predictor = load_predictor("./test_output")
        >>> predictions = predictor.predict(structures)
    """
    return ComformerPredictor.from_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )


def load_predictor_hf(
    repo_id: str = "jwchen25/MatFlow",
    subfolder: str = "property_prediction/test",
    checkpoint_name: str = "best_model.pt",
    device: str = None,
) -> ComformerPredictor:
    """
    Load a trained ComFormer predictor from HuggingFace Hub.

    This function downloads the model checkpoint and configuration from a HuggingFace
    repository and loads them for prediction.

    Args:
        repo_id: HuggingFace repository ID (default: "jwchen25/MatFlow")
        subfolder: Path to the model files within the repository
                  (default: "property_prediction/band_gap")
        checkpoint_name: Name of checkpoint file to load (default: "best_model.pt")
        device: Device to run predictions on ('cuda' or 'cpu').
               If None, automatically selects GPU if available.

    Returns:
        ComformerPredictor instance ready for making predictions

    Raises:
        FileNotFoundError: If config.json or checkpoint file not found in the repository
        ValueError: If model configuration is invalid
        ConnectionError: If unable to connect to HuggingFace Hub

    Example:
        >>> from comformer.predict import load_predictor_hf
        >>> predictor = load_predictor_hf()  # Uses default repo and subfolder
        >>> # Predict with default batch_size=32
        >>> predictions = predictor.predict(structures)
        >>> # Or predict with custom batch size
        >>> predictions = predictor.predict(structures, batch_size=64)
        >>>
        >>> # Or specify custom repository
        >>> predictor = load_predictor_hf(
        ...     repo_id="your_username/your_repo",
        ...     subfolder="path/to/model"
        ... )
        >>> predictions = predictor.predict(structures, batch_size=128)
    """
    # Download the entire model folder from HuggingFace Hub
    print(f"Downloading model from HuggingFace: {repo_id}/{subfolder}")

    try:
        model_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=None,
            allow_patterns=None if not subfolder else f"{subfolder}/*"
        )
        print(f"Model downloaded to: {model_dir}")
    except Exception as e:
        raise ConnectionError(
            f"Failed to download model from HuggingFace Hub. "
            f"Repo: {repo_id}, Error: {e}"
        )

    # Construct path to the checkpoint directory
    if subfolder:
        checkpoint_dir = Path(model_dir) / subfolder
    else:
        checkpoint_dir = Path(model_dir)

    # Check if checkpoint directory exists
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}. "
            f"Please ensure the subfolder '{subfolder}' exists in the repository '{repo_id}'"
        )

    # Load configuration
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please ensure 'config.json' exists in {checkpoint_dir}"
        )

    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Find checkpoint file
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            f"Please ensure '{checkpoint_name}' exists in {checkpoint_dir}"
        )

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "iComformer")

    if model_name == "iComformer":
        model_cls = iComformer
        config_cls = iComformerConfig
    elif model_name == "eComformer":
        model_cls = eComformer
        config_cls = eComformerConfig
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Create model config object
    model_config_obj = config_cls(**model_config)
    model = model_cls(config=model_config_obj)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        # New format: {"model": state_dict}
        model.load_state_dict(checkpoint["model"])
    else:
        # Old format from Checkpoint handler (for backward compatibility)
        to_load = {"model": model}
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

    print(f"Model loaded successfully from HuggingFace: {model_name}")

    return ComformerPredictor(model=model, config=config, device=device)
