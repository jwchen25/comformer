"""
Prediction interface for trained ComFormer models.

This module provides a simple interface to load trained models and predict
properties for new crystal structures.
"""

import json
from pathlib import Path
from typing import List, Union, Dict
import torch
from pymatgen.core import Structure

from comformer.models.comformer import iComformer, eComformer, iComformerConfig, eComformerConfig
from comformer.custom_train import pymatgen_to_jarvis
from comformer.graphs import PygGraph, PygStructureDataset
from jarvis.core.atoms import Atoms
from ignite.handlers import Checkpoint


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
        return_std: bool = False,
    ) -> Union[List[float], tuple]:
        """
        Predict properties for a list of crystal structures.

        Args:
            structures: List of pymatgen Structure objects
            return_std: If True, return (predictions, stds) tuple.
                       Currently returns zeros for std as single model doesn't have uncertainty.

        Returns:
            List of predicted property values, or (predictions, stds) if return_std=True

        Example:
            >>> structures = [structure1, structure2, structure3]
            >>> predictions = predictor.predict(structures)
            >>> print(predictions)
            [0.123, 0.456, 0.789]
        """
        if not structures:
            return [] if not return_std else ([], [])

        predictions = []

        print(f"Predicting properties for {len(structures)} structures...")

        with torch.no_grad():
            for i, structure in enumerate(structures):
                try:
                    # Convert structure to graph
                    graph = self._structure_to_graph(structure)

                    # Move to device
                    graph = graph.to(self.device)

                    # Create line graph (copy of graph for this architecture)
                    lg = graph

                    # Predict
                    # Model expects [data, ldata, lattice]
                    output = self.model([graph, lg, lg])

                    # Extract prediction
                    pred_value = output.cpu().item()
                    predictions.append(pred_value)

                except Exception as e:
                    print(f"Warning: Failed to predict for structure {i}: {e}")
                    predictions.append(float('nan'))

        print(f"Prediction completed: {len(predictions)} values")

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
        predictions = self.predict([structure])
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
