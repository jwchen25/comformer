"""Custom training interface for iComformer with pymatgen Structure inputs."""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice as JarvisLattice
from pymatgen.core import Structure
from comformer.graphs import PygGraph, PygStructureDataset
from comformer.train import train_main
from torch.utils.data import DataLoader
from jarvis.db.jsonutils import dumpjson
import os
import random


def pymatgen_to_jarvis(structure: Structure) -> dict:
    """
    Convert a pymatgen Structure to a jarvis Atoms dictionary.

    Args:
        structure: pymatgen Structure object

    Returns:
        Dictionary representation compatible with jarvis Atoms.from_dict()
    """
    # Extract lattice parameters
    lattice = structure.lattice
    lattice_mat = lattice.matrix.tolist()

    # Extract atomic information
    elements = [str(site.specie) for site in structure]
    coords = [site.frac_coords.tolist() for site in structure]

    # Create jarvis-compatible dictionary
    atoms_dict = {
        "lattice_mat": lattice_mat,
        "coords": coords,
        "elements": elements,
        "abc": [lattice.a, lattice.b, lattice.c],
        "angles": [lattice.alpha, lattice.beta, lattice.gamma],
        "cartesian": False,  # We're using fractional coordinates
        "props": ["", "", ""],
    }

    return atoms_dict


def prepare_custom_dataset(
    strucs: List[Structure],
    labels: List[float],
    id_prefix: str = "custom",
) -> pd.DataFrame:
    """
    Prepare a custom dataset from pymatgen Structures and labels.

    Args:
        strucs: List of pymatgen Structure objects
        labels: List of target property values
        id_prefix: Prefix for generating sample IDs

    Returns:
        pandas DataFrame with 'atoms', 'target', and 'id' columns
    """
    if len(strucs) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(strucs)} structures but {len(labels)} labels"
        )

    # Convert all structures to jarvis format
    print(f"Converting {len(strucs)} structures to jarvis format...")
    atoms_dicts = []
    for i, struc in enumerate(strucs):
        try:
            atoms_dict = pymatgen_to_jarvis(struc)
            atoms_dicts.append(atoms_dict)
        except Exception as e:
            print(f"Warning: Failed to convert structure {i}: {e}")
            atoms_dicts.append(None)

    # Create DataFrame
    data = []
    for i, (atoms_dict, label) in enumerate(zip(atoms_dicts, labels)):
        if atoms_dict is not None:
            data.append({
                "atoms": atoms_dict,
                "target": label,
                "id": f"{id_prefix}_{i}",
            })

    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} valid samples")

    return df


def load_pyg_graphs_from_df(
    df: pd.DataFrame,
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    """
    Convert atoms dictionaries to PyTorch Geometric graphs.

    Args:
        df: DataFrame with 'atoms' column containing jarvis-format dicts
        neighbor_strategy: Strategy for neighbor finding
        cutoff: Distance cutoff for neighbors (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
        use_lattice: Whether to include lattice information
        use_angle: Whether to include angle information

    Returns:
        List of PyTorch Geometric Data objects
    """
    from tqdm import tqdm

    def atoms_to_graph(atoms_dict):
        """Convert structure dict to PyG graph."""
        structure = Atoms.from_dict(atoms_dict)
        return PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=False,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )

    print("Building graphs from structures...")
    graphs = []
    for atoms_dict in tqdm(df["atoms"].values):
        try:
            graph = atoms_to_graph(atoms_dict)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Failed to build graph: {e}")
            graphs.append(None)

    # Filter out failed conversions
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    graphs = [graphs[i] for i in valid_indices]

    return graphs, valid_indices


def train_custom_icomformer(
    strucs: List[Structure],
    labels: List[float],
    # Training parameters
    learning_rate: float = 0.001,
    batch_size: int = 64,
    n_epochs: int = 500,
    # Data split parameters
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 123,
    # Model parameters
    model_name: str = "iComformer",
    atom_features: str = "cgcnn",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_lattice: bool = False,
    use_angle: bool = False,
    # Other parameters
    output_dir: str = "./custom_output",
    num_workers: int = 4,
    classification: bool = False,
    classification_threshold: Optional[float] = None,
    random_seed: Optional[int] = None,
    **kwargs,
):
    """
    Train iComformer model on custom dataset.

    Args:
        strucs: List of pymatgen Structure objects
        labels: List of target property values (same length as strucs)
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        split_seed: Random seed for data splitting
        model_name: Model architecture ("iComformer" or "eComformer")
        atom_features: Type of atomic features ("cgcnn", "atomic_number", etc.)
        cutoff: Distance cutoff for neighbor finding (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
        use_lattice: Whether to include lattice vector information
        use_angle: Whether to include bond angle information
        output_dir: Directory to save results and checkpoints
        num_workers: Number of workers for data loading
        classification: Whether this is a classification task
        classification_threshold: Threshold for binary classification
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters passed to the model config

    Returns:
        Dictionary with training results including train/val/test metrics
    """
    # Validate inputs
    if len(strucs) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(strucs)} structures but {len(labels)} labels"
        )

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("train_ratio, val_ratio, test_ratio must be between 0 and 1")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset
    print("\n" + "="*60)
    print("Preparing custom dataset...")
    print("="*60)
    df = prepare_custom_dataset(strucs, labels, id_prefix="custom")

    # Split dataset
    total_size = len(df)
    n_train = int(train_ratio * total_size)
    n_val = int(val_ratio * total_size)
    n_test = total_size - n_train - n_val

    print(f"\nDataset split:")
    print(f"  Total samples: {total_size}")
    print(f"  Train: {n_train} ({train_ratio*100:.1f}%)")
    print(f"  Val: {n_val} ({val_ratio*100:.1f}%)")
    print(f"  Test: {n_test} ({test_ratio*100:.1f}%)")

    # Shuffle and split
    indices = list(range(total_size))
    random.seed(split_seed)
    random.shuffle(indices)

    id_train = indices[:n_train]
    id_val = indices[n_train:n_train + n_val]
    id_test = indices[n_train + n_val:]

    # Save split indices
    ids_train_val_test = {
        "id_train": [df.iloc[i]["id"] for i in id_train],
        "id_val": [df.iloc[i]["id"] for i in id_val],
        "id_test": [df.iloc[i]["id"] for i in id_test],
    }
    dumpjson(
        data=ids_train_val_test,
        filename=os.path.join(output_dir, "ids_train_val_test.json"),
    )

    # Create train/val/test datasets
    df_train = df.iloc[id_train].reset_index(drop=True)
    df_val = df.iloc[id_val].reset_index(drop=True)
    df_test = df.iloc[id_test].reset_index(drop=True)

    # Build graphs
    print("\n" + "="*60)
    print("Building graph representations...")
    print("="*60)
    graphs_train, valid_train = load_pyg_graphs_from_df(
        df_train,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    graphs_val, valid_val = load_pyg_graphs_from_df(
        df_val,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    graphs_test, valid_test = load_pyg_graphs_from_df(
        df_test,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )

    # Filter dataframes to keep only valid graphs
    df_train = df_train.iloc[valid_train].reset_index(drop=True)
    df_val = df_val.iloc[valid_val].reset_index(drop=True)
    df_test = df_test.iloc[valid_test].reset_index(drop=True)

    # Compute normalization statistics from training set
    train_labels = df_train["target"].values
    mean_train = np.mean(train_labels)
    std_train = np.std(train_labels)

    print(f"\nTarget statistics (training set):")
    print(f"  Mean: {mean_train:.4f}")
    print(f"  Std: {std_train:.4f}")
    print(f"  Min: {np.min(train_labels):.4f}")
    print(f"  Max: {np.max(train_labels):.4f}")

    # Create PyTorch datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = PygStructureDataset(
        df_train,
        graphs_train,
        target="target",
        atom_features=atom_features,
        line_graph=True,
        id_tag="id",
        classification=classification,
        neighbor_strategy="k-nearest",
        mean_train=mean_train,
        std_train=std_train,
    )

    val_dataset = PygStructureDataset(
        df_val,
        graphs_val,
        target="target",
        atom_features=atom_features,
        line_graph=True,
        id_tag="id",
        classification=classification,
        neighbor_strategy="k-nearest",
        mean_train=mean_train,
        std_train=std_train,
    )

    test_dataset = PygStructureDataset(
        df_test,
        graphs_test,
        target="target",
        atom_features=atom_features,
        line_graph=True,
        id_tag="id",
        classification=classification,
        neighbor_strategy="k-nearest",
        mean_train=mean_train,
        std_train=std_train,
    )

    # Create data loaders
    collate_fn = train_dataset.collate_line_graph

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Prepare config for training
    config = {
        "dataset": "custom",
        "target": "target",
        "epochs": n_epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "criterion": "mse" if not classification else "ce",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "write_predictions": True,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "atom_features": atom_features,
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
        "pyg_input": True,
        "use_lattice": use_lattice,
        "use_angle": use_angle,
        "neighbor_strategy": "k-nearest",
        "output_dir": output_dir,
        "id_tag": "id",
        "model": {
            "name": model_name,
            "use_angle": use_angle,
        },
    }

    if random_seed is not None:
        config["random_seed"] = random_seed

    # Add any additional kwargs to config
    config.update(kwargs)

    # Prepare custom loaders to pass to train_main
    train_val_test_loaders = [
        train_loader,
        val_loader,
        test_loader,
        train_dataset.prepare_batch,
        mean_train,
        std_train,
    ]

    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    result = train_main(
        config,
        train_val_test_loaders=train_val_test_loaders,
        use_save=True
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    # Extract final MAE values from history
    # result is a dict with structure: {"train": {"mae": [...]}, "validation": {"mae": [...]}}
    train_mae = result['train'].get('mae', [])
    val_mae = result['validation'].get('mae', [])

    if train_mae:
        print(f"Train MAE (final): {train_mae[-1]:.4f}")
        print(f"Train MAE (best): {min(train_mae):.4f}")
    if val_mae:
        print(f"Validation MAE (final): {val_mae[-1]:.4f}")
        print(f"Validation MAE (best): {min(val_mae):.4f}")

    print(f"Results saved to: {output_dir}")

    # Return summary with best metrics for easier access
    summary = {
        'history': result,
        'train_mae_final': train_mae[-1] if train_mae else None,
        'train_mae_best': min(train_mae) if train_mae else None,
        'val_mae_final': val_mae[-1] if val_mae else None,
        'val_mae_best': min(val_mae) if val_mae else None,
    }

    return summary


# Example usage function
def example_usage():
    """Example of how to use the custom training interface."""
    from pymatgen.core import Lattice, Structure

    # Create some example structures
    print("Creating example structures...")

    # Example 1: Simple cubic structure
    lattice1 = Lattice.cubic(4.0)
    strucs = [
        Structure(lattice1, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(lattice1, ["Cu", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.2), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]

    # Example labels (e.g., formation energies)
    labels = [-2.5, -1.8, -2.1]

    # Train the model
    results = train_custom_icomformer(
        strucs=strucs,
        labels=labels,
        learning_rate=0.001,
        batch_size=2,
        n_epochs=10,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        output_dir="./test_output",
    )

    return results


if __name__ == "__main__":
    # Run example if executed directly
    print("Running example usage...")
    example_usage()
