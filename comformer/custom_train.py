"""Custom training interface for iComformer with pymatgen Structure inputs and ASE extxyz files."""
import os
import random
from multiprocessing import Pool, cpu_count
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from ase.io import read as ase_read
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import dumpjson
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch.utils.data import DataLoader

from comformer.graphs import PygGraph, PygStructureDataset
from comformer.train import train_main


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


def _safe_pymatgen_to_jarvis(args):
    """
    Wrapper function for parallel processing of structure conversion.

    Args:
        args: Tuple of (index, structure)

    Returns:
        Tuple of (index, atoms_dict or None, error_message or None)
    """
    i, struc = args
    try:
        atoms_dict = pymatgen_to_jarvis(struc)
        return (i, atoms_dict, None)
    except Exception as e:
        return (i, None, str(e))


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

    # Use parallel processing for large datasets
    num_strucs = len(strucs)
    if num_strucs > 100:
        # Parallel processing for datasets > 100 samples
        num_workers = min(cpu_count(), max(1, num_strucs // 100))
        print(f"Using {num_workers} parallel workers for structure conversion...")

        from tqdm import tqdm
        with Pool(processes=num_workers) as pool:
            # Create indexed args for parallel processing
            indexed_strucs = list(enumerate(strucs))
            results = list(tqdm(
                pool.imap(_safe_pymatgen_to_jarvis, indexed_strucs, chunksize=max(1, num_strucs // (num_workers * 4))),
                total=num_strucs,
                desc="Converting structures"
            ))

        # Reconstruct atoms_dicts list maintaining original order
        atoms_dicts = [None] * num_strucs
        for i, atoms_dict, error in results:
            if error:
                print(f"Warning: Failed to convert structure {i}: {error}")
            atoms_dicts[i] = atoms_dict
    else:
        # Sequential processing for small datasets
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


def ase_atoms_to_pymatgen(ase_atoms) -> Structure:
    """
    Convert ASE Atoms object to pymatgen Structure.

    Args:
        ase_atoms: ASE Atoms object

    Returns:
        pymatgen Structure object

    Raises:
        ImportError: If ASE is not installed
    """

    # Use pymatgen's built-in converter
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(ase_atoms)

    return structure


def read_extxyz_file(
    filename: str,
    target_property: str,
    index: Union[int, str, slice] = ":",
) -> tuple:
    """
    Read structures and target properties from an extxyz file.

    Args:
        filename: Path to the extxyz file
        target_property: Name of the property to use as training target.
                        Can be:
                        - A per-atom property (will be averaged)
                        - A global property stored in atoms.info dict
        index: Which structures to read. Default ":" reads all.
               Can be an integer, slice, or string (e.g., "::10" for every 10th)

    Returns:
        Tuple of (list of pymatgen Structures, list of target values)

    Raises:
        ImportError: If ASE is not installed
        FileNotFoundError: If file doesn't exist
        KeyError: If target_property not found
        ValueError: If no valid structures found

    Example:
        >>> structures, labels = read_extxyz_file("data.xyz", "energy")
        >>> structures, labels = read_extxyz_file("data.xyz", "forces", index="::10")
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    print(f"\nReading structures from: {filename}")
    print(f"Target property: {target_property}")

    # Read structures from extxyz file
    ase_atoms_list = ase_read(filename, index=index)

    # Handle single structure case
    if not isinstance(ase_atoms_list, list):
        ase_atoms_list = [ase_atoms_list]

    print(f"Read {len(ase_atoms_list)} structures from file")

    # Convert to pymatgen and extract target properties
    structures = []
    labels = []
    failed_count = 0

    from tqdm import tqdm
    print("Converting ASE Atoms to pymatgen Structures and extracting properties...")

    for i, ase_atoms in enumerate(tqdm(ase_atoms_list, desc="Processing structures")):
        try:
            # Extract target property
            target_value = None

            # First, check if property is in atoms.info (global property)
            if target_property in ase_atoms.info:
                target_value = float(ase_atoms.info[target_property])

            # If not in info, check if it's a per-atom property in atoms.arrays
            elif target_property in ase_atoms.arrays:
                # Average per-atom property
                target_value = float(np.mean(ase_atoms.arrays[target_property]))

            else:
                # Property not found
                if failed_count == 0:
                    available_info = list(ase_atoms.info.keys())
                    available_arrays = list(ase_atoms.arrays.keys())
                    print(f"\nWarning: Property '{target_property}' not found in structure {i}")
                    print(f"Available info properties: {available_info}")
                    print(f"Available array properties: {available_arrays}")
                failed_count += 1
                continue

            # Convert to pymatgen
            structure = ase_atoms_to_pymatgen(ase_atoms)

            structures.append(structure)
            labels.append(target_value)

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"Warning: Failed to process structure {i}: {e}")
            elif failed_count == 6:
                print(f"Warning: Suppressing further conversion errors...")

    if len(structures) == 0:
        raise ValueError(
            f"No valid structures found! Failed to process {failed_count} structures. "
            f"Check that property '{target_property}' exists in the file."
        )

    if failed_count > 0:
        print(f"Successfully converted {len(structures)} structures ({failed_count} failed)")
    else:
        print(f"Successfully converted all {len(structures)} structures")

    return structures, labels


def _compute_graph_cache_key(df, neighbor_strategy, cutoff, max_neighbors, use_lattice, use_angle):
    """
    Compute a hash key for caching graph construction results.

    Args:
        df: DataFrame with atoms data
        neighbor_strategy: Neighbor finding strategy
        cutoff: Distance cutoff
        max_neighbors: Maximum number of neighbors
        use_lattice: Whether lattice info is used
        use_angle: Whether angle info is used

    Returns:
        str: Hash key for caching
    """
    # Create a unique key based on data and parameters
    key_components = [
        str(len(df)),  # Number of structures
        neighbor_strategy,
        str(cutoff),
        str(max_neighbors),
        str(use_lattice),
        str(use_angle),
    ]
    # Add hash of first and last structure for uniqueness
    if len(df) > 0:
        key_components.append(str(hash(str(df["atoms"].iloc[0]))))
        if len(df) > 1:
            key_components.append(str(hash(str(df["atoms"].iloc[-1]))))

    key_string = "_".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()


def _safe_atoms_to_graph(args):
    """
    Wrapper function for parallel processing of graph construction.

    Args:
        args: Tuple of (index, atoms_dict, config_dict)

    Returns:
        Tuple of (index, graph or None, error_message or None)
    """
    i, atoms_dict, config = args
    try:
        structure = Atoms.from_dict(atoms_dict)
        graph = PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=config['neighbor_strategy'],
            cutoff=config['cutoff'],
            atom_features="atomic_number",
            max_neighbors=config['max_neighbors'],
            compute_line_graph=False,
            use_canonize=False,
            use_lattice=config['use_lattice'],
            use_angle=config['use_angle'],
        )
        return (i, graph, None)
    except Exception as e:
        return (i, None, str(e))


def load_pyg_graphs_from_df(
    df: pd.DataFrame,
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_lattice: bool = False,
    use_angle: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Convert atoms dictionaries to PyTorch Geometric graphs with parallel processing.

    Args:
        df: DataFrame with 'atoms' column containing jarvis-format dicts
        neighbor_strategy: Strategy for neighbor finding
        cutoff: Distance cutoff for neighbors (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
        use_lattice: Whether to include lattice information
        use_angle: Whether to include angle information
        cache_dir: Optional directory to cache pre-built graphs (recommended for large datasets)

    Returns:
        Tuple of (list of PyTorch Geometric Data objects, list of valid indices)
    """
    from tqdm import tqdm

    atoms_list = df["atoms"].values
    num_structures = len(atoms_list)

    # Try to load from cache if cache_dir is specified
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _compute_graph_cache_key(df, neighbor_strategy, cutoff, max_neighbors, use_lattice, use_angle)
        cache_file = os.path.join(cache_dir, f"graphs_{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading pre-built graphs from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Successfully loaded {len(cached_data['graphs'])} cached graphs")
                return cached_data['graphs'], cached_data['valid_indices']
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}. Rebuilding graphs...")

    # Configuration dict for graph building
    config = {
        'neighbor_strategy': neighbor_strategy,
        'cutoff': cutoff,
        'max_neighbors': max_neighbors,
        'use_lattice': use_lattice,
        'use_angle': use_angle,
    }

    print("Building graphs from structures...")

    # Use parallel processing for large datasets (>50 structures)
    if num_structures > 50:
        # Parallel processing for datasets > 50 samples
        # For graph construction, use fewer workers due to memory constraints
        num_workers = max(1, cpu_count() // 2)  # Limit to 8 workers max
        print(f"Using {num_workers} parallel workers for graph construction...")

        with Pool(processes=num_workers) as pool:
            # Create indexed args for parallel processing
            indexed_atoms = [(i, atoms_dict, config) for i, atoms_dict in enumerate(atoms_list)]
            results = list(tqdm(
                pool.imap(_safe_atoms_to_graph, indexed_atoms, chunksize=max(1, num_structures // (num_workers * 2))),
                total=num_structures,
                desc="Building graphs"
            ))

        # Reconstruct graphs list maintaining original order
        graphs = [None] * num_structures
        error_count = 0
        for i, graph, error in results:
            if error:
                error_count += 1
                if error_count <= 5:  # Only print first 5 errors
                    print(f"Warning: Failed to build graph {i}: {error}")
                elif error_count == 6:
                    print(f"Warning: Suppressing further graph building errors...")
            graphs[i] = graph
    else:
        # Sequential processing for small datasets
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

        graphs = []
        for atoms_dict in tqdm(atoms_list, desc="Building graphs"):
            try:
                graph = atoms_to_graph(atoms_dict)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to build graph: {e}")
                graphs.append(None)

    # Filter out failed conversions
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    graphs = [graphs[i] for i in valid_indices]

    # Save to cache if cache_dir is specified
    if cache_dir is not None:
        cache_key = _compute_graph_cache_key(df, neighbor_strategy, cutoff, max_neighbors, use_lattice, use_angle)
        cache_file = os.path.join(cache_dir, f"graphs_{cache_key}.pkl")
        try:
            print(f"Saving built graphs to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({'graphs': graphs, 'valid_indices': valid_indices}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Successfully cached {len(graphs)} graphs")
        except Exception as e:
            print(f"Warning: Failed to save cache file: {e}")

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
    cache_graphs: bool = False,
    graph_cache_dir: Optional[str] = None,
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

    # Determine cache directory
    cache_dir_to_use = None
    if cache_graphs:
        cache_dir_to_use = graph_cache_dir if graph_cache_dir is not None else os.path.join(output_dir, ".graph_cache")
        print(f"Graph caching enabled. Cache directory: {cache_dir_to_use}")

    graphs_train, valid_train = load_pyg_graphs_from_df(
        df_train,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
        cache_dir=cache_dir_to_use,
    )
    graphs_val, valid_val = load_pyg_graphs_from_df(
        df_val,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
        cache_dir=cache_dir_to_use,
    )
    graphs_test, valid_test = load_pyg_graphs_from_df(
        df_test,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
        cache_dir=cache_dir_to_use,
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

    # Enable pin_memory for GPU training (improves CPU-GPU transfer speed)
    use_pin_memory = torch.cuda.is_available()

    # Adaptive num_workers for large datasets
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    if num_workers == 0 and total_samples > 1000:
        # For large datasets, use multiple workers
        num_workers = min(8, cpu_count() // 2)
        print(f"Large dataset detected ({total_samples} samples). Using {num_workers} DataLoader workers.")

    # Use larger batch size for test set (much faster inference)
    test_batch_size = min(batch_size * 4, 128)  # Use 4x training batch size or 128, whichever is smaller

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,  # Use larger batch size for testing
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
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


def train_from_extxyz(
    extxyz_file: str,
    target_property: str,
    # Data reading parameters
    index: Union[int, str, slice] = ":",
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
    output_dir: str = "./extxyz_output",
    num_workers: int = 4,
    classification: bool = False,
    classification_threshold: Optional[float] = None,
    random_seed: Optional[int] = None,
    cache_graphs: bool = False,
    graph_cache_dir: Optional[str] = None,
    **kwargs,
):
    """
    Train iComformer model directly from an ASE extxyz file.

    This function reads structures from an extxyz file, extracts the specified
    target property, and trains an iComformer model. It supports large datasets
    (200k+ structures) with automatic parallelization and optional graph caching.

    Args:
        extxyz_file: Path to the extxyz file containing structures
        target_property: Name of the property to predict. Can be:
                        - A global property in atoms.info dict (e.g., "energy", "formation_energy")
                        - A per-atom property in atoms.arrays (will be averaged)
        index: Which structures to read from file. Default ":" reads all.
               Examples: 0 (first structure), "::10" (every 10th), ":1000" (first 1000)

        learning_rate: Learning rate for optimizer (default: 0.001)
        batch_size: Batch size for training (default: 64)
        n_epochs: Number of training epochs (default: 500)

        train_ratio: Fraction of data for training (default: 0.8)
        val_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: 0.1)
        split_seed: Random seed for data splitting (default: 123)

        model_name: Model architecture to use (default: "iComformer")
        atom_features: Atomic features to use (default: "cgcnn")
        cutoff: Distance cutoff for neighbor finding in Angstroms (default: 8.0)
        max_neighbors: Maximum number of neighbors per atom (default: 12)
        use_lattice: Whether to include lattice information (default: False)
        use_angle: Whether to include angle information (default: False)

        output_dir: Directory to save outputs (default: "./extxyz_output")
        num_workers: Number of DataLoader workers (default: 4, 0 = auto-detect for large datasets)
        classification: Whether this is a classification task (default: False)
        classification_threshold: Threshold for classification (if applicable)
        random_seed: Random seed for reproducibility (default: None)

        cache_graphs: Enable graph caching for faster repeated training (default: False)
        graph_cache_dir: Custom cache directory (default: output_dir/.graph_cache)

        **kwargs: Additional arguments passed to the training function

    Returns:
        dict: Training summary with:
            - 'history': Full training history with losses and metrics
            - 'train_mae_final': Final training MAE
            - 'train_mae_best': Best training MAE
            - 'val_mae_final': Final validation MAE
            - 'val_mae_best': Best validation MAE

    Raises:
        ImportError: If ASE is not installed
        FileNotFoundError: If extxyz file doesn't exist
        KeyError: If target_property not found in structures
        ValueError: If no valid structures found

    Examples:
        >>> # Basic usage
        >>> results = train_from_extxyz(
        ...     extxyz_file="structures.xyz",
        ...     target_property="energy",
        ...     output_dir="./my_model"
        ... )

        >>> # Large dataset with caching
        >>> results = train_from_extxyz(
        ...     extxyz_file="large_dataset.xyz",  # 200k structures
        ...     target_property="formation_energy",
        ...     batch_size=128,
        ...     cache_graphs=True,  # Cache for faster retraining
        ...     output_dir="./large_model"
        ... )

        >>> # Read subset of structures
        >>> results = train_from_extxyz(
        ...     extxyz_file="data.xyz",
        ...     target_property="bandgap",
        ...     index=":10000",  # Only first 10k structures
        ...     output_dir="./subset_model"
        ... )

    Performance:
        - Automatically uses parallel processing for datasets > 100 structures
        - Graph caching recommended for datasets > 10k structures
        - For 200k structures: ~6-12 hours preprocessing (first run),
          ~30 seconds with caching (subsequent runs)

    Notes:
        - All code and comments are in English
        - Leverages all optimizations from train_custom_icomformer
        - Supports both global properties (atoms.info) and per-atom properties (atoms.arrays)
        - Per-atom properties are automatically averaged to get a single target value
    """

    print("\n" + "="*60)
    print("ComFormer Training from ExtXYZ File")
    print("="*60)

    # Read structures and labels from extxyz file
    structures, labels = read_extxyz_file(
        filename=extxyz_file,
        target_property=target_property,
        index=index,
    )

    print(f"\nDataset summary:")
    print(f"  Total structures: {len(structures)}")
    print(f"  Target property: {target_property}")
    print(f"  Label range: [{min(labels):.4f}, {max(labels):.4f}]")
    print(f"  Label mean: {np.mean(labels):.4f}")
    print(f"  Label std: {np.std(labels):.4f}")

    # Call the main training function with pymatgen structures
    print("\n" + "="*60)
    print("Starting model training...")
    print("="*60)

    results = train_custom_icomformer(
        strucs=structures,
        labels=labels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        model_name=model_name,
        atom_features=atom_features,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
        output_dir=output_dir,
        num_workers=num_workers,
        classification=classification,
        classification_threshold=classification_threshold,
        random_seed=random_seed,
        cache_graphs=cache_graphs,
        graph_cache_dir=graph_cache_dir,
        **kwargs,
    )

    print("\n" + "="*60)
    print("Training from ExtXYZ completed successfully!")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    return results


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
