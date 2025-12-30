#!/usr/bin/env python
"""
Example script for multi-GPU distributed training with ComFormer.

This script demonstrates how to use ComFormer's distributed training capabilities
on multiple GPUs using PyTorch's DistributedDataParallel (DDP).

Requirements:
    - PyTorch >= 2.6
    - Multiple CUDA-capable GPUs
    - ComFormer installed

Usage:
    # Single node, multiple GPUs (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train_multi_gpu.py

    # Multi-node training (e.g., 2 nodes with 4 GPUs each)
    # On node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 train_multi_gpu.py
    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=29500 train_multi_gpu.py

Notes:
    - torchrun automatically sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables
    - Each GPU process will handle a subset of the training data
    - Gradients are synchronized across all GPUs after each backward pass
    - Only rank 0 process saves checkpoints and prints logs to reduce I/O overhead
"""

import os
import sys
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure

# Add parent directory to path to import comformer
sys.path.insert(0, str(Path(__file__).parent.parent))

from comformer import train_from_list


def create_example_dataset(n_samples=1000):
    """
    Create a synthetic dataset for demonstration.

    Args:
        n_samples: Number of structures to generate

    Returns:
        Tuple of (structures, labels)
    """
    print(f"Generating {n_samples} example structures...")

    structures = []
    labels = []

    # Create diverse crystal structures
    elements_list = [
        ["Fe", "O"],
        ["Cu", "O"],
        ["Ni", "O"],
        ["Co", "O"],
        ["Mn", "O"],
        ["Cr", "O"],
        ["Ti", "O"],
        ["V", "O"],
    ]

    for i in range(n_samples):
        # Vary lattice parameter
        a = 3.5 + (i % 20) * 0.05  # 3.5 to 4.5 Angstrom

        lattice = Lattice.cubic(a)
        elements = elements_list[i % len(elements_list)]

        structure = Structure(
            lattice,
            elements,
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )

        # Synthetic target: formation energy as function of lattice parameter
        # This is just for demonstration
        formation_energy = -2.5 + 0.3 * (a - 4.0) + 0.1 * np.random.randn()

        structures.append(structure)
        labels.append(formation_energy)

    print(f"Generated {len(structures)} structures")
    print(f"Label range: [{min(labels):.3f}, {max(labels):.3f}]")

    return structures, labels


def main():
    """Main training function."""

    # Check if distributed environment variables are set
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only print from rank 0 to avoid duplicate messages
    is_main = (rank == 0)

    if is_main:
        print("="*70)
        print("ComFormer Multi-GPU Distributed Training Example")
        print("="*70)
        print(f"World size: {world_size}")
        print(f"Current rank: {rank}")
        print(f"Local rank: {local_rank}")
        print("="*70)

    # Create example dataset
    # Note: All processes create the same dataset (deterministic with seed)
    structures, labels = create_example_dataset(n_samples=1000)

    # Training configuration
    output_dir = "./multi_gpu_training_output"

    if is_main:
        print(f"\nTraining configuration:")
        print(f"  Output directory: {output_dir}")
        print(f"  Batch size per GPU: 32")
        print(f"  Effective batch size: {32 * world_size}")
        print(f"  Epochs: 100")
        print(f"  Learning rate: 0.001")
        print(f"  Model: iComformer")
        print(f"  Cutoff: 6.0 Angstrom")
        print(f"  Max neighbors: 25")
        print("")

    # Train with distributed training enabled
    results = train_from_list(
        strucs=structures,
        labels=labels,
        # Training parameters
        learning_rate=0.001,
        batch_size=32,  # Batch size per GPU
        n_epochs=100,
        # Data split
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        # Model parameters
        model_name="iComformer",
        atom_features="cgcnn",
        cutoff=6.0,
        max_neighbors=25,
        use_lattice=True,
        use_angle=False,
        # Output
        output_dir=output_dir,
        num_workers=4,  # DataLoader workers per GPU
        random_seed=42,
        # Enable distributed training
        distributed=True,
    )

    # Only rank 0 prints final results
    if is_main:
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)

        if results and 'val_mae_best' in results:
            print(f"Best validation MAE: {results['val_mae_best']:.4f}")
            print(f"Final validation MAE: {results['val_mae_final']:.4f}")

        print(f"\nResults saved to: {output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()
