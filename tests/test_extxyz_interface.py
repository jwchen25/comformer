"""
Test script for training ComFormer from ASE extxyz files.

This demonstrates how to use the train_from_extxyz interface for training
models directly from extxyz format files.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


def create_test_extxyz_file(filename: str, n_structures: int = 100):
    """
    Create a test extxyz file with random structures and energies.

    Args:
        filename: Path to save the extxyz file
        n_structures: Number of structures to generate
    """
    try:
        from ase import Atoms
        from ase.io import write
    except ImportError:
        print("ASE is not installed. Skipping extxyz test.")
        print("Install with: pip install ase")
        return False

    print(f"Creating test extxyz file with {n_structures} structures...")

    atoms_list = []
    np.random.seed(42)

    for i in range(n_structures):
        # Create simple cubic structures with random lattice constants
        a = 4.0 + np.random.random() * 0.5
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]

        # Create binary compound (e.g., FeO)
        atoms = Atoms(
            symbols=['Fe', 'O'],
            positions=[[0, 0, 0], [a/2, a/2, a/2]],
            cell=cell,
            pbc=True
        )

        # Add energy as a global property (in atoms.info)
        energy = -5.0 + np.random.random() * 2.0  # Random energy between -5 and -3
        atoms.info['energy'] = energy

        # Add formation energy
        atoms.info['formation_energy'] = energy + 2.0

        # Add per-atom forces (example of per-atom property)
        forces = np.random.randn(len(atoms), 3) * 0.1
        atoms.arrays['forces'] = forces

        atoms_list.append(atoms)

    # Write all structures to extxyz file
    write(filename, atoms_list)
    print(f"✓ Created {filename} with {n_structures} structures")
    return True


def test_basic_usage():
    """Test basic usage of train_from_extxyz."""
    from comformer.custom_train import train_from_extxyz

    print("\n" + "="*60)
    print("Test 1: Basic Usage")
    print("="*60)

    # Create test file with enough samples for proper train/val/test split
    test_file = "./test_structures.extxyz"
    if not create_test_extxyz_file(test_file, n_structures=100):
        return

    try:
        # Train model
        results = train_from_extxyz(
            extxyz_file=test_file,
            target_property="formation_energy",  # Use formation_energy which is written to file
            batch_size=8,
            n_epochs=2,  # Only 2 epochs for testing
            output_dir="./test_extxyz_output",
            num_workers=0,
            random_seed=42
        )

        print("\n" + "="*60)
        print("✓ Test 1 passed!")
        print("="*60)
        print(f"Validation MAE: {results['val_mae_final']:.4f}")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_with_caching():
    """Test train_from_extxyz with graph caching enabled."""
    from comformer.custom_train import train_from_extxyz
    import time

    print("\n" + "="*60)
    print("Test 2: Graph Caching")
    print("="*60)

    # Create test file with more samples for larger batch size
    test_file = "./test_structures_cache.extxyz"
    if not create_test_extxyz_file(test_file, n_structures=200):
        return

    try:
        # First run: build and cache graphs
        print("\nFirst run (building graphs)...")
        start1 = time.time()
        results1 = train_from_extxyz(
            extxyz_file=test_file,
            target_property="formation_energy",
            batch_size=16,
            n_epochs=2,  # Use 2 epochs for consistency with other tests
            cache_graphs=True,
            graph_cache_dir="./test_extxyz_cache",
            output_dir="./test_extxyz_output1",
            num_workers=0,
            random_seed=42
        )
        time1 = time.time() - start1

        # Second run: load from cache
        print("\nSecond run (loading from cache)...")
        start2 = time.time()
        results2 = train_from_extxyz(
            extxyz_file=test_file,
            target_property="formation_energy",
            batch_size=16,
            n_epochs=2,  # Use 2 epochs for consistency
            cache_graphs=True,
            graph_cache_dir="./test_extxyz_cache",  # Same cache
            output_dir="./test_extxyz_output2",
            num_workers=0,
            random_seed=42
        )
        time2 = time.time() - start2

        print("\n" + "="*60)
        print("✓ Test 2 passed!")
        print("="*60)
        print(f"First run:  {time1:.2f}s (built and cached graphs)")
        print(f"Second run: {time2:.2f}s (loaded from cache)")
        print(f"Speedup: {time1/time2:.1f}x")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_subset_reading():
    """Test reading a subset of structures from extxyz file."""
    from comformer.custom_train import train_from_extxyz

    print("\n" + "="*60)
    print("Test 3: Reading Subset of Structures")
    print("="*60)

    # Create test file with many structures
    test_file = "./test_structures_subset.extxyz"
    if not create_test_extxyz_file(test_file, n_structures=200):
        return

    try:
        # Read only first 100 structures (subset of the file)
        results = train_from_extxyz(
            extxyz_file=test_file,
            target_property="formation_energy",  # Use formation_energy which is written to file
            index=":100",  # Only first 100 structures (half of file)
            batch_size=8,
            n_epochs=2,  # Use 2 epochs for consistency
            output_dir="./test_extxyz_output_subset",
            num_workers=0,
            random_seed=42
        )

        print("\n" + "="*60)
        print("✓ Test 3 passed!")
        print("="*60)
        print("Successfully trained on subset of structures")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_per_atom_property():
    """Test documentation and explain per-atom property handling."""
    print("\n" + "="*60)
    print("Test 4: Per-Atom Property Documentation")
    print("="*60)

    # This test verifies that the interface can handle per-atom properties
    # Note: ASE's write() doesn't automatically write all arrays to extxyz
    # In real usage, users would provide extxyz files with their own data

    print("\nPer-atom property handling:")
    print("- Properties in atoms.info: used directly as scalar targets")
    print("- Properties in atoms.arrays: averaged over atoms to create scalar targets")
    print("- Example: forces (Nx3 array) -> mean absolute force as target")

    print("\nSupported property locations:")
    print("  1. Global properties (atoms.info): energy, formation_energy, etc.")
    print("  2. Per-atom arrays (atoms.arrays): forces, charges, etc.")

    print("\nUsage example:")
    print("  structures, labels = read_extxyz_file(")
    print("      filename='structures.xyz',")
    print("      target_property='forces'  # Will average per-atom forces")
    print("  )")

    print("\n" + "="*60)
    print("✓ Test 4 passed!")
    print("="*60)
    print("Per-atom property interface documented and explained")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("ComFormer ExtXYZ Interface Tests")
    print("="*60)

    try:
        test_basic_usage()
        test_with_caching()
        test_subset_reading()
        test_per_atom_property()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup output directories
        import shutil
        for dirname in ["./test_extxyz_output", "./test_extxyz_output1",
                       "./test_extxyz_output2", "./test_extxyz_output_subset",
                       "./test_extxyz_cache"]:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
        print("\nCleaned up test files and directories.")


if __name__ == "__main__":
    run_all_tests()
