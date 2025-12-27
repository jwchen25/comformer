#!/usr/bin/env python
"""Test script for custom training interface."""

from pymatgen.core import Lattice, Structure
from comformer.custom_train import train_custom_icomformer, pymatgen_to_jarvis
import numpy as np


def test_structure_conversion():
    """Test pymatgen to jarvis conversion."""
    print("="*60)
    print("Testing structure conversion...")
    print("="*60)

    # Create a simple structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Convert to jarvis format
    atoms_dict = pymatgen_to_jarvis(structure)

    print(f"Original structure: {structure}")
    print(f"\nJarvis dictionary:")
    print(f"  Elements: {atoms_dict['elements']}")
    print(f"  Coordinates: {atoms_dict['coords']}")
    print(f"  Lattice parameters: {atoms_dict['abc']}")
    print(f"  Lattice angles: {atoms_dict['angles']}")
    print("\n✓ Structure conversion successful!\n")


def test_small_dataset():
    """Test training on a small synthetic dataset."""
    print("="*60)
    print("Testing small dataset training...")
    print("="*60)

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 50

    strucs = []
    labels = []

    for i in range(n_samples):
        # Vary lattice parameter slightly
        a = 3.5 + 1.0 * np.random.random()
        lattice = Lattice.cubic(a)

        # Create structure
        if i % 2 == 0:
            struc = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            label = -2.5 + 0.1 * a  # Synthetic formation energy
        else:
            struc = Structure(lattice, ["Cu", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            label = -1.8 + 0.15 * a

        strucs.append(struc)
        labels.append(label)

    print(f"\nCreated {n_samples} synthetic structures")
    print(f"Label range: [{min(labels):.2f}, {max(labels):.2f}]")

    # Train model with minimal epochs for testing
    print("\nStarting training (test mode, only 5 epochs)...\n")

    try:
        results = train_custom_icomformer(
            strucs=strucs,
            labels=labels,
            learning_rate=0.001,
            batch_size=8,
            n_epochs=5,  # Very few epochs for testing
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            cutoff=6.0,
            max_neighbors=10,
            use_lattice=False,
            use_angle=False,
            output_dir="./test_output",
            num_workers=0,  # Avoid multiprocessing issues
            random_seed=42
        )

        print("\n" + "="*60)
        print("Testing completed!")
        print("="*60)
        if results.get('train_mae_final') is not None:
            print(f"✓ Training MAE (final): {results['train_mae_final']:.4f}")
        if results.get('val_mae_final') is not None:
            print(f"✓ Validation MAE (final): {results['val_mae_final']:.4f}")
        print("\nAll tests passed! ✓")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_various_structures():
    """Test with various crystal structures."""
    print("="*60)
    print("Testing various crystal structure types...")
    print("="*60)

    strucs = []
    labels = []

    # 1. Simple cubic
    lattice1 = Lattice.cubic(4.0)
    strucs.append(Structure(lattice1, ["Fe"], [[0, 0, 0]]))
    labels.append(-1.5)

    # 2. FCC-like
    lattice2 = Lattice.cubic(4.2)
    strucs.append(Structure(lattice2, ["Cu", "Cu", "Cu", "Cu"],
                           [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]))
    labels.append(-2.0)

    # 3. Binary compound
    lattice3 = Lattice.cubic(5.0)
    strucs.append(Structure(lattice3, ["Ti", "O", "O"],
                           [[0, 0, 0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]]))
    labels.append(-3.2)

    # 4. Orthorhombic
    lattice4 = Lattice.orthorhombic(4.0, 5.0, 6.0)
    strucs.append(Structure(lattice4, ["Si", "O", "O"],
                           [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]))
    labels.append(-2.8)

    # 5. Hexagonal
    lattice5 = Lattice.hexagonal(3.0, 5.0)
    strucs.append(Structure(lattice5, ["C", "C"],
                           [[0, 0, 0], [0.333, 0.667, 0.5]]))
    labels.append(-1.0)

    # Duplicate to have enough data
    strucs = strucs * 10
    labels = labels * 10

    print(f"Created {len(strucs)} structures of different types")
    print("Structure types:")
    print("  - Simple cubic")
    print("  - FCC structure")
    print("  - Binary compound")
    print("  - Orthorhombic")
    print("  - Hexagonal")

    print("\n✓ Structure creation successful!")
    print(f"Ready to train {len(strucs)} samples...\n")

    return strucs, labels


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ComFormer Custom Dataset Interface Test")
    print("="*60 + "\n")

    # Test 1: Structure conversion
    test_structure_conversion()

    # Test 2: Various structures
    strucs, labels = test_various_structures()

    # Test 3: Small dataset training
    success = test_small_dataset()

    if success:
        print("\n" + "="*60)
        print("All tests passed! Interface is working correctly.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Tests failed, please check error messages.")
        print("="*60 + "\n")
