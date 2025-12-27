"""
Example script demonstrating how to use the ComFormer predictor.

This script shows how to:
1. Load a trained model
2. Prepare crystal structures
3. Make predictions
"""

from comformer.predict import load_predictor
from pymatgen.core import Lattice, Structure


def create_example_structures():
    """Create some example crystal structures for testing."""
    structures = []

    # Example 1: Simple cubic structure (e.g., simple metal)
    lattice1 = Lattice.cubic(3.0)
    structure1 = Structure(
        lattice1,
        ["Fe"],
        [[0, 0, 0]],
    )
    structures.append(structure1)

    # Example 2: FCC structure
    lattice2 = Lattice.cubic(4.05)
    structure2 = Structure(
        lattice2,
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    structures.append(structure2)

    # Example 3: Binary compound (e.g., NaCl-like)
    lattice3 = Lattice.cubic(5.64)
    structure3 = Structure(
        lattice3,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    structures.append(structure3)

    return structures


def main():
    """Main prediction workflow."""
    print("=" * 60)
    print("ComFormer Prediction Example")
    print("=" * 60)

    # Step 1: Load trained model
    print("\n[Step 1] Loading trained model...")
    checkpoint_dir = "./test_output"  # Path to your trained model directory

    try:
        predictor = load_predictor(
            checkpoint_dir=checkpoint_dir,
            # checkpoint_name defaults to "best_model.pt"
            device=None,  # Auto-select GPU if available
        )
        print("✓ Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease make sure you have:")
        print("  1. Trained a model first using the custom_train interface")
        print("  2. The file 'best_model.pt' exists in './test_output'")
        return

    # Step 2: Prepare structures
    print("\n[Step 2] Preparing crystal structures...")
    structures = create_example_structures()
    print(f"✓ Created {len(structures)} example structures")

    for i, struct in enumerate(structures, 1):
        composition = struct.composition.reduced_formula
        print(f"  Structure {i}: {composition} ({len(struct)} atoms)")

    # Step 3: Make predictions
    print("\n[Step 3] Making predictions...")
    predictions = predictor.predict(structures)

    # Step 4: Display results
    print("\n[Step 4] Results:")
    print("-" * 60)
    for i, (struct, pred) in enumerate(zip(structures, predictions), 1):
        composition = struct.composition.reduced_formula
        print(f"Structure {i} ({composition}): {pred:.4f}")
    print("-" * 60)

    # Step 5: Single structure prediction
    print("\n[Step 5] Single structure prediction example:")
    single_pred = predictor.predict_single(structures[0])
    print(f"Single prediction: {single_pred:.4f}")

    print("\n" + "=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
