#!/usr/bin/env python3
"""
Example script showing how to load and use the Materials Project data.

This demonstrates reading the extxyz file and performing simple analysis
on the structures and band gap data.
"""

import numpy as np
from pathlib import Path
from ase.io import read


def load_structures(filepath):
    """
    Load structures from extxyz file.

    Parameters:
    -----------
    filepath : str or Path
        Path to the extxyz file

    Returns:
    --------
    list of ase.Atoms
        List of atomic structures with associated properties
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading structures from {filepath}...")
    structures = read(filepath, index=':')

    # Ensure it's a list
    if not isinstance(structures, list):
        structures = [structures]

    return structures


def analyze_structures(structures):
    """
    Perform basic analysis on structures.

    Parameters:
    -----------
    structures : list of ase.Atoms
        List of atomic structures
    """

    print(f"\n{'='*60}")
    print(f"Total structures: {len(structures)}")
    print(f"{'='*60}\n")

    band_gaps = []
    direct_gap_count = 0
    formulas = set()

    for i, atoms in enumerate(structures):
        bg = atoms.info.get('band_gap', None)
        if bg is not None:
            band_gaps.append(bg)

        if atoms.info.get('is_gap_direct', False):
            direct_gap_count += 1

        formula = atoms.info.get('formula', '')
        if formula:
            formulas.add(formula)

    # Statistics
    if band_gaps:
        band_gaps = np.array(band_gaps)
        print("Band Gap Statistics:")
        print(f"  Minimum: {band_gaps.min():.4f} eV")
        print(f"  Maximum: {band_gaps.max():.4f} eV")
        print(f"  Mean:    {band_gaps.mean():.4f} eV")
        print(f"  Median:  {np.median(band_gaps):.4f} eV")
        print(f"  Std Dev: {band_gaps.std():.4f} eV")
        print(f"  Direct gap ratio: {direct_gap_count}/{len(structures)} "
              f"({100*direct_gap_count/len(structures):.1f}%)")
        print(f"  Unique formulas: {len(formulas)}")

        # Distribution
        print("\nBand Gap Distribution:")
        bins = np.arange(0, band_gaps.max() + 0.5, 0.5)
        hist, _ = np.histogram(band_gaps, bins=bins)
        for i in range(len(hist)):
            bar = '█' * int(hist[i] / max(hist) * 40)
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}] eV: {bar} ({int(hist[i])})")

    print()


def filter_by_bandgap(structures, min_gap=0, max_gap=float('inf')):
    """
    Filter structures by band gap range.

    Parameters:
    -----------
    structures : list of ase.Atoms
        List of atomic structures
    min_gap : float
        Minimum band gap (eV)
    max_gap : float
        Maximum band gap (eV)

    Returns:
    --------
    list of ase.Atoms
        Filtered structures
    """
    filtered = []

    for atoms in structures:
        bg = atoms.info.get('band_gap', None)
        if bg is not None and min_gap <= bg <= max_gap:
            filtered.append(atoms)

    return filtered


def print_sample_structures(structures, n_samples=5):
    """
    Print information about sample structures.

    Parameters:
    -----------
    structures : list of ase.Atoms
        List of atomic structures
    n_samples : int
        Number of samples to print
    """

    print(f"\nSample Structures (first {min(n_samples, len(structures))}):")
    print(f"{'-'*60}")

    for i, atoms in enumerate(structures[:n_samples]):
        print(f"\n{i+1}. Material ID: {atoms.info.get('material_id', 'N/A')}")
        print(f"   Formula:      {atoms.info.get('formula', 'N/A')}")
        print(f"   Band Gap:     {atoms.info.get('band_gap', 'N/A'):.4f} eV")
        print(f"   Direct Gap:   {atoms.info.get('is_gap_direct', False)}")
        print(f"   Atoms:        {len(atoms)}")
        print(f"   Cell:         {atoms.get_cell().lengths()}")


def main():
    """Main function."""

    # Define data file path
    data_file = Path(__file__).parent / 'materials_project.extxyz'

    # Check if file exists
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print(f"Please run 'python download_mp_data.py' first to download the data.")
        return 1

    try:
        # Load structures
        structures = load_structures(data_file)

        # Analyze structures
        analyze_structures(structures)

        # Print sample structures
        print_sample_structures(structures)

        # Example: Filter by band gap
        print(f"\n{'='*60}")
        print("Semiconductors with band gap between 1.0 and 3.0 eV:")
        print(f"{'='*60}")
        semiconductors = filter_by_bandgap(structures, 1.0, 3.0)
        print(f"Found {len(semiconductors)} semiconductors")

        if semiconductors:
            print("\nFirst few semiconductors:")
            print_sample_structures(semiconductors, n_samples=3)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
