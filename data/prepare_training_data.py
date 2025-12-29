#!/usr/bin/env python3
"""
Prepare Materials Project data for use with comformer training.

This script loads the downloaded structures and prepares them for use
with the comformer machine learning pipeline.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import read, write

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess crystal structure data for training."""

    def __init__(self, input_file: Path, output_file: Path = None):
        """
        Initialize data preprocessor.

        Parameters:
        -----------
        input_file : Path
            Path to input extxyz file
        output_file : Path, optional
            Path to output file (default: input_file_processed.extxyz)
        """
        self.input_file = Path(input_file)
        self.output_file = output_file or self.input_file.parent / (
            self.input_file.stem + '_processed.extxyz'
        )

    def load_structures(self) -> List[Atoms]:
        """Load structures from extxyz file."""
        logger.info(f"Loading structures from {self.input_file}...")

        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {self.input_file}")

        structures = read(self.input_file, index=':')

        if not isinstance(structures, list):
            structures = [structures]

        logger.info(f"Loaded {len(structures)} structures")
        return structures

    def validate_structure(self, atoms: Atoms) -> bool:
        """
        Validate structure for training.

        Parameters:
        -----------
        atoms : Atoms
            Structure to validate

        Returns:
        --------
        bool
            True if structure is valid
        """
        # Check for valid geometry
        if atoms.get_cell().volume < 1.0:
            logger.warning(f"Invalid cell volume: {atoms.get_cell().volume}")
            return False

        # Check for valid band gap
        bg = atoms.info.get('band_gap', None)
        if bg is None or bg < 0:
            logger.warning("No valid band gap found")
            return False

        # Check for reasonable number of atoms
        if len(atoms) < 1 or len(atoms) > 1000:
            logger.warning(f"Unusual number of atoms: {len(atoms)}")
            return False

        return True

    def normalize_structures(self, structures: List[Atoms]) -> List[Atoms]:
        """
        Normalize structures for training.

        Parameters:
        -----------
        structures : List[Atoms]
            List of structures

        Returns:
        --------
        List[Atoms]
            Normalized structures
        """
        logger.info("Normalizing structures...")

        normalized = []

        for atoms in structures:
            # Ensure periodic boundary conditions
            atoms.set_pbc(True)

            # Wrap atoms into unit cell
            atoms.wrap()

            normalized.append(atoms)

        return normalized

    def compute_statistics(self, structures: List[Atoms]) -> dict:
        """
        Compute statistics on dataset.

        Parameters:
        -----------
        structures : List[Atoms]
            List of structures

        Returns:
        --------
        dict
            Dictionary of statistics
        """
        logger.info("Computing dataset statistics...")

        band_gaps = []
        natoms_list = []
        volumes = []

        for atoms in structures:
            bg = atoms.info.get('band_gap', None)
            if bg is not None:
                band_gaps.append(bg)

            natoms_list.append(len(atoms))
            volumes.append(atoms.get_cell().volume)

        stats = {
            'total_structures': len(structures),
            'band_gap': {
                'min': float(np.min(band_gaps)),
                'max': float(np.max(band_gaps)),
                'mean': float(np.mean(band_gaps)),
                'median': float(np.median(band_gaps)),
                'std': float(np.std(band_gaps)),
            },
            'atoms_per_structure': {
                'min': int(np.min(natoms_list)),
                'max': int(np.max(natoms_list)),
                'mean': float(np.mean(natoms_list)),
            },
            'cell_volume': {
                'min': float(np.min(volumes)),
                'max': float(np.max(volumes)),
                'mean': float(np.mean(volumes)),
            },
        }

        return stats

    def filter_structures(
        self,
        structures: List[Atoms],
        min_bandgap: float = 0.0,
        max_bandgap: float = float('inf'),
        max_atoms: int = 1000,
    ) -> Tuple[List[Atoms], int]:
        """
        Filter structures based on criteria.

        Parameters:
        -----------
        structures : List[Atoms]
            Input structures
        min_bandgap : float
            Minimum band gap (eV)
        max_bandgap : float
            Maximum band gap (eV)
        max_atoms : int
            Maximum number of atoms

        Returns:
        --------
        tuple
            (filtered_structures, number_removed)
        """
        logger.info(f"Filtering structures...")
        logger.info(f"  Band gap range: {min_bandgap} - {max_bandgap} eV")
        logger.info(f"  Max atoms: {max_atoms}")

        filtered = []
        removed = 0

        for atoms in structures:
            bg = atoms.info.get('band_gap', None)

            if bg is None:
                removed += 1
                continue

            if not (min_bandgap <= bg <= max_bandgap):
                removed += 1
                continue

            if len(atoms) > max_atoms:
                removed += 1
                continue

            filtered.append(atoms)

        logger.info(f"Kept {len(filtered)}/{len(structures)} structures "
                   f"({100*len(filtered)/len(structures):.1f}%)")

        return filtered, removed

    def split_dataset(
        self,
        structures: List[Atoms],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Atoms], List[Atoms], List[Atoms]]:
        """
        Split dataset into train/val/test sets.

        Parameters:
        -----------
        structures : List[Atoms]
            Input structures
        train_ratio : float
            Training set ratio
        val_ratio : float
            Validation set ratio
        test_ratio : float
            Test set ratio
        seed : int
            Random seed

        Returns:
        --------
        tuple
            (train_structures, val_structures, test_structures)
        """
        logger.info(f"Splitting dataset (train/val/test = "
                   f"{train_ratio}/{val_ratio}/{test_ratio})")

        np.random.seed(seed)

        n = len(structures)
        indices = np.arange(n)
        np.random.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train = [structures[i] for i in train_idx]
        val = [structures[i] for i in val_idx]
        test = [structures[i] for i in test_idx]

        logger.info(f"Train set: {len(train)} structures")
        logger.info(f"Validation set: {len(val)} structures")
        logger.info(f"Test set: {len(test)} structures")

        return train, val, test

    def process(
        self,
        output_splits: bool = False,
        min_bandgap: float = 0.0,
        max_bandgap: float = float('inf'),
        max_atoms: int = 1000,
        validate: bool = True,
    ):
        """
        Process dataset.

        Parameters:
        -----------
        output_splits : bool
            Whether to output train/val/test splits
        min_bandgap : float
            Minimum band gap filter
        max_bandgap : float
            Maximum band gap filter
        max_atoms : int
            Maximum atoms filter
        validate : bool
            Whether to validate structures
        """
        try:
            # Load
            structures = self.load_structures()

            # Validate
            if validate:
                logger.info("Validating structures...")
                valid_structures = []
                for atoms in structures:
                    if self.validate_structure(atoms):
                        valid_structures.append(atoms)

                logger.info(f"Valid structures: {len(valid_structures)}/"
                           f"{len(structures)}")
                structures = valid_structures

            # Normalize
            structures = self.normalize_structures(structures)

            # Filter
            structures, n_removed = self.filter_structures(
                structures,
                min_bandgap=min_bandgap,
                max_bandgap=max_bandgap,
                max_atoms=max_atoms,
            )

            # Statistics
            stats = self.compute_statistics(structures)
            self._print_statistics(stats)

            # Save processed data
            logger.info(f"Writing processed data to {self.output_file}...")
            write(self.output_file, structures, format='extxyz')

            # Split dataset
            if output_splits:
                self._save_splits(structures)

            logger.info("Processing completed!")

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

    def _print_statistics(self, stats: dict):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total structures: {stats['total_structures']}")
        print(f"\nBand Gap (eV):")
        for key, value in stats['band_gap'].items():
            print(f"  {key:6s}: {value:8.4f}")
        print(f"\nAtoms per structure:")
        for key, value in stats['atoms_per_structure'].items():
            print(f"  {key:6s}: {value:8.2f}")
        print(f"\nCell Volume (Å³):")
        for key, value in stats['cell_volume'].items():
            print(f"  {key:6s}: {value:8.2f}")
        print(f"{'='*60}\n")

    def _save_splits(self, structures: List[Atoms]):
        """Save train/val/test splits."""
        train, val, test = self.split_dataset(structures)

        output_dir = self.output_file.parent

        # Save splits
        train_file = output_dir / (self.output_file.stem + '_train.extxyz')
        val_file = output_dir / (self.output_file.stem + '_val.extxyz')
        test_file = output_dir / (self.output_file.stem + '_test.extxyz')

        write(train_file, train, format='extxyz')
        write(val_file, val, format='extxyz')
        write(test_file, test, format='extxyz')

        logger.info(f"Saved splits:")
        logger.info(f"  Train: {train_file}")
        logger.info(f"  Val:   {val_file}")
        logger.info(f"  Test:  {test_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Prepare Materials Project data for comformer training'
    )
    parser.add_argument(
        'input',
        help='Input extxyz file'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output file (default: input_processed.extxyz)'
    )
    parser.add_argument(
        '--splits',
        action='store_true',
        help='Output train/val/test splits'
    )
    parser.add_argument(
        '--min-bandgap',
        type=float,
        default=0.0,
        help='Minimum band gap (eV)'
    )
    parser.add_argument(
        '--max-bandgap',
        type=float,
        default=float('inf'),
        help='Maximum band gap (eV)'
    )
    parser.add_argument(
        '--max-atoms',
        type=int,
        default=1000,
        help='Maximum atoms per structure'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation'
    )

    args = parser.parse_args()

    try:
        preprocessor = DataPreprocessor(args.input, args.output)
        preprocessor.process(
            output_splits=args.splits,
            min_bandgap=args.min_bandgap,
            max_bandgap=args.max_bandgap,
            max_atoms=args.max_atoms,
            validate=not args.no_validate,
        )
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
