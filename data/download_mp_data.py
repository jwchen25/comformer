"""
Download crystal structures and band gap data from Materials Project.

This script fetches all structures with computed band gap values from the
Materials Project database and saves them to an ASE extxyz file.

Requirements:
    - pymatgen>=2024.1.1
    - mp-api>=0.35.0
    - ase>=3.23.0
    - Materials Project API key (set as MP_API_KEY environment variable)

Usage:
    python download_mp_data.py [--output output.extxyz] [--target band_gap]
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


TARGET_ARGS = {
    'band_gap': ['band_gap', (-0.01, 10)],
    'bulk_modulus': ['k_vrh', (0, 500)],
    'shear_modulus': ['g_vrh', (0, 500)],
    'eform': ['formation_energy', (-6, 6)],
    'ehull': ['energy_above_hull', (-0.2, 0.5)],
    'total_dielectric_constant': ['e_total', (0, 120)],
    'young_modulus': ['k_vrh', (0, 1000)],
    'total_mag_per_atom': ['total_magnetization_normalized_vol', (-0.01, 0.25)],
}

DEFAULT_ARGS = {
    'deprecated': False,
    'fields': ['formula_pretty', 'volume', 'density', 'symmetry', 'material_id', 'structure', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy'],
}

PROP_DICT = {
    'band_gap': 'band_gap',
    'eform': 'formation_energy_per_atom',
    'ehull': 'energy_above_hull',
    'total_dielectric_constant': 'e_total',
    'total_mag_per_atom': 'total_magnetization',
}


def download_mp_structures_with_bandgap(api_key: str = None, target: str = 'band_gap') -> List[Dict[str, Any]]:
    """
    Download crystal structures with band gap from Materials Project.

    Parameters:
    -----------
    api_key : str, optional
        Materials Project API key. If None, tries to read from MP_API_KEY
        environment variable.

    Returns:
    --------
    list of dict
        List containing structure data with keys:
        - 'structure': pymatgen.Structure object
        - 'material_id': str
        - 'band_gap': float (eV)
        - 'is_gap_direct': bool
        - 'formula_pretty': str
    """

    # Get API key
    if api_key is None:
        api_key = os.environ.get('MP_API_KEY')

    if not api_key:
        raise ValueError(
            "Materials Project API key not found. "
            "Please set MP_API_KEY environment variable or pass it as argument."
        )

    logger.info("Connecting to Materials Project using mp-api...")
    query_args = {
        TARGET_ARGS[target][0]: TARGET_ARGS[target][1],
    }
    query_args.update(DEFAULT_ARGS)

    with MPRester(api_key="VrfB6q2B47s34fzkWcNXu9pHAVJUPRHP") as mpr:
        docs = mpr.materials.summary.search(**query_args)

    results = []
    for doc in docs:
        if doc.structure is None:
            continue

        if target == 'bulk_modulus':
            prop = doc.bulk_modulus['vrh']
        elif target == 'shear_modulus':
            prop = doc.shear_modulus['vrh']
        elif target == 'total_mag_per_atom':
            prop = doc.total_magnetization / len(doc.structure)
        else:
            prop = doc.get(PROP_DICT[target], None)

        if isinstance(prop, float):
            results.append(
                {
                    'mpid': str(doc.material_id),
                    'structure': doc.structure,
                    'formula_pretty': doc.formula_pretty,
                    'target_name': target,
                    'target_value': prop,
                }
            )

    logger.info(f"Downloaded {len(results)} structures with target data")
    return results


def save_to_extxyz(structures_data: List[dict], output_file: str):
    """
    Save structures with band gap information to ASE extxyz file.

    Parameters:
    -----------
    structures_data : list of dict
        List of structure data from download_mp_structures_with_bandgap
    output_file : str or Path
        Output file path
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {len(structures_data)} structures...")

    atoms_list = []

    for i, data in enumerate(structures_data):
        try:
            pmg_structure = data['structure']

            # Convert to ASE Atoms
            atoms = AseAtomsAdaptor.get_atoms(pmg_structure)

            # Store band gap and other properties
            # extxyz format supports key=value pairs in the comment line
            atoms.info['mpid'] = data['mpid']
            atoms.info['formula'] = data.get('formula_pretty', '')
            atoms.info[data['target_name']] = float(data['target_value'])

            atoms_list.append(atoms)

        except Exception as e:
            logger.warning(
                f"Failed to convert {data.get('mpid', 'unknown')}: {e}"
            )
            continue

    # Write to extxyz format
    # ASE's extxyz writer automatically includes info dict as key=value pairs
    write(output_file, atoms_list, format='extxyz', append=False)

    logger.info(f"Successfully saved {len(atoms_list)} structures to {output_file}")

    return len(atoms_list)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download Materials Project structures with band gap data'
    )
    parser.add_argument(
        '--output',
        default='materials_project.extxyz',
        help='Output extxyz file path (default: materials_project.extxyz)'
    )
    parser.add_argument(
        '--target',
        default='band_gap',
        help='Target material property (default: band_gap)'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='Materials Project API key (default: reads from MP_API_KEY env var)'
    )

    args = parser.parse_args()

    try:
        # Download data
        structures_data = download_mp_structures_with_bandgap(
            api_key=args.api_key,
            target=args.target,
        )

        # Save to extxyz
        count = save_to_extxyz(structures_data, args.output)

        logger.info(f"\nCompleted! Saved {count} structures to {args.output}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
