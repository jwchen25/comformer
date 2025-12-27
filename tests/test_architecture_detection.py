#!/usr/bin/env python3
"""
Test script for dependency verification.

This script verifies that all required dependencies are properly installed
and shows system information for debugging purposes.
"""
import platform
import sys


def test_dependencies():
    """Test that all required dependencies are installed"""

    print("\n" + "="*70)
    print("ComFormer Dependency Verification")
    print("="*70)

    # System Information
    print(f"\nSystem Information:")
    print(f"  Platform:      {platform.system()}")
    print(f"  Architecture:  {platform.machine()}")
    print(f"  Python:        {platform.python_version()}")

    print(f"\n{'='*70}")
    print("Checking Required Dependencies")
    print("="*70)

    # List of required packages
    required_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("jarvis-tools", "jarvis"),
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("pymatgen", "pymatgen"),
        ("e3nn", "e3nn"),
        ("sympy", "sympy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("pytorch-ignite", "ignite"),
        ("pydantic", "pydantic"),
    ]

    missing_packages = []
    installed_packages = []

    for package_name, import_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            installed_packages.append((package_name, version))
            print(f"  ✓ {package_name:20s} version: {version}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ✗ {package_name:20s} [NOT INSTALLED]")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print("="*70)

    total = len(required_packages)
    installed = len(installed_packages)
    missing = len(missing_packages)

    print(f"\n  Total packages:     {total}")
    print(f"  Installed:          {installed}")
    print(f"  Missing:            {missing}")

    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("\n  To install missing packages, run:")
        print("    pip install -e .")

    print(f"\n{'='*70}\n")

    return len(missing_packages) == 0


def main():
    """Main function"""
    try:
        all_installed = test_dependencies()

        if all_installed:
            print("✓ All dependencies are installed correctly\n")
            return 0
        else:
            print("✗ Some dependencies are missing\n")
            return 1

    except Exception as e:
        print(f"\n❌ Error during dependency check: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
