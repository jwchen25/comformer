# ComFormer Platform Compatibility Guide

## Overview

This document provides comprehensive information about ComFormer's compatibility with different hardware architectures, especially NVIDIA GH200 Grace Hopper Superchip and ARM aarch64 platforms.

---

## Supported Architectures

### ✅ Fully Supported

| Architecture | Platform | Status | Notes |
|--------------|----------|--------|-------|
| **x86_64** | Intel Xeon, AMD EPYC | ✅ Fully Supported | Primary development platform |
| **aarch64** | ARM Neoverse, NVIDIA Grace | ✅ Supported | See ARM-specific notes below |
| **GH200** | NVIDIA Grace Hopper Superchip | ✅ Supported | Requires special PyTorch setup |

---

## NVIDIA GH200 Grace Hopper Support

### Architecture Overview

The [NVIDIA GH200 Grace Hopper Superchip](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) combines:
- **Grace CPU**: 72-core ARM Neoverse V2 processor (aarch64 architecture)
- **Hopper GPU**: H100 GPU with 96GB HBM3 memory
- **NVLink-C2C**: 900 GB/s bidirectional bandwidth between CPU and GPU

### Compatibility Status

| Component | Compatibility | Notes |
|-----------|---------------|-------|
| Python | ✅ Compatible | Python 3.10-3.12 fully supported on aarch64 |
| NumPy | ✅ Compatible | Native aarch64 wheels available |
| SciPy | ✅ Compatible | Native aarch64 wheels available |
| PyTorch | ⚠️ Requires Setup | ARM64+CUDA requires special installation |
| PyG | ✅ Compatible | Works with PyTorch aarch64 |
| ASE | ✅ Compatible | Pure Python, works on all platforms |

### PyTorch Installation on GH200

**Important**: Default `pip install torch` on ARM64 systems installs CPU-only version, even with GPU present.

#### Method 1: NVIDIA NGC Container (Recommended)

```bash
# Pull NVIDIA's pre-configured PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.12-py3
```

**Advantages**:
- Pre-configured with ARM64 CUDA support
- Optimized for Grace Hopper
- Includes cuDNN, NCCL, and other GPU libraries
- Officially supported by NVIDIA

#### Method 2: Manual Installation with CUDA Support

```bash
# Install PyTorch with CUDA 12.x for ARM64
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

**Note**: As of 2025, PyTorch 2.7+ includes official ARM64+CUDA support, but the wheels are hosted on a separate index.

#### Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Expected output on GH200**:
```
PyTorch version: 2.7.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA H100 80GB HBM3
```

### Known Issues and Workarounds

#### Issue 1: PyTorch CUDA Not Available After Installation

**Symptoms**: `torch.cuda.is_available()` returns `False` despite having a GPU.

**Cause**: Installed CPU-only PyTorch from PyPI.

**Solution**: Reinstall from PyTorch's CUDA index:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

**References**:
- [PyTorch Forums: Installing PyTorch on Grace Hopper](https://discuss.pytorch.org/t/installing-pytorch-on-a-grace-hopper-gh200-node-with-gpu-support/216836)
- [GitHub Issue: Package manager install on Grace Hopper](https://github.com/pytorch/pytorch/issues/123835)

#### Issue 2: Binary Wheel Not Available for Some Packages

**Symptoms**: `pip` builds packages from source, compilation errors.

**Cause**: Some packages don't provide pre-built aarch64 wheels.

**Solution**: Install build dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential python3-dev

# Install packages that may need compilation
pip install --no-binary :all: <package-name>
```

---

## ARM aarch64 General Support

### Python on ARM

Python has [official Tier 1 support for Linux aarch64](https://www.python.org/success-stories/python-on-arm-2025-update/) as of PEP 11. This means:
- ✅ All standard library modules work
- ✅ Binary wheels available for most popular packages
- ✅ Performance optimizations for ARM architecture
- ✅ [Python 3.13+ includes ARM-optimized JIT](https://newsroom.arm.com/blog/how-the-python-software-foundation-future-proofed-its-infrastructure-with-arm)

### Scientific Python Ecosystem on ARM

| Package | aarch64 Support | Notes |
|---------|-----------------|-------|
| **numpy** | ✅ Excellent | Native wheels, BLAS optimizations available |
| **scipy** | ✅ Excellent | Native wheels, LAPACK optimizations |
| **pandas** | ✅ Excellent | Native wheels |
| **matplotlib** | ✅ Excellent | Native wheels |
| **scikit-learn** | ✅ Excellent | Native wheels |
| **torch** | ✅ Good | CPU version from PyPI, CUDA from special index |
| **torch-geometric** | ✅ Good | Works with PyTorch aarch64 |

### ASE (Atomic Simulation Environment) on ARM

**Status**: ✅ Fully Compatible

ASE is primarily a pure Python package with minimal compiled dependencies:
- **Installation**: Standard `pip install ase` works on aarch64
- **Dependencies**: Only requires numpy and scipy (both have aarch64 wheels)
- **Performance**: No significant performance difference vs x86_64
- **Version**: ASE 3.22.0+ recommended for best compatibility

**Installation**:
```bash
pip install ase>=3.22.0
```

**Verification**:
```python
from ase.io import read, write
from ase import Atoms

# Test basic functionality
atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
print(f"Created {len(atoms)} atom structure")
```

---

## ComFormer Installation

### Standard Installation (x86_64)

```bash
# Clone repository
git clone https://github.com/jwchen25/comformer.git
cd comformer

# Install with all dependencies
pip install -e .

# Or install with ExtXYZ support
pip install -e ".[extxyz]"
```

### Installation on ARM/GH200

```bash
# 1. Install PyTorch with CUDA support (if using GPU)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install ComFormer
pip install -e ".[extxyz]"

# 3. Verify installation
python -c "
import torch
import comformer
print('PyTorch CUDA:', torch.cuda.is_available())
print('ComFormer version:', comformer.__version__)
"
```

### Docker Installation (Recommended for GH200)

```dockerfile
# Use NVIDIA's PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Install ComFormer
WORKDIR /workspace
COPY . .
RUN pip install -e ".[extxyz]"

# Set up environment
ENV PYTHONUNBUFFERED=1
```

Build and run:
```bash
docker build -t comformer:gh200 .
docker run --gpus all -it comformer:gh200
```

---

## Dependency Version Compatibility

### Core Dependencies

| Package | Version Range | ARM Support | Notes |
|---------|---------------|-------------|-------|
| **Python** | 3.10-3.12 | ✅ Yes | 3.13 experimental JIT on ARM |
| **numpy** | ≥1.26.0 | ✅ Yes | Native aarch64 wheels |
| **scipy** | ≥1.11.0 | ✅ Yes | Native aarch64 wheels |
| **torch** | ≥2.6.0 | ⚠️ Special | Requires CUDA index for GPU |
| **torch-geometric** | ≥2.7.0 | ✅ Yes | Works with torch aarch64 |
| **pymatgen** | ≥2024.1.1 | ✅ Yes | Pure Python + compiled extensions |
| **jarvis-tools** | ≥2024.1.1 | ✅ Yes | Works on aarch64 |

### Optional Dependencies

| Package | Version Range | ARM Support | Use Case |
|---------|---------------|-------------|----------|
| **ase** | ≥3.22.0, <3.25.0 | ✅ Yes | ExtXYZ file support |
| **pandarallel** | ≥1.6.0 | ✅ Yes | Parallel processing |

### ASE Version Selection

**Chosen version**: `ase>=3.22.0,<3.25.0`

**Rationale**:
- **Lower bound (3.22.0)**:
  - Compatible with numpy ≥1.19, scipy ≥1.6
  - Stable API for extxyz reading/writing
  - Released 2022-06, well-tested

- **Upper bound (<3.25.0)**:
  - Prevents automatic updates to untested versions
  - Ensures API stability
  - Can be updated after testing new releases

**Compatibility check**:
- ✅ numpy 1.26+ (required by ComFormer) > 1.19 (required by ASE)
- ✅ scipy 1.11+ (required by ComFormer) > 1.6 (required by ASE)
- ✅ Python 3.10-3.12 (required by ComFormer) ⊂ Python 3.8+ (supported by ASE)

---

## Performance Considerations

### GH200 Specific Optimizations

1. **NVLink-C2C Bandwidth**:
   - 900 GB/s between CPU and GPU
   - Faster than PCIe Gen 5 by 9x
   - Benefits: Reduced data transfer overhead

2. **Unified Memory**:
   - Grace CPU can directly access H100 GPU memory
   - Simplifies memory management for large graphs
   - Recommendation: Use `pin_memory=True` in DataLoaders (already enabled)

3. **ARM Neoverse V2 Optimizations**:
   - SVE (Scalable Vector Extension) support
   - NumPy/SciPy can leverage ARM SIMD instructions
   - Recommendation: Use optimized BLAS libraries (OpenBLAS, ARM Performance Libraries)

### Benchmark Results

Based on [published benchmarks](https://www.phoronix.com/review/nvidia-gh200-gptshop-benchmark/5) and [HPC reports](https://hprc.tamu.edu/files/training/2025/Spring/ACES_Intro_to_the_Grace_Hopper_Superchip_jchegwidden.pdf):

| Workload | x86_64 (Xeon/EPYC) | GH200 | Notes |
|----------|-------------------|-------|-------|
| Graph Construction | Baseline | 1.1-1.3x | ARM CPU slightly faster |
| PyTorch Training | Baseline | 1.0-1.2x | Similar with proper setup |
| Memory Bandwidth | Baseline | 1.5-2.0x | NVLink-C2C advantage |

---

## Troubleshooting

### Q1: How do I check if I'm on an ARM system?

```bash
uname -m
# Output: aarch64 = ARM 64-bit
#         x86_64  = Intel/AMD 64-bit
```

```python
import platform
print(platform.machine())  # 'aarch64' or 'x86_64'
```

### Q2: PyTorch not detecting GPU on GH200

**Check CUDA installation**:
```bash
nvidia-smi  # Should show H100 GPU
nvcc --version  # Should show CUDA toolkit
```

**Verify PyTorch was built with CUDA**:
```python
import torch
print(torch.version.cuda)  # Should show '12.1' or similar, not 'None'
```

**Solution**: Reinstall PyTorch from CUDA index (see installation section above).

### Q3: ASE import error on ARM

**Error**: `ImportError: cannot import name 'Atoms' from 'ase'`

**Solution**: Ensure numpy and scipy are installed first:
```bash
pip install numpy scipy
pip install ase
```

### Q4: Building from source takes too long

**Cause**: Missing binary wheels for some packages on aarch64.

**Solution**: Use NVIDIA NGC container with pre-built packages, or install build dependencies:
```bash
sudo apt-get install -y python3-dev build-essential gfortran
```

---

## Testing Platform Compatibility

### Compatibility Check Script

Save as `check_compatibility.py`:

```python
"""
ComFormer Platform Compatibility Checker
Verifies system compatibility with ComFormer and identifies potential issues.
"""

import sys
import platform
import subprocess

def check_architecture():
    """Check CPU architecture."""
    arch = platform.machine()
    print(f"\n{'='*60}")
    print("System Architecture Check")
    print(f"{'='*60}")
    print(f"Architecture: {arch}")

    if arch == 'aarch64':
        print("✅ ARM64 detected (GH200 compatible)")
    elif arch == 'x86_64':
        print("✅ x86_64 detected (standard platform)")
    else:
        print(f"⚠️  Untested architecture: {arch}")

    return arch

def check_python_version():
    """Check Python version compatibility."""
    print(f"\n{'='*60}")
    print("Python Version Check")
    print(f"{'='*60}")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if 3.10 <= version.major + version.minor/10 <= 3.12:
        print("✅ Python version compatible (3.10-3.12)")
        return True
    else:
        print("❌ Python version not in supported range (3.10-3.12)")
        return False

def check_gpu():
    """Check GPU availability."""
    print(f"\n{'='*60}")
    print("GPU Check")
    print(f"{'='*60}")

    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Try to detect GH200
            if 'H100' in result.stdout or 'GH200' in result.stdout:
                print("✅ NVIDIA GH200/H100 detected!")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected (CPU-only mode)")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  nvidia-smi not found (CPU-only mode)")
        return False

def check_torch():
    """Check PyTorch installation and CUDA support."""
    print(f"\n{'='*60}")
    print("PyTorch Check")
    print(f"{'='*60}")

    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

            # Check if ARM + CUDA (GH200)
            if platform.machine() == 'aarch64':
                print("✅ ARM64 + CUDA support confirmed (GH200 compatible)")
        else:
            if platform.machine() == 'aarch64':
                print("⚠️  ARM64 system but CUDA not available")
                print("    Install PyTorch with CUDA:")
                print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("⚠️  CUDA not available (CPU-only PyTorch)")

        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check core dependencies."""
    print(f"\n{'='*60}")
    print("Core Dependencies Check")
    print(f"{'='*60}")

    packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'torch': 'torch',
        'torch_geometric': 'torch-geometric',
        'pymatgen': 'pymatgen',
        'ase': 'ase (optional, for ExtXYZ)',
    }

    results = {}
    for import_name, display_name in packages.items():
        try:
            module = __import__(import_name.replace('_', '.'))
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            results[import_name] = True
        except ImportError:
            if 'optional' in display_name:
                print(f"⚠️  {display_name}: Not installed (optional)")
            else:
                print(f"❌ {display_name}: Not installed")
            results[import_name] = False

    return results

def main():
    """Run all compatibility checks."""
    print(f"\n{'='*60}")
    print("ComFormer Platform Compatibility Checker")
    print(f"{'='*60}")

    arch = check_architecture()
    python_ok = check_python_version()
    gpu_available = check_gpu()
    torch_ok = check_torch()
    deps = check_dependencies()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    all_ok = python_ok and torch_ok and all(
        deps.get(k, False) for k in ['numpy', 'scipy', 'torch']
    )

    if all_ok:
        print("✅ System is compatible with ComFormer!")
        if arch == 'aarch64' and gpu_available:
            print("✅ GH200 Grace Hopper configuration detected")
    else:
        print("⚠️  Some compatibility issues found (see above)")

    if arch == 'aarch64' and torch_ok and not gpu_available:
        print("\n💡 Tip: For GH200, install PyTorch with CUDA support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")

    if not deps.get('ase', False):
        print("\n💡 Tip: To use ExtXYZ interface, install ASE:")
        print("   pip install 'comformer[extxyz]'")

if __name__ == "__main__":
    main()
```

### Running the Check

```bash
python check_compatibility.py
```

---

## References

### NVIDIA GH200 Documentation
- [NVIDIA GH200 Grace Hopper Superchip](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
- [NVIDIA Grace CPU Documentation](https://docs.nvidia.com/dccpu/index.html)
- [Grace CPU Superchip Architecture](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/)

### PyTorch on ARM/GH200
- [PyTorch Forum: Installing PyTorch on Grace Hopper](https://discuss.pytorch.org/t/installing-pytorch-on-a-grace-hopper-gh200-node-with-gpu-support/216836)
- [PyTorch CUDA Setup for GH200](https://michaelbommarito.com/wiki/programming/languages/python/libraries/pytorch-gh200-arm64/)
- [GitHub Issue: Package manager install on Grace Hopper](https://github.com/pytorch/pytorch/issues/123835)

### Python on ARM
- [Python on ARM: 2025 Update](https://www.python.org/success-stories/python-on-arm-2025-update/)
- [Python Software Foundation on ARM](https://newsroom.arm.com/blog/how-the-python-software-foundation-future-proofed-its-infrastructure-with-arm)

### ASE Documentation
- [ASE Installation Guide](https://wiki.fysik.dtu.dk/ase/install.html)
- [ASE PyPI Package](https://pypi.org/project/ase/)
- [ASE GitLab Repository](https://gitlab.com/ase/ase)

### Benchmarks and Performance
- [NVIDIA GH200 Performance Benchmarks](https://www.phoronix.com/review/nvidia-gh200-gptshop-benchmark/5)
- [Grace Hopper Superchip Introduction (TAMU)](https://hprc.tamu.edu/files/training/2025/Spring/ACES_Intro_to_the_Grace_Hopper_Superchip_jchegwidden.pdf)
- [Grace Hopper Benchmarking (QMUL)](https://blog.hpc.qmul.ac.uk/benchmarking-grace-hopper-nodes/)

---

## Support

For platform-specific issues:

1. **Check documentation**: Review relevant sections above
2. **Run compatibility checker**: `python check_compatibility.py`
3. **Review logs**: Check for architecture-specific warnings
4. **Container option**: Consider using NVIDIA NGC containers for GH200

For bugs or feature requests, please open an issue on the GitHub repository.

---

**Last Updated**: December 2025
**Document Version**: 1.0
