# Multi-GPU Distributed Training Guide

This guide explains how to use ComFormer's multi-GPU distributed training capabilities with PyTorch >= 2.6.

## Overview

ComFormer supports distributed training using PyTorch's DistributedDataParallel (DDP), which enables:
- **Faster training**: Distribute workload across multiple GPUs
- **Larger batch sizes**: Effective batch size = batch_size × number of GPUs
- **Scalability**: Support for single-node and multi-node training
- **Efficiency**: Automatic gradient synchronization and optimized communication

## Requirements

- PyTorch >= 2.6
- Multiple CUDA-capable GPUs
- NCCL backend (automatically used for CUDA devices)

## Quick Start

### 1. Basic Multi-GPU Training

The simplest way to use multiple GPUs on a single machine:

```bash
# Train on 4 GPUs
torchrun --nproc_per_node=4 your_train_script.py
```

### 2. Python Script Setup

Enable distributed training by setting `distributed=True`:

```python
from comformer import train_from_list

results = train_from_list(
    strucs=structures,
    labels=labels,
    batch_size=32,  # Batch size per GPU
    n_epochs=500,
    distributed=True,  # Enable multi-GPU training
    output_dir="./output"
)
```

### 3. Complete Example

See `examples/train_multi_gpu.py` for a complete working example:

```bash
cd examples
torchrun --nproc_per_node=4 train_multi_gpu.py
```

## Detailed Usage

### Single Node, Multiple GPUs

Train on a single machine with multiple GPUs:

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train_script.py

# 4 GPUs
torchrun --nproc_per_node=4 train_script.py

# 8 GPUs
torchrun --nproc_per_node=8 train_script.py
```

### Multi-Node Training

For training across multiple machines:

**Node 0 (Master):**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train_script.py
```

**Node 1 (Worker):**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train_script.py
```

### Environment Variables

`torchrun` automatically sets these variables:
- `RANK`: Global rank (0 to world_size-1)
- `LOCAL_RANK`: Local rank on current node (0 to nproc_per_node-1)
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Master node port

## API Reference

### train_from_list

```python
train_from_list(
    strucs,
    labels,
    distributed=False,  # Enable/disable distributed training
    **kwargs
)
```

### train_from_extxyz

```python
train_from_extxyz(
    extxyz_file,
    target_property,
    distributed=False,  # Enable/disable distributed training
    **kwargs
)
```

## Performance Tips

### 1. Batch Size Selection

- **Per-GPU batch size**: Choose based on GPU memory (typically 32-128)
- **Effective batch size**: `batch_size × num_gpus`
- Example: 4 GPUs × 64 batch_size = 256 effective batch size

```python
# Good: Clear per-GPU batch size
train_from_list(
    strucs=structures,
    labels=labels,
    batch_size=64,  # Per GPU
    distributed=True
)
```

### 2. Learning Rate Scaling

When using larger effective batch sizes, consider scaling the learning rate:

```python
base_lr = 0.001
num_gpus = 4
scaled_lr = base_lr * num_gpus  # Linear scaling rule

train_from_list(
    strucs=structures,
    labels=labels,
    learning_rate=scaled_lr,
    batch_size=64,
    distributed=True
)
```

### 3. DataLoader Workers

Adjust `num_workers` based on available CPU cores:

```python
import os

num_workers = min(8, os.cpu_count() // num_gpus)

train_from_list(
    strucs=structures,
    labels=labels,
    num_workers=num_workers,
    distributed=True
)
```

### 4. Gradient Accumulation

For very large effective batch sizes, use gradient accumulation:

```python
# Effective batch size = batch_size × num_gpus × accumulation_steps
train_from_list(
    strucs=structures,
    labels=labels,
    batch_size=32,
    distributed=True,
    # Note: Gradient accumulation can be added as a custom parameter
)
```

## Technical Details

### Architecture

ComFormer's distributed training uses:
1. **DistributedDataParallel (DDP)**: Model replication across GPUs
2. **DistributedSampler**: Data partitioning for each GPU
3. **NCCL Backend**: Optimized GPU-to-GPU communication
4. **Process Groups**: Automatic synchronization

### Data Distribution

Each GPU processes a unique subset of data:
```
Total dataset: 10,000 samples
4 GPUs: Each GPU sees ~2,500 samples per epoch

GPU 0: samples [0, 2500)
GPU 1: samples [2500, 5000)
GPU 2: samples [5000, 7500)
GPU 3: samples [7500, 10000)
```

### Gradient Synchronization

After each backward pass:
1. Each GPU computes local gradients
2. Gradients are averaged across all GPUs
3. All GPUs update with synchronized gradients
4. Model parameters remain identical across GPUs

### Checkpointing

Only rank 0 (main process) saves checkpoints to avoid:
- Redundant I/O operations
- File write conflicts
- Storage overhead

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce per-GPU batch size
```python
batch_size=32  # Try smaller values: 16, 8, etc.
```

### Issue: Slow training with multiple GPUs

**Possible causes**:
1. Small dataset (overhead > speedup)
2. Too few `num_workers` (data loading bottleneck)
3. Large models (communication overhead)

**Solutions**:
```python
# Increase DataLoader workers
num_workers=8

# Reduce synchronization frequency
# (requires custom implementation)
```

### Issue: Process hangs or times out

**Solutions**:
1. Check firewall settings for multi-node training
2. Verify all nodes can reach master address
3. Ensure consistent PyTorch and NCCL versions
4. Set timeout environment variable:
   ```bash
   export NCCL_TIMEOUT=1800  # 30 minutes
   ```

### Issue: Different results on different GPUs

**This should NOT happen**. If it does:
1. Check random seed is set consistently
2. Verify model parameters are synchronized
3. Ensure DistributedSampler is used correctly

## Benchmarks

Approximate speedup on typical workloads:

| GPUs | Speedup | Notes |
|------|---------|-------|
| 1    | 1.0×    | Baseline |
| 2    | 1.8×    | ~90% efficiency |
| 4    | 3.4×    | ~85% efficiency |
| 8    | 6.4×    | ~80% efficiency |

*Actual speedup depends on model size, batch size, and hardware.*

## Best Practices

1. **Always set random seed** for reproducibility:
   ```python
   random_seed=42
   ```

2. **Monitor GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Use mixed precision training** (if supported):
   ```python
   # Future feature - not yet implemented
   use_amp=True
   ```

4. **Profile first epoch** to identify bottlenecks:
   ```bash
   python -m torch.profiler train_script.py
   ```

5. **Start with single GPU** to debug, then scale up:
   ```bash
   # Debug on 1 GPU
   python train_script.py

   # Scale to multiple GPUs
   torchrun --nproc_per_node=4 train_script.py
   ```

## Example: Training on 200k Structures

```python
from comformer import train_from_extxyz

# Train on large dataset with 8 GPUs
results = train_from_extxyz(
    extxyz_file="large_dataset.xyz",
    target_property="formation_energy_per_atom",
    # Hardware: 8 × A100 GPUs
    batch_size=64,  # 64 per GPU = 512 effective
    learning_rate=0.004,  # Scaled for large batch
    n_epochs=500,
    num_workers=8,
    # Enable optimizations
    distributed=True,
    cache_graphs=True,  # Cache preprocessed graphs
    use_lattice=True,
    use_angle=False,
    output_dir="./large_scale_training"
)
```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL GitHub](https://github.com/NVIDIA/nccl)
- ComFormer Examples: `examples/train_multi_gpu.py`

## Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review example scripts in `examples/`
3. Open an issue on GitHub with:
   - PyTorch version
   - CUDA version
   - Number of GPUs
   - Error message and traceback
