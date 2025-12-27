"""
PyTorch native implementations of scatter operations.

This module provides pure PyTorch implementations of scatter operations,
eliminating external dependencies and ensuring compatibility across all
platforms (x86_64, ARM64, macOS, Windows).

These implementations use optimized PyTorch built-in operations and provide
good performance across different hardware architectures.
"""
import torch
from typing import Optional, Literal


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: Literal['sum', 'mean', 'max', 'min'] = 'sum'
) -> torch.Tensor:
    """
    PyTorch native implementation of scatter operation.

    Reduces all values from `src` into `out` at the indices specified
    in `index` along the given axis `dim`.

    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to scatter
        dim_size: Size of output along dimension `dim`
        reduce: Reduction operation ('sum', 'mean', 'max', 'min')

    Returns:
        Reduced tensor of shape [..., dim_size, ...]

    Example:
        >>> src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> index = torch.tensor([0, 0, 1, 1, 1])
        >>> scatter(src, index, dim=0, reduce='sum')
        tensor([3.0, 12.0])  # [1+2, 3+4+5]
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Prepare output shape
    size = list(src.size())
    size[dim] = dim_size

    if reduce == 'sum':
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.index_add_(dim, index, src)

    elif reduce == 'mean':
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, dtype=torch.long, device=src.device)

        out.index_add_(dim, index, src)
        count.index_add_(0, index, torch.ones_like(index))

        # Reshape count for broadcasting
        count = count.clamp(min=1)
        if dim != 0:
            # Add dimensions for broadcasting
            count_shape = [1] * len(size)
            count_shape[dim] = dim_size
            count = count.view(count_shape)
        else:
            count = count.view(-1, *([1] * (out.dim() - 1)))

        return out / count

    elif reduce == 'max':
        out = torch.full(size, float('-inf'), dtype=src.dtype, device=src.device)

        # PyTorch 1.12+ has scatter_reduce
        if hasattr(torch.Tensor, 'scatter_reduce_'):
            # Expand index to match src shape if needed
            if dim == 0:
                index_expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
            else:
                index_expanded = index
            out.scatter_reduce_(dim, index_expanded, src, reduce='amax')
        else:
            # Fallback for older PyTorch versions
            for i in range(dim_size):
                mask = (index == i)
                if mask.any():
                    values = src[mask]
                    if dim == 0:
                        out[i] = values.max(dim=0)[0]
                    else:
                        # More complex indexing for other dimensions
                        out.index_copy_(dim, torch.tensor([i], device=src.device),
                                       values.max(dim=0, keepdim=True)[0])
        return out

    elif reduce == 'min':
        out = torch.full(size, float('inf'), dtype=src.dtype, device=src.device)

        if hasattr(torch.Tensor, 'scatter_reduce_'):
            if dim == 0:
                index_expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
            else:
                index_expanded = index
            out.scatter_reduce_(dim, index_expanded, src, reduce='amin')
        else:
            for i in range(dim_size):
                mask = (index == i)
                if mask.any():
                    values = src[mask]
                    if dim == 0:
                        out[i] = values.min(dim=0)[0]
                    else:
                        out.index_copy_(dim, torch.tensor([i], device=src.device),
                                       values.min(dim=0, keepdim=True)[0])
        return out

    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


def segment_csr(
    src: torch.Tensor,
    ptr: torch.Tensor,
    reduce: Literal['sum', 'mean', 'max', 'min'] = 'sum'
) -> torch.Tensor:
    """
    Segment reduction in CSR (Compressed Sparse Row) format.

    Reduces values in `src` within segments defined by `ptr`.

    Args:
        src: Source tensor of shape [N, ...]
        ptr: CSR pointer tensor of shape [num_segments + 1]
        reduce: Reduction operation

    Returns:
        Reduced tensor of shape [num_segments, ...]

    Example:
        >>> src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> ptr = torch.tensor([0, 2, 5])  # Two segments: [0:2] and [2:5]
        >>> segment_csr(src, ptr, reduce='sum')
        tensor([3.0, 12.0])  # [1+2, 3+4+5]
    """
    if reduce == 'sum':
        # Cumulative sum approach
        cumsum = torch.cat([torch.zeros(1, *src.shape[1:], device=src.device, dtype=src.dtype),
                           torch.cumsum(src, dim=0)])
        result = cumsum[ptr[1:]] - cumsum[ptr[:-1]]
        return result

    elif reduce == 'max':
        # Segment-wise maximum
        result = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            if start < end:
                result.append(src[start:end].max(dim=0)[0])
            else:
                # Empty segment
                result.append(torch.full_like(src[0], float('-inf')))
        return torch.stack(result)

    elif reduce == 'min':
        # Segment-wise minimum
        result = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            if start < end:
                result.append(src[start:end].min(dim=0)[0])
            else:
                # Empty segment
                result.append(torch.full_like(src[0], float('inf')))
        return torch.stack(result)

    elif reduce == 'mean':
        # Segment-wise mean
        result = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            if start < end:
                result.append(src[start:end].mean(dim=0))
            else:
                # Empty segment
                result.append(torch.zeros_like(src[0]))
        return torch.stack(result)

    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


def gather_csr(src: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """
    Gather values from CSR format.

    Expands compressed `src` values according to CSR pointer `ptr`.

    Args:
        src: Compressed source tensor of shape [num_segments, ...]
        ptr: CSR pointer tensor of shape [num_segments + 1]

    Returns:
        Expanded tensor of shape [N, ...] where N = ptr[-1]

    Example:
        >>> src = torch.tensor([1.0, 2.0])
        >>> ptr = torch.tensor([0, 2, 5])  # First value repeated 2x, second 3x
        >>> gather_csr(src, ptr)
        tensor([1.0, 1.0, 2.0, 2.0, 2.0])
    """
    # Calculate repetition counts for each segment
    counts = ptr[1:] - ptr[:-1]

    # Use repeat_interleave to expand
    return src.repeat_interleave(counts, dim=0)


# Alias for compatibility
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter sum operation. Alias for scatter(..., reduce='sum')."""
    return scatter(src, index, dim, dim_size, reduce='sum')


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter mean operation. Alias for scatter(..., reduce='mean')."""
    return scatter(src, index, dim, dim_size, reduce='mean')


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter max operation. Alias for scatter(..., reduce='max')."""
    return scatter(src, index, dim, dim_size, reduce='max')


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter min operation. Alias for scatter(..., reduce='min')."""
    return scatter(src, index, dim, dim_size, reduce='min')
