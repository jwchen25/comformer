"""
Test script for scatter operations.

This script verifies that the PyTorch native scatter implementations
in scatter_fallback.py produce correct results across all platforms.
"""
import torch
import sys


def test_scatter_sum():
    """Test scatter sum operation"""
    from comformer.models.scatter_fallback import scatter

    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    result = scatter(src, index, dim=0, dim_size=2, reduce='sum')
    expected = torch.tensor([3.0, 12.0])  # [1+2, 3+4+5]

    assert torch.allclose(result, expected), \
        f"scatter sum failed: got {result}, expected {expected}"
    print("✓ scatter sum test passed")


def test_scatter_mean():
    """Test scatter mean operation"""
    from comformer.models.scatter_fallback import scatter

    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    result = scatter(src, index, dim=0, dim_size=2, reduce='mean')
    expected = torch.tensor([1.5, 4.0])  # [(1+2)/2, (3+4+5)/3]

    assert torch.allclose(result, expected), \
        f"scatter mean failed: got {result}, expected {expected}"
    print("✓ scatter mean test passed")


def test_scatter_max():
    """Test scatter max operation"""
    from comformer.models.scatter_fallback import scatter

    src = torch.tensor([1.0, 5.0, 3.0, 2.0, 7.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    result = scatter(src, index, dim=0, dim_size=2, reduce='max')
    expected = torch.tensor([5.0, 7.0])  # [max(1,5), max(3,2,7)]

    assert torch.allclose(result, expected), \
        f"scatter max failed: got {result}, expected {expected}"
    print("✓ scatter max test passed")


def test_segment_csr_sum():
    """Test segment CSR sum operation"""
    from comformer.models.scatter_fallback import segment_csr

    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ptr = torch.tensor([0, 2, 5])  # Two segments: [0:2] and [2:5]
    result = segment_csr(src, ptr, reduce='sum')
    expected = torch.tensor([3.0, 12.0])  # [1+2, 3+4+5]

    assert torch.allclose(result, expected), \
        f"segment_csr sum failed: got {result}, expected {expected}"
    print("✓ segment_csr sum test passed")


def test_segment_csr_max():
    """Test segment CSR max operation"""
    from comformer.models.scatter_fallback import segment_csr

    src = torch.tensor([1.0, 5.0, 3.0, 2.0, 7.0])
    ptr = torch.tensor([0, 2, 5])
    result = segment_csr(src, ptr, reduce='max')
    expected = torch.tensor([5.0, 7.0])  # [max(1,5), max(3,2,7)]

    assert torch.allclose(result, expected), \
        f"segment_csr max failed: got {result}, expected {expected}"
    print("✓ segment_csr max test passed")


def test_segment_csr_mean():
    """Test segment CSR mean operation"""
    from comformer.models.scatter_fallback import segment_csr

    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ptr = torch.tensor([0, 2, 5])
    result = segment_csr(src, ptr, reduce='mean')
    expected = torch.tensor([1.5, 4.0])  # [(1+2)/2, (3+4+5)/3]

    assert torch.allclose(result, expected), \
        f"segment_csr mean failed: got {result}, expected {expected}"
    print("✓ segment_csr mean test passed")


def test_gather_csr():
    """Test gather CSR operation"""
    from comformer.models.scatter_fallback import gather_csr

    src = torch.tensor([1.0, 2.0])
    ptr = torch.tensor([0, 2, 5])  # First value repeated 2x, second 3x
    result = gather_csr(src, ptr)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0])

    assert torch.allclose(result, expected), \
        f"gather_csr failed: got {result}, expected {expected}"
    print("✓ gather_csr test passed")


def test_multidimensional():
    """Test scatter with multi-dimensional tensors"""
    from comformer.models.scatter_fallback import scatter

    # 2D case: [5, 3] tensor
    src = torch.randn(5, 3)
    index = torch.tensor([0, 0, 1, 1, 1])

    # Test sum
    result = scatter(src, index, dim=0, dim_size=2, reduce='sum')
    expected = torch.zeros(2, 3)
    expected[0] = src[0] + src[1]
    expected[1] = src[2] + src[3] + src[4]

    assert torch.allclose(result, expected), \
        "multi-dimensional scatter sum failed"
    print("✓ multi-dimensional scatter test passed")


def main():
    """Run all tests"""
    print("Testing scatter fallback implementations...\n")

    try:
        test_scatter_sum()
        test_scatter_mean()
        test_scatter_max()
        test_segment_csr_sum()
        test_segment_csr_max()
        test_segment_csr_mean()
        test_gather_csr()
        test_multidimensional()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nAll scatter operations are working correctly.")
        print("ComFormer uses pure PyTorch implementations for scatter operations.")
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
