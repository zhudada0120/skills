"""
Example Ascend NPU Kernels using triton-ascend

This file demonstrates common NPU kernel patterns optimized for Ascend hardware.
Each example includes detailed comments explaining NPU-specific considerations.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Example 1: Vector Addition (Element-wise Operation)
# ============================================================================

@triton.jit
def vector_add_kernel(
    x_ptr,  # Input tensor X pointer
    y_ptr,  # Input tensor Y pointer
    output_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    """
    Vector addition kernel for Ascend NPU.

    NPU-specific considerations:
    - No thread ID, only block ID
    - Data flows: GM → UB → Vector → UB → GM
    - Use 32B alignment for Vector operations
    """
    # Get block ID (not thread ID!)
    pid = tl.program_id(axis=0)

    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for handling boundary elements
    mask = offsets < n_elements

    # Load data from Global Memory to Unified Buffer
    # Note: triton-ascend handles UB allocation automatically
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute in Vector Unit
    # Data path: UB → Vector → UB
    output = x + y

    # Store back from UB to Global Memory
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor):
    """
    Host function for vector addition.

    Args:
        x: Input tensor (on NPU)
        y: Input tensor (on NPU)

    Returns:
        output: x + y
    """
    # Allocate output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Launch kernel
    BLOCK_SIZE = 256  # 32B aligned (256 * 4 bytes = 1024B)
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ============================================================================
# Example 2: Vector Reduction (Sum)
# ============================================================================

@triton.jit
def reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduction kernel: sum all elements in each block.

    NPU-specific considerations:
    - Reduction performed in Vector Unit
    - Each block produces one output value
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load block data
    block = tl.load(input_ptr + offsets, mask=mask)

    # Reduce within block (Vector Unit)
    # Note: Use tl.reduce for efficient reduction
    accumulator = tl.zeros([], dtype=block.dtype)
    accumulator = tl.reduce(block, axis=0)

    # Store single result per block
    tl.store(output_ptr + pid, accumulator)


def reduce_sum(input: torch.Tensor):
    """
    Host function for sum reduction.

    Args:
        input: Input tensor (on NPU)

    Returns:
        output: Sum of all elements
    """
    n_elements = input.numel()
    num_blocks = triton.cdiv(n_elements, 256)

    # Allocate intermediate results
    partial_sums = torch.empty(num_blocks, dtype=input.dtype, device=input.device)

    # Launch reduction kernel
    BLOCK_SIZE = 256
    grid = (num_blocks,)

    reduce_sum_kernel[grid](
        input,
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Final reduction on host (or launch another kernel)
    result = partial_sums.sum().item()

    return result


# ============================================================================
# Example 3: Softmax (Element-wise + Reduction)
# ============================================================================

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel operating on rows independently.

    NPU-specific considerations:
    - Two-pass algorithm: find max, then compute exp and normalize
    - Both max and sum are reductions
    """
    # Row ID (each row processed independently)
    row_id = tl.program_id(axis=0)

    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Row pointer
    row_ptr = input_ptr + row_id * n_cols

    # === Pass 1: Find maximum ===
    row = tl.load(row_ptr + col_offsets, mask=mask)
    row_max = tl.reduce(row, axis=0)  # Max reduction

    # === Pass 2: Compute exp and normalize ===
    # Load again (or cache if register pressure allows)
    row = tl.load(row_ptr + col_offsets, mask=mask)

    # exp(x - max) for numerical stability
    row_exp = tl.exp(row - row_max)

    # Sum of exps
    exp_sum = tl.reduce(row_exp, axis=0)

    # Normalize
    softmax = row_exp / exp_sum

    # Store result
    output_ptr_row = output_ptr + row_id * n_cols
    tl.store(output_ptr_row + col_offsets, softmax, mask=mask)


def softmax(input: torch.Tensor):
    """
    Host function for softmax.

    Args:
        input: 2D tensor (rows, cols) on NPU

    Returns:
        output: Softmax of input (applied to each row independently)
    """
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape

    # Launch kernel (one block per row)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    softmax_kernel[grid](
        input,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ============================================================================
# Example 4: Matrix Multiplication (NPU-Optimized)
# ============================================================================

@triton.jit
def matmul_kernel(
    a_ptr,  # Matrix A (M x K)
    b_ptr,  # Matrix B (K x N)
    c_ptr,  # Matrix C (M x N) - output
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Matrix multiplication kernel optimized for Ascend NPU.

    NPU-specific considerations:
    - Uses Cube Unit for matrix multiplication (tl.dot)
    - Data flows: GM → UB → L0A/L0B → Cube → L0C → UB → GM
    - 64B alignment recommended for Cube operations
    - Multi-buffering can overlap compute and transfer
    """
    # Grid navigation
    pid = tl.program_id(axis=0)
    pid_m = pid // GROUP_SIZE
    pid_n = pid % GROUP_SIZE

    # Block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to A and B blocks
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Initialize accumulator (in registers/UB)
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles to Unified Buffer
        # MTE3 handles GM → UB transfer
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k and offs_bn[None, :] < N, other=0.0)

        # Matrix multiply using Cube Unit
        # Data path: UB → L0A/L0B → Cube → L0C → UB
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * N + offs_bn[None, :])
    tl.store(c_ptrs, accumulator, mask=offs_am[:, None] < M and offs_bn[None, :] < N)


def matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Host function for matrix multiplication.

    Args:
        a: Matrix A (M x K) on NPU
        b: Matrix B (K x N) on NPU

    Returns:
        c: Matrix C (M x N) = A @ B
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Dimension mismatch"

    # Allocate output
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    # Block sizes (tuned for NPU)
    BLOCK_SIZE_M = 64  # 64B aligned
    BLOCK_SIZE_N = 64  # 64B aligned
    BLOCK_SIZE_K = 32
    GROUP_SIZE = 8

    # Grid
    grid = lambda meta: (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE=GROUP_SIZE,
    )

    return c


# ============================================================================
# Testing Functions
# ============================================================================

def test_vector_add():
    """Test vector addition kernel."""
    print("Testing vector_add...")

    # Test data
    x = torch.randn(1024, device='npu')
    y = torch.randn(1024, device='npu')

    # Custom kernel
    z_custom = vector_add(x, y)

    # Reference (PyTorch)
    z_ref = x + y

    # Verify
    assert torch.allclose(z_custom, z_ref, rtol=1e-3, atol=1e-3)
    print("✓ vector_add test passed")


def test_reduce_sum():
    """Test sum reduction kernel."""
    print("Testing reduce_sum...")

    # Test data
    x = torch.randn(1024, device='npu')

    # Custom kernel
    sum_custom = reduce_sum(x)

    # Reference (PyTorch)
    sum_ref = x.sum().item()

    # Verify
    assert abs(sum_custom - sum_ref) < 1e-3
    print("✓ reduce_sum test passed")


def test_softmax():
    """Test softmax kernel."""
    print("Testing softmax...")

    # Test data
    x = torch.randn(32, 128, device='npu')

    # Custom kernel
    y_custom = softmax(x)

    # Reference (PyTorch)
    y_ref = torch.softmax(x, dim=-1)

    # Verify
    assert torch.allclose(y_custom, y_ref, rtol=1e-3, atol=1e-3)
    print("✓ softmax test passed")


def test_matmul():
    """Test matrix multiplication kernel."""
    print("Testing matmul...")

    # Test data
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device='npu')
    b = torch.randn(K, N, device='npu')

    # Custom kernel
    c_custom = matmul(a, b)

    # Reference (PyTorch)
    c_ref = a @ b

    # Verify
    assert torch.allclose(c_custom, c_ref, rtol=1e-2, atol=1e-2)
    print("✓ matmul test passed")


if __name__ == "__main__":
    """
    Run all tests.

    Note: Ensure you have NPU hardware or emulation enabled.
    """
    print("=" * 60)
    print("Ascend NPU Kernel Examples")
    print("=" * 60)
    print()

    try:
        test_vector_add()
        test_reduce_sum()
        test_softmax()
        test_matmul()

        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise
