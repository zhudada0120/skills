# triton-ascend Development Guide

Comprehensive guide for developing kernels with triton-ascend, from basic concepts to advanced optimization techniques.

## Overview

triton-ascend is a port of OpenAI's Triton language for Huawei Ascend NPU. It provides a Python-based domain-specific language for writing high-performance NPU kernels without dealing with complex hardware details.

## Getting Started

### Installation

```bash
# Install triton-ascend
pip install triton-ascend

# Or from source
git clone https://github.com/Ascend/triton-ascend.git
cd triton-ascend
pip install -e .
```

### First Kernel: Vector Addition

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block ID
    pid = tl.program_id(axis=0)

    # Compute offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary handling
    mask = offsets < n_elements

    # Load data (masked for safety)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    z = x + y

    # Store result
    tl.store(z_ptr + offsets, z, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Allocate output
    z = torch.empty_like(x)
    n_elements = z.numel()

    # Launch kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_kernel[grid](
        x, y, z,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return z
```

## Kernel Development Workflow

### 1. Problem Analysis

Understand the computation:
- What are the inputs and outputs?
- What is the computation pattern?
- What are the memory access patterns?

### 2. Tiling Strategy

Choose appropriate block sizes based on:
- **NPU constraints**: 32B alignment for Vector, 64B for Cube
- **Register pressure**: Don't exceed on-chip storage
- **Data reuse**: Maximize reuse of loaded data

**Common tile sizes:**
```python
BLOCK_SIZE_M = 64   # Rows
BLOCK_SIZE_N = 64   # Columns
BLOCK_SIZE_K = 32   # Reduction dimension
```

### 3. Data Movement Plan

Plan data flow through storage hierarchy:

```
GM → UB (load tiles)
UB → Vector/Cube (compute)
Vector/Cube → UB (store results)
UB → GM (write back)
```

### 4. Kernel Implementation

Write the kernel following NPU best practices:
- Use `tl.constexpr` for compile-time constants
- Mask loads/stores for boundary handling
- Ensure proper alignment

### 5. Host Function

Write the wrapper that:
- Allocates output tensors
- Calculates grid size
- Launches kernel
- Handles edge cases

### 6. Testing and Verification

```python
import torch

def test_kernel():
    # Small test case
    x = torch.randn(128, device='npu')
    y = torch.randn(128, device='npu')

    # Custom kernel
    z_custom = add(x, y)

    # Reference (PyTorch)
    z_ref = x + y

    # Verify
    assert torch.allclose(z_custom, z_ref)
    print("✓ Test passed")

test_kernel()
```

## Common Patterns

### Element-wise Operations

```python
@triton.jit
def elementwise_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask)
    output = data * 2 + 1  # Example operation
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Reduction Operations

```python
@triton.jit
def reduction_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load block
    block = tl.load(input_ptr + offsets, mask=mask)

    # Reduce within block
    accumulator = tl.zeros([], dtype=block.dtype)
    accumulator = tl.reduce(block, axis=0)

    # Write to output (single value per block)
    tl.store(output_ptr + pid, accumulator)
```

### Matrix Multiplication (NPU-optimized)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Grid navigation
    pid = tl.program_id(axis=0)
    pid_m = pid // GROUP_SIZE
    pid_n = pid % GROUP_SIZE

    # Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Accumulator
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_bn[None, :] < N)

        # Accumulate (uses Cube unit when possible)
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * N + offs_bn[None, :])
    tl.store(c_ptrs, accumulator, mask=offs_am[:, None] < M and offs_bn[None, :] < N)
```

## Performance Optimization

### 1. Tile Size Selection

Guidelines for choosing tile sizes:

| Factor | Recommendation |
|--------|---------------|
| **Alignment** | Use multiples of 32 (Vector) or 64 (Cube) |
| **Cache** | Tiles should fit in UB (1-2MB) |
| **Balance** | Balance M, N, K dimensions |
| **Experiment** | Autotune for best performance |

### 2. Memory Access Optimization

```python
# Bad: Non-contiguous access
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * stride

# Good: Contiguous access
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.contiguous(tl.load(ptr + offsets))
```

### 3. Loop Unrolling

```python
# Let triton-ascend handle unrolling
for k in range(0, K, BLOCK_SIZE_K):
    # ... computation ...
    # Compiler will unroll based on BLOCK_SIZE_K
```

### 4. Autotuning

Use `triton.testing` for automatic tuning:

```python
from triton import testing

configs = [
    triton.Config(
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
        num_stages=4,
        num_warps=2,
    ),
    triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
        num_stages=4,
        num_warps=4,
    ),
    # More configurations...
]

@triton.autotune(configs, key=['M', 'N', 'K'])
def matmul_kernel(...):
    ...
```

## Debugging Tips

### 1. Verify Intermediate Results

```python
@triton.jit
def debug_kernel(...):
    # Load and verify
    data = tl.load(ptr + offsets)

    # Debug output (if supported)
    tl.debug_print("data:", data)

    # Continue computation
    ...
```

### 2. Boundary Testing

```python
# Test various input sizes
sizes = [1, 7, 32, 64, 127, 128, 1024, 1025]
for size in sizes:
    x = torch.randn(size, device='npu')
    result = kernel(x)
    verify(result)
```

### 3. Performance Profiling

```python
import time

# Warmup
for _ in range(10):
    _ = kernel(x)

# Time
start = time.time()
for _ in range(100):
    result = kernel(x)
end = time.time()

# Compute bandwidth
elapsed = (end - start) / 100
bandwidth = x.numel() * x.element_size() / elapsed / 1e9
print(f"Bandwidth: {bandwidth:.2f} GB/s")
```

## Migration from GPU Triton

### Key Changes

| Aspect | GPU Triton | triton-ascend |
|--------|-----------|---------------|
| **Thread ID** | `tl.program_id` is thread/block ID | `tl.program_id` is block ID only |
| **Shared memory** | `tl.shared_memory` | Unified Buffer (implicit) |
| **Atomic operations** | `tl.atomic_*` | Limited support, check docs |
| **Dot product** | `tl.dot` | `tl.dot` (maps to Cube) |
| **Synchronization** | `tl.barrier()` | Different semantics |

### Migration Checklist

1. **Remove thread-level parallelism**: NPU uses block-based execution
2. **Replace shared memory**: Use standard load/store with UB
3. **Update synchronization**: Check NPU-specific barriers
4. **Verify alignment**: Ensure 32B/64B alignment
5. **Test thoroughly**: NPU behavior may differ from GPU

## Reference Links

- [Architecture Difference](https://github.com/Ascend/triton-ascend/blob/main/docs/en/migration_guide/architecture_difference.md)
- [Migration from GPU](https://github.com/Ascend/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md)
- [Performance Guidelines](https://github.com/Ascend/triton-ascend/blob/main/docs/en/migration_guide/performance_guidelines.md)
- [Core Features](https://github.com/Ascend/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)
