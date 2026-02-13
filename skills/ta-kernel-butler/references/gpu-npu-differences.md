# GPU vs NPU: Comprehensive Comparison

Detailed comparison between GPU and Ascend NPU architectures to help avoid confusion when migrating code or answering questions.

## Executive Summary

| Aspect | GPU (NVIDIA) | Ascend NPU | Key Difference |
|--------|-------------|------------|----------------|
| **Compute Unit** | Streaming Multiprocessor (SM) | AI Core | Different organization |
| **Matrix Unit** | Tensor Core | Cube Unit | Similar purpose, different interface |
| **Vector Unit** | CUDA Core | Vector Unit | Similar, different execution model |
| **Thread Model** | Thread blocks, Warps (32) | Blocks only | No warp concept in NPU |
| **Memory** | Global, Shared, Register | GM, UB, L1, L0A/B/C | More levels in NPU |
| **Synchronization** | `__syncthreads()`, warp sync | PipeBarrier, event-based | Different primitives |
| **Programming** | CUDA, Triton | CANN, triton-ascend | Different frameworks |

## Architecture Comparison

### Compute Unit Organization

#### GPU (NVIDIA)

```
┌─────────────────────────────────────────────────────────┐
│                    GPU (e.g., A100)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │   SM    │  │   SM    │  │   SM    │  │   SM    │    │
│  │         │  │         │  │         │  │         │    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │    │
│  │ │Tensor│ │  │ │Tensor│ │  │ │Tensor│ │  │ │Tensor│ │    │
│  │ │Core │ │  │ │Core │ │  │ │Core │ │  │ │Core │ │    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │    │
│  │ ┌───────────────────────┐ │         │  │         │    │
│  │ │     CUDA Cores        │ │         │  │         │    │
│  │ └───────────────────────┘ │         │  │         │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Characteristics:**
- 100+ SMs per GPU
- 4 Tensor Cores per SM
- 64 CUDA Cores per SM
- Warps of 32 threads

#### Ascend NPU

```
┌─────────────────────────────────────────────────────────┐
│              Ascend NPU (e.g., Ascend 910B)              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │AI Core 1│  │AI Core 2│  │AI Core 3│  │AI Core 4│ ...│
│  │         │  │         │  │         │  │         │    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │    │
│  │ │Cube │ │  │ │Cube │ │  │ │Cube │ │  │ │Cube │ │    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │    │
│  │ │Vector│ │  │ │Vector│ │  │ │Vector│ │  │ │Vector│ │    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │    │
│  │ │Scalar│ │  │ │Scalar│ │  │ │Scalar│ │  │ │Scalar│ │    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Characteristics:**
- 10-30 AI Cores per NPU
- 1 Cube unit per AI Core
- 1 Vector unit per AI Core
- 1-2 Scalar units per AI Core

### Memory Hierarchy

#### GPU Memory

| Level | Size | Speed | Purpose |
|-------|------|-------|---------|
| **Register** | ~64KB/SM | Fastest | Thread-local data |
| **Shared Memory** | 48KB/SM | Very Fast | Block-level sharing |
| **L1 Cache** | 128KB/SM | Fast | Automatic caching |
| **L2 Cache** | 40-50MB | Medium | Global cache |
| **Global Memory** | 40-80GB | Slow | Main storage |

**Key features:**
- Unified address space
- Software-managed shared memory
- Hardware-managed caches

#### NPU Memory

| Level | Size | Speed | Purpose |
|-------|------|-------|---------|
| **L0A/L0B** | ~64KB/AI Core | Very Fast | Cube input |
| **L0C** | ~64KB/AI Core | Very Fast | Cube output |
| **L1 Buffer** | ~256KB/AI Core | Fast | Data staging |
| **Unified Buffer** | 1-2MB/AI Core | Medium | General purpose |
| **Global Memory** | 32-64GB | Slow | Main storage |

**Key features:**
- Explicit data movement (MTE units)
- No unified address space between levels
- Software-managed all levels

## Threading Model

### GPU: Warps and SIMT

```python
# GPU Triton
@triton.jit
def gpu_kernel(x_ptr, output_ptr, n_elements):
    # Block ID
    pid = tl.program_id(axis=0)

    # Thread ID within block
    tid = tl.arange(0, BLOCK_SIZE)

    # Warp ID and lane ID
    warp_id = tid // 32
    lane_id = tid % 32

    # Warp-level communication
    # (NPU doesn't have this!)
    data = tl.shfl_xor_sync(mask, value, lane_mask)

    offsets = pid * BLOCK_SIZE + tid
    x = tl.load(x_ptr + offsets)
    output = x * 2
    tl.store(output_ptr + offsets, output)
```

**GPU characteristics:**
- Threads execute in warps (32 threads)
- SIMT: Single Instruction, Multiple Threads
- Warp-level primitives: `shfl`, `vote`, `match`
- Thread divergence is expensive

### NPU: Block-Based Execution

```python
# NPU triton-ascend
@triton.jit
def npu_kernel(x_ptr, output_ptr, n_elements):
    # Block ID only (no thread ID)
    pid = tl.program_id(axis=0)

    # Offset generation
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # No warp-level primitives!
    # All work-items execute in lockstep

    x = tl.load(x_ptr + offsets)
    output = x * 2
    tl.store(output_ptr + offsets, output)
```

**NPU characteristics:**
- No warp concept
- No SIMT execution
- All work-items in block execute together
- No thread divergence

## Synchronization

### GPU Synchronization

```cuda
// CUDA
__syncthreads();              // Block-level barrier
__syncwarp(mask);             // Warp-level barrier
__threadfence();              // Memory fence
atomicAdd(&counter, 1);       // Atomic operation
```

### NPU Synchronization

```python
# NPU (triton-ascend)
# Note: Exact syntax may vary, check triton-ascend docs

# Block-level barrier
tl.debug_barrier()  # If available

# Event-based synchronization
# (Different from GPU)
```

**Key differences:**
- No warp-level sync in NPU
- Different event model
- Check triton-ascend documentation for exact APIs

## Code Comparison

### Vector Addition

#### GPU Triton

```python
@triton.jit
def gpu_add(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Block and thread IDs
    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)

    # Offsets
    offsets = pid * BLOCK_SIZE + tid

    # Mask for boundary
    mask = offsets < n_elements

    # Load from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    z = x + y

    # Store
    tl.store(z_ptr + offsets, z, mask=mask)

    # Optional: use shared memory
    # (NPU handles this differently)
    # shared_x = tl.shared_memory((BLOCK_SIZE,), dtype=tl.float32)
```

#### NPU triton-ascend

```python
@triton.jit
def npu_add(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Block ID only (no thread/warp concept)
    pid = tl.program_id(axis=0)

    # Offsets (similar, but semantics differ)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary
    mask = offsets < n_elements

    # Load to Unified Buffer (implicit)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute in Vector Unit
    z = x + y

    # Store from Unified Buffer
    tl.store(z_ptr + offsets, z, mask=mask)

    # No explicit shared memory management
    # UB handles intermediate storage
```

### Matrix Multiplication

#### GPU Triton

```python
@triton.jit
def gpu_matmul(a_ptr, b_ptr, c_ptr, M, N, K,
               BLOCK_SIZE_M: tl.constexpr,
               BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Shared memory for tiling
    a_block = tl.shared_memory((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    b_block = tl.shared_memory((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulator
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load to shared memory
        tl.debug_barrier()  # Wait for previous iteration
        tl.store(a_block, tl.load(a_ptrs))
        tl.store(b_block, tl.load(b_ptrs))
        tl.debug_barrier()  # Wait for load to complete

        # Compute
        accumulator += tl.dot(a_block, b_block)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * N + offs_bn[None, :])
    tl.store(c_ptrs, accumulator)
```

#### NPU triton-ascend

```python
@triton.jit
def npu_matmul(a_ptr, b_ptr, c_ptr, M, N, K,
               BLOCK_SIZE_M: tl.constexpr,
               BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Accumulator (in registers/UB, not shared memory)
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load directly to UB (no explicit shared memory)
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_bn[None, :] < N)

        # Compute using Cube unit
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * N + offs_bn[None, :])
    tl.store(c_ptrs, accumulator, mask=offs_am[:, None] < M and offs_bn[None, :] < N)
```

**Key differences:**
- GPU: Explicit shared memory management
- NPU: Implicit UB usage (no `tl.shared_memory`)
- Synchronization may differ

## Terminology Mapping

| Concept | GPU Term | NPU Term | Notes |
|---------|----------|----------|-------|
| Compute unit | Streaming Multiprocessor (SM) | AI Core | Different scale |
| Matrix accelerator | Tensor Core | Cube Unit | Similar purpose |
| Vector processor | CUDA Core | Vector Unit | Similar |
| Control unit | Warp Scheduler | Scalar Unit | Different scale |
| On-chip memory | Shared Memory | Unified Buffer | Different management |
| Fast memory | Register File | L0A/L0B/L0C | NPU has more levels |
| Transfer | Memory Controller | MTE1/MTE2/MTE3 | NPU is more explicit |
| Thread group | Warp (32) | Block | NPU has no warps |
| Parallelism | SIMT | Vector + Cube | Different model |

## Common Confusion Points

### 1. "Shared Memory"

**GPU**: Explicitly allocated and managed
```python
shared = tl.shared_memory((128,), dtype=tl.float32)
```

**NPU**: Implicit through Unified Buffer
```python
# No explicit allocation
# UB handles intermediate storage
data = tl.load(ptr)  # GM → UB
result = data * 2    # UB → Vector → UB
tl.store(out, result) # UB → GM
```

### 2. "Thread ID"

**GPU**: Thread ID within block
```python
tid = tl.arange(0, BLOCK_SIZE)  # Thread ID
warp_id = tid // 32              # Warp ID
```

**NPU**: Block ID only (no threads)
```python
pid = tl.program_id(axis=0)  # Block ID only
# No thread ID or warp ID
```

### 3. "Warp-level Primitives"

**GPU**: Available
```python
# Shuffle within warp
value = tl.shfl_sync(mask, value, src_lane)

# Warp vote
pred = tl.all_sync(mask, condition)
```

**NPU**: Not available
- No warps = no warp primitives
- Use block-level operations instead

### 4. "Atomic Operations"

**GPU**: Rich atomic support
```python
# Atomic add
tl.atomic_add(counter_ptr, 1)

# Atomic CAS
old = tl.atomic_cas(ptr, expected, desired)
```

**NPU**: Limited support
- Check triton-ascend documentation for availability
- May need alternative approaches

## Migration Checklist

When answering NPU questions or migrating GPU code:

- [ ] Remove GPU-specific terminology (warp, SM, shared memory)
- [ ] Update to NPU terminology (AI Core, Cube, UB)
- [ ] Remove thread-level parallelism (use block-level only)
- [ ] Remove warp-level primitives (shuffle, vote, match)
- [ ] Update synchronization (check triton-ascend docs)
- [ ] Verify memory alignment (32B for Vector, 64B for Cube)
- [ ] Check atomic operation availability
- [ ] Update shared memory usage (use UB implicitly)
- [ ] Verify data paths through storage hierarchy
- [ ] Test with small inputs first

## Reference Links

- [Architecture Difference (Official)](https://github.com/Ascend/triton-ascend/blob/main/docs/en/migration_guide/architecture_difference.md)
- [Ascend Hardware Architecture](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [Abstract Hardware Architecture](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
