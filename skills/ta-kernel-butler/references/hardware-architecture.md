# Ascend NPU Hardware Architecture

Detailed reference for Ascend NPU hardware architecture, covering AI Core structure, memory hierarchy, and execution model.

## AI Core Architecture

### Computing Units

The AI Core consists of four main functional units:

#### 1. Cube Unit (矩阵计算单元)

Dedicated matrix multiplication accelerator:
- **Computation**: FP16 16×16 matrix multiplication per cycle
- **Latency**: Multi-cycle operation with pipelining
- **Use cases**: Deep learning matrix operations (convolutions, linear layers)
- **Data path**: GM → L1 → L0A/L0B → Cube → L0C → GM

**Performance characteristics:**
| Data Type | Throughput |
|-----------|-----------|
| FP16 | 16×16 × 2 matrices/cycle |
| FP32 | Limited support (via accumulation) |

#### 2. Vector Unit (矢量计算单元)

SIMD vector processing unit:
- **Computation**: Element-wise operations, reductions, data type conversion
- **Latency**: Typically 1-4 cycles
- **Use cases**: Activation functions, normalization, element-wise math
- **Data path**: GM → UB → Vector → UB → GM

**Supported operations:**
- Arithmetic: `+`, `-`, `*`, `/`
- Math functions: `exp`, `log`, `sqrt`, `pow`
- Logical: `and`, `or`, `xor`
- Comparisons: `eq`, `ne`, `lt`, `gt`, `le`, `ge`

#### 3. Scalar Unit (标量计算单元)

Control flow and instruction scheduling:
- **Function**: Branch handling, address calculation, loop control
- **Performance**: Limited compute throughput (not for heavy computation)
- **Programming implication**: Minimize Scalar-side calculations

**Best practices:**
- Precompute constants at compile time using `tl.constexpr`
- Avoid runtime-dependent branching
- Keep Scalar logic simple

#### 4. Data Transfer Units

Multiple transfer engines manage data movement:

| Unit | Source → Destination | Notes |
|------|---------------------|-------|
| **MTE3** | GM ↔ UB | Largest transfer granularity |
| **MTE2** | UB ↔ L1 | Intermediate staging |
| **MTE1** | L1 ↔ L0A/L0B/L0C | Fine-grained control |
| **FixPipe** | L0C → UB/L1/GM | Format conversion on-the-fly |

### Operating Modes

#### Coupled Mode (A1 Series)

```
        Scalar Unit
         ↙         ↘
    Cube Unit   Vector Unit
```

- Single Scalar schedules both units
- Simpler programming model
- Lower parallelism potential

#### Decoupled Mode (A2/A3 Series)

```
Scalar_Cube    Scalar_Vector
     ↓               ↓
Cube Unit     Vector Unit
```

- Independent Scalar units
- Higher parallelism (compute overlap)
- More complex programming model

## Memory Hierarchy

### Storage Levels

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (GM)                        │
│              (External DDR, Large capacity)                  │
└────────────────────┬────────────────────────────────────────┘
                     │ MTE3
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                 Unified Buffer (UB)                          │
│           (On-chip, 1-2MB, General purpose)                │
└────────────────────┬────────────────────────────────────────┘
                     │ MTE2
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    L1 Buffer                                │
│              (On-chip, Data staging)                         │
└───┬───────────────────────┬────────────────────────────────┘
    │ MTE1                   │ MTE1
    ↓                       ↓
┌─────────┐           ┌─────────┐
│  L0A    │           │  L0B    │
│(Cube in)│           │(Cube in)│
└─────────┘           └─────────┘
         ↘           ↙
          ┌───────┐
          │  Cube │
          └───┬───┘
              ↓
         ┌─────────┐
         │  L0C    │
         │(Cubeout)│
         └────┬────┘
              │ FixPipe
              ↓
         (UB/L1/GM with format conversion)
```

### Storage Characteristics

| Level | Capacity | Access | Purpose |
|-------|----------|--------|---------|
| **GM** | GBs | Slow | Large data storage |
| **UB** | 1-2MB | Medium | Data staging, Vector compute |
| **L1** | ~256KB | Fast | Cube compute staging |
| **L0A** | ~64KB | Very Fast | Cube left input |
| **L0B** | ~64KB | Very Fast | Cube right input |
| **L0C** | ~64KB | Very Fast | Cube output |

### Alignment Requirements

Proper alignment is critical for performance:

| Operation | Alignment | Penalty if violated |
|-----------|-----------|---------------------|
| Vector load/store | 32B | Severe performance loss |
| Cube operation | 64B (Cache Line) | Reduced efficiency |
| GM access | 32B minimum | Bank conflicts |

**Alignment techniques:**
```python
# Ensure contiguous allocation
data = tl.contiguous(data)

# Pad to alignment
BLOCK_SIZE = 64  # 64B aligned

# Check alignment at runtime
assert ptr % 32 == 0, "Must be 32B aligned"
```

## Execution Model

### Block-Based Execution

NPU executes programs in blocks (similar to GPU thread blocks, but different semantics):

```python
# Get block ID
pid = tl.program_id(axis=0)  # Not thread ID!
```

**Key differences from GPU:**
- No warp concept
- No SIMT execution within block
- All work-items in block execute in lockstep
- Synchronization is block-level

### Instruction Scheduling

```
┌───────────────────────────────────────┐
│           Scalar Unit                 │
│  (Control flow, address calculation)  │
└────────┬──────────────────────┬───────┘
         │                      │
    ┌────↓─────┐          ┌─────↓────┐
    │ Cube     │          │ Vector   │
    │ Pipeline │          │ Pipeline │
    └──────────┘          └──────────┘
```

1. Scalar decodes instructions
2. Cube instructions → Cube pipeline
3. Vector instructions → Vector pipeline
4. Instructions from different pipelines can execute in parallel

### Pipeline Synchronization

Use PipeBarrier for pipeline coordination:

```python
# Wait for data availability
tl.debug_barrier()

# NPU-specific synchronization (if available)
# Note: This is hardware-specific, check triton-ascend docs
```

## Performance Considerations

### Compute Efficiency

**Cube Unit:**
- Maximize utilization with large matrices (16×16 or larger)
- Batch small operations
- Avoid partial tile utilization

**Vector Unit:**
- Use SIMD-friendly operations
- Avoid divergent paths within block
- Minimize Scalar-side computation

### Memory Bandwidth

**Optimization strategies:**
1. **Data reuse**: Load once, compute multiple times
2. **Double buffering**: Overlap compute and transfer
3. **Padding**: Align to Cache Line boundaries (64B)
4. **Contiguous access**: Ensure `tl.contiguous()` layout

### Latency Hiding

Overlap computation with data transfer:
- Use multi-buffering technique
- Schedule independent operations
- Pipeline Cube and Vector operations

## Reference Links

- [Ascend Basic Architecture](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [Abstract Hardware Architecture](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
