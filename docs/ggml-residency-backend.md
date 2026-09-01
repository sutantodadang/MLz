# Official GGML Residency Backend Integration

MLz provides a host-compatible `ggml_backend_buffer_type_t` selected through
llama.cpp's official `llama_model_params.tensor_buft_overrides` API. The current
milestone connects that buffer to the bounded file-backed residency manager and
the synchronized native GGML CPU node boundary.

## Execution model

In file-backed mode:

1. llama.cpp builds its ordinary model graph and places model tensors in the
   `MLzResidency` buffer type.
2. The buffer reserves inaccessible virtual address space for stable tensor
   identity pointers; it does not commit or upload model bytes.
3. During model load, `set_tensor` resolves each tensor name to its GGUF file
   offset and records the source span.
4. Before a CPU graph node runs, GGML thread 0 either acquires ordinary
   backed sources as whole spans or selects an op-specific bounded path:
   - regular `MUL_MAT` maps complete output-weight rows in budget-sized tiles;
   - `MUL_MAT_ID` maps only selected routed-expert rows in budget-sized tiles;
   - 2-D `GET_ROWS` maps the row envelope requested by the index tensor.
5. Thread-pool barriers publish each mapping. The stock GGML conversion,
   dequantization, and vec-dot kernels execute against the mapped GGUF bytes.
6. After all workers finish a node or tile, thread 0 restores the reserved
   identity pointer and releases the pin. LRU may then evict that mapping.

Fusion is disabled only while the residency hooks are enabled so that each
pre/post callback corresponds to one stock node. Ordinary builds retain the
upstream fused execution path.

The implementation lives in:

- `src/ggml_residency_backend.h`
- `src/ggml_residency_backend.c`
- `src/residency_ggml_bridge.zig`
- `src/patch_ggml_residency.zig`
- `src/residency_llama_reference.zig`
- `src/validate_ggml_backend.zig`

## Validation

Build hooks and disable CPU repack for a strict bit-identical comparison:

```sh
zig build validate-ggml-backend \
  -Doptimize=ReleaseFast \
  -Dsimd-backend=false \
  -Dggml-residency-hooks=true \
  -Dcpu-repack=false -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 1 4
```

The third positional argument enables file-backed mode and specifies the mapped
weight budget in MiB.

Observed on the 762.81 MiB Llama-3.2-1B Q4_K_M model:

```text
logits: exact=true, max-error=0, mean-error=0, argmax=11/11
uploads=147, uploaded=0.00 MiB
node hooks: pre=358, post=358, active=0, peak-active=1
residency: budget=4.00 MiB, peak-resident=4.00 MiB
           faults=319, hits=16, evictions=318
```

The same native backend was validated on the 27.2 GiB
`Qwen3-Coder-Next-Q2_K.gguf` hybrid DeltaNet+MoE model:

```text
logits: exact=true, max-error=0, mean-error=0, argmax=3830/3830
uploads=843, uploaded=0.00 MiB
node hooks: pre=3066, post=3066, active=0, peak-active=1
residency: budget=4.00 MiB, peak-resident=4.00 MiB
           faults=2332, hits=0, evictions=2331
```

For Qwen, ordinary llama.cpp still builds the architecture-specific DeltaNet
and routed-MoE graph. MLz only controls source mapping lifetime; selected expert
arithmetic remains the stock `MUL_MAT_ID` GGML kernel.

The backend allocation statistic still reports 762.81 MiB because it measures
the logical GGML buffer/virtual address reservation. In backed mode those pages
are `MEM_RESERVE/PAGE_NOACCESS` on Windows or `PROT_NONE` on POSIX and are not a
762.81 MiB committed weight upload. `uploaded=0` plus the residency manager's
`peak_resident_bytes` are the relevant physical mapping gates.

The legacy heap-upload mode remains available by omitting the budget argument.
It is used as a compatibility/control path and still uploads the complete
model.

## Correctness and safety invariants

The validator requires:

- finite logits and exact equality when CPU repack is disabled;
- nonzero, balanced node pre/post calls and a zero final active-node gauge;
- zero uploaded weight bytes in backed mode;
- balanced residency acquires/releases;
- manager peak mapped bytes no greater than the requested budget;
- use of the custom buffer type for all model tensors.

The C backend records each node pin explicitly by tensor, source ID, and
reserved identity offset. This allows tied/shared sources and GGML views to be
restored correctly. Reserved address space intentionally faults if a stock
kernel accesses a backed weight outside the synchronized node lifetime.

## Tiled-op scope and current limitations

The validated native paths allow budgets below the largest tensor by tiling
regular 2-D `MUL_MAT`, routed 3-D `MUL_MAT_ID`, and 2-D `GET_ROWS`. Their key
invariant is that each canonical GGML dot/dequantization still consumes one
complete physical row, so arithmetic and reduction order are unchanged.

Unsupported `GET_ROWS` layouts (for example model-weight views or sources with
more than two dimensions) fall back to whole-source mapping. Sparse multi-row
lookups currently map the contiguous envelope from the minimum to maximum row;
if that envelope exceeds the budget, a future per-row `ops.cpp` hook is needed.

Other current limitations:

- CPU host backend only; no CUDA/Metal/Vulkan residency bridge yet;
- one active bridge/model instance per process;
- graph execution for one bridge must be externally serialized. Callback state
  is mutex-protected against data races, but the current source-ID release ABI
  permits only one open view per source, so concurrent graphs sharing one model
  are rejected rather than independently pinned;
- `ggml-residency-hooks` and the custom SIMD source patch are mutually exclusive
  because both currently rewrite the same vendored `ggml-cpu.c` at build time;
- default CPU_REPACK uses a different packed layout/kernel than this arbitrary
  host buffer, so strict bit equality is validated with `-Dcpu-repack=false`.
