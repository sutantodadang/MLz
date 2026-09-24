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
   - `GET_ROWS` maps each selected row independently, including sparse token
     IDs at opposite ends of the vocabulary.
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

For the sparse-row exact gate, pass comma-separated token IDs instead of one ID:

```sh
zig build validate-ggml-backend -Doptimize=ReleaseFast -Dsimd-backend=false \
  -Dggml-residency-hooks=true -Dcpu-repack=false -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 1,128000 4
```

The dependency is pinned to llama.cpp/GGML tag `b9106` in `build.zig.zon`.
`src/patch_ggml_residency.zig` requires each upstream hook marker exactly once;
the hooks-enabled build fails when the vendored CPU source shape drifts.

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
regular 2-D `MUL_MAT`, routed 3-D `MUL_MAT_ID`, and sparse `GET_ROWS`. Their key
invariant is that each canonical GGML dot/dequantization still consumes one
complete physical row, so arithmetic and reduction order are unchanged.

The sparse `GET_ROWS` path uses GGML index and tensor strides and accounts for
source view offsets. Opposite-row Llama and Qwen model gates pass exactly at a
4 MiB budget. Uncommon view and multidimensional layouts still need dedicated
fixtures; unsupported layouts fail with a diagnostic rather than silently
mapping a whole source when a bounded row path is available.

## Concurrent graphs and pin lifetimes

Every acquisition returns a unique pin token and is released by that token, so
two contexts can pin the same source independently. A node's whole-span sources
are pinned together through `acquire_many`: either all are pinned or none are,
and the caller only waits for admission while it holds no pin from that
attempt, so two graphs can never each hold half of what the other needs. Node
source clones, node pins, and tile pins are thread-local to GGML thread 0 of
each graph; shared model tensors keep their reserved identity pointer.

Admission waits up to 30 s for other graphs to release pins. A request that
cannot fit even with no other pin open fails immediately (`budget_impossible`).

## Failure handling

A weight that cannot be mapped no longer aborts the process. The failing hook
releases every pin it took, restores the node's sources, and the patched graph
loop stops all workers at that node with `GGML_STATUS_FAILED`; `llama_decode`
returns `-3` and rolls back the failed ubatch. The engine then clears the whole
sequence (single-stream) or every in-flight slot (scheduler) so no partial
prefix is reused, and the HTTP request gets `503 residency_error`. Only
bookkeeping violations (releasing an unknown pin token) still abort.

Failure causes are counted separately: `budget_impossible`, `admission_timeout`,
`io` (mapping failed), `invalid` (bad span/range), and `injected` (tests). The
last C-side reason names the operation and tensor, e.g.
`MUL_MAT_ID tile acquisition failed for 'blk.2.ffn_up_exps.weight'`.

`MLZ_RESIDENCY_INJECT_FAILURE=<n>` (test only) fails the n-th acquisition once.

## Non-weight memory policy

The weight budget bounds mapped immutable weights only. `state_budget_mib` is a
hard limit for everything else a context allocates, planned before any
allocation by simulating the model and context with
`llama_model_params.no_alloc` and reading llama.cpp's own memory breakdown
(`src/llama_memory_shim.cpp`):

| Category | Source |
|---|---|
| Mapped weights | residency manager, `weight_budget_mib` |
| KV cache + recurrent/DeltaNet state | breakdown `context` |
| Graph workspace | breakdown `compute` |
| Logits buffer | `n_vocab * n_seq_max * 4` |

A plan above the limit fails startup with a per-category message; above 90% it
logs a warning. After the real context exists, the allocated sizes are read
again and must not exceed the plan. The estimator is architecture-independent
(Llama, Qwen3, Qwen3.5/Qwen3-Next hybrids, Gemma 3). Page cache behind weight
mappings, allocator slack, thread stacks, and tokenizer/sampler heap are not
covered by any hard limit.

## Normal server path

```toml
[residency]
enabled = true
backend = "ggml"
weight_budget_mib = 256
state_budget_mib = 1024
```

or `--residency --weight-budget-mib 256 --state-budget-mib 1024`. Chat,
completion, streaming, and the continuous-batching scheduler all run on the
backed model. `GET /v1/residency/metrics` reports weight mapping (budget,
current/peak mapped, faults, hits, evictions, bytes mapped/evicted), acquire
latency and admission waits, hook/pin balance, failure counts with the last
reason, and planned vs allocated memory per category. No per-tensor labels are
exported.

`tools/residency_server_smoke.py` checks the server end to end against the
ordinary llama.cpp path, including failure injection, state-budget rejection,
and graceful shutdown.

## Validator modes

The fourth validator argument selects a backed mode:

| Mode | Gate |
|---|---|
| `single` (default) | exact logits vs ordinary llama.cpp |
| `concurrent` | two contexts decode the same model at once; identical logits |
| `cancel` | one of two concurrent contexts is aborted mid-graph, then recovers |
| `fail` | acquisitions 1, 2, 3, 17, 50, 200 fail in turn; each decode returns an error with zero open pins and balanced hooks, then the context decodes exactly |

## Current limitations

- CPU host backend only; no CUDA/Metal/Vulkan residency bridge yet;
- one active bridge/model instance per process;
- `ggml-residency-hooks` and the custom SIMD source patch are mutually exclusive
  because both currently rewrite the same vendored `ggml-cpu.c` at build time;
- default CPU_REPACK uses a different packed layout/kernel than this arbitrary
  host buffer, so strict bit equality is validated with `-Dcpu-repack=false`.
