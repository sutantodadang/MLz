# Official GGML Residency Backend Integration

MLz now ships a host-compatible `ggml_backend_buffer_type_t` that can be
selected through llama.cpp's official `llama_model_params.tensor_buft_overrides`
API. This is the first integration milestone between bounded-residency work and
the native llama.cpp/GGML graph runner.

## What this milestone proves

With the override enabled:

- llama.cpp creates model tensors in the `MLzResidency` buffer type;
- model loading reaches the custom buffer through `set_tensor`;
- the standard GGML CPU scheduler accepts it as ordinary host memory;
- inference executes the original llama.cpp graph and stock GGML CPU kernels;
- no model graph or arithmetic is reimplemented by the validator.

The implementation lives in:

- `src/ggml_residency_backend.h`
- `src/ggml_residency_backend.c`
- `src/residency_llama_reference.zig`
- `src/validate_ggml_backend.zig`

The backend is compiled into the GGML static library, uses platform-appropriate
aligned allocation (`_aligned_malloc`/`_aligned_free` on Windows and
`posix_memalign`/`free` on POSIX), and validates tensor/buffer bounds for all
copy callbacks. Statistics are process-global atomics; reset clears interval
counters while preserving the live allocation gauge, so freeing a pre-existing
buffer cannot underflow accounting.

## Validation

Run with CPU repack disabled for a strict bit-identical comparison:

```sh
zig build validate-ggml-backend \
  -Doptimize=ReleaseFast \
  -Dsimd-backend=false \
  -Dcpu-repack=false -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 1
```

Observed on the real 762.81 MiB Llama model:

```text
logits: exact=true, max-error=0, argmax=11/11
backend buffers: allocated=1, tensors=147, uploads=147
uploaded=762.81 MiB
```

With the default CPU repack build, the ordinary reference uses CPU_REPACK while
the custom override is not a CPU_REPACK buffer type and therefore retains
canonical CPU layout. That changes packed layouts and reduction kernels; it is
not evidence of a faulty copy or a bounded-residency approximation. The
validator accepts exact results immediately, otherwise requires finite logits,
max error <= 0.1, mean error <= 0.02, and identical top-1. The observed result
was max error 0.080871, mean error 0.013434, top-1 11/11.

## Current limitation: allocation is still pin-per-model

This milestone intentionally allocates the complete packed model buffer and
keeps every weight pointer stable for the model lifetime. It proves official
backend selection and native graph execution, but does **not yet enforce the
bounded mapping budget**. On the 762.81 MiB test model, peak custom-buffer
allocation is 762.81 MiB.

This is not something buffer callbacks alone can fix: stock CPU kernels directly
dereference `ggml_tensor.data`; `get_tensor` is a host-copy API, not a compute
fault callback. Replacing `tensor.data` after graph construction is unsafe
without synchronizing every graph worker.

## Next step: node-lifetime pinning

The next integration stage must add an execution-lifetime boundary around the
native CPU graph:

1. Before a node runs, thread 0 resolves its weight sources to GGUF descriptors,
   acquires the required residency windows, and assigns stable data pointers.
2. A threadpool barrier publishes those pointers to every GGML worker.
3. The stock GGML op executes unchanged.
4. After all workers finish the node, thread 0 releases the views; only then may
   LRU eviction occur.
5. Fused nodes need the union of all source weights in the fused group.

A whole tensor larger than the budget cannot use ordinary stock GGML kernels,
because they expect contiguous `tensor.data`. Such tensors still require one of:

- the existing MLz tiled executor;
- an op-specific tiled CPU hook;
- a budget at least as large as the largest tensor used by one node.

Therefore the immediate backend target is **bounded per-node residency** with a
well-defined minimum budget, followed by per-op tiled hooks for tensors larger
than that budget.
