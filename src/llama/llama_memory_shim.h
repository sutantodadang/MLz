#ifndef MLZ_LLAMA_MEMORY_SHIM_H
#define MLZ_LLAMA_MEMORY_SHIM_H

#include <stdbool.h>
#include <stddef.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Byte totals from llama.cpp's own per-buffer-type memory breakdown, summed
// over all buffer types. With llama_model_params.no_alloc the same numbers are
// the simulated allocation sizes, so they serve as an exact preflight.
struct mlz_llama_memory {
    size_t model;   // weight buffers (excluding `exclude_model_buft`)
    size_t context; // KV cache and recurrent/DeltaNet state
    size_t compute; // graph workspace reserved by the scheduler
};

// `exclude_model_buft` names a buffer type whose model bytes are accounted
// elsewhere (the file-backed residency buffer is reserved address space, not
// memory); pass NULL to include every weight buffer.
bool mlz_llama_memory_breakdown(
        const struct llama_context * ctx,
        ggml_backend_buffer_type_t exclude_model_buft,
        struct mlz_llama_memory * out);

#ifdef __cplusplus
}
#endif

#endif
