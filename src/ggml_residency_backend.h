#ifndef MLZ_GGML_RESIDENCY_BACKEND_H
#define MLZ_GGML_RESIDENCY_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct mlz_ggml_residency_stats {
    uint64_t buffers_allocated;
    uint64_t buffers_freed;
    uint64_t tensors_initialized;
    uint64_t tensor_uploads;
    uint64_t uploaded_bytes;
    uint64_t current_allocated_bytes;
    uint64_t peak_allocated_bytes;
};

// Host-compatible buffer type selected through
// llama_model_params.tensor_buft_overrides. Tensors allocated from this type
// execute on the stock GGML CPU backend and its native kernels.
ggml_backend_buffer_type_t mlz_ggml_residency_buffer_type(void);

// Reset interval counters. The live current-allocation gauge is preserved; its
// corresponding peak is reset to the current value.
void mlz_ggml_residency_reset_stats(void);
struct mlz_ggml_residency_stats mlz_ggml_residency_get_stats(void);

#ifdef __cplusplus
}
#endif

#endif
