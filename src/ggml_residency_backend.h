#ifndef MLZ_GGML_RESIDENCY_BACKEND_H
#define MLZ_GGML_RESIDENCY_BACKEND_H

#include <stdbool.h>
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
    uint64_t node_pre_calls;
    uint64_t node_post_calls;
    uint64_t current_active_nodes;
    uint64_t peak_active_nodes;
};

// Host-compatible buffer type selected through
// llama_model_params.tensor_buft_overrides. Tensors allocated from this type
// execute on the stock GGML CPU backend and its native kernels.
ggml_backend_buffer_type_t mlz_ggml_residency_buffer_type(void);

// Reset interval counters. Live allocation/node gauges are preserved and their
// corresponding peaks are reset to the current values.
void mlz_ggml_residency_reset_stats(void);
struct mlz_ggml_residency_stats mlz_ggml_residency_get_stats(void);

// Synchronized CPU node hooks are compiled in only when the build uses
// -Dggml-residency-hooks=true. These APIs remain available in every build so
// callers can detect support without conditional ABI bindings.
bool mlz_ggml_residency_node_hooks_available(void);
// Enable or disable hooks between graph executions. A running graph samples
// this switch once so all of its node boundaries use one barrier protocol.
void mlz_ggml_residency_set_node_hooks_enabled(bool enabled);
bool mlz_ggml_residency_node_hooks_enabled(void);
void mlz_ggml_residency_node_pre(struct ggml_tensor * node);
void mlz_ggml_residency_node_post(struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif

#endif
