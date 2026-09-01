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
    uint64_t residency_acquires;
    uint64_t residency_releases;
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

/* ---- File-backed residency mode (buffer bytes live in the GGUF file) ---- */

/* Bridge callbacks, implemented by the Zig residency layer:
 *   acquire: map the tensor's GGUF span; returns the mapped address (NULL on
 *            failure, e.g. when the span exceeds the residency budget).
 *   release: unmap/release the pinned view for this source id; returns false
 *            when the id has no matching open view.
 *   span:    resolve a tensor name to its GGUF file span. Returns false when
 *            the model has no such tensor.
 *   acquire_range/range_capacity: map and size a tensor-relative subrange for
 *            synchronized native kernel tiling. */
typedef void * (*mlz_ggml_residency_acquire_fn)(uint32_t source_id, uint64_t file_offset, size_t byte_len);
typedef bool (*mlz_ggml_residency_release_fn)(uint32_t source_id);
typedef bool (*mlz_ggml_residency_span_fn)(const char * tensor_name, uint64_t * file_offset, size_t * byte_len);
typedef void * (*mlz_ggml_residency_acquire_range_fn)(uint32_t source_id, size_t tensor_offset, size_t byte_len);
typedef size_t (*mlz_ggml_residency_range_capacity_fn)(uint32_t source_id, size_t tensor_offset);

void mlz_ggml_residency_set_bridge(
        mlz_ggml_residency_acquire_fn acquire,
        mlz_ggml_residency_release_fn release,
        mlz_ggml_residency_span_fn span,
        mlz_ggml_residency_acquire_range_fn acquire_range,
        mlz_ggml_residency_range_capacity_fn range_capacity);

/* Internal CPU-hook API used by the generated ggml-cpu.c integration. These
 * functions are dormant unless backed mode and node hooks are both enabled. */
bool mlz_ggml_residency_should_tile_mul_mat(struct ggml_tensor * node);
bool mlz_ggml_residency_should_tile_mul_mat_id(struct ggml_tensor * node);
size_t mlz_ggml_residency_tile_capacity(struct ggml_tensor * tensor, size_t tensor_offset);
bool mlz_ggml_residency_tile_acquire(struct ggml_tensor * tensor, size_t tensor_offset, size_t byte_len);
bool mlz_ggml_residency_tile_release(struct ggml_tensor * tensor);

/* Backed mode switches how set_tensor behaves: instead of copying weight
 * bytes into the buffer, it registers the tensor's GGUF span and keeps
 * tensor->data pointing at reserved (inaccessible) address space. Must be set
 * before model load. */
void mlz_ggml_residency_set_backed_mode(bool enabled);
bool mlz_ggml_residency_backed_mode(void);

/* Clears the source-span registry and residency acquire/release counters.
 * Call between model runs, never while a backed buffer or graph is alive. */
void mlz_ggml_residency_registry_reset(void);

/* Number of spans registered so far; source ids are 1-based indices. */
size_t mlz_ggml_residency_registry_count(void);

/* Returns the span registered at 1-based `index`. */
bool mlz_ggml_residency_registry_span(
        size_t index, uint64_t * file_offset, size_t * byte_len);

#ifdef __cplusplus
}
#endif

#endif
