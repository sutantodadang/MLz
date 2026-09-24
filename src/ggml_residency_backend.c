#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#    define _POSIX_C_SOURCE 200112L
#endif

#include "ggml_residency_backend.h"

#include "ggml-backend-impl.h"

#if !defined(__STDC_NO_ATOMICS__)
#    include <stdatomic.h>
#else
#    error "MLz GGML residency backend requires C11 atomics"
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    include <malloc.h>
#    include <windows.h>
#else
#    include <sys/mman.h>
#endif

#define MLZ_GGML_RESIDENCY_ALIGNMENT 32u

struct mlz_buffer_context {
    void * base;
    size_t size;
    /* Backed mode: base is reserved-only address space; tensor bytes live in
     * the GGUF file and are mapped per node through the residency manager.
     * 0 = heap-upload (milestone 1), 1 = file-backed. */
    int backed;
};

static struct ggml_backend_buffer_type g_mlz_buft;

static atomic_uint_fast64_t g_buffers_allocated;
static atomic_uint_fast64_t g_buffers_freed;
static atomic_uint_fast64_t g_tensors_initialized;
static atomic_uint_fast64_t g_tensor_uploads;
static atomic_uint_fast64_t g_uploaded_bytes;
static atomic_uint_fast64_t g_current_allocated_bytes;
static atomic_uint_fast64_t g_peak_allocated_bytes;
static atomic_bool g_node_hooks_enabled;
static atomic_uint_fast64_t g_node_pre_calls;
static atomic_uint_fast64_t g_node_post_calls;
static atomic_uint_fast64_t g_current_active_nodes;
static atomic_uint_fast64_t g_peak_active_nodes;
static atomic_uint_fast64_t g_residency_acquires;
static atomic_uint_fast64_t g_residency_releases;
static atomic_uint_fast64_t g_graph_failures;

/* Last failure reason for operators. Written rarely (failure path only), so a
 * spin lock is sufficient. */
static char g_last_failure[256];
static atomic_flag g_last_failure_lock = ATOMIC_FLAG_INIT;
/* Set by operations running on graph thread 0 and consumed by the node post
 * hook on the same thread. */
static _Thread_local bool g_graph_failed;

/* ---- Backed-mode registry: which buffer offset maps to which GGUF span ---- */

struct mlz_tensor_source {
    /* Buffer-relative offset. In backed mode tensor->data is a reserved
     * (inaccessible) identity address: base + buffer_offset. */
    const void * buffer_base;
    size_t buffer_offset;
    uint64_t file_offset;
    size_t byte_len;
    uint32_t source_id;
};

/* Sorted by buffer_offset. Filled during model load by set_tensor; read-only
 * once loading finishes, so lookups need no lock. source_id is 1-based and
 * matches the index in this array. */
static struct mlz_tensor_source * g_registry;
static size_t g_registry_len;
static size_t g_registry_cap;
static atomic_bool g_backed_mode;

/* Bridged residency manager callbacks (implemented in Zig). */
static mlz_ggml_residency_acquire_fn g_acquire;
static mlz_ggml_residency_release_fn g_release;
static mlz_ggml_residency_span_fn g_span;
static mlz_ggml_residency_acquire_range_fn g_acquire_range;
static mlz_ggml_residency_range_capacity_fn g_range_capacity;
static mlz_ggml_residency_acquire_many_fn g_acquire_many;

void mlz_ggml_residency_set_bridge(
        mlz_ggml_residency_acquire_fn acquire,
        mlz_ggml_residency_release_fn release,
        mlz_ggml_residency_span_fn span,
        mlz_ggml_residency_acquire_range_fn acquire_range,
        mlz_ggml_residency_range_capacity_fn range_capacity,
        mlz_ggml_residency_acquire_many_fn acquire_many) {
    g_acquire = acquire;
    g_release = release;
    g_span = span;
    g_acquire_range = acquire_range;
    g_range_capacity = range_capacity;
    g_acquire_many = acquire_many;
}

void mlz_ggml_residency_mark_failed(const struct ggml_tensor * tensor, const char * what) {
    const char * name = tensor != NULL ? tensor->name : "?";
    fprintf(stderr, "mlz backed: %s for '%s'\n", what, name);
    while (atomic_flag_test_and_set_explicit(&g_last_failure_lock, memory_order_acquire)) {
    }
    snprintf(g_last_failure, sizeof(g_last_failure), "%s for '%s'", what, name);
    atomic_flag_clear_explicit(&g_last_failure_lock, memory_order_release);
    g_graph_failed = true;
}

size_t mlz_ggml_residency_last_failure(char * out, size_t capacity) {
    if (out == NULL || capacity == 0) {
        return 0;
    }
    while (atomic_flag_test_and_set_explicit(&g_last_failure_lock, memory_order_acquire)) {
    }
    const size_t len = strlen(g_last_failure);
    const size_t copied = len < capacity - 1 ? len : capacity - 1;
    memcpy(out, g_last_failure, copied);
    out[copied] = '\0';
    atomic_flag_clear_explicit(&g_last_failure_lock, memory_order_release);
    return copied;
}

void mlz_ggml_residency_set_backed_mode(bool enabled) {
    atomic_store_explicit(&g_backed_mode, enabled, memory_order_release);
}

bool mlz_ggml_residency_backed_mode(void) {
    return atomic_load_explicit(&g_backed_mode, memory_order_acquire);
}

/* Registers one source span. source_id must be (current length + 1) so the
 * bridge can index views by (source_id - 1). Called by the loader path only;
 * registry is read-only once graph execution begins. */
void mlz_ggml_residency_registry_add(
        const void * buffer_base,
        size_t buffer_offset, uint64_t file_offset, size_t byte_len) {
    if (g_registry_len == g_registry_cap) {
        const size_t next_cap = g_registry_cap != 0 ? g_registry_cap * 2 : 256;
        struct mlz_tensor_source * grown =
            realloc(g_registry, next_cap * sizeof(*grown));
        if (grown == NULL) {
            abort();
        }
        g_registry = grown;
        g_registry_cap = next_cap;
    }
    g_registry[g_registry_len].buffer_base = buffer_base;
    g_registry[g_registry_len].buffer_offset = buffer_offset;
    g_registry[g_registry_len].file_offset = file_offset;
    g_registry[g_registry_len].byte_len = byte_len;
    g_registry[g_registry_len].source_id = (uint32_t) (g_registry_len + 1);
    g_registry_len += 1;
}

size_t mlz_ggml_residency_registry_count(void) {
    return g_registry_len;
}

/* Returns the span registered at 1-based `index`. */
bool mlz_ggml_residency_registry_span(
        size_t index, uint64_t * file_offset, size_t * byte_len) {
    if (index == 0 || index > g_registry_len ||
        file_offset == NULL || byte_len == NULL) {
        return false;
    }
    *file_offset = g_registry[index - 1].file_offset;
    *byte_len = g_registry[index - 1].byte_len;
    return true;
}

/* Finds the source span starting at `buffer_offset`. Registrations follow the
 * loader's tensor order, not necessarily buffer-offset order, so this must
 * not use binary search unless the registry is explicitly sorted. Model load
 * is a one-time path and the registry is small enough for a linear scan. */
static const struct mlz_tensor_source * mlz_registry_find(
        const void * buffer_base, size_t buffer_offset, size_t byte_len) {
    for (size_t index = 0; index < g_registry_len; ++index) {
        const struct mlz_tensor_source * entry = &g_registry[index];
        if (entry->buffer_base == buffer_base &&
            entry->buffer_offset == buffer_offset &&
            byte_len <= entry->byte_len) {
            return entry;
        }
    }
    return NULL;
}

static const struct mlz_tensor_source * mlz_registry_find_file(
        uint64_t file_offset, size_t byte_len) {
    for (size_t index = 0; index < g_registry_len; ++index) {
        const struct mlz_tensor_source * entry = &g_registry[index];
        if (entry->file_offset == file_offset && byte_len <= entry->byte_len) {
            return entry;
        }
    }
    return NULL;
}

static void mlz_update_atomic_peak(atomic_uint_fast64_t * peak_counter, uint_fast64_t current) {
    uint_fast64_t peak = atomic_load_explicit(peak_counter, memory_order_relaxed);
    while (peak < current &&
           !atomic_compare_exchange_weak_explicit(
               peak_counter, &peak, current,
               memory_order_relaxed, memory_order_relaxed)) {
    }
}

static void mlz_update_peak(uint_fast64_t current) {
    mlz_update_atomic_peak(&g_peak_allocated_bytes, current);
}

static void * mlz_aligned_alloc(size_t alignment, size_t size) {
    if (size == 0) {
        return NULL;
    }
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void * ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void mlz_aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* Reserve (but do not commit/map) address space for backed buffers.
 * tensor->data points here as a stable identity address; any access without
 * a live node-hook mapping faults, which is exactly the safety property we
 * want: bytes only exist while a kernel is executing. */
static void * mlz_reserved_alloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
#if defined(_WIN32)
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
#else
    void * ptr = mmap(NULL, size, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return ptr == MAP_FAILED ? NULL : ptr;
#endif
}

static void mlz_reserved_free(void * ptr, size_t size) {
    if (ptr == NULL) {
        return;
    }
#if defined(_WIN32)
    (void) size;
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);
#endif
}

static bool mlz_tensor_range_is_valid(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        size_t offset,
        size_t size) {
    if (buffer == NULL || tensor == NULL || buffer->context == NULL ||
        buffer->buft != &g_mlz_buft) {
        return false;
    }

    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) buffer->context;
    const size_t tensor_size = ggml_nbytes(tensor);
    if (offset > tensor_size || size > tensor_size - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    if (context->base == NULL || tensor->data == NULL) {
        return false;
    }

    const uintptr_t base = (uintptr_t) context->base;
    const uintptr_t data = (uintptr_t) tensor->data;
    if (data < base) {
        return false;
    }
    const uintptr_t relative_u = data - base;
    const size_t relative = (size_t) relative_u;
    if (relative > context->size || offset > context->size - relative) {
        return false;
    }
    return size <= context->size - relative - offset;
}

static void mlz_require_tensor_range(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        size_t offset,
        size_t size) {
    if (!mlz_tensor_range_is_valid(buffer, tensor, offset, size)) {
        abort();
    }
}

static const char * mlz_buft_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return "MLzResidency";
}

static void mlz_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL || buffer->buft != &g_mlz_buft) {
        return;
    }
    struct mlz_buffer_context * context =
        (struct mlz_buffer_context *) buffer->context;
    if (context == NULL) {
        return;
    }

    if (context->backed) {
        mlz_reserved_free(context->base, context->size);
    } else {
        mlz_aligned_free(context->base);
    }
    atomic_fetch_add_explicit(&g_buffers_freed, 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(
        &g_current_allocated_bytes, context->size, memory_order_relaxed);
    free(context);
}

static void * mlz_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (buffer == NULL || buffer->buft != &g_mlz_buft) {
        return NULL;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) buffer->context;
    return context != NULL ? context->base : NULL;
}

static enum ggml_status mlz_buffer_init_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor) {
    if (!mlz_tensor_range_is_valid(buffer, tensor, 0, ggml_nbytes(tensor))) {
        return GGML_STATUS_FAILED;
    }
    atomic_fetch_add_explicit(&g_tensors_initialized, 1, memory_order_relaxed);
    return GGML_STATUS_SUCCESS;
}

static void mlz_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    mlz_require_tensor_range(buffer, tensor, offset, size);
    struct mlz_buffer_context * context =
        (struct mlz_buffer_context *) buffer->context;
    if (context != NULL && context->backed && size != 0) {
        /* Weight bytes live in the GGUF file; memset has no meaning here. */
        abort();
    }
    if (size != 0) {
        memset((uint8_t *) tensor->data + offset, value, size);
    }
}

static void mlz_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    mlz_require_tensor_range(buffer, tensor, offset, size);
    if (size != 0) {
        if (data == NULL) {
            abort();
        }
        struct mlz_buffer_context * context =
            (struct mlz_buffer_context *) buffer->context;
        if (context != NULL && context->backed) {
            /* Backed mode: no bytes are copied. tensor->data is a reserved
             * identity address (base + buffer_offset); resolve the GGUF span
             * from the bridge (by tensor name) and register it so node hooks
             * can map the bytes at execution time. Uploads must cover the
             * whole tensor in one call, matching how llama-model-loader loads
             * each weight. */
            if (offset != 0 || size != ggml_nbytes(tensor)) {
                fprintf(stderr, "mlz backed: partial upload for '%s'\n", tensor->name);
                abort();
            }
            if (g_span == NULL) {
                fprintf(stderr, "mlz backed: span callback missing\n");
                abort();
            }
            uint64_t file_offset = 0;
            size_t byte_len = 0;
            if (!g_span(tensor->name, &file_offset, &byte_len) ||
                byte_len != size) {
                fprintf(stderr, "mlz backed: no span for '%s' (len %zu vs %zu)\n",
                        tensor->name, byte_len, size);
                abort();
            }
            const uintptr_t base = (uintptr_t) context->base;
            const uintptr_t data_addr = (uintptr_t) tensor->data;
            if (data_addr < base) {
                abort();
            }
            mlz_ggml_residency_registry_add(
                context->base,
                (size_t) (data_addr - base), file_offset, size);
            atomic_fetch_add_explicit(&g_tensor_uploads, 1, memory_order_relaxed);
            /* Nothing was uploaded in backed mode: only the span was
             * registered. */
            return;
        }
        memmove((uint8_t *) tensor->data + offset, data, size);
    }
    atomic_fetch_add_explicit(&g_tensor_uploads, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_uploaded_bytes, size, memory_order_relaxed);
}

static void mlz_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    mlz_require_tensor_range(buffer, tensor, offset, size);
    struct mlz_buffer_context * context =
        (struct mlz_buffer_context *) buffer->context;
    if (context != NULL && context->backed && size != 0) {
        /* Read-back path (only used for e.g. output embeddings copy). Map the
         * tensor's GGUF span temporarily through the bridge. */
        if (g_acquire == NULL || g_release == NULL ||
            context->base == NULL || tensor->data == NULL) {
            abort();
        }
        const uintptr_t base = (uintptr_t) context->base;
        const uintptr_t tensor_data = (uintptr_t) tensor->data;
        if (tensor_data < base) {
            abort();
        }
        const struct mlz_tensor_source * source = mlz_registry_find(
            context->base, (size_t) (tensor_data - base), ggml_nbytes(tensor));
        if (source == NULL || source->byte_len < offset + size) {
            abort();
        }
        uint64_t pin_token = 0;
        void * mapped = g_acquire(
            source->source_id, source->file_offset, source->byte_len, &pin_token);
        if (mapped == NULL) {
            fprintf(stderr, "mlz backed get: acquire failed for '%s'\n", tensor->name);
            abort();
        }
        memmove(data, (const uint8_t *) mapped + offset, size);
        if (!g_release(pin_token)) {
            abort();
        }
        return;
    }
    if (size != 0) {
        if (data == NULL) {
            abort();
        }
        memmove(data, (const uint8_t *) tensor->data + offset, size);
    }
}

static bool mlz_tensors_have_same_layout(
        const struct ggml_tensor * src,
        const struct ggml_tensor * dst) {
    if (src == NULL || dst == NULL || src->type != dst->type) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (src->ne[dimension] != dst->ne[dimension] ||
            src->nb[dimension] != dst->nb[dimension]) {
            return false;
        }
    }
    return true;
}

static bool mlz_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst) {
    if (src == NULL || dst == NULL || src->data == NULL ||
        !mlz_tensors_have_same_layout(src, dst)) {
        return false;
    }
    ggml_backend_buffer_t src_buffer =
        src->view_src != NULL ? src->view_src->buffer : src->buffer;
    if (src_buffer == NULL || !ggml_backend_buffer_is_host(src_buffer)) {
        return false;
    }

    struct mlz_buffer_context * dst_context =
        (struct mlz_buffer_context *) buffer->context;
    if (dst_context != NULL && dst_context->backed) {
        /* Backed destination means immutable GGUF-backed weights; copying
         * into them is not a supported flow. */
        return false;
    }

    const size_t src_size = ggml_nbytes(src);
    const size_t dst_size = ggml_nbytes(dst);
    if (src_size != dst_size || !mlz_tensor_range_is_valid(buffer, dst, 0, src_size)) {
        return false;
    }

    const ggml_backend_buffer_t src_owner =
        src->buffer != NULL ? src->buffer : src_buffer;
    struct mlz_buffer_context * src_context =
        src_owner != NULL && src_owner->buft == &g_mlz_buft
            ? (struct mlz_buffer_context *) src_owner->context
            : NULL;
    if (src_context != NULL && src_context->backed) {
        /* Read a GGUF-backed source through a temporary mapping. */
        if (g_acquire == NULL || g_release == NULL ||
            src_context->base == NULL || src->data == NULL) {
            return false;
        }
        const uintptr_t base = (uintptr_t) src_context->base;
        const uintptr_t src_data = (uintptr_t) src->data;
        if (src_data < base) {
            return false;
        }
        const struct mlz_tensor_source * source = mlz_registry_find(
            src_context->base, (size_t) (src_data - base), src_size);
        if (source == NULL || source->byte_len != src_size) {
            return false;
        }
        uint64_t pin_token = 0;
        void * mapped = g_acquire(
            source->source_id, source->file_offset, source->byte_len, &pin_token);
        if (mapped == NULL) {
            return false;
        }
        memmove(dst->data, mapped, src_size);
        if (!g_release(pin_token)) {
            abort();
        }
        return true;
    }

    if (src_size != 0) {
        memmove(dst->data, src->data, src_size);
    }
    atomic_fetch_add_explicit(&g_tensor_uploads, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_uploaded_bytes, src_size, memory_order_relaxed);
    return true;
}

static void mlz_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer == NULL || buffer->buft != &g_mlz_buft) {
        return;
    }
    struct mlz_buffer_context * context =
        (struct mlz_buffer_context *) buffer->context;
    if (context != NULL && context->base != NULL) {
        if (context->backed) {
            /* Reserved address space is not accessible by design. */
            return;
        }
        memset(context->base, value, context->size);
    }
}

static const struct ggml_backend_buffer_i mlz_buffer_iface = {
    /* .free_buffer   = */ mlz_buffer_free,
    /* .get_base      = */ mlz_buffer_get_base,
    /* .init_tensor   = */ mlz_buffer_init_tensor,
    /* .memset_tensor = */ mlz_buffer_memset_tensor,
    /* .set_tensor    = */ mlz_buffer_set_tensor,
    /* .get_tensor    = */ mlz_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ mlz_buffer_cpy_tensor,
    /* .clear         = */ mlz_buffer_clear,
    /* .reset         = */ NULL,
};

static ggml_backend_buffer_t mlz_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    if (buft != &g_mlz_buft) {
        return NULL;
    }
    struct mlz_buffer_context * context =
        (struct mlz_buffer_context *) calloc(1, sizeof(*context));
    if (context == NULL) {
        return NULL;
    }

    context->size = size;
    context->backed = atomic_load_explicit(&g_backed_mode, memory_order_relaxed) ? 1 : 0;
    if (context->backed) {
        context->base = mlz_reserved_alloc(size);
    } else {
        context->base = mlz_aligned_alloc(MLZ_GGML_RESIDENCY_ALIGNMENT, size);
    }
    if (size != 0 && context->base == NULL) {
        free(context);
        return NULL;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, mlz_buffer_iface, context, size);
    if (buffer == NULL) {
        mlz_aligned_free(context->base);
        free(context);
        return NULL;
    }

    atomic_fetch_add_explicit(&g_buffers_allocated, 1, memory_order_relaxed);
    const uint_fast64_t current = atomic_fetch_add_explicit(
        &g_current_allocated_bytes, size, memory_order_relaxed) + size;
    mlz_update_peak(current);
    return buffer;
}

static size_t mlz_buft_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return MLZ_GGML_RESIDENCY_ALIGNMENT;
}

static bool mlz_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return true;
}

static struct ggml_backend_buffer_type g_mlz_buft = {
    /* .iface = */ {
        /* .get_name       = */ mlz_buft_name,
        /* .alloc_buffer   = */ mlz_buft_alloc_buffer,
        /* .get_alignment  = */ mlz_buft_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ NULL,
        /* .is_host        = */ mlz_buft_is_host,
    },
    /* .device  = */ NULL,
    /* .context = */ NULL,
};

ggml_backend_buffer_type_t mlz_ggml_residency_buffer_type(void) {
    return &g_mlz_buft;
}

void mlz_ggml_residency_registry_reset(void) {
    free(g_registry);
    g_registry = NULL;
    g_registry_len = 0;
    g_registry_cap = 0;
    atomic_store_explicit(&g_residency_acquires, 0, memory_order_relaxed);
    atomic_store_explicit(&g_residency_releases, 0, memory_order_relaxed);
}

void mlz_ggml_residency_reset_stats(void) {
    atomic_store_explicit(&g_buffers_allocated, 0, memory_order_relaxed);
    atomic_store_explicit(&g_buffers_freed, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tensors_initialized, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tensor_uploads, 0, memory_order_relaxed);
    atomic_store_explicit(&g_uploaded_bytes, 0, memory_order_relaxed);
    atomic_store_explicit(&g_node_pre_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&g_node_post_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&g_graph_failures, 0, memory_order_relaxed);

    // Live gauges are not interval counters. Preserve them so a reset during
    // allocation or node execution cannot make a later decrement underflow.
    const uint_fast64_t current = atomic_load_explicit(
        &g_current_allocated_bytes, memory_order_relaxed);
    atomic_store_explicit(&g_peak_allocated_bytes, current, memory_order_relaxed);
    mlz_update_atomic_peak(
        &g_peak_allocated_bytes,
        atomic_load_explicit(&g_current_allocated_bytes, memory_order_relaxed));
    const uint_fast64_t active = atomic_load_explicit(
        &g_current_active_nodes, memory_order_relaxed);
    atomic_store_explicit(&g_peak_active_nodes, active, memory_order_relaxed);
    mlz_update_atomic_peak(
        &g_peak_active_nodes,
        atomic_load_explicit(&g_current_active_nodes, memory_order_relaxed));
}

struct mlz_ggml_residency_stats mlz_ggml_residency_get_stats(void) {
    struct mlz_ggml_residency_stats stats;
    stats.buffers_allocated = atomic_load_explicit(&g_buffers_allocated, memory_order_relaxed);
    stats.buffers_freed = atomic_load_explicit(&g_buffers_freed, memory_order_relaxed);
    stats.tensors_initialized = atomic_load_explicit(&g_tensors_initialized, memory_order_relaxed);
    stats.tensor_uploads = atomic_load_explicit(&g_tensor_uploads, memory_order_relaxed);
    stats.uploaded_bytes = atomic_load_explicit(&g_uploaded_bytes, memory_order_relaxed);
    stats.current_allocated_bytes = atomic_load_explicit(&g_current_allocated_bytes, memory_order_relaxed);
    stats.peak_allocated_bytes = atomic_load_explicit(&g_peak_allocated_bytes, memory_order_relaxed);
    stats.node_pre_calls = atomic_load_explicit(&g_node_pre_calls, memory_order_relaxed);
    stats.node_post_calls = atomic_load_explicit(&g_node_post_calls, memory_order_relaxed);
    stats.current_active_nodes = atomic_load_explicit(&g_current_active_nodes, memory_order_relaxed);
    stats.peak_active_nodes = atomic_load_explicit(&g_peak_active_nodes, memory_order_relaxed);
    stats.residency_acquires = atomic_load_explicit(&g_residency_acquires, memory_order_relaxed);
    stats.residency_releases = atomic_load_explicit(&g_residency_releases, memory_order_relaxed);
    stats.graph_failures = atomic_load_explicit(&g_graph_failures, memory_order_relaxed);
    return stats;
}

bool mlz_ggml_residency_node_hooks_available(void) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    return true;
#else
    return false;
#endif
}

void mlz_ggml_residency_set_node_hooks_enabled(bool enabled) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    atomic_store_explicit(&g_node_hooks_enabled, enabled, memory_order_release);
#else
    (void) enabled;
#endif
}

bool mlz_ggml_residency_node_hooks_enabled(void) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    // A graph samples the switch once so all workers follow the same barrier
    // protocol even if another thread requests disable while it is running.
    return atomic_load_explicit(&g_node_hooks_enabled, memory_order_acquire);
#else
    return false;
#endif
}

#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
/* ---- Backed-mode tensor rebase/restore ------------------------------------ */

/* Finds the registry entry whose identity address matches tensor->data and
 * whose byte length covers the tensor's bytes. */
static const struct mlz_tensor_source * mlz_registry_for_tensor(
        const struct ggml_tensor * tensor) {
    const struct ggml_tensor * owner =
        tensor->view_src != NULL ? tensor->view_src : tensor;
    const ggml_backend_buffer_t buffer = owner->buffer;
    if (buffer == NULL || buffer->buft != &g_mlz_buft ||
        buffer->context == NULL) {
        return NULL;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) buffer->context;
    if (!context->backed || context->base == NULL || tensor->data == NULL) {
        return NULL;
    }
    const uintptr_t base = (uintptr_t) context->base;
    const uintptr_t data = (uintptr_t) tensor->data;
    if (data < base || (size_t) (data - base) > context->size) {
        return NULL;
    }
    const struct mlz_tensor_source * source = mlz_registry_find(
        context->base, (size_t) (data - base), ggml_nbytes(tensor));
    if (source != NULL) {
        return source;
    }
    /* GGML views can start inside a registered model tensor. Resolve those by
     * absolute GGUF span using the tensor name callback. */
    if (tensor->view_src != NULL && g_span != NULL) {
        uint64_t file_offset = 0;
        size_t byte_len = 0;
        if (g_span(tensor->view_src->name, &file_offset, &byte_len)) {
            return mlz_registry_find_file(file_offset, byte_len);
        }
    }
    return NULL;
}

struct mlz_node_pin {
    uint32_t source_id;
    uint64_t token;
    void * mapped_base;
};

static _Thread_local struct mlz_node_pin g_node_pins[GGML_MAX_SRC];
static _Thread_local size_t g_node_pin_count;
static _Thread_local struct ggml_tensor g_node_clones[GGML_MAX_SRC];
static _Thread_local struct ggml_tensor * g_node_original_src[GGML_MAX_SRC];

/* Byte offset of a backed tensor inside its registered source span, or
 * SIZE_MAX when the tensor does not lie completely within the span. */
static size_t mlz_source_delta(
        const struct ggml_tensor * tensor,
        const struct mlz_tensor_source * source) {
    const struct ggml_tensor * owner =
        tensor->view_src != NULL ? tensor->view_src : tensor;
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) owner->buffer->context;
    const size_t identity_offset =
        (size_t) ((uintptr_t) tensor->data - (uintptr_t) context->base);
    if (identity_offset < source->buffer_offset ||
        identity_offset - source->buffer_offset > source->byte_len ||
        ggml_nbytes(tensor) > source->byte_len -
            (identity_offset - source->buffer_offset)) {
        return SIZE_MAX;
    }
    return identity_offset - source->buffer_offset;
}

/* A graph node is context-local; model weight tensors are shared. Replace only
 * this node's source pointer with a thread-local copy so another context never
 * observes a transient tensor->data rebase. `mapped` is the pinned base of the
 * source span, or NULL for tiled sources that are mapped per tile. */
static void mlz_clone_source(
        struct ggml_tensor * node, int source_index, void * mapped, size_t delta) {
    struct ggml_tensor * tensor = node->src[source_index];
    g_node_original_src[source_index] = tensor;
    g_node_clones[source_index] = *tensor;
    node->src[source_index] = &g_node_clones[source_index];
    if (mapped != NULL) {
        g_node_clones[source_index].data = (uint8_t *) mapped + delta;
    }
}

static void mlz_release_node_pins(void) {
    for (size_t index = g_node_pin_count; index > 0; --index) {
        const struct mlz_node_pin * pin = &g_node_pins[index - 1];
        if (!g_release(pin->token)) {
            /* A token the bridge does not know is a bookkeeping bug, not a
             * recoverable resource failure. */
            fprintf(stderr, "mlz backed: pin release failed for source %u\n", pin->source_id);
            abort();
        }
        atomic_fetch_add_explicit(&g_residency_releases, 1, memory_order_relaxed);
    }
    g_node_pin_count = 0;
}

static void mlz_restore_node_sources(struct ggml_tensor * node) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (g_node_original_src[i] != NULL) {
            node->src[i] = g_node_original_src[i];
            g_node_original_src[i] = NULL;
        }
    }
}

static bool mlz_is_backed_source(const struct ggml_tensor * src) {
    const struct ggml_tensor * owner = src->view_src != NULL ? src->view_src : src;
    return owner->buffer != NULL &&
        owner->buffer->buft == &g_mlz_buft &&
        owner->buffer->context != NULL &&
        ((const struct mlz_buffer_context *) owner->buffer->context)->backed;
}

#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */

bool mlz_ggml_residency_should_tile_mul_mat(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (node == NULL || node->op != GGML_OP_MUL_MAT ||
        !atomic_load_explicit(&g_backed_mode, memory_order_acquire) ||
        !atomic_load_explicit(&g_node_hooks_enabled, memory_order_acquire) ||
        g_acquire_range == NULL || g_range_capacity == NULL || g_release == NULL) {
        return false;
    }
    struct ggml_tensor * src0 = node->src[0];
    if (src0 == NULL || src0->view_src != NULL || src0->data == NULL ||
        ggml_n_dims(src0) != 2 ||
        src0->nb[0] != ggml_type_size(src0->type) ||
        src0->nb[1] != ggml_row_size(src0->type, src0->ne[0]) ||
        src0->nb[2] != src0->nb[1] * (size_t) src0->ne[1] ||
        src0->nb[3] != src0->nb[2]) {
        return false;
    }
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(src0);
    if (source == NULL) {
        return false;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) src0->buffer->context;
    if (context == NULL || !context->backed || context->base == NULL) {
        return false;
    }
    const size_t identity_offset =
        (size_t) ((uintptr_t) src0->data - (uintptr_t) context->base);
    if (identity_offset < source->buffer_offset) {
        return false;
    }
    const size_t source_offset = identity_offset - source->buffer_offset;
    const size_t tensor_bytes = ggml_nbytes(src0);
    if (source_offset > source->byte_len ||
        tensor_bytes > source->byte_len - source_offset) {
        return false;
    }
    const size_t capacity = g_range_capacity(source->source_id, source_offset);
    return capacity < tensor_bytes && capacity >= src0->nb[1];
#else
    (void) node;
    return false;
#endif
}

bool mlz_ggml_residency_should_tile_mul_mat_id(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (node == NULL || node->op != GGML_OP_MUL_MAT_ID ||
        !atomic_load_explicit(&g_backed_mode, memory_order_acquire) ||
        !atomic_load_explicit(&g_node_hooks_enabled, memory_order_acquire) ||
        g_acquire_range == NULL || g_range_capacity == NULL || g_release == NULL) {
        return false;
    }
    struct ggml_tensor * src0 = node->src[0];
    if (src0 == NULL || src0->view_src != NULL || src0->data == NULL ||
        src0->ne[3] != 1 ||
        src0->nb[0] != ggml_type_size(src0->type) ||
        src0->nb[1] != ggml_row_size(src0->type, src0->ne[0]) ||
        src0->nb[2] != src0->nb[1] * (size_t) src0->ne[1] ||
        src0->nb[3] != src0->nb[2] * (size_t) src0->ne[2]) {
        return false;
    }
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(src0);
    if (source == NULL) {
        return false;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) src0->buffer->context;
    if (context == NULL || !context->backed || context->base == NULL) {
        return false;
    }
    const size_t identity_offset =
        (size_t) ((uintptr_t) src0->data - (uintptr_t) context->base);
    if (identity_offset < source->buffer_offset) {
        return false;
    }
    const size_t source_offset = identity_offset - source->buffer_offset;
    const size_t tensor_bytes = ggml_nbytes(src0);
    if (source_offset > source->byte_len ||
        tensor_bytes > source->byte_len - source_offset) {
        return false;
    }
    const size_t capacity = g_range_capacity(source->source_id, source_offset);
    return capacity < tensor_bytes && capacity >= src0->nb[1];
#else
    (void) node;
    return false;
#endif
}

size_t mlz_ggml_residency_tile_capacity(
        struct ggml_tensor * tensor, size_t tensor_offset) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (tensor == NULL || g_range_capacity == NULL) {
        return 0;
    }
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(tensor);
    if (source == NULL) {
        return 0;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) tensor->buffer->context;
    if (context == NULL || !context->backed || context->base == NULL) {
        return 0;
    }
    const size_t identity_offset =
        (size_t) ((uintptr_t) tensor->data - (uintptr_t) context->base);
    if (identity_offset < source->buffer_offset) {
        return 0;
    }
    const size_t tensor_source_offset = identity_offset - source->buffer_offset;
    if (tensor_source_offset > source->byte_len ||
        tensor_offset >= ggml_nbytes(tensor) ||
        tensor_offset > source->byte_len - tensor_source_offset) {
        return 0;
    }
    const size_t source_offset = tensor_source_offset + tensor_offset;
    return g_range_capacity(source->source_id, source_offset);
#else
    (void) tensor;
    (void) tensor_offset;
    return 0;
#endif
}

#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
struct mlz_tile_pin {
    struct ggml_tensor * tensor;
    uint64_t token;
    size_t identity_offset;
    bool active;
};
static _Thread_local struct mlz_tile_pin g_tile_pin;
#endif

bool mlz_ggml_residency_tile_acquire(
        struct ggml_tensor * tensor, size_t tensor_offset, size_t byte_len) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (tensor == NULL || byte_len == 0 || g_tile_pin.active ||
        g_acquire_range == NULL || tensor_offset > ggml_nbytes(tensor) ||
        byte_len > ggml_nbytes(tensor) - tensor_offset) {
        return false;
    }
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(tensor);
    if (source == NULL) {
        return false;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) tensor->buffer->context;
    if (context == NULL || !context->backed || context->base == NULL) {
        return false;
    }
    const size_t identity_offset =
        (size_t) ((uintptr_t) tensor->data - (uintptr_t) context->base);
    if (identity_offset < source->buffer_offset) {
        return false;
    }
    const size_t tensor_source_offset = identity_offset - source->buffer_offset;
    if (tensor_source_offset > source->byte_len ||
        tensor_offset > source->byte_len - tensor_source_offset ||
        byte_len > source->byte_len - tensor_source_offset - tensor_offset) {
        return false;
    }
    uint64_t token = 0;
    void * mapped = g_acquire_range(
        source->source_id, tensor_source_offset + tensor_offset, byte_len, &token);
    if (mapped == NULL) {
        return false;
    }
    /* one_chunk indexes rows using their absolute ir0, so bias the base back
     * by the tile's tensor-relative offset. It only dereferences addresses
     * inside the acquired tile. */
    tensor->data = (void *) ((uintptr_t) mapped - tensor_offset);
    g_tile_pin = (struct mlz_tile_pin) {
        .tensor = tensor,
        .token = token,
        .identity_offset = identity_offset,
        .active = true,
    };
    atomic_fetch_add_explicit(&g_residency_acquires, 1, memory_order_relaxed);
    return true;
#else
    (void) tensor;
    (void) tensor_offset;
    (void) byte_len;
    return false;
#endif
}

bool mlz_ggml_residency_tile_release(struct ggml_tensor * tensor) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (!g_tile_pin.active || tensor == NULL || g_tile_pin.tensor != tensor ||
        g_release == NULL) {
        return false;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) tensor->buffer->context;
    if (context == NULL || !context->backed || context->base == NULL) {
        return false;
    }
    if (!g_release(g_tile_pin.token)) {
        return false;
    }
    tensor->data = (void *) ((uintptr_t) context->base + g_tile_pin.identity_offset);
    g_tile_pin.active = false;
    atomic_fetch_add_explicit(&g_residency_releases, 1, memory_order_relaxed);
    return true;
#else
    (void) tensor;
    return false;
#endif
}

#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
static bool mlz_can_range_get_rows(struct ggml_tensor * node) {
    if (node == NULL || node->op != GGML_OP_GET_ROWS ||
        g_acquire_range == NULL || g_range_capacity == NULL || g_release == NULL ||
        node->src[0] == NULL || node->src[1] == NULL ||
        node->src[1]->type != GGML_TYPE_I32) {
        return false;
    }
    const struct ggml_tensor * index_owner =
        node->src[1]->view_src != NULL ? node->src[1]->view_src : node->src[1];
    if (index_owner->buffer != NULL &&
        index_owner->buffer->buft == &g_mlz_buft &&
        index_owner->buffer->context != NULL &&
        ((const struct mlz_buffer_context *) index_owner->buffer->context)->backed) {
        return false;
    }
    const struct ggml_tensor * tensor = node->src[0];
    return tensor->nb[0] == ggml_type_size(tensor->type) &&
        mlz_registry_for_tensor(tensor) != NULL;
}
#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */

bool mlz_ggml_residency_get_rows_sparse(struct ggml_tensor * node, int ith) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    if (!mlz_can_range_get_rows(node)) return false;
    if (ith != 0) return true; // one pin per source; other workers meet at node-post barrier

    const struct ggml_tensor * tensor = node->src[0];
    const struct ggml_tensor * indices = node->src[1];
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(tensor);
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) (tensor->view_src != NULL ?
            tensor->view_src : tensor)->buffer->context;
    const size_t identity_offset =
        (size_t) ((uintptr_t) tensor->data - (uintptr_t) context->base);
    const size_t source_offset = identity_offset >= source->buffer_offset ?
        identity_offset - source->buffer_offset : SIZE_MAX;
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const int64_t count = ggml_nelements(indices);
    const struct ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (identity_offset < source->buffer_offset || source_offset > source->byte_len ||
        row_bytes == 0 || count < 1 ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32 &&
         tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16 &&
         (traits == NULL || traits->to_float == NULL))) {
        mlz_ggml_residency_mark_failed(tensor, "unsupported GET_ROWS layout");
        return true;
    }

    for (int64_t i = 0; i < count; ++i) {
        const int64_t i12 = i / (indices->ne[1] * indices->ne[0]);
        const int64_t i11 = (i - i12 * indices->ne[1] * indices->ne[0]) / indices->ne[0];
        const int64_t i10 = i - i12 * indices->ne[1] * indices->ne[0] - i11 * indices->ne[0];
        const int32_t row = *(const int32_t *) ((const char *) indices->data +
            i10 * indices->nb[0] + i11 * indices->nb[1] + i12 * indices->nb[2]);
        const size_t available = source->byte_len - source_offset;
        if (tensor->nb[1] == 0 || tensor->nb[2] == 0 || tensor->nb[3] == 0 ||
            row < 0 || row >= tensor->ne[1] ||
            (size_t) row > available / tensor->nb[1] ||
            (size_t) i11 > available / tensor->nb[2] ||
            (size_t) i12 > available / tensor->nb[3]) {
            mlz_ggml_residency_mark_failed(tensor, "invalid GET_ROWS index/layout");
            return true;
        }
        const size_t row_offset = (size_t) row * tensor->nb[1];
        const size_t plane_offset = (size_t) i11 * tensor->nb[2];
        const size_t volume_offset = (size_t) i12 * tensor->nb[3];
        if (plane_offset > available - row_offset ||
            volume_offset > available - row_offset - plane_offset) {
            mlz_ggml_residency_mark_failed(tensor, "GET_ROWS strides exceed source span");
            return true;
        }
        const size_t offset = row_offset + plane_offset + volume_offset;
        if (offset > available || row_bytes > available - offset ||
            g_range_capacity(source->source_id, source_offset + offset) < row_bytes) {
            mlz_ggml_residency_mark_failed(tensor, "GET_ROWS row exceeds weight budget or source span");
            return true;
        }
        uint64_t token = 0;
        void * mapped = g_acquire_range(source->source_id, source_offset + offset, row_bytes, &token);
        if (mapped == NULL) {
            mlz_ggml_residency_mark_failed(tensor, "GET_ROWS acquire failed");
            return true;
        }
        atomic_fetch_add_explicit(&g_residency_acquires, 1, memory_order_relaxed);
        void * dst = (char *) node->data + i10 * node->nb[1] +
            i11 * node->nb[2] + i12 * node->nb[3];
        if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
            memcpy(dst, mapped, row_bytes);
        } else if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) mapped, (float *) dst, tensor->ne[0]);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) mapped, (float *) dst, tensor->ne[0]);
        } else {
            traits->to_float(mapped, (float *) dst, tensor->ne[0]);
        }
        if (!g_release(token)) {
            fprintf(stderr, "mlz backed: GET_ROWS release failed for '%s'\n", tensor->name);
            abort();
        }
        atomic_fetch_add_explicit(&g_residency_releases, 1, memory_order_relaxed);
    }
    return true;
#else
    (void) node;
    (void) ith;
    return false;
#endif
}

bool mlz_ggml_residency_node_pre(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    atomic_fetch_add_explicit(&g_node_pre_calls, 1, memory_order_relaxed);
    const uint_fast64_t active = atomic_fetch_add_explicit(
        &g_current_active_nodes, 1, memory_order_relaxed) + 1;
    mlz_update_atomic_peak(&g_peak_active_nodes, active);

    g_node_pin_count = 0;
    g_graph_failed = false;
    memset(g_node_original_src, 0, sizeof(g_node_original_src));
    /* Backed mode: clone each shared model-weight source into this graph node
     * before mapping it. Other contexts keep the original identity pointer. */
    if (!atomic_load_explicit(&g_backed_mode, memory_order_relaxed) ||
        g_acquire_many == NULL || g_release == NULL) {
        return true;
    }
    const bool tiled_mul_mat = mlz_ggml_residency_should_tile_mul_mat(node);
    const bool tiled_mul_mat_id =
        mlz_ggml_residency_should_tile_mul_mat_id(node);
    const bool ranged_get_rows = mlz_can_range_get_rows(node);
    if (node->op == GGML_OP_GET_ROWS && node->src[0] != NULL &&
        mlz_registry_for_tensor(node->src[0]) != NULL && !ranged_get_rows) {
        mlz_ggml_residency_mark_failed(node->src[0], "unsupported GET_ROWS layout");
        g_graph_failed = false;
        atomic_fetch_add_explicit(&g_graph_failures, 1, memory_order_relaxed);
        return false;
    }

    /* Pass 1: collect distinct whole-span sources and pin them together so a
     * node never holds part of its inputs while waiting for the rest. */
    uint32_t source_ids[GGML_MAX_SRC];
    size_t deltas[GGML_MAX_SRC];
    int source_slot[GGML_MAX_SRC];
    size_t source_count = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        source_slot[i] = -1;
        struct ggml_tensor * src = node->src[i];
        if (src == NULL) {
            break;
        }
        if (i == 0 && (tiled_mul_mat || tiled_mul_mat_id || ranged_get_rows)) {
            continue;
        }
        if (!mlz_is_backed_source(src)) {
            continue;
        }
        const struct mlz_tensor_source * source = mlz_registry_for_tensor(src);
        const size_t delta = source != NULL ? mlz_source_delta(src, source) : SIZE_MAX;
        if (delta == SIZE_MAX) {
            mlz_ggml_residency_mark_failed(src, "unregistered or out-of-span weight view");
            g_graph_failed = false;
            atomic_fetch_add_explicit(&g_graph_failures, 1, memory_order_relaxed);
            return false;
        }
        deltas[i] = delta;
        size_t slot = 0;
        while (slot < source_count && source_ids[slot] != source->source_id) {
            ++slot;
        }
        if (slot == source_count) {
            source_ids[source_count++] = source->source_id;
        }
        source_slot[i] = (int) slot;
    }

    void * mapped[GGML_MAX_SRC];
    uint64_t tokens[GGML_MAX_SRC];
    if (source_count != 0) {
        if (!g_acquire_many(source_count, source_ids, mapped, tokens)) {
            const struct ggml_tensor * weight = node;
            for (int i = 0; i < GGML_MAX_SRC && node->src[i] != NULL; ++i) {
                if (source_slot[i] == 0) {
                    weight = node->src[i];
                    break;
                }
            }
            mlz_ggml_residency_mark_failed(weight, "weight acquisition failed");
            g_graph_failed = false;
            atomic_fetch_add_explicit(&g_graph_failures, 1, memory_order_relaxed);
            return false;
        }
        for (size_t slot = 0; slot < source_count; ++slot) {
            g_node_pins[slot] = (struct mlz_node_pin) {
                .source_id = source_ids[slot],
                .token = tokens[slot],
                .mapped_base = mapped[slot],
            };
        }
        g_node_pin_count = source_count;
        atomic_fetch_add_explicit(&g_residency_acquires, source_count, memory_order_relaxed);
    }

    /* Pass 2: rebase node-local clones. Nothing below can fail. */
    for (int i = 0; i < GGML_MAX_SRC && node->src[i] != NULL; ++i) {
        if (i == 0 && (tiled_mul_mat || tiled_mul_mat_id)) {
            mlz_clone_source(node, i, NULL, 0);
        } else if (source_slot[i] >= 0) {
            mlz_clone_source(node, i, mapped[source_slot[i]], deltas[i]);
        }
    }
    return true;
#else
    (void) node;
    return true;
#endif
}

bool mlz_ggml_residency_node_post(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    atomic_fetch_add_explicit(&g_node_post_calls, 1, memory_order_relaxed);
    uint_fast64_t active = atomic_load_explicit(
        &g_current_active_nodes, memory_order_relaxed);
    while (active != 0 &&
           !atomic_compare_exchange_weak_explicit(
               &g_current_active_nodes, &active, active - 1,
               memory_order_relaxed, memory_order_relaxed)) {
    }

    if (atomic_load_explicit(&g_backed_mode, memory_order_relaxed) && g_release != NULL) {
        mlz_release_node_pins();
    }
    mlz_restore_node_sources(node);
    const bool failed = g_graph_failed;
    g_graph_failed = false;
    if (failed) {
        atomic_fetch_add_explicit(&g_graph_failures, 1, memory_order_relaxed);
    }
    return !failed;
#else
    (void) node;
    return true;
#endif
}
