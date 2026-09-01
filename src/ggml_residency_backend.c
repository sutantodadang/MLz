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
typedef void * (*mlz_residency_acquire_fn)(uint32_t source_id, uint64_t file_offset, size_t byte_len);
typedef bool (*mlz_residency_release_fn)(uint32_t source_id);
typedef bool (*mlz_residency_span_fn)(const char * tensor_name, uint64_t * file_offset, size_t * byte_len);

static mlz_residency_acquire_fn g_acquire;
static mlz_residency_release_fn g_release;
static mlz_residency_span_fn g_span;

void mlz_ggml_residency_set_bridge(
        mlz_residency_acquire_fn acquire,
        mlz_residency_release_fn release,
        mlz_residency_span_fn span) {
    g_acquire = acquire;
    g_release = release;
    g_span = span;
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
        void * mapped = g_acquire(
            source->source_id, source->file_offset, source->byte_len);
        if (mapped == NULL) {
            fprintf(stderr, "mlz backed get: acquire failed for '%s'\n", tensor->name);
            abort();
        }
        memmove(data, (const uint8_t *) mapped + offset, size);
        if (!g_release(source->source_id)) {
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
        void * mapped = g_acquire(
            source->source_id, source->file_offset, source->byte_len);
        if (mapped == NULL) {
            return false;
        }
        memmove(dst->data, mapped, src_size);
        if (!g_release(source->source_id)) {
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
    struct ggml_tensor * tensor;
    uint32_t source_id;
    size_t identity_offset;
    bool releases_source;
};

static _Thread_local struct mlz_node_pin g_node_pins[GGML_MAX_SRC];
static _Thread_local size_t g_node_pin_count;

/* Acquires the live mapping for a weight tensor and repoints tensor->data.
 * The acquired view is held in the bridge, keyed by source_id, and released
 * by the matching post-hook. Returns false when the source is unregistered or
 * the residency manager cannot map it (e.g. budget too small). */
static bool mlz_rebase_tensor(struct ggml_tensor * tensor) {
    for (size_t index = 0; index < g_node_pin_count; ++index) {
        if (g_node_pins[index].tensor == tensor) {
            return true; /* duplicate source pointer in the same node */
        }
    }
    const struct mlz_tensor_source * source = mlz_registry_for_tensor(tensor);
    if (source == NULL || g_acquire == NULL ||
        g_node_pin_count >= GGML_MAX_SRC) {
        return false;
    }
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
        return false;
    }
    void * mapped = NULL;
    bool releases_source = true;
    for (size_t index = 0; index < g_node_pin_count; ++index) {
        if (g_node_pins[index].source_id == source->source_id) {
            mapped = (uint8_t *) g_node_pins[index].tensor->data -
                (g_node_pins[index].identity_offset - source->buffer_offset);
            releases_source = false;
            break;
        }
    }
    if (mapped == NULL) {
        mapped = g_acquire(
            source->source_id,
            source->file_offset,
            source->byte_len);
    }
    if (mapped == NULL) {
        return false;
    }
    const size_t source_delta = identity_offset - source->buffer_offset;
    tensor->data = (uint8_t *) mapped + source_delta;
    g_node_pins[g_node_pin_count++] = (struct mlz_node_pin) {
        .tensor = tensor,
        .source_id = source->source_id,
        .identity_offset = identity_offset,
        .releases_source = releases_source,
    };
    if (releases_source) {
        atomic_fetch_add_explicit(&g_residency_acquires, 1, memory_order_relaxed);
    }
    return true;
}

/* Restores the reserved identity address and releases the pinned source view. */
static bool mlz_restore_tensor(
        struct ggml_tensor * tensor,
        uint32_t source_id,
        size_t identity_offset) {
    if (g_release == NULL || source_id == 0 || source_id > g_registry_len) {
        return false;
    }
    const struct ggml_tensor * owner =
        tensor->view_src != NULL ? tensor->view_src : tensor;
    const ggml_backend_buffer_t buffer = owner->buffer;
    if (buffer == NULL || buffer->buft != &g_mlz_buft ||
        buffer->context == NULL) {
        return false;
    }
    const struct mlz_buffer_context * context =
        (const struct mlz_buffer_context *) buffer->context;
    if (!context->backed || context->base == NULL || !g_release(source_id)) {
        return false;
    }
    tensor->data = (void *) ((uintptr_t) context->base + identity_offset);
    atomic_fetch_add_explicit(&g_residency_releases, 1, memory_order_relaxed);
    return true;
}

#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */

void mlz_ggml_residency_node_pre(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    atomic_fetch_add_explicit(&g_node_pre_calls, 1, memory_order_relaxed);
    const uint_fast64_t active = atomic_fetch_add_explicit(
        &g_current_active_nodes, 1, memory_order_relaxed) + 1;
    mlz_update_atomic_peak(&g_peak_active_nodes, active);

    /* Backed mode: map every model-weight source and rebase tensor->data.
     * Thread 0 records exact identity offsets for post-hook restoration. */
    if (!atomic_load_explicit(&g_backed_mode, memory_order_relaxed) ||
        g_acquire == NULL) {
        return;
    }
    g_node_pin_count = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = node->src[i];
        if (src == NULL) {
            break;
        }
        struct ggml_tensor * owner =
            src->view_src != NULL ? src->view_src : src;
        if (owner->buffer == NULL ||
            owner->buffer->buft != &g_mlz_buft ||
            owner->buffer->context == NULL) {
            continue;
        }
        struct mlz_buffer_context * context =
            (struct mlz_buffer_context *) owner->buffer->context;
        if (!context->backed) {
            continue;
        }
        /* Each source is tracked explicitly by pre-hook, so tied mappings
         * and GGML views restore the exact original identity pointer. */
        if (!mlz_rebase_tensor(src)) {
            fprintf(stderr, "mlz backed: rebase failed for '%s'\n", src->name);
            abort();
        }
    }
#endif
}

void mlz_ggml_residency_node_post(struct ggml_tensor * node) {
#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS
    atomic_fetch_add_explicit(&g_node_post_calls, 1, memory_order_relaxed);
    uint_fast64_t active = atomic_load_explicit(
        &g_current_active_nodes, memory_order_relaxed);
    while (active != 0 &&
           !atomic_compare_exchange_weak_explicit(
               &g_current_active_nodes, &active, active - 1,
               memory_order_relaxed, memory_order_relaxed)) {
    }

    if (!atomic_load_explicit(&g_backed_mode, memory_order_relaxed) ||
        g_release == NULL) {
        return;
    }
    (void) node;
    for (size_t index = g_node_pin_count; index > 0; --index) {
        const struct mlz_node_pin * pin = &g_node_pins[index - 1];
        if (pin->releases_source) {
            if (!mlz_restore_tensor(
                    pin->tensor, pin->source_id, pin->identity_offset)) {
                fprintf(stderr, "mlz backed: restore failed for '%s'\n",
                        pin->tensor->name);
                abort();
            }
        } else {
            const struct mlz_buffer_context * context =
                (const struct mlz_buffer_context *) pin->tensor->buffer->context;
            pin->tensor->data =
                (void *) ((uintptr_t) context->base + pin->identity_offset);
        }
    }
    g_node_pin_count = 0;
#endif
}
