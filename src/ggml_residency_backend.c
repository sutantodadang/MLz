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
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    include <malloc.h>
#endif

#define MLZ_GGML_RESIDENCY_ALIGNMENT 32u

struct mlz_buffer_context {
    void * base;
    size_t size;
};

static struct ggml_backend_buffer_type g_mlz_buft;

static atomic_uint_fast64_t g_buffers_allocated;
static atomic_uint_fast64_t g_buffers_freed;
static atomic_uint_fast64_t g_tensors_initialized;
static atomic_uint_fast64_t g_tensor_uploads;
static atomic_uint_fast64_t g_uploaded_bytes;
static atomic_uint_fast64_t g_current_allocated_bytes;
static atomic_uint_fast64_t g_peak_allocated_bytes;

static void mlz_update_peak(uint_fast64_t current) {
    uint_fast64_t peak = atomic_load_explicit(
        &g_peak_allocated_bytes, memory_order_relaxed);
    while (peak < current &&
           !atomic_compare_exchange_weak_explicit(
               &g_peak_allocated_bytes, &peak, current,
               memory_order_relaxed, memory_order_relaxed)) {
    }
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

    mlz_aligned_free(context->base);
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

    const size_t src_size = ggml_nbytes(src);
    const size_t dst_size = ggml_nbytes(dst);
    if (src_size != dst_size || !mlz_tensor_range_is_valid(buffer, dst, 0, src_size)) {
        return false;
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
    context->base = mlz_aligned_alloc(MLZ_GGML_RESIDENCY_ALIGNMENT, size);
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

void mlz_ggml_residency_reset_stats(void) {
    atomic_store_explicit(&g_buffers_allocated, 0, memory_order_relaxed);
    atomic_store_explicit(&g_buffers_freed, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tensors_initialized, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tensor_uploads, 0, memory_order_relaxed);
    atomic_store_explicit(&g_uploaded_bytes, 0, memory_order_relaxed);

    // current_allocated_bytes is a live gauge, not an interval counter. Keep it
    // intact so resetting while buffers exist cannot underflow it on free.
    const uint_fast64_t current = atomic_load_explicit(
        &g_current_allocated_bytes, memory_order_relaxed);
    atomic_store_explicit(&g_peak_allocated_bytes, current, memory_order_relaxed);
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
    return stats;
}
