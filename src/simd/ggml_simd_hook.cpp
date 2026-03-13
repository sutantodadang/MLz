#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-threading.h"
#include "ggml-backend-impl.h"
#include "quants.h"
#include "simd_matmul.h"
#include "flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <mutex>

// -----------------------------------------------------------------------------
// Assembly Kernel Declarations
// -----------------------------------------------------------------------------
extern "C" {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM NEON Kernels
    void simd_vec_dot_q4_0_q8_0_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_0_q8_0_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q2_k_q8_k_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q3_k_q8_k_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q4_k_q8_k_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q6_k_q8_k_neon(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_k_q8_k_neon(int n, float* result, const void* vx, const void* vy);
#else
    // AVX2 Kernels
    void simd_vec_dot_q4_0_q8_0_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_0_q8_0_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q2_k_q8_k_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q3_k_q8_k_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q4_k_q8_k_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q6_k_q8_k_avx2(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_k_q8_k_avx2(int n, float* result, const void* vx, const void* vy);

    // AVX512 Kernels (if built)
    void simd_vec_dot_q4_0_q8_0_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_0_q8_0_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q2_k_q8_k_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q3_k_q8_k_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q4_k_q8_k_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q6_k_q8_k_avx512(int n, float* result, const void* vx, const void* vy);
    void simd_vec_dot_q8_k_q8_k_avx512(int n, float* result, const void* vx, const void* vy);
#endif
}

// -----------------------------------------------------------------------------
// Activation Quantization Cache
// -----------------------------------------------------------------------------
// Avoids re-quantizing the same F32 activations (src1) across multiple matmul
// calls within the same layer/token.  Two caches: one for Q8_0, one for Q8_K.
//
// Thread safety: ggml dispatches mul_mat with thread 0 doing the quantization
// before the work starts (barrier), so the cache is only written by one thread
// at a time.  We protect mutations with a lightweight spinlock anyway.
// -----------------------------------------------------------------------------

struct quant_cache {
    uint8_t * buf      = nullptr;   // Quantised activation buffer
    size_t    buf_cap  = 0;         // Allocated capacity in bytes
    const void * src1_data = nullptr; // Pointer we quantised from
    int64_t   K        = 0;
    int64_t   N        = 0;
    size_t    row_size  = 0;        // Bytes per quantised row

    // Ensure buffer is at least `need` bytes; returns buf pointer.
    uint8_t * ensure(size_t need) {
        if (need > buf_cap) {
            // Over-allocate by 25% to reduce future reallocs
            size_t alloc = need + (need >> 2);
            uint8_t * p = (uint8_t *)realloc(buf, alloc);
            if (!p) { p = (uint8_t *)realloc(buf, need); alloc = need; }
            buf = p;
            buf_cap = alloc;
        }
        return buf;
    }

    // Check whether the cache is still valid for this src1 tensor.
    bool valid_for(const void * data, int64_t k, int64_t n) const {
        return src1_data == data && K == k && N == n;
    }

    void tag(const void * data, int64_t k, int64_t n, size_t rs) {
        src1_data = data;
        K = k;
        N = n;
        row_size = rs;
    }
};

// Per-format caches (one per quant family).
// Static lifetime — buffer freed on program exit.
static quant_cache g_cache_q8_0;
static quant_cache g_cache_q8_k;
static std::mutex  g_cache_mtx;

// -----------------------------------------------------------------------------
// Hook Implementation
// -----------------------------------------------------------------------------
extern "C" int ggml_simd_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (dst->type != GGML_TYPE_F32) return 0;

    const struct ggml_tensor * src0 = dst->src[0]; // Weights (usually quantized)
    const struct ggml_tensor * src1 = dst->src[1]; // Activations (F32)

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    if (src1->ne[0] != K) return 0; 
    if (src1->type != GGML_TYPE_F32) return 0; // Only handle F32 activations for now

    // Check Hardware Support
#if defined(__aarch64__) || defined(_M_ARM64)
    bool use_neon = simd_check_neon();
    if (!use_neon) return 0;
#else
    bool use_avx2 = simd_check_avx2();
    bool use_avx512 = simd_check_avx512();
    if (!use_avx2 && !use_avx512) return 0;
#endif

    // Threading
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Partition M rows among threads
    const int64_t m_per_thread = (M + nth - 1) / nth;
    const int64_t m_start = std::min((int64_t)(ith * m_per_thread), M);
    const int64_t m_end = std::min((int64_t)(m_start + m_per_thread), M);

    if (m_start >= m_end) return 1;

    // Dispatch based on Weight Type
    if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0) {
        // ---------------------------------------------------------------------
        // Legacy Quants (Q4_0, Q8_0) -> Use Q8_0 Activations
        // ---------------------------------------------------------------------
        const size_t q8_0_bs = 34;
        const int block_k = 32;
        const size_t q8_row_size = (size_t)(K / block_k) * q8_0_bs;
        const size_t total_size = (size_t)N * q8_row_size;

        // Thread 0 handles quantization; others skip to compute after barrier
        uint8_t * src1_q8;
        {
            std::lock_guard<std::mutex> lock(g_cache_mtx);
            if (!g_cache_q8_0.valid_for(src1->data, K, N)) {
                src1_q8 = g_cache_q8_0.ensure(total_size);
                for (int64_t j = 0; j < N; j++) {
                    const float * src1_col = (const float *)((char *)src1->data + j * src1->nb[1]);
                    void * dst_q = src1_q8 + j * q8_row_size;
                    quantize_row_q8_0(src1_col, dst_q, K);
                }
                g_cache_q8_0.tag(src1->data, K, N, q8_row_size);
            } else {
                src1_q8 = g_cache_q8_0.buf;
            }
        }

        // Compute
        for (int64_t m = m_start; m < m_end; m++) {
            const void * w_row = (const char *)src0->data + m * src0->nb[1];
            for (int64_t n = 0; n < N; n++) {
                const void * a_row = src1_q8 + n * q8_row_size;
                float * dst_val = (float *)((char *)dst->data + m * dst->nb[1] + n * dst->nb[0]);

                float sum = 0.0f;
                if (src0->type == GGML_TYPE_Q4_0) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q4_0_q8_0_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q4_0_q8_0_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q4_0_q8_0_avx2(K, &sum, w_row, a_row);
#endif
                } else { // Q8_0
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q8_0_q8_0_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q8_0_q8_0_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q8_0_q8_0_avx2(K, &sum, w_row, a_row);
#endif
                }
                *dst_val = sum;
            }
        }
        return 1;
    }
    else if (src0->type == GGML_TYPE_Q2_K || src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q8_K) {
        // ---------------------------------------------------------------------
        // K-Quants (Q2_K, Q3_K, Q4_K, Q6_K, Q8_K) -> Use Q8_K Activations
        // ---------------------------------------------------------------------
        const size_t q8_k_bs = 292; // 4 + 256 + 32
        const int block_k = 256;

        if (K % block_k != 0) return 0;

        const size_t q8_k_row_size = (size_t)(K / block_k) * q8_k_bs;
        const size_t total_size = (size_t)N * q8_k_row_size;

        uint8_t * src1_q8k;
        {
            std::lock_guard<std::mutex> lock(g_cache_mtx);
            if (!g_cache_q8_k.valid_for(src1->data, K, N)) {
                src1_q8k = g_cache_q8_k.ensure(total_size);
                for (int64_t j = 0; j < N; j++) {
                    const float * src1_col = (const float *)((char *)src1->data + j * src1->nb[1]);
                    void * dst_q = src1_q8k + j * q8_k_row_size;
                    quantize_row_q8_K(src1_col, dst_q, K);
                }
                g_cache_q8_k.tag(src1->data, K, N, q8_k_row_size);
            } else {
                src1_q8k = g_cache_q8_k.buf;
            }
        }

        // Compute
        for (int64_t m = m_start; m < m_end; m++) {
            const void * w_row = (const char *)src0->data + m * src0->nb[1];
            for (int64_t n = 0; n < N; n++) {
                const void * a_row = src1_q8k + n * q8_k_row_size;
                float * dst_val = (float *)((char *)dst->data + m * dst->nb[1] + n * dst->nb[0]);

                float sum = 0.0f;
                if (src0->type == GGML_TYPE_Q2_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q2_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q2_k_q8_k_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q2_k_q8_k_avx2(K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q3_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q3_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q3_k_q8_k_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q3_k_q8_k_avx2(K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q4_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q4_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q4_k_q8_k_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q4_k_q8_k_avx2(K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q6_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q6_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q6_k_q8_k_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q6_k_q8_k_avx2(K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q8_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q8_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    if (use_avx512) simd_vec_dot_q8_k_q8_k_avx512(K, &sum, w_row, a_row);
                    else            simd_vec_dot_q8_k_q8_k_avx2(K, &sum, w_row, a_row);
#endif
                }
                *dst_val = sum;
            }
        }
        return 1;
    }
    
    // F32 Fallback
    if (src0->type == GGML_TYPE_F32) {
        simd_matmul_f32(
            (const float *)src0->data,
            (const float *)src1->data,
            (float *)dst->data,
            M, N, K, ith, nth
        );
        return 1;
    }

    return 0; // Default fallback for other types
}

extern "C" int ggml_simd_try_flash_attn(const struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    const struct ggml_tensor * q_tensor   = tensor->src[0];
    const struct ggml_tensor * k_tensor   = tensor->src[1];
    const struct ggml_tensor * v_tensor   = tensor->src[2];
    const struct ggml_tensor * mask_tensor = tensor->src[3];

    if (!q_tensor || !k_tensor || !v_tensor) return 0;
    if (q_tensor->type != GGML_TYPE_F32) return 0;

    flash_attn_config_t config;
    memset(&config, 0, sizeof(config));

    config.head_dim_k = q_tensor->ne[0];
    config.head_dim_v = v_tensor->ne[0];
    config.n_queries  = q_tensor->ne[1];
    config.n_kv       = k_tensor->ne[1];
    config.n_head_q   = q_tensor->ne[2];
    config.n_head_kv  = k_tensor->ne[2];
    config.batch_size = q_tensor->ne[3];

    float scale = 0.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float*)tensor->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float*)tensor->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float*)tensor->op_params + 2, sizeof(float));

    config.scale         = scale;
    config.max_bias      = max_bias;
    config.logit_softcap = logit_softcap;

    uint32_t mode = 0;
    if (config.n_head_kv < config.n_head_q) mode |= FA_MODE_GQA_BATCH;
    if (max_bias != 0.0f) mode |= FA_MODE_ALIBI;
    if (logit_softcap != 0.0f) mode |= FA_MODE_SOFTCAP;
    config.mode = mode;
    config.window_size = 0;

    config.q    = (const float*)q_tensor->data;
    config.k    = k_tensor->data;
    config.v    = v_tensor->data;
    config.mask = mask_tensor ? (const float*)mask_tensor->data : nullptr;
    config.dst  = (float*)tensor->data;

    for (int i = 0; i < 4; i++) {
        config.q_nb[i]   = q_tensor->nb[i];
        config.k_nb[i]   = k_tensor->nb[i];
        config.v_nb[i]   = v_tensor->nb[i];
        config.dst_nb[i] = tensor->nb[i];
    }
    if (mask_tensor) {
        for (int i = 0; i < 4; i++) config.mask_nb[i] = mask_tensor->nb[i];
    }

    config.k_type = k_tensor->type;
    config.v_type = v_tensor->type;

    config.ith = params->ith;
    config.nth = params->nth;

    return flash_attn_dispatch(&config);
}