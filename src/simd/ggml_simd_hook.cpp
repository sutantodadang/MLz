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
#include <atomic>
#include <mutex>
#include <vector>

// -----------------------------------------------------------------------------
// Runtime Kill-Switch & Trace
// -----------------------------------------------------------------------------
// MLZ_SIMD=0     : disable all custom SIMD hooks at runtime (per PLAN-ASSEMBLY-REWRITE)
// MLZ_SIMD_TRACE=1 : print every dispatched op to stderr
//
// The env vars are read once on first call and cached.  A non-zero default
// keeps the historical behaviour (hooks active when the build flag is on).
// -----------------------------------------------------------------------------
namespace {
    std::atomic<int> g_simd_enabled{-1};   // -1 = uninit, 0 = off, 1 = on
    std::atomic<int> g_simd_trace{-1};

    inline bool simd_runtime_enabled() {
        int v = g_simd_enabled.load(std::memory_order_relaxed);
        if (v < 0) {
            const char * s = std::getenv("MLZ_SIMD");
            v = (s && s[0] == '0' && s[1] == '\0') ? 0 : 1;
            g_simd_enabled.store(v, std::memory_order_relaxed);
        }
        return v != 0;
    }

    inline bool simd_trace_enabled() {
        int v = g_simd_trace.load(std::memory_order_relaxed);
        if (v < 0) {
            const char * s = std::getenv("MLZ_SIMD_TRACE");
            v = (s && s[0] != '0') ? 1 : 0;
            g_simd_trace.store(v, std::memory_order_relaxed);
        }
        return v != 0;
    }
}

// -----------------------------------------------------------------------------
// Assembly Kernel Declarations
// -----------------------------------------------------------------------------
// Symbols are declared in `simd_kernels_manifest.h`, which is generated at
// build time by `generateSimdManifestHeader` in build.zig.  The single source
// of truth is `src/simd/kernels/manifest.txt` (mirrored in build.zig's
// `simd_manifest` array).  Add new kernels there, never inline here.
#include "simd_kernels_manifest.h"

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
    // Runtime kill-switch — MLZ_SIMD=0 forces ggml default path (zero-rebuild rollback).
    if (!simd_runtime_enabled()) return 0;

    if (dst->type != GGML_TYPE_F32) return 0;

    const struct ggml_tensor * src0 = dst->src[0]; // Weights (usually quantized)
    const struct ggml_tensor * src1 = dst->src[1]; // Activations (F32)

    // Dispatch gate: if Repack (or any extra-buffer backend) has claimed this
    // tensor, defer to it.  Repack repacks weights at load time and is faster
    // than our generic kernels for K-quants — don't fight it on its own turf.
    if (src0->extra != nullptr) return 0;
    if (src1->extra != nullptr) return 0;

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    if (src1->ne[0] != K) return 0;
    if (src1->type != GGML_TYPE_F32) return 0; // Only handle F32 activations for now

    // Require plain contiguous 2D layout (per repo memory: mlz-mul-mat-layout-fix)
    if (dst->ne[2] != 1 || dst->ne[3] != 1) return 0;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return 0;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return 0;

    if (simd_trace_enabled()) {
        fprintf(stderr, "[mlz-simd] mul_mat type=%d M=%lld N=%lld K=%lld\n",
                (int)src0->type, (long long)M, (long long)N, (long long)K);
    }

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
                    // Handwritten NASM AVX2 / AVX-512 — verified by U1
                    // (zig build test-simd) at K ∈ {32, 256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_{avx2,avx512}.asm
                    extern void simd_vec_dot_q4_0_q8_0_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q4_0_q8_0_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q4_0_q8_0_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q4_0_q8_0_avx2((int)K, &sum, w_row, a_row);
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
    else if (src0->type == GGML_TYPE_Q2_K || src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q8_K) {
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
                    // Handwritten NASM AVX2 / AVX-512 — verified by U1
                    // (zig build test-simd) at K ∈ {256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_{avx2,avx512}.asm
                    extern void simd_vec_dot_q2_k_q8_k_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q2_k_q8_k_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q2_k_q8_k_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q2_k_q8_k_avx2((int)K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q3_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q3_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    // Handwritten NASM AVX2 / AVX-512 — verified by U1
                    // (zig build test-simd) at K ∈ {256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_{avx2,avx512}.asm
                    extern void simd_vec_dot_q3_k_q8_k_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q3_k_q8_k_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q3_k_q8_k_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q3_k_q8_k_avx2((int)K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q4_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q4_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    // Handwritten NASM AVX2 / AVX-512 — verified by U1
                    // (zig build test-simd) at K ∈ {256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_{avx2,avx512}.asm
                    extern void simd_vec_dot_q4_k_q8_k_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q4_k_q8_k_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q4_k_q8_k_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q4_k_q8_k_avx2((int)K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q5_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    // Handwritten NEON .S kernel — bit-for-bit reference port.
                    // Source: src/simd/kernels/aarch64/vec/vec_dot_q5_k_q8_k_neon.S
                    extern void simd_vec_dot_q5_k_q8_k_neon(int n, float * r, const void * vx, const void * vy);
                    simd_vec_dot_q5_k_q8_k_neon((int)K, &sum, w_row, a_row);
#else
                    // Handwritten NASM AVX2 / AVX-512 kernels — verified by
                    // U1 (zig build test-simd) at K ∈ {256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_{avx2,avx512}.asm
                    extern void simd_vec_dot_q5_k_q8_k_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q5_k_q8_k_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q5_k_q8_k_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q5_k_q8_k_avx2((int)K, &sum, w_row, a_row);
#endif
                } else if (src0->type == GGML_TYPE_Q6_K) {
#if defined(__aarch64__) || defined(_M_ARM64)
                    simd_vec_dot_q6_k_q8_k_neon(K, &sum, w_row, a_row);
#else
                    // Handwritten NASM AVX2 / AVX-512 — verified by U1
                    // (zig build test-simd) at K ∈ {256, 1024, 4096}.
                    // Sources: src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_{avx2,avx512}.asm
                    extern void simd_vec_dot_q6_k_q8_k_avx2(int n, float * r, const void * vx, const void * vy);
                    extern void simd_vec_dot_q6_k_q8_k_avx512(int n, float * r, const void * vx, const void * vy);
                    if (use_avx512) simd_vec_dot_q6_k_q8_k_avx512((int)K, &sum, w_row, a_row);
                    else            simd_vec_dot_q6_k_q8_k_avx2((int)K, &sum, w_row, a_row);
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
    if (!simd_runtime_enabled()) return 0;

    const struct ggml_tensor * q_tensor   = tensor->src[0];
    const struct ggml_tensor * k_tensor   = tensor->src[1];
    const struct ggml_tensor * v_tensor   = tensor->src[2];
    const struct ggml_tensor * mask_tensor = tensor->src[3];

    if (!q_tensor || !k_tensor || !v_tensor) return 0;
    if (q_tensor->type != GGML_TYPE_F32) return 0;

    // Defer to extra-buffer backends (Repack etc.) when present.
    if (k_tensor->extra != nullptr || v_tensor->extra != nullptr) return 0;

    if (simd_trace_enabled()) {
        fprintf(stderr, "[mlz-simd] flash_attn k_type=%d v_type=%d\n",
                (int)k_tensor->type, (int)v_tensor->type);
    }

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

// =============================================================================
// Unary hook: ggml_simd_try_rms_norm
// =============================================================================
//
// Replacement for `ggml_compute_forward_rms_norm_f32` (ops.cpp:3713).
//
// y[i] = x[i] * scale,  scale = 1 / sqrt(sum(x^2)/n + eps)
//
// Only handles GGML_TYPE_F32 with contiguous innermost stride.  Defers (returns
// 0) for any other dtype/layout, so upstream remains the safety net.
//
// Opt-in: gated behind `MLZ_SIMD_RMS_NORM=1` because parallel f64 reduction is
// not bit-exact with the upstream serial-f64 accumulator (typically ≤ 2 ULP off
// per element), which can make E1's SHA256-identity gate flake at very long
// horizons.  U1 (`zig build test-simd`) checks ULP-bounded equivalence.
// =============================================================================

namespace {
    std::atomic<int> g_simd_rms_norm{-1};

    inline bool simd_rms_norm_enabled() {
        int v = g_simd_rms_norm.load(std::memory_order_relaxed);
        if (v < 0) {
            const char * s = std::getenv("MLZ_SIMD_RMS_NORM");
            v = (s && s[0] == '1') ? 1 : 0;
            g_simd_rms_norm.store(v, std::memory_order_relaxed);
        }
        return v != 0;
    }
}

extern "C" int ggml_simd_try_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!simd_runtime_enabled())  return 0;
    if (!simd_rms_norm_enabled()) return 0;

    const struct ggml_tensor * src0 = dst->src[0];
    if (!src0) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    if (src0->extra != nullptr) return 0;
    // Innermost stride must be packed f32.
    if (src0->nb[0] != sizeof(float)) return 0;
    if (dst ->nb[0] != sizeof(float)) return 0;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    if (ne00 <= 0) return 0;

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));
    if (eps < 0.0f) return 0;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (simd_trace_enabled()) {
        fprintf(stderr, "[mlz-simd] rms_norm ne=%lldx%lldx%lldx%lld eps=%g (neon)\n",
                (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03, (double)eps);
    }
    const int ith = params->ith;
    const int nth = params->nth;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (const float *)((const char *)src0->data
                                                  + i01 * src0->nb[1]
                                                  + i02 * src0->nb[2]
                                                  + i03 * src0->nb[3]);
                float * y       = (float *)((char *)dst->data
                                            + i01 * dst->nb[1]
                                            + i02 * dst->nb[2]
                                            + i03 * dst->nb[3]);
                simd_rms_norm_f32_neon((int)ne00, eps, x, y);
            }
        }
    }
    return 1;
#else
    static const bool g_have_avx512 = simd_check_avx512() != 0;
    static const bool g_have_avx2   = simd_check_avx2()   != 0;
    if (!g_have_avx2) return 0;

    if (simd_trace_enabled()) {
        fprintf(stderr, "[mlz-simd] rms_norm ne=%lldx%lldx%lldx%lld eps=%g\n",
                (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03, (double)eps);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    // Mirror upstream's row partitioning: outer loops over (i03,i02,i01),
    // i01 striped across nth threads.
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (const float *)((const char *)src0->data
                                                  + i01 * src0->nb[1]
                                                  + i02 * src0->nb[2]
                                                  + i03 * src0->nb[3]);
                float * y       = (float *)((char *)dst->data
                                            + i01 * dst->nb[1]
                                            + i02 * dst->nb[2]
                                            + i03 * dst->nb[3]);
                if (g_have_avx512) {
                    simd_rms_norm_f32_avx512((int)ne00, eps, x, y);
                } else {
                    simd_rms_norm_f32_avx2  ((int)ne00, eps, x, y);
                }
            }
        }
    }
    return 1;
#endif
}

// =============================================================================
// Unary hook: ggml_simd_try_rope (NEOX f32 fast-path only).
// =============================================================================
//
// Scope: src0 type == F32, mode == GGML_ROPE_TYPE_NEOX (2), no MROPE / VISION /
// IMROPE multi-position arrangements.  Re-implements the per-row cache via the
// same libm `cosf`/`sinf`/`logf` calls used by upstream's
// `ggml_rope_cache_init` so the cos/sin lookup table is bit-exact.  The inner
// rotation `rotate_pairs` is replaced by a NASM AVX2 / AVX-512 kernel.
//
// Opt-in: gated behind `MLZ_SIMD_ROPE=1`.  Defaults OFF until the hook is
// proven against multiple architectures and rope variants.
//
// Other rope modes (NORMAL pair stride, MROPE, VISION, IMROPE) defer to
// upstream untouched.
// =============================================================================

namespace {
    std::atomic<int> g_simd_rope{-1};

    inline bool simd_rope_enabled() {
        int v = g_simd_rope.load(std::memory_order_relaxed);
        if (v < 0) {
            const char * s = std::getenv("MLZ_SIMD_ROPE");
            v = (s && s[0] == '1') ? 1 : 0;
            g_simd_rope.store(v, std::memory_order_relaxed);
        }
        return v != 0;
    }

    // Mirrors ops.cpp:rope_yarn_ramp / rope_yarn (bit-exact: same libm calls,
    // same operation order).
    static inline float mlz_rope_yarn_ramp(float low, float high, int i0) {
        const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
        return 1.0f - std::min(1.0f, std::max(0.0f, y));
    }

    static inline void mlz_rope_yarn(
            float theta_extrap, float freq_scale, const float corr_dims[2], int64_t i0,
            float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        if (ext_factor != 0.0f) {
            float ramp_mix = mlz_rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
        }
        *cos_theta = std::cos(theta) * mscale;
        *sin_theta = std::sin(theta) * mscale;
    }

    // Mirrors ops.cpp:ggml_rope_cache_init (NEOX path: cache[i0]=cos,
    // cache[i0+1]=sin*sin_sign for i0 in [0, ne0) step 2).
    static inline void mlz_rope_cache_init(
            float theta_base, float freq_scale, const float * freq_factors,
            const float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
            float * cache, float sin_sign, float theta_scale) {
        float theta = theta_base;
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
            mlz_rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale,
                          &cache[i0 + 0], &cache[i0 + 1]);
            cache[i0 + 1] *= sin_sign;
            theta *= theta_scale;
        }
    }
}

extern "C" int ggml_simd_try_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!simd_runtime_enabled()) return 0;
    if (!simd_rope_enabled())    return 0;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    if (!src0 || !src1) return 0;
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return 0;
    if (src1->type != GGML_TYPE_I32) return 0;
    if (src0->extra != nullptr) return 0;
    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return 0;

    const int n_dims     = ((const int32_t *) dst->op_params)[1];
    const int mode       = ((const int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) dst->op_params)[4];

    // Fast-path: NEOX-style only.  Defer for NORMAL (mode==0, pair stride 1),
    // MROPE/VISION/IMROPE (multi-position with sections layout).
    if (mode != GGML_ROPE_TYPE_NEOX) return 0;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (const int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 10, sizeof(float));

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    if (n_dims <= 0 || (n_dims & 1) != 0) return 0;
    if (n_dims > ne0) return 0;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        if (src2->type != GGML_TYPE_F32) return 0;
        if (src2->ne[0] < n_dims/2) return 0;
        freq_factors = (const float *) src2->data;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    using rope_kernel_t = void (*)(long long, const float *, const float *, float *);
    rope_kernel_t rope_kernel = &simd_rope_neox_f32_neon;
#else
    static const bool g_have_avx512 = simd_check_avx512() != 0;
    static const bool g_have_avx2   = simd_check_avx2()   != 0;
    if (!g_have_avx2) return 0;
    using rope_kernel_t = void (*)(long long, const float *, const float *, float *);
    rope_kernel_t rope_kernel = g_have_avx512
        ? &simd_rope_neox_f32_avx512
        : &simd_rope_neox_f32_avx2;
#endif

    if (simd_trace_enabled()) {
        fprintf(stderr, "[mlz-simd] rope ne=%lldx%lldx%lldx%lld n_dims=%d mode=%d\n",
                (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3, n_dims, mode);
    }

    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float sin_sign = 1.0f;  // forward
    const int32_t * pos = (const int32_t *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;
    const int64_t nr = ne1 * ne2 * ne3;
    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = std::min(ir0 + dr, nr);

    // Per-thread cache scratch: ne0 floats, allocated on stack once.
    // ne0 is per-head dimensionality (typ. 64..256).
    constexpr size_t CACHE_MAX = 1024;
    float cache_stack[CACHE_MAX];
    float * cache;
    std::vector<float> cache_heap;
    if ((size_t)ne0 <= CACHE_MAX) {
        cache = cache_stack;
    } else {
        cache_heap.resize((size_t)ne0);
        cache = cache_heap.data();
    }

    int64_t last_i2 = -1;
    int64_t ir = 0;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            // Lazily build the cache when this thread first touches a new i2.
            bool cache_built = (last_i2 == i2);
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir > ir1) goto done;

                if (!cache_built) {
                    const int64_t p = pos[i2];
                    mlz_rope_cache_init((float)p, freq_scale, freq_factors, corr_dims,
                                        ne0, ext_factor, attn_factor, cache, sin_sign,
                                        theta_scale);
                    last_i2 = i2;
                    cache_built = true;
                }

                const float * src = (const float *)((const char *)src0->data
                                                    + i3*src0->nb[3]
                                                    + i2*src0->nb[2]
                                                    + i1*src0->nb[1]);
                float * dst_data  = (float *)((char *)dst->data
                                              + i3*dst->nb[3]
                                              + i2*dst->nb[2]
                                              + i1*dst->nb[1]);

                // NEOX rotation on the first n_dims channels (n_dims/2 pairs).
                rope_kernel((long long)(n_dims/2), cache, src, dst_data);

                // Pass-through the remaining channels [n_dims, ne0) in pairs.
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    dst_data[i0 + 0] = src[i0 + 0];
                    dst_data[i0 + 1] = src[i0 + 1];
                }
            }
        }
    }
done:
    return 1;
}
