#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-threading.h"
#include "ggml-backend-impl.h"
#include "simd_matmul.h"
#include "flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

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
// Helper: Quantize F32 row to Q8_0
// -----------------------------------------------------------------------------
// block_q8_0: 2 bytes (d: fp16) + 32 bytes (qs: int8) = 34 bytes
static void quantize_row_q8_0_reference(const float* x, void* y, int k) {
    const int block_size = 32;
    struct block_q8_0 {
        uint16_t d;     // fp16
        int8_t qs[32];
    };
    
    block_q8_0* y_blocks = (block_q8_0*)y;
    int nb = k / block_size;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;

        for (int j = 0; j < block_size; j++) {
            float v = x[i * block_size + j];
            if (std::abs(v) > amax) {
                amax = std::abs(v);
                max = v;
            }
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        // Simple float->fp16 conversion
        uint16_t d_fp16;
        {
            uint32_t x_u32;
            memcpy(&x_u32, &d, 4);
            uint32_t sign = (x_u32 >> 16) & 0x8000;
            int32_t exp = ((x_u32 >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (x_u32 >> 13) & 0x3FF;
            
            if (exp <= 0) {
                 d_fp16 = (uint16_t)(sign); 
            } else if (exp >= 31) {
                 d_fp16 = (uint16_t)(sign | 0x7C00);
            } else {
                 d_fp16 = (uint16_t)(sign | (exp << 10) | mant);
            }
        }
        
        y_blocks[i].d = d_fp16;

        for (int j = 0; j < block_size; j++) {
            const float x0 = x[i * block_size + j] * id;
            y_blocks[i].qs[j] = (int8_t)std::round(x0);
        }
    }
}

// -----------------------------------------------------------------------------
// Helper: Quantize F32 row to Q8_K
// -----------------------------------------------------------------------------
// block_q8_K: 4 bytes (d: float) + 256 bytes (qs: int8) + 32 bytes (bsums: int16) = 292 bytes
static void quantize_row_q8_k_reference(const float* x, void* y, int k) {
    const int block_size = 256;
    struct block_q8_K {
        float d;            // float32
        int8_t qs[256];
        int16_t bsums[16];
    };

    block_q8_K* y_blocks = (block_q8_K*)y;
    int nb = k / block_size;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;

        for (int j = 0; j < block_size; j++) {
            float v = x[i * block_size + j];
            if (std::abs(v) > amax) amax = std::abs(v);
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        
        y_blocks[i].d = d;

        for (int j = 0; j < block_size; j++) {
            const float x0 = x[i * block_size + j] * id;
            y_blocks[i].qs[j] = (int8_t)std::round(x0);
        }

        // Calculate bsums (sum of 16 quants)
        for (int j = 0; j < 16; j++) {
            int sum = 0;
            for (int l = 0; l < 16; l++) {
                sum += y_blocks[i].qs[j * 16 + l];
            }
            y_blocks[i].bsums[j] = (int16_t)sum;
        }
    }
}

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
        size_t q8_row_size = (K / block_k) * q8_0_bs;
        
        // Thread-local quantization buffer for src1 columns
        std::vector<uint8_t> src1_q8(N * q8_row_size);

        // Quantize src1 (F32 -> Q8_0)
        for (int64_t j = 0; j < N; j++) {
            const float* src1_col = (const float*)((char*)src1->data + j * src1->nb[1]);
            void* dst_q = src1_q8.data() + j * q8_row_size;
            quantize_row_q8_0_reference(src1_col, dst_q, K);
        }

        // Compute
        for (int64_t m = m_start; m < m_end; m++) {
            const void* w_row = (const char*)src0->data + m * src0->nb[1];
            for (int64_t n = 0; n < N; n++) {
                const void* a_row = src1_q8.data() + n * q8_row_size;
                float* dst_val = (float*)((char*)dst->data + m * dst->nb[1] + n * dst->nb[0]);
                
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
        
        if (K % block_k != 0) return 0; // Should not happen for valid K-quant tensors

        size_t q8_k_row_size = (K / block_k) * q8_k_bs;
        std::vector<uint8_t> src1_q8k(N * q8_k_row_size);

        // Quantize src1 (F32 -> Q8_K)
        for (int64_t j = 0; j < N; j++) {
            const float* src1_col = (const float*)((char*)src1->data + j * src1->nb[1]);
            void* dst_q = src1_q8k.data() + j * q8_k_row_size;
            quantize_row_q8_k_reference(src1_col, dst_q, K);
        }

        // Compute
        for (int64_t m = m_start; m < m_end; m++) {
            const void* w_row = (const char*)src0->data + m * src0->nb[1];
            for (int64_t n = 0; n < N; n++) {
                const void* a_row = src1_q8k.data() + n * q8_k_row_size;
                float* dst_val = (float*)((char*)dst->data + m * dst->nb[1] + n * dst->nb[0]);
                
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