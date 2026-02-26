/**
 * @file flash_attention.h
 * @brief Flash Attention 2 SIMD Kernels for MLz
 *
 * Tiled Flash Attention implementation with hand-written AVX2/AVX-512 assembly.
 * Supports:
 *   - Flash Attention 2 (tiled online softmax, O(N) memory)
 *   - Grouped Query Attention (GQA) head mapping
 *   - Sliding Window Attention (skip out-of-window KV tiles)
 *   - ALiBi positional bias
 *   - Logit soft-capping
 *   - F32 Q/K/V (primary path via assembly kernels)
 *   - Quantized K via function-pointer dispatch (C++ path)
 *
 * License: MIT
 */

#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONSTANTS
 *============================================================================*/

/** KV tile sizes (tuned for L1/L2 cache) */
#define FA_TILE_KV_AVX2     32
#define FA_TILE_KV_AVX512   64
#define FA_TILE_KV_NEON     32

/** Maximum supported head dimension */
#define FA_MAX_HEAD_DIM     256

/** PagedAttention defaults */
#define FA_DEFAULT_PAGE_SIZE 16

/** Attention mode flags (combinable via bitwise OR) */
#define FA_MODE_STANDARD     0x00u
#define FA_MODE_SLIDING_WIN  0x01u
#define FA_MODE_PAGED        0x02u
#define FA_MODE_GQA_BATCH    0x04u
#define FA_MODE_ALIBI        0x08u
#define FA_MODE_SOFTCAP      0x10u

/*============================================================================
 * FLASH ATTENTION CONFIGURATION
 *
 * WARNING: This struct layout is shared with assembly code.
 * The offset constants below MUST match the struct field offsets exactly.
 * Do NOT reorder, add, or remove fields without updating the ASM kernels.
 *============================================================================*/

/**
 * @brief Complete configuration for one flash attention invocation.
 *
 * Populated by the ggml hook from tensor metadata and op params.
 * All pointers and strides reference the original ggml tensor data.
 */
typedef struct {
    /* ---- Dimensions ---- */
    int64_t head_dim_k;     /**< Key/Query head dimension (DK)                  */
    int64_t head_dim_v;     /**< Value head dimension (DV), often == DK         */
    int64_t n_queries;      /**< Number of query positions this call            */
    int64_t n_kv;           /**< Number of key/value positions                  */
    int64_t n_head_q;       /**< Number of query attention heads                */
    int64_t n_head_kv;      /**< Number of KV attention heads (GQA: < n_head_q) */
    int64_t batch_size;     /**< Batch dimension                                */

    /* ---- Attention parameters ---- */
    float   scale;          /**< QK scale, typically 1/sqrt(DK)                 */
    float   logit_softcap;  /**< Logit soft-cap value, 0 = disabled             */
    float   max_bias;       /**< ALiBi max positional bias, 0 = disabled        */

    /* ---- Mode ---- */
    uint32_t mode;          /**< Bitwise OR of FA_MODE_* flags                  */

    /* ---- Sliding window ---- */
    int64_t window_size;    /**< Attend only to [pos-window, pos], 0 = all      */

    /* ---- Data pointers ---- */
    const float*  q;        /**< Query data  — always F32                       */
    const void*   k;        /**< Key data    — F32, F16, or quantized           */
    const void*   v;        /**< Value data  — F32 or F16                       */
    const float*  mask;     /**< Mask tensor — [n_kv, n_q, ...], NULL if none   */
    float*        dst;      /**< Output      — [DV, n_q, n_head_q, batch]       */

    /* ---- Byte strides [dim0, dim1, dim2, dim3] ---- */
    size_t q_nb[4];
    size_t k_nb[4];
    size_t v_nb[4];
    size_t mask_nb[4];
    size_t dst_nb[4];

    /* ---- Element types (GGML_TYPE_* enum values) ---- */
    int k_type;             /**< Key element type                               */
    int v_type;             /**< Value element type                             */

    /* ---- PagedAttention (when FA_MODE_PAGED set) ---- */
    int            page_size;        /**< KV positions per physical page         */
    const int32_t* page_table;       /**< Logical→physical page index mapping   */
    int            n_pages_logical;  /**< Number of logical pages                */
    const void**   k_pages;          /**< Array of K page data pointers          */
    const void**   v_pages;          /**< Array of V page data pointers          */

    /* ---- Threading ---- */
    int ith;                /**< This thread's index                            */
    int nth;                /**< Total threads                                  */
} flash_attn_config_t;

/*============================================================================
 * CONFIG STRUCT OFFSET CONSTANTS (for assembly kernels)
 *
 * These must be kept in sync with the struct above. Verified by
 * static_assert in flash_attention.cpp.
 *============================================================================*/

#define FA_CFG_HEAD_DIM_K     0
#define FA_CFG_HEAD_DIM_V     8
#define FA_CFG_N_QUERIES      16
#define FA_CFG_N_KV           24
#define FA_CFG_N_HEAD_Q       32
#define FA_CFG_N_HEAD_KV      40
#define FA_CFG_BATCH_SIZE     48
#define FA_CFG_SCALE          56
#define FA_CFG_LOGIT_SOFTCAP  60
#define FA_CFG_MAX_BIAS       64
#define FA_CFG_MODE           68
#define FA_CFG_WINDOW_SIZE    72
#define FA_CFG_Q_PTR          80
#define FA_CFG_K_PTR          88
#define FA_CFG_V_PTR          96
#define FA_CFG_MASK_PTR       104
#define FA_CFG_DST_PTR        112
#define FA_CFG_Q_NB           120
#define FA_CFG_K_NB           152
#define FA_CFG_V_NB           184
#define FA_CFG_MASK_NB        216
#define FA_CFG_DST_NB         248
#define FA_CFG_K_TYPE         280
#define FA_CFG_V_TYPE         284

/*============================================================================
 * VEC-DOT FUNCTION POINTER (for quantized K)
 *============================================================================*/

/**
 * @brief Quantized dot-product function pointer (matches ggml_vec_dot_t).
 */
typedef void (*fa_vec_dot_fn)(int n, float* s, size_t bs,
                              const void* vx, size_t bx,
                              const void* vy, size_t by, int nrc);

/**
 * @brief Quantize-from-float function pointer.
 */
typedef void (*fa_from_float_fn)(const float* src, void* dst, int64_t k);

/*============================================================================
 * ASSEMBLY KERNEL DECLARATIONS
 *
 * These are the hand-written NASM assembly entry points.
 * Each takes a single pointer to flash_attn_config_t and performs the
 * complete FA2 tiled computation for the thread's assigned work items.
 *============================================================================*/

/** F32 Q, F32 K, F32 V — AVX2 kernel */
void simd_flash_attn_f32_avx2(const flash_attn_config_t* config);

/** F32 Q, F32 K, F32 V — AVX-512 kernel */
void simd_flash_attn_f32_avx512(const flash_attn_config_t* config);

/** Q4_0 K/V — AVX2 kernel */
void simd_flash_attn_q4_0_avx2(const flash_attn_config_t* config);

/** Q4_0 K/V — AVX-512 kernel */
void simd_flash_attn_q4_0_avx512(const flash_attn_config_t* config);

/** Q8_0 K/V — AVX2 kernel */
void simd_flash_attn_q8_0_avx2(const flash_attn_config_t* config);

/** Q8_0 K/V — AVX-512 kernel */
void simd_flash_attn_q8_0_avx512(const flash_attn_config_t* config);

/** F16 K/V — AVX2 kernel */
void simd_flash_attn_f16_avx2(const flash_attn_config_t* config);

/** F16 K/V — AVX-512 kernel */
void simd_flash_attn_f16_avx512(const flash_attn_config_t* config);

/** Q4_1 K/V — AVX2 kernel */
void simd_flash_attn_q4_1_avx2(const flash_attn_config_t* config);
/** Q4_1 K/V — AVX-512 kernel */
void simd_flash_attn_q4_1_avx512(const flash_attn_config_t* config);

/** Q5_0 K/V — AVX2 kernel */
void simd_flash_attn_q5_0_avx2(const flash_attn_config_t* config);
/** Q5_0 K/V — AVX-512 kernel */
void simd_flash_attn_q5_0_avx512(const flash_attn_config_t* config);

/** Q5_1 K/V — AVX2 kernel */
void simd_flash_attn_q5_1_avx2(const flash_attn_config_t* config);
/** Q5_1 K/V — AVX-512 kernel */
void simd_flash_attn_q5_1_avx512(const flash_attn_config_t* config);

/** IQ4_NL K/V — AVX2 kernel */
void simd_flash_attn_iq4_nl_avx2(const flash_attn_config_t* config);
/** IQ4_NL K/V — AVX-512 kernel */
void simd_flash_attn_iq4_nl_avx512(const flash_attn_config_t* config);

/** Q2_K K/V — AVX2 kernel */
void simd_flash_attn_q2_k_avx2(const flash_attn_config_t* config);
/** Q2_K K/V — AVX-512 kernel */
void simd_flash_attn_q2_k_avx512(const flash_attn_config_t* config);

/** Q3_K K/V — AVX2 kernel */
void simd_flash_attn_q3_k_avx2(const flash_attn_config_t* config);
/** Q3_K K/V — AVX-512 kernel */
void simd_flash_attn_q3_k_avx512(const flash_attn_config_t* config);

/** Q4_K K/V — AVX2 kernel */
void simd_flash_attn_q4_k_avx2(const flash_attn_config_t* config);
/** Q4_K K/V — AVX-512 kernel */
void simd_flash_attn_q4_k_avx512(const flash_attn_config_t* config);

/** Q5_K K/V — AVX2 kernel */
void simd_flash_attn_q5_k_avx2(const flash_attn_config_t* config);
/** Q5_K K/V — AVX-512 kernel */
void simd_flash_attn_q5_k_avx512(const flash_attn_config_t* config);

/** Q6_K K/V — AVX2 kernel */
void simd_flash_attn_q6_k_avx2(const flash_attn_config_t* config);
/** Q6_K K/V — AVX-512 kernel */
void simd_flash_attn_q6_k_avx512(const flash_attn_config_t* config);

/** Q8_K K/V — AVX2 kernel */
void simd_flash_attn_q8_k_avx2(const flash_attn_config_t* config);
/** Q8_K K/V — AVX-512 kernel */
void simd_flash_attn_q8_k_avx512(const flash_attn_config_t* config);

/*============================================================================
 * ARM NEON ASSEMBLY KERNEL DECLARATIONS (AArch64)
 *
 * Hand-written GAS assembly kernels for ARM NEON (128-bit SIMD).
 * Targeting Raspberry Pi 5 (Cortex-A76, ARMv8.2-A).
 *============================================================================*/

/** F32 Q, F32 K, F32 V — NEON kernel */
void simd_flash_attn_f32_neon(const flash_attn_config_t* config);

/** F16 K/V — NEON kernel */
void simd_flash_attn_f16_neon(const flash_attn_config_t* config);

/** Q4_0 K/V — NEON kernel */
void simd_flash_attn_q4_0_neon(const flash_attn_config_t* config);

/** Q4_1 K/V — NEON kernel */
void simd_flash_attn_q4_1_neon(const flash_attn_config_t* config);

/** Q5_0 K/V — NEON kernel */
void simd_flash_attn_q5_0_neon(const flash_attn_config_t* config);

/** Q5_1 K/V — NEON kernel */
void simd_flash_attn_q5_1_neon(const flash_attn_config_t* config);

/** Q8_0 K/V — NEON kernel */
void simd_flash_attn_q8_0_neon(const flash_attn_config_t* config);

/** IQ4_NL K/V — NEON kernel */
void simd_flash_attn_iq4_nl_neon(const flash_attn_config_t* config);

/** Q2_K K/V — NEON kernel */
void simd_flash_attn_q2_k_neon(const flash_attn_config_t* config);

/** Q3_K K/V — NEON kernel */
void simd_flash_attn_q3_k_neon(const flash_attn_config_t* config);

/** Q4_K K/V — NEON kernel */
void simd_flash_attn_q4_k_neon(const flash_attn_config_t* config);

/** Q5_K K/V — NEON kernel */
void simd_flash_attn_q5_k_neon(const flash_attn_config_t* config);

/** Q6_K K/V — NEON kernel */
void simd_flash_attn_q6_k_neon(const flash_attn_config_t* config);

/** Q8_K K/V — NEON kernel */
void simd_flash_attn_q8_k_neon(const flash_attn_config_t* config);

/*============================================================================
 * DISPATCH
 *============================================================================*/

/**
 * @brief Main flash attention dispatch.
 *
 * Selects optimal SIMD kernel based on CPU features and data types.
 * For F32 K/V: dispatches to assembly kernels.
 * For quantized K: uses C++ quantized path with function pointers.
 *
 * @return 1 if handled, 0 to fall back to default implementation.
 */
int flash_attn_dispatch(const flash_attn_config_t* config);

/*============================================================================
 * QUANTIZED-KEY KERNEL  (F32 Q, quantized K, F16/F32 V)
 *============================================================================*/

/**
 * @brief Flash attention with quantized K (C++ implementation).
 *
 * Converts Q rows to vec_dot format, then uses vec_dot for QK products.
 * V accumulation handled as F32 (with F16→F32 conversion if needed).
 *
 * @param config      Attention configuration
 * @param vec_dot     Function for dot(K_quantized, Q_quantized)
 * @param from_float  Function to convert Q from F32 to vec_dot format
 * @param q_buf_size  Size in bytes of one quantized Q row
 */
void flash_attn_quantized(const flash_attn_config_t* config,
                          fa_vec_dot_fn   vec_dot,
                          fa_from_float_fn from_float,
                          size_t           q_buf_size);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_H */
