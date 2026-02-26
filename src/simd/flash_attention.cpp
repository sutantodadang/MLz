#include "flash_attention.h"
#include "simd_matmul.h"
#include <cstddef>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <algorithm>

// Verify struct offsets match the constants used by assembly kernels
static_assert(offsetof(flash_attn_config_t, head_dim_k)    == FA_CFG_HEAD_DIM_K,     "head_dim_k offset mismatch");
static_assert(offsetof(flash_attn_config_t, head_dim_v)    == FA_CFG_HEAD_DIM_V,     "head_dim_v offset mismatch");
static_assert(offsetof(flash_attn_config_t, n_queries)     == FA_CFG_N_QUERIES,      "n_queries offset mismatch");
static_assert(offsetof(flash_attn_config_t, n_kv)          == FA_CFG_N_KV,           "n_kv offset mismatch");
static_assert(offsetof(flash_attn_config_t, n_head_q)      == FA_CFG_N_HEAD_Q,       "n_head_q offset mismatch");
static_assert(offsetof(flash_attn_config_t, n_head_kv)     == FA_CFG_N_HEAD_KV,      "n_head_kv offset mismatch");
static_assert(offsetof(flash_attn_config_t, batch_size)    == FA_CFG_BATCH_SIZE,     "batch_size offset mismatch");
static_assert(offsetof(flash_attn_config_t, scale)         == FA_CFG_SCALE,          "scale offset mismatch");
static_assert(offsetof(flash_attn_config_t, logit_softcap) == FA_CFG_LOGIT_SOFTCAP,  "logit_softcap offset mismatch");
static_assert(offsetof(flash_attn_config_t, max_bias)      == FA_CFG_MAX_BIAS,       "max_bias offset mismatch");
static_assert(offsetof(flash_attn_config_t, mode)          == FA_CFG_MODE,           "mode offset mismatch");
static_assert(offsetof(flash_attn_config_t, window_size)   == FA_CFG_WINDOW_SIZE,    "window_size offset mismatch");
static_assert(offsetof(flash_attn_config_t, q)             == FA_CFG_Q_PTR,          "q offset mismatch");
static_assert(offsetof(flash_attn_config_t, k)             == FA_CFG_K_PTR,          "k offset mismatch");
static_assert(offsetof(flash_attn_config_t, v)             == FA_CFG_V_PTR,          "v offset mismatch");
static_assert(offsetof(flash_attn_config_t, mask)          == FA_CFG_MASK_PTR,       "mask offset mismatch");
static_assert(offsetof(flash_attn_config_t, dst)           == FA_CFG_DST_PTR,        "dst offset mismatch");
static_assert(offsetof(flash_attn_config_t, q_nb)          == FA_CFG_Q_NB,           "q_nb offset mismatch");
static_assert(offsetof(flash_attn_config_t, k_nb)          == FA_CFG_K_NB,           "k_nb offset mismatch");
static_assert(offsetof(flash_attn_config_t, v_nb)          == FA_CFG_V_NB,           "v_nb offset mismatch");
static_assert(offsetof(flash_attn_config_t, mask_nb)       == FA_CFG_MASK_NB,        "mask_nb offset mismatch");
static_assert(offsetof(flash_attn_config_t, dst_nb)        == FA_CFG_DST_NB,         "dst_nb offset mismatch");
static_assert(offsetof(flash_attn_config_t, k_type)        == FA_CFG_K_TYPE,         "k_type offset mismatch");
static_assert(offsetof(flash_attn_config_t, v_type)        == FA_CFG_V_TYPE,         "v_type offset mismatch");

enum {
    GGML_TYPE_F32_  = 0,
    GGML_TYPE_F16_  = 1,
    GGML_TYPE_Q4_0_ = 2,
    GGML_TYPE_Q8_0_ = 8,
    GGML_TYPE_Q2_K_ = 10,
    GGML_TYPE_Q3_K_ = 11,
    GGML_TYPE_Q4_K_ = 12,
    GGML_TYPE_Q6_K_ = 14,
    GGML_TYPE_Q4_1_ = 3,
    GGML_TYPE_Q5_0_ = 6,
    GGML_TYPE_Q5_1_ = 7,
    GGML_TYPE_Q5_K_ = 13,
    GGML_TYPE_Q8_K_ = 15,
    GGML_TYPE_IQ4_NL_ = 20,
};

extern "C" int flash_attn_dispatch(const flash_attn_config_t* config) {
    if (!config) return 0;
    if (config->head_dim_k <= 0 || config->head_dim_k > FA_MAX_HEAD_DIM) return 0;
    if (config->n_queries <= 0 || config->n_kv <= 0) return 0;

#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM NEON dispatch — always available on AArch64
    int alignment = 4;  // NEON needs 4-float alignment (16 bytes)
    if (config->head_dim_k % alignment != 0) return 0;

    if (config->k_type == GGML_TYPE_F32_ && config->v_type == GGML_TYPE_F32_) {
        simd_flash_attn_f32_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_F16_ && config->v_type == GGML_TYPE_F16_) {
        simd_flash_attn_f16_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q4_0_ && config->v_type == GGML_TYPE_Q4_0_) {
        simd_flash_attn_q4_0_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q4_1_ && config->v_type == GGML_TYPE_Q4_1_) {
        simd_flash_attn_q4_1_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q5_0_ && config->v_type == GGML_TYPE_Q5_0_) {
        simd_flash_attn_q5_0_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q5_1_ && config->v_type == GGML_TYPE_Q5_1_) {
        simd_flash_attn_q5_1_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q8_0_ && config->v_type == GGML_TYPE_Q8_0_) {
        simd_flash_attn_q8_0_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_IQ4_NL_ && config->v_type == GGML_TYPE_IQ4_NL_) {
        simd_flash_attn_iq4_nl_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q2_K_ && config->v_type == GGML_TYPE_Q2_K_) {
        simd_flash_attn_q2_k_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q3_K_ && config->v_type == GGML_TYPE_Q3_K_) {
        simd_flash_attn_q3_k_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q4_K_ && config->v_type == GGML_TYPE_Q4_K_) {
        simd_flash_attn_q4_k_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q5_K_ && config->v_type == GGML_TYPE_Q5_K_) {
        simd_flash_attn_q5_k_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q6_K_ && config->v_type == GGML_TYPE_Q6_K_) {
        simd_flash_attn_q6_k_neon(config); return 1;
    }
    if (config->k_type == GGML_TYPE_Q8_K_ && config->v_type == GGML_TYPE_Q8_K_) {
        simd_flash_attn_q8_k_neon(config); return 1;
    }
    return 0;

#elif defined(__x86_64__) || defined(_M_X64)
    // x86_64 AVX2/AVX-512 dispatch
    int has_avx512 = simd_check_avx512();
    int has_avx2   = simd_check_avx2();
    if (!has_avx2 && !has_avx512) return 0;

    int alignment = has_avx512 ? 16 : 8;
    if (config->head_dim_k % alignment != 0) return 0;

    if (config->k_type == GGML_TYPE_F32_ && config->v_type == GGML_TYPE_F32_) {
        if (has_avx512) {
            simd_flash_attn_f32_avx512(config);
        } else {
            simd_flash_attn_f32_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q4_0_ && config->v_type == GGML_TYPE_Q4_0_) {
        if (has_avx512) {
            simd_flash_attn_q4_0_avx512(config);
        } else {
            simd_flash_attn_q4_0_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q8_0_ && config->v_type == GGML_TYPE_Q8_0_) {
        if (has_avx512) {
            simd_flash_attn_q8_0_avx512(config);
        } else {
            simd_flash_attn_q8_0_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_F16_ && config->v_type == GGML_TYPE_F16_) {
        if (has_avx512) {
            simd_flash_attn_f16_avx512(config);
        } else {
            simd_flash_attn_f16_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q4_1_ && config->v_type == GGML_TYPE_Q4_1_) {
        if (has_avx512) {
            simd_flash_attn_q4_1_avx512(config);
        } else {
            simd_flash_attn_q4_1_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q5_0_ && config->v_type == GGML_TYPE_Q5_0_) {
        if (has_avx512) {
            simd_flash_attn_q5_0_avx512(config);
        } else {
            simd_flash_attn_q5_0_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q5_1_ && config->v_type == GGML_TYPE_Q5_1_) {
        if (has_avx512) {
            simd_flash_attn_q5_1_avx512(config);
        } else {
            simd_flash_attn_q5_1_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_IQ4_NL_ && config->v_type == GGML_TYPE_IQ4_NL_) {
        if (has_avx512) {
            simd_flash_attn_iq4_nl_avx512(config);
        } else {
            simd_flash_attn_iq4_nl_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q2_K_ && config->v_type == GGML_TYPE_Q2_K_) {
        if (has_avx512) {
            simd_flash_attn_q2_k_avx512(config);
        } else {
            simd_flash_attn_q2_k_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q3_K_ && config->v_type == GGML_TYPE_Q3_K_) {
        if (has_avx512) {
            simd_flash_attn_q3_k_avx512(config);
        } else {
            simd_flash_attn_q3_k_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q4_K_ && config->v_type == GGML_TYPE_Q4_K_) {
        if (has_avx512) {
            simd_flash_attn_q4_k_avx512(config);
        } else {
            simd_flash_attn_q4_k_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q5_K_ && config->v_type == GGML_TYPE_Q5_K_) {
        if (has_avx512) {
            simd_flash_attn_q5_k_avx512(config);
        } else {
            simd_flash_attn_q5_k_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q6_K_ && config->v_type == GGML_TYPE_Q6_K_) {
        if (has_avx512) {
            simd_flash_attn_q6_k_avx512(config);
        } else {
            simd_flash_attn_q6_k_avx2(config);
        }
        return 1;
    }

    if (config->k_type == GGML_TYPE_Q8_K_ && config->v_type == GGML_TYPE_Q8_K_) {
        if (has_avx512) {
            simd_flash_attn_q8_k_avx512(config);
        } else {
            simd_flash_attn_q8_k_avx2(config);
        }
        return 1;
    }

    return 0;
#else
    // Unsupported architecture
    (void)config;
    return 0;
#endif
}

// ============================================================================
// Scalar fallback helpers for quantized path
// ============================================================================

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400u;
    } else if (exp == 31) {
        uint32_t v = sign | 0x7F800000u | (mant << 13);
        float r; memcpy(&r, &v, 4); return r;
    }

    uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &v, 4); return r;
}

static inline float fast_exp_scalar(float x) {
    if (x < -88.0f) return 0.0f;
    if (x >  88.0f) return HUGE_VALF;
    return expf(x);
}

static inline float alibi_slope(int head_idx, int n_heads) {
    int closest_pow2 = 1;
    while (closest_pow2 < n_heads) closest_pow2 <<= 1;
    float base = (closest_pow2 == n_heads)
        ? powf(2.0f, -8.0f / (float)n_heads)
        : powf(2.0f, -8.0f / (float)closest_pow2);
    return powf(base, (float)(head_idx + 1));
}

// ============================================================================
// Quantized-key flash attention (C++ — not worth ASM due to function pointers)
// ============================================================================

extern "C" void flash_attn_quantized(const flash_attn_config_t* config,
                                     fa_vec_dot_fn   vec_dot,
                                     fa_from_float_fn from_float,
                                     size_t           q_buf_size)
{
    const int64_t DK  = config->head_dim_k;
    const int64_t DV  = config->head_dim_v;
    const int64_t NQ  = config->n_queries;
    const int64_t NKV = config->n_kv;
    const int64_t NHQ = config->n_head_q;
    const int64_t NHKV = config->n_head_kv;
    const int64_t BS  = config->batch_size;
    const float   sc  = config->scale;

    const int ith = config->ith;
    const int nth = config->nth;

    const int64_t total_work = NQ * NHQ * BS;
    const int64_t work_per_thread = (total_work + nth - 1) / nth;
    const int64_t w_start = std::min((int64_t)ith * work_per_thread, total_work);
    const int64_t w_end   = std::min(w_start + work_per_thread, total_work);

    if (w_start >= w_end) return;

    uint8_t q_buf_stack[4096];
    uint8_t* q_buf = (q_buf_size <= sizeof(q_buf_stack)) ? q_buf_stack : new uint8_t[q_buf_size];

    float scores[FA_TILE_KV_AVX2];
    float O[FA_MAX_HEAD_DIM];

    for (int64_t w = w_start; w < w_end; w++) {
        const int64_t ib  = w / (NQ * NHQ);
        const int64_t rem = w % (NQ * NHQ);
        const int64_t ihq = rem / NQ;
        const int64_t iq  = rem % NQ;
        const int64_t ihkv = ihq * NHKV / NHQ;

        const float* Q_row = (const float*)((const char*)config->q + iq * config->q_nb[1] + ihq * config->q_nb[2] + ib * config->q_nb[3]);
        const char* K_base = (const char*)config->k + ihkv * config->k_nb[2] + ib * config->k_nb[3];
        const char* V_base = (const char*)config->v + ihkv * config->v_nb[2] + ib * config->v_nb[3];
        float* dst_row     = (float*)((char*)config->dst + iq * config->dst_nb[1] + ihq * config->dst_nb[2] + ib * config->dst_nb[3]);

        from_float(Q_row, q_buf, DK);

        memset(O, 0, sizeof(float) * DV);
        float row_max = -FLT_MAX;
        float row_sum = 0.0f;

        const float slope = (config->mode & FA_MODE_ALIBI) ? alibi_slope((int)ihq, (int)NHQ) : 0.0f;

        for (int64_t t = 0; t < NKV; t += FA_TILE_KV_AVX2) {
            const int64_t tile_end = std::min(t + (int64_t)FA_TILE_KV_AVX2, NKV);
            const int64_t tile_len = tile_end - t;

            for (int64_t j = 0; j < tile_len; j++) {
                const void* K_row = K_base + (t + j) * config->k_nb[1];
                float s = 0.0f;
                vec_dot((int)DK, &s, 0, K_row, 0, q_buf, 0, 1);
                s *= sc;

                if (config->mask) {
                    const float* m = (const float*)((const char*)config->mask + iq * config->mask_nb[0] + (t + j) * config->mask_nb[1]);
                    s += *m;
                }

                if (config->mode & FA_MODE_SOFTCAP) {
                    s = config->logit_softcap * tanhf(s / config->logit_softcap);
                }

                if (config->mode & FA_MODE_ALIBI) {
                    s += slope * (float)((t + j) - iq);
                }

                scores[j] = s;
            }

            float tile_max = -FLT_MAX;
            for (int64_t j = 0; j < tile_len; j++) {
                if (scores[j] > tile_max) tile_max = scores[j];
            }

            float new_max = std::max(row_max, tile_max);
            float correction = fast_exp_scalar(row_max - new_max);

            for (int64_t d = 0; d < DV; d++) O[d] *= correction;
            row_sum *= correction;
            row_max = new_max;

            float tile_sum = 0.0f;
            for (int64_t j = 0; j < tile_len; j++) {
                float p = fast_exp_scalar(scores[j] - row_max);
                scores[j] = p;
                tile_sum += p;
            }
            row_sum += tile_sum;

            for (int64_t j = 0; j < tile_len; j++) {
                float p = scores[j];
                if (p == 0.0f) continue;

                if (config->v_type == GGML_TYPE_F32_) {
                    const float* V_row = (const float*)(V_base + (t + j) * config->v_nb[1]);
                    for (int64_t d = 0; d < DV; d++) O[d] += p * V_row[d];
                } else if (config->v_type == GGML_TYPE_F16_) {
                    const uint16_t* V_row = (const uint16_t*)(V_base + (t + j) * config->v_nb[1]);
                    for (int64_t d = 0; d < DV; d++) O[d] += p * fp16_to_fp32(V_row[d]);
                }
            }
        }

        if (row_sum > 0.0f) {
            float inv_sum = 1.0f / row_sum;
            for (int64_t d = 0; d < DV; d++) dst_row[d] = O[d] * inv_sum;
        } else {
            memset(dst_row, 0, sizeof(float) * DV);
        }
    }

    if (q_buf != q_buf_stack) delete[] q_buf;
}
