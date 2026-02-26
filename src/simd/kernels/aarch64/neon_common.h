/**
 * @file neon_common.h
 * @brief Common constants and macros for AArch64 NEON assembly kernels
 *
 * Shared definitions for all ARM NEON flash attention, vec_dot, and
 * matrix multiplication kernels. Included by .S files via C preprocessor.
 *
 * AArch64 Calling Convention (AAPCS64):
 *   x0-x7:   argument/result registers (caller-saved)
 *   x8:      indirect result location
 *   x9-x15:  temporary (caller-saved)
 *   x16-x17: IP0/IP1 intra-procedure scratch
 *   x18:     platform register (reserved)
 *   x19-x28: callee-saved
 *   x29:     frame pointer (FP)
 *   x30:     link register (LR)
 *   sp:      stack pointer (16-byte aligned)
 *
 *   v0-v7:   argument/result NEON (caller-saved)
 *   v8-v15:  callee-saved (lower 64-bit d8-d15 only)
 *   v16-v31: temporary (caller-saved)
 *
 * License: MIT
 */

#ifndef NEON_COMMON_H
#define NEON_COMMON_H

/* ============================================================================
 * Config struct field offsets (flash_attn_config_t)
 * Must match flash_attention.h exactly.
 * All int64_t = 8 bytes, float = 4 bytes, pointers = 8 bytes, size_t = 8 bytes
 * ============================================================================ */
#define CFG_HEAD_DIM_K      0
#define CFG_HEAD_DIM_V      8
#define CFG_N_QUERIES       16
#define CFG_N_KV            24
#define CFG_N_HEAD_Q        32
#define CFG_N_HEAD_KV       40
#define CFG_BATCH_SIZE      48
#define CFG_SCALE           56
#define CFG_LOGIT_SOFTCAP   60
#define CFG_MAX_BIAS        64
#define CFG_MODE            68
#define CFG_WINDOW_SIZE     72
#define CFG_Q_PTR           80
#define CFG_K_PTR           88
#define CFG_V_PTR           96
#define CFG_MASK_PTR        104
#define CFG_DST_PTR         112
/* q_nb[4] at 120, k_nb[4] at 152, v_nb[4] at 184, mask_nb[4] at 216, dst_nb[4] at 248 */
#define CFG_Q_NB0           120
#define CFG_Q_NB1           128
#define CFG_Q_NB2           136
#define CFG_Q_NB3           144
#define CFG_K_NB0           152
#define CFG_K_NB1           160
#define CFG_K_NB2           168
#define CFG_K_NB3           176
#define CFG_V_NB0           184
#define CFG_V_NB1           192
#define CFG_V_NB2           200
#define CFG_V_NB3           208
#define CFG_MASK_NB0        216
#define CFG_MASK_NB1        224
#define CFG_MASK_NB2        232
#define CFG_MASK_NB3        240
#define CFG_DST_NB0         248
#define CFG_DST_NB1         256
#define CFG_DST_NB2         264
#define CFG_DST_NB3         272
#define CFG_K_TYPE          280
#define CFG_V_TYPE          284
/* Page table fields at 288-320, ith/nth at 328/332 */
#define CFG_ITH             328
#define CFG_NTH             332

/* ============================================================================
 * FA2 Algorithm Constants
 * ============================================================================ */
#define FA_TILE_KV_NEON     32      /* KV tile size for NEON (matches AVX2) */
#define FA_MAX_HEAD_DIM     256     /* Maximum head dimension */

/* ============================================================================
 * Attention Mode Flags
 * ============================================================================ */
#define FA_MODE_STANDARD    0x00
#define FA_MODE_SLIDING_WIN 0x01
#define FA_MODE_PAGED       0x02
#define FA_MODE_GQA_BATCH   0x04
#define FA_MODE_ALIBI       0x08
#define FA_MODE_SOFTCAP     0x10

/* ============================================================================
 * Stack Frame Layout (16-byte aligned for NEON)
 * ============================================================================ */
#define STK_O               0       /* float[256] output accumulator = 1024 bytes */
#define STK_SCORES          1024    /* float[32]  QK scores = 128 bytes */
#define STK_PROBS           1152    /* float[32]  softmax probs = 128 bytes */
/* Saved locals (1280+) */
#define STK_CONFIG          1280
#define STK_Q_ROW           1288
#define STK_K_BASE          1296
#define STK_V_BASE          1304
#define STK_DST_ROW         1312
#define STK_MASK_PTR_L      1320
#define STK_HEAD_DIM_K      1328
#define STK_HEAD_DIM_V      1336
#define STK_N_KV_L          1344
#define STK_N_QUERIES       1352
#define STK_N_HEAD_Q        1360
#define STK_N_HEAD_KV       1368
#define STK_BATCH_SIZE      1376
/* Strides */
#define STK_Q_NB1           1384
#define STK_K_NB1           1392
#define STK_V_NB1           1400
#define STK_DST_NB1         1408
#define STK_MASK_NB1        1416
#define STK_Q_NB2           1424
#define STK_K_NB2           1432
#define STK_V_NB2           1440
#define STK_DST_NB2         1448
#define STK_Q_NB3           1456
#define STK_K_NB3           1464
#define STK_V_NB3           1472
#define STK_DST_NB3         1480
/* Work decomposition */
#define STK_WORK_START      1488
#define STK_WORK_END        1496
/* Scale (broadcast-ready) */
#define STK_SCALE_VAL       1504
/* Quant block counts */
#define STK_K_NBLOCKS       1512
#define STK_V_NBLOCKS       1520
/* Saved iq for mask offset */
#define STK_IQ_SAVE         1528
/* FP save */
#define STK_FP_SAVE         1536
#define STK_LR_SAVE         1544
#define FRAME_SIZE          1552    /* Must be 16-byte aligned */

/* ============================================================================
 * Quantization Block Sizes (bytes)
 * ============================================================================ */
#define Q4_0_BLOCK_BYTES    18      /* 2 (fp16 d) + 16 (32 nibbles) */
#define Q4_0_BLOCK_VALUES   32
#define Q8_0_BLOCK_BYTES    34      /* 2 (fp16 d) + 32 (int8) */
#define Q8_0_BLOCK_VALUES   32
#define Q4_1_BLOCK_BYTES    20      /* 2 (fp16 d) + 2 (fp16 m) + 16 (nibbles) */
#define Q4_1_BLOCK_VALUES   32
#define Q5_0_BLOCK_BYTES    22      /* 2 (fp16 d) + 4 (high bits) + 16 (nibbles) */
#define Q5_0_BLOCK_VALUES   32
#define Q5_1_BLOCK_BYTES    24      /* 2 (fp16 d) + 2 (fp16 m) + 4 (high) + 16 */
#define Q5_1_BLOCK_VALUES   32
#define Q2_K_BLOCK_BYTES    84      /* Variable: scales+qs+d+dmin */
#define Q2_K_BLOCK_VALUES   256
#define Q3_K_BLOCK_BYTES    110     /* hmask + qs + scales + d */
#define Q3_K_BLOCK_VALUES   256
#define Q4_K_BLOCK_BYTES    144     /* d + dmin + scales + qs */
#define Q4_K_BLOCK_VALUES   256
#define Q5_K_BLOCK_BYTES    176     /* d + dmin + scales + qh + qs */
#define Q5_K_BLOCK_VALUES   256
#define Q6_K_BLOCK_BYTES    210     /* ql + qh + scales + d */
#define Q6_K_BLOCK_VALUES   256
#define Q8_K_BLOCK_BYTES    292     /* d(f32) + qs(256) + bsums(32) */
#define Q8_K_BLOCK_VALUES   256
#define IQ4_NL_BLOCK_BYTES  18      /* 2 (fp16 d) + 16 (nibbles) */
#define IQ4_NL_BLOCK_VALUES 32
#define F16_BLOCK_BYTES     64      /* 32 * 2 (fp16 values) */
#define F16_BLOCK_VALUES    32

/* ============================================================================
 * Helper Macros for GAS AArch64 Assembly
 * ============================================================================ */

/* Symbol visibility for ELF targets */
#ifdef __APPLE__
#define GLOBAL_SYMBOL(name) .globl _##name ; _##name:
#define SYMBOL_REF(name) _##name
#else
#define GLOBAL_SYMBOL(name) .globl name ; .type name, %function ; name:
#define SYMBOL_REF(name) name
#endif

/* Function end marker for ELF */
#ifdef __APPLE__
#define END_FUNCTION(name)
#else
#define END_FUNCTION(name) .size name, .-name
#endif

#endif /* NEON_COMMON_H */
