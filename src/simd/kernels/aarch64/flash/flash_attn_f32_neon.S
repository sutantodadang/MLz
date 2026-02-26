// ===========================================================================
//                 FLASH ATTENTION 2 - F32 AArch64 NEON KERNEL
// ===========================================================================
//
// High-performance Flash Attention 2 using AArch64 NEON (128-bit SIMD).
// Implements full tiled online-softmax algorithm (Tri Dao, 2023).
//
// F32 Query, F32 Key, F32 Value.
//
// Calling convention: AAPCS64
//   x0 = const flash_attn_config_t* config
//
// Register allocation:
//   x19 = config pointer (callee-saved, persistent)
//   x20 = work_idx
//   x21 = head_dim_k
//   x22 = head_dim_v
//   x23 = n_kv
//   x24 = Q row pointer
//   x25 = K base pointer
//   x26 = V base pointer
//   x27 = dst row pointer
//   x28 = scratch
//
// NEON registers:
//   v8-v15: callee-saved (lower 64-bit d8-d15)
//   s30: row_sum
//   s31: row_max
//
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .arch armv8-a
    .text
    .align 4

// ===========================================================================
// Data section - constants
// ===========================================================================
    .section .rodata
    .align 4
neon_neg_inf:
    .long 0xFF800000           // -infinity (IEEE 754)
    .long 0xFF800000
    .long 0xFF800000
    .long 0xFF800000
neon_neg_inf_scalar:
    .word  0xFF800000           // -infinity as 32-bit word
neon_exp_log2e:
    .float 1.44269504          // log2(e)
neon_exp_ln2:
    .float 0.693147180         // ln(2)
neon_exp_c1:
    .float 0.166666667         // 1/6
neon_exp_c2:
    .float 0.041666668         // 1/24
neon_exp_c3:
    .float 0.008333334         // 1/120
neon_one:
    .float 1.0
neon_half:
    .float 0.5
neon_exp_clamp_hi:
    .float 88.7228
neon_exp_clamp_lo:
    .float -103.278

    .text
    .align 4

// ===========================================================================
// Main kernel: simd_flash_attn_f32_neon
// ===========================================================================
GLOBAL_SYMBOL(simd_flash_attn_f32_neon)

    // === Prologue: save callee-saved registers ===
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    // Save NEON callee-saved (d8-d15)
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    // x0 = config pointer
    mov     x19, x0

    // === Allocate stack frame (16-byte aligned) ===
    // O[256] = 1024, scores[32] = 128, probs[32] = 128, locals = ~256
    // Total = 1536, round up to 1552 (16-byte aligned)
    sub     sp, sp, #FRAME_SIZE

    // === Load config dimensions into callee-saved regs & stack ===
    ldr     x21, [x19, #CFG_HEAD_DIM_K]    // head_dim_k
    ldr     x22, [x19, #CFG_HEAD_DIM_V]    // head_dim_v
    ldr     x23, [x19, #CFG_N_KV]          // n_kv
    ldr     x3,  [x19, #CFG_N_QUERIES]     // n_queries
    ldr     x4,  [x19, #CFG_N_HEAD_Q]      // n_head_q
    ldr     x5,  [x19, #CFG_N_HEAD_KV]     // n_head_kv
    ldr     x6,  [x19, #CFG_BATCH_SIZE]    // batch_size

    str     x21, [sp, #STK_HEAD_DIM_K]
    str     x22, [sp, #STK_HEAD_DIM_V]
    str     x23, [sp, #STK_N_KV_L]
    str     x3,  [sp, #STK_N_QUERIES]
    str     x4,  [sp, #STK_N_HEAD_Q]
    str     x5,  [sp, #STK_N_HEAD_KV]
    str     x6,  [sp, #STK_BATCH_SIZE]
    str     x19, [sp, #STK_CONFIG]

    // Load scale
    ldr     s0,  [x19, #CFG_SCALE]
    str     s0,  [sp, #STK_SCALE_VAL]

    // Load mask pointer
    ldr     x7,  [x19, #CFG_MASK_PTR]
    str     x7,  [sp, #STK_MASK_PTR_L]

    // Load strides nb1
    ldr     x7,  [x19, #CFG_Q_NB1]
    str     x7,  [sp, #STK_Q_NB1]
    ldr     x7,  [x19, #CFG_K_NB1]
    str     x7,  [sp, #STK_K_NB1]
    ldr     x7,  [x19, #CFG_V_NB1]
    str     x7,  [sp, #STK_V_NB1]
    ldr     x7,  [x19, #CFG_DST_NB1]
    str     x7,  [sp, #STK_DST_NB1]
    ldr     x7,  [x19, #CFG_MASK_NB1]
    str     x7,  [sp, #STK_MASK_NB1]
    // nb2
    ldr     x7,  [x19, #CFG_Q_NB2]
    str     x7,  [sp, #STK_Q_NB2]
    ldr     x7,  [x19, #CFG_K_NB2]
    str     x7,  [sp, #STK_K_NB2]
    ldr     x7,  [x19, #CFG_V_NB2]
    str     x7,  [sp, #STK_V_NB2]
    ldr     x7,  [x19, #CFG_DST_NB2]
    str     x7,  [sp, #STK_DST_NB2]
    // nb3
    ldr     x7,  [x19, #CFG_Q_NB3]
    str     x7,  [sp, #STK_Q_NB3]
    ldr     x7,  [x19, #CFG_K_NB3]
    str     x7,  [sp, #STK_K_NB3]
    ldr     x7,  [x19, #CFG_V_NB3]
    str     x7,  [sp, #STK_V_NB3]
    ldr     x7,  [x19, #CFG_DST_NB3]
    str     x7,  [sp, #STK_DST_NB3]

    // === Thread work partitioning ===
    // total_work = n_queries * n_head_q * batch_size
    ldr     x3,  [sp, #STK_N_QUERIES]
    ldr     x4,  [sp, #STK_N_HEAD_Q]
    ldr     x6,  [sp, #STK_BATCH_SIZE]
    mul     x7,  x3, x4
    mul     x7,  x7, x6              // x7 = total_work

    ldrsw   x8,  [x19, #CFG_ITH]     // ith (thread index)
    ldrsw   x9,  [x19, #CFG_NTH]     // nth (total threads)

    // work_per_thread = (total_work + nth - 1) / nth
    add     x10, x7, x9
    sub     x10, x10, #1
    udiv    x10, x10, x9              // x10 = work_per_thread

    // w_start = min(ith * work_per_thread, total_work)
    mul     x11, x8, x10
    cmp     x11, x7
    csel    x11, x11, x7, lo         // x11 = w_start

    // w_end = min(w_start + work_per_thread, total_work)
    add     x12, x11, x10
    cmp     x12, x7
    csel    x12, x12, x7, lo         // x12 = w_end

    str     x11, [sp, #STK_WORK_START]
    str     x12, [sp, #STK_WORK_END]

    // If w_start >= w_end, exit
    cmp     x11, x12
    b.ge    .Lf32_epilogue

    // === Main work loop ===
    mov     x20, x11                  // x20 = work_idx = w_start

.Lf32_work_loop:
    cmp     x20, x12
    b.ge    .Lf32_epilogue

    // Decompose work_idx: ib = w / (NQ * NHQ), rem = w % (NQ * NHQ)
    // ihq = rem / NQ, iq = rem % NQ
    ldr     x3,  [sp, #STK_N_QUERIES]
    ldr     x4,  [sp, #STK_N_HEAD_Q]
    mul     x5,  x3, x4              // x5 = NQ * NHQ

    udiv    x6,  x20, x5             // x6 = ib (batch index)
    msub    x7,  x6, x5, x20         // x7 = rem = w - ib * (NQ * NHQ)

    udiv    x8,  x7, x3              // x8 = ihq (query head index)
    msub    x9,  x8, x3, x7          // x9 = iq (query position)

    str     x9,  [sp, #STK_IQ_SAVE]

    // ihkv = ihq * n_head_kv / n_head_q
    ldr     x10, [sp, #STK_N_HEAD_KV]
    mul     x11, x8, x10
    udiv    x11, x11, x4             // x11 = ihkv

    // === Compute pointers ===
    // Q_row = q + iq*q_nb1 + ihq*q_nb2 + ib*q_nb3
    ldr     x24, [x19, #CFG_Q_PTR]
    ldr     x12, [sp, #STK_Q_NB1]
    madd    x24, x9, x12, x24
    ldr     x12, [sp, #STK_Q_NB2]
    madd    x24, x8, x12, x24
    ldr     x12, [sp, #STK_Q_NB3]
    madd    x24, x6, x12, x24

    // K_base = k + ihkv*k_nb2 + ib*k_nb3
    ldr     x25, [x19, #CFG_K_PTR]
    ldr     x12, [sp, #STK_K_NB2]
    madd    x25, x11, x12, x25
    ldr     x12, [sp, #STK_K_NB3]
    madd    x25, x6, x12, x25

    // V_base = v + ihkv*v_nb2 + ib*v_nb3
    ldr     x26, [x19, #CFG_V_PTR]
    ldr     x12, [sp, #STK_V_NB2]
    madd    x26, x11, x12, x26
    ldr     x12, [sp, #STK_V_NB3]
    madd    x26, x6, x12, x26

    // dst_row = dst + iq*dst_nb1 + ihq*dst_nb2 + ib*dst_nb3
    ldr     x27, [x19, #CFG_DST_PTR]
    ldr     x12, [sp, #STK_DST_NB1]
    madd    x27, x9, x12, x27
    ldr     x12, [sp, #STK_DST_NB2]
    madd    x27, x8, x12, x27
    ldr     x12, [sp, #STK_DST_NB3]
    madd    x27, x6, x12, x27

    // === Zero O accumulator ===
    add     x0, sp, #STK_O
    movi    v0.4s, #0
    mov     x1, x22                   // DV
    lsr     x1, x1, #2               // DV / 4 (process 4 floats at a time)
.Lf32_zero_O:
    st1     {v0.4s}, [x0], #16
    subs    x1, x1, #1
    b.gt    .Lf32_zero_O

    // Initialize row_max = -inf, row_sum = 0
    mov     w0, #0xFF800000           // -infinity
    fmov    s31, w0                   // s31 = row_max
    movi    d30, #0                   // s30 = row_sum = 0.0

    // Load scale for this work item
    ldr     s29, [sp, #STK_SCALE_VAL] // s29 = scale

    // === KV tile loop ===
    mov     x14, #0                   // x14 = tile_start = 0

.Lf32_tile_loop:
    cmp     x14, x23                  // tile_start >= n_kv?
    b.ge    .Lf32_normalize

    // tile_end = min(tile_start + FA_TILE_KV_NEON, n_kv)
    add     x15, x14, #FA_TILE_KV_NEON
    cmp     x15, x23
    csel    x15, x15, x23, lo        // x15 = tile_end
    sub     x16, x15, x14            // x16 = tile_len

    // === Compute QK^T for this tile ===
    mov     x0, #0                    // j = 0 (tile offset)
.Lf32_qk_loop:
    cmp     x0, x16
    b.ge    .Lf32_qk_done

    // K_row = K_base + (tile_start + j) * k_nb1
    add     x1, x14, x0              // kv_idx = tile_start + j
    ldr     x2, [sp, #STK_K_NB1]
    mul     x1, x1, x2
    add     x1, x25, x1              // x1 = K_row pointer

    // Dot product: sum = Q_row . K_row over head_dim_k
    movi    v0.4s, #0                 // accumulator
    movi    v1.4s, #0                 // accumulator (for unrolled)
    mov     x2, x24                   // Q pointer
    mov     x3, x1                    // K pointer
    mov     x4, x21                   // dim_k remaining

.Lf32_dot_loop:
    cmp     x4, #8
    b.lt    .Lf32_dot_tail4

    // Load 8 floats from Q and K (2x4)
    ld1     {v16.4s, v17.4s}, [x2], #32
    ld1     {v18.4s, v19.4s}, [x3], #32

    fmla    v0.4s, v16.4s, v18.4s
    fmla    v1.4s, v17.4s, v19.4s

    sub     x4, x4, #8
    b       .Lf32_dot_loop

.Lf32_dot_tail4:
    cmp     x4, #4
    b.lt    .Lf32_dot_tail1

    ld1     {v16.4s}, [x2], #16
    ld1     {v18.4s}, [x3], #16
    fmla    v0.4s, v16.4s, v18.4s
    sub     x4, x4, #4
    b       .Lf32_dot_tail4

.Lf32_dot_tail1:
    // Handle remaining 1-3 elements with scalar
    cbz     x4, .Lf32_dot_reduce
.Lf32_dot_scalar:
    ldr     s16, [x2], #4
    ldr     s18, [x3], #4
    fmadd   s2, s16, s18, s2         // scalar accumulate into s2
    subs    x4, x4, #1
    b.gt    .Lf32_dot_scalar

.Lf32_dot_reduce:
    // Reduce v0 + v1 to scalar
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s     // pairwise add
    faddp   s0, v0.2s               // final scalar sum in s0

    // Apply scale
    fmul    s0, s0, s29              // s0 = score * scale

    // Apply mask if present
    ldr     x2, [sp, #STK_MASK_PTR_L]
    cbz     x2, .Lf32_no_mask

    ldr     x3, [sp, #STK_IQ_SAVE]
    ldr     x4, [sp, #STK_MASK_NB1]
    add     x5, x14, x0              // kv_idx
    // mask_addr = mask + iq * mask_nb0 + kv_idx * mask_nb1
    ldr     x6, [x19, #CFG_MASK_NB0]
    mul     x3, x3, x6
    mul     x5, x5, x4
    add     x2, x2, x3
    add     x2, x2, x5
    ldr     s16, [x2]
    fadd    s0, s0, s16

.Lf32_no_mask:
    // Apply logit softcap if enabled
    ldr     w2, [x19, #CFG_MODE]
    tst     w2, #FA_MODE_SOFTCAP
    b.eq    .Lf32_no_softcap

    ldr     s16, [x19, #CFG_LOGIT_SOFTCAP]
    // s0 = softcap * tanh(s0 / softcap)
    fdiv    s17, s0, s16
    // Approximate tanh using NEON: tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
    fmul    s18, s17, s17            // x²
    mov     w7, #0x41D80000          // 27.0f
    fmov    s19, w7
    mov     w7, #0x41100000          // 9.0f
    fmov    s20, w7
    fmul    s20, s20, s18            // 9*x²
    fadd    s21, s19, s18            // 27 + x²
    fadd    s20, s19, s20            // 27 + 9*x²
    fdiv    s21, s21, s20
    fmul    s17, s17, s21            // x * (27+x²)/(27+9x²)
    fmul    s0, s16, s17             // softcap * tanh_approx

.Lf32_no_softcap:
    // Store score
    add     x2, sp, #STK_SCORES
    str     s0, [x2, x0, lsl #2]

    add     x0, x0, #1
    b       .Lf32_qk_loop

.Lf32_qk_done:
    // === Find tile max ===
    add     x0, sp, #STK_SCORES
    mov     w1, #0xFF800000
    fmov    s0, w1                   // tile_max = -inf

    mov     x1, #0
.Lf32_tile_max_loop:
    cmp     x1, x16
    b.ge    .Lf32_tile_max_done
    ldr     s16, [x0, x1, lsl #2]
    fmax    s0, s0, s16
    add     x1, x1, #1
    b       .Lf32_tile_max_loop
.Lf32_tile_max_done:
    // s0 = tile_max

    // === Online softmax correction ===
    // new_max = max(row_max, tile_max)
    fmax    s1, s31, s0              // s1 = new_max

    // correction = exp(row_max - new_max)
    fsub    s2, s31, s1              // old_max - new_max
    // Fast exp approximation for scalar
    bl      .Lf32_exp_scalar         // s0 = exp(s2), input in s2, output in s0
    // s0 = correction factor

    // Scale O accumulator by correction
    add     x0, sp, #STK_O
    dup     v16.4s, v0.s[0]          // broadcast correction
    mov     x1, x22
    lsr     x1, x1, #2
.Lf32_correct_O:
    ld1     {v17.4s}, [x0]
    fmul    v17.4s, v17.4s, v16.4s
    st1     {v17.4s}, [x0], #16
    subs    x1, x1, #1
    b.gt    .Lf32_correct_O

    // row_sum *= correction
    fmul    s30, s30, s0
    // row_max = new_max
    fmov    s31, s1

    // === Compute softmax probabilities for tile ===
    add     x0, sp, #STK_SCORES
    add     x1, sp, #STK_PROBS
    movi    d28, #0                   // tile_sum = 0 (s28)

    mov     x2, #0
.Lf32_prob_loop:
    cmp     x2, x16
    b.ge    .Lf32_prob_done

    ldr     s2, [x0, x2, lsl #2]
    fsub    s2, s2, s31              // score - row_max
    bl      .Lf32_exp_scalar         // s0 = exp(s2)
    str     s0, [x1, x2, lsl #2]    // probs[j] = exp(score - max)
    fadd    s28, s28, s0             // tile_sum += prob

    add     x2, x2, #1
    b       .Lf32_prob_loop
.Lf32_prob_done:

    // row_sum += tile_sum
    fadd    s30, s30, s28

    // === Accumulate P * V ===
    add     x1, sp, #STK_PROBS
    mov     x2, #0                   // j = 0

.Lf32_pv_loop:
    cmp     x2, x16
    b.ge    .Lf32_pv_done

    // Load prob[j]
    ldr     s0, [x1, x2, lsl #2]
    // Skip if prob == 0
    fcmp    s0, #0.0
    b.eq    .Lf32_pv_skip

    // V_row = V_base + (tile_start + j) * v_nb1
    add     x3, x14, x2
    ldr     x4, [sp, #STK_V_NB1]
    mul     x3, x3, x4
    add     x3, x26, x3             // x3 = V_row

    // O[d] += prob * V_row[d] for d in [0, DV)
    add     x4, sp, #STK_O
    dup     v16.4s, v0.s[0]          // broadcast prob
    mov     x5, x22
    lsr     x5, x5, #2

.Lf32_pv_inner:
    ld1     {v17.4s}, [x3], #16     // V_row[d..d+3]
    ld1     {v18.4s}, [x4]          // O[d..d+3]
    fmla    v18.4s, v17.4s, v16.4s  // O += prob * V
    st1     {v18.4s}, [x4], #16
    subs    x5, x5, #1
    b.gt    .Lf32_pv_inner

.Lf32_pv_skip:
    add     x2, x2, #1
    b       .Lf32_pv_loop

.Lf32_pv_done:
    // Next tile
    add     x14, x14, #FA_TILE_KV_NEON
    b       .Lf32_tile_loop

.Lf32_normalize:
    // === Normalize: dst[d] = O[d] / row_sum ===
    fcmp    s30, #0.0
    b.le    .Lf32_zero_output

    // inv_sum = 1.0 / row_sum
    fmov    s0, #1.0
    fdiv    s0, s0, s30
    dup     v16.4s, v0.s[0]          // broadcast inv_sum

    add     x0, sp, #STK_O
    mov     x1, x27                  // dst_row
    mov     x2, x22
    lsr     x2, x2, #2

.Lf32_norm_loop:
    ld1     {v17.4s}, [x0], #16
    fmul    v17.4s, v17.4s, v16.4s
    st1     {v17.4s}, [x1], #16
    subs    x2, x2, #1
    b.gt    .Lf32_norm_loop
    b       .Lf32_next_work

.Lf32_zero_output:
    movi    v0.4s, #0
    mov     x1, x27
    mov     x2, x22
    lsr     x2, x2, #2
.Lf32_zero_out_loop:
    st1     {v0.4s}, [x1], #16
    subs    x2, x2, #1
    b.gt    .Lf32_zero_out_loop

.Lf32_next_work:
    add     x20, x20, #1
    ldr     x12, [sp, #STK_WORK_END]
    b       .Lf32_work_loop

.Lf32_epilogue:
    // === Epilogue: restore callee-saved registers ===
    add     sp, sp, #FRAME_SIZE

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ===========================================================================
// Scalar exp approximation subroutine
// Input: s2
// Output: s0 = exp(s2)
// Clobbers: s16, s17, s18, s19, v20, v21, w7
// Preserves: s28, s29, s30, s31, x0-x28
// ===========================================================================
.Lf32_exp_scalar:
    // Clamp input
    adrp    x7, neon_exp_clamp_hi
    add     x7, x7, :lo12:neon_exp_clamp_hi
    ldr     s16, [x7]
    fmin    s2, s2, s16
    adrp    x7, neon_exp_clamp_lo
    add     x7, x7, :lo12:neon_exp_clamp_lo
    ldr     s16, [x7]
    fmax    s2, s2, s16

    // exp(x) using polynomial approximation:
    // t = x * log2(e)
    // n = round(t)
    // r = x - n * ln(2)
    // exp(x) = 2^n * (1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120)
    adrp    x7, neon_exp_log2e
    add     x7, x7, :lo12:neon_exp_log2e
    ldr     s16, [x7]
    fmul    s17, s2, s16             // t = x * log2(e)
    frintn  s18, s17                 // n = round(t)

    // r = x - n * ln(2)
    adrp    x7, neon_exp_ln2
    add     x7, x7, :lo12:neon_exp_ln2
    ldr     s16, [x7]
    fnmsub  s19, s18, s16, s2        // r = x - n * ln(2)

    // Polynomial: p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))
    adrp    x7, neon_exp_c3
    add     x7, x7, :lo12:neon_exp_c3
    ldr     s0, [x7]                // 1/120
    adrp    x7, neon_exp_c2
    add     x7, x7, :lo12:neon_exp_c2
    ldr     s16, [x7]               // 1/24
    fmadd   s0, s0, s19, s16        // 1/24 + r/120
    adrp    x7, neon_exp_c1
    add     x7, x7, :lo12:neon_exp_c1
    ldr     s16, [x7]               // 1/6
    fmadd   s0, s0, s19, s16        // 1/6 + r*(1/24 + r/120)
    adrp    x7, neon_half
    add     x7, x7, :lo12:neon_half
    ldr     s16, [x7]               // 0.5
    fmadd   s0, s0, s19, s16        // 0.5 + r*(...)
    adrp    x7, neon_one
    add     x7, x7, :lo12:neon_one
    ldr     s16, [x7]               // 1.0
    fmadd   s0, s0, s19, s16        // 1.0 + r*(...)
    fmadd   s0, s0, s19, s16        // 1.0 + r*(1.0 + r*(...))

    // Multiply by 2^n: convert n to int, shift into exponent
    fcvtzs  w7, s18                  // n as int
    add     w7, w7, #127             // bias
    lsl     w7, w7, #23              // shift to exponent position
    fmov    s16, w7                  // 2^n as float bits
    fmul    s0, s0, s16

    ret

END_FUNCTION(simd_flash_attn_f32_neon)
