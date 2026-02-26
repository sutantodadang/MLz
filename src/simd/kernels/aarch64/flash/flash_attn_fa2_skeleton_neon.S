// ===========================================================================
// Flash Attention 2 - AArch64 NEON Skeleton (Shared Include)
// ===========================================================================
//
// This file implements the full FA2 online-softmax algorithm for NEON.
// Each quantized kernel file defines type-specific macros then includes
// this skeleton via:
//   #include "flash_attn_fa2_skeleton_neon.S"
//
// REQUIRED DEFINITIONS (C preprocessor macros):
//   KERNEL_NAME          - global symbol name
//   K_BLOCK_BYTES        - bytes per K quant block
//   V_BLOCK_BYTES        - bytes per V quant block
//   K_BLOCK_VALUES       - values per K block
//   V_BLOCK_VALUES       - values per V block
//
// REQUIRED ASSEMBLY MACROS (GAS .macro):
//   dequant_k_dot_block  - dot(Q_f32, dequant(K_block)), add to s_acc
//       Args: k_ptr, q_ptr, s_acc (scalar float reg)
//       Clobbers: v0-v7, v16-v23, x9-x15
//       Preserves: x19-x28, v8-v15, v28-v31
//
//   dequant_v_accum_block - dequant V block, multiply by prob, add to O
//       Args: v_ptr, prob_reg (scalar float), o_ptr
//       Clobbers: v0-v7, v16-v23, x9-x15
//       Preserves: x19-x28, v8-v15, v28-v31
//
// Register allocation:
//   x19: config pointer
//   x20: work_idx
//   x21: head_dim_k
//   x22: head_dim_v
//   x23: n_kv
//   x24: Q row
//   x25: K base
//   x26: V base
//   x27: dst row
//   s29: scale
//   s30: row_sum
//   s31: row_max
//   s28: tile_sum (temporary)
//
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .arch armv8-a
    .text
    .align 4

// ===========================================================================
// Exp approximation constants (shared)
// ===========================================================================
    .section .rodata
    .align 4
.Lskel_exp_clamp_hi:
    .float 88.7228
.Lskel_exp_clamp_lo:
    .float -103.278
.Lskel_exp_log2e:
    .float 1.44269504
.Lskel_exp_ln2:
    .float 0.693147180
.Lskel_exp_c1:
    .float 0.166666667
.Lskel_exp_c2:
    .float 0.041666668
.Lskel_exp_c3:
    .float 0.008333334
.Lskel_one:
    .float 1.0
.Lskel_half:
    .float 0.5

    .text
    .align 4

// ===========================================================================
// Main skeleton kernel
// ===========================================================================
GLOBAL_SYMBOL(KERNEL_NAME)

    // Prologue
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    mov     x19, x0
    sub     sp, sp, #FRAME_SIZE

    // Load config
    ldr     x21, [x19, #CFG_HEAD_DIM_K]
    ldr     x22, [x19, #CFG_HEAD_DIM_V]
    ldr     x23, [x19, #CFG_N_KV]
    ldr     x3,  [x19, #CFG_N_QUERIES]
    ldr     x4,  [x19, #CFG_N_HEAD_Q]
    ldr     x5,  [x19, #CFG_N_HEAD_KV]
    ldr     x6,  [x19, #CFG_BATCH_SIZE]

    str     x21, [sp, #STK_HEAD_DIM_K]
    str     x22, [sp, #STK_HEAD_DIM_V]
    str     x23, [sp, #STK_N_KV_L]
    str     x3,  [sp, #STK_N_QUERIES]
    str     x4,  [sp, #STK_N_HEAD_Q]
    str     x5,  [sp, #STK_N_HEAD_KV]
    str     x6,  [sp, #STK_BATCH_SIZE]
    str     x19, [sp, #STK_CONFIG]

    ldr     s29, [x19, #CFG_SCALE]
    str     s29, [sp, #STK_SCALE_VAL]

    ldr     x7,  [x19, #CFG_MASK_PTR]
    str     x7,  [sp, #STK_MASK_PTR_L]

    // Strides
    ldr     x7, [x19, #CFG_Q_NB1];   str x7, [sp, #STK_Q_NB1]
    ldr     x7, [x19, #CFG_K_NB1];   str x7, [sp, #STK_K_NB1]
    ldr     x7, [x19, #CFG_V_NB1];   str x7, [sp, #STK_V_NB1]
    ldr     x7, [x19, #CFG_DST_NB1]; str x7, [sp, #STK_DST_NB1]
    ldr     x7, [x19, #CFG_MASK_NB1]; str x7, [sp, #STK_MASK_NB1]
    ldr     x7, [x19, #CFG_Q_NB2];   str x7, [sp, #STK_Q_NB2]
    ldr     x7, [x19, #CFG_K_NB2];   str x7, [sp, #STK_K_NB2]
    ldr     x7, [x19, #CFG_V_NB2];   str x7, [sp, #STK_V_NB2]
    ldr     x7, [x19, #CFG_DST_NB2]; str x7, [sp, #STK_DST_NB2]
    ldr     x7, [x19, #CFG_Q_NB3];   str x7, [sp, #STK_Q_NB3]
    ldr     x7, [x19, #CFG_K_NB3];   str x7, [sp, #STK_K_NB3]
    ldr     x7, [x19, #CFG_V_NB3];   str x7, [sp, #STK_V_NB3]
    ldr     x7, [x19, #CFG_DST_NB3]; str x7, [sp, #STK_DST_NB3]

    // K/V block counts = head_dim_k / K_BLOCK_VALUES
    mov     x7, #K_BLOCK_VALUES
    udiv    x8, x21, x7
    str     x8, [sp, #STK_K_NBLOCKS]
    mov     x7, #V_BLOCK_VALUES
    udiv    x8, x22, x7
    str     x8, [sp, #STK_V_NBLOCKS]

    // Thread work partitioning
    mul     x7, x3, x4
    mul     x7, x7, x6
    ldrsw   x8, [x19, #CFG_ITH]
    ldrsw   x9, [x19, #CFG_NTH]
    add     x10, x7, x9
    sub     x10, x10, #1
    udiv    x10, x10, x9
    mul     x11, x8, x10
    cmp     x11, x7
    csel    x11, x11, x7, lo
    add     x12, x11, x10
    cmp     x12, x7
    csel    x12, x12, x7, lo
    str     x11, [sp, #STK_WORK_START]
    str     x12, [sp, #STK_WORK_END]

    cmp     x11, x12
    b.ge    .Lskel_epilogue

    mov     x20, x11

.Lskel_work_loop:
    ldr     x12, [sp, #STK_WORK_END]
    cmp     x20, x12
    b.ge    .Lskel_epilogue

    // Decompose work_idx
    ldr     x3, [sp, #STK_N_QUERIES]
    ldr     x4, [sp, #STK_N_HEAD_Q]
    mul     x5, x3, x4
    udiv    x6, x20, x5
    msub    x7, x6, x5, x20
    udiv    x8, x7, x3
    msub    x9, x8, x3, x7
    str     x9, [sp, #STK_IQ_SAVE]

    ldr     x10, [sp, #STK_N_HEAD_KV]
    mul     x11, x8, x10
    udiv    x11, x11, x4

    // Compute pointers
    ldr     x24, [x19, #CFG_Q_PTR]
    ldr     x12, [sp, #STK_Q_NB1]; madd x24, x9, x12, x24
    ldr     x12, [sp, #STK_Q_NB2]; madd x24, x8, x12, x24
    ldr     x12, [sp, #STK_Q_NB3]; madd x24, x6, x12, x24

    ldr     x25, [x19, #CFG_K_PTR]
    ldr     x12, [sp, #STK_K_NB2]; madd x25, x11, x12, x25
    ldr     x12, [sp, #STK_K_NB3]; madd x25, x6, x12, x25

    ldr     x26, [x19, #CFG_V_PTR]
    ldr     x12, [sp, #STK_V_NB2]; madd x26, x11, x12, x26
    ldr     x12, [sp, #STK_V_NB3]; madd x26, x6, x12, x26

    ldr     x27, [x19, #CFG_DST_PTR]
    ldr     x12, [sp, #STK_DST_NB1]; madd x27, x9, x12, x27
    ldr     x12, [sp, #STK_DST_NB2]; madd x27, x8, x12, x27
    ldr     x12, [sp, #STK_DST_NB3]; madd x27, x6, x12, x27

    // Zero O
    add     x0, sp, #STK_O
    movi    v0.4s, #0
    mov     x1, x22
    lsr     x1, x1, #2
.Lskel_zero_O:
    st1     {v0.4s}, [x0], #16
    subs    x1, x1, #1
    b.gt    .Lskel_zero_O

    mov     w0, #0xFF800000
    fmov    s31, w0
    movi    d30, #0
    ldr     s29, [sp, #STK_SCALE_VAL]

    // KV tile loop
    mov     x14, #0

.Lskel_tile_loop:
    cmp     x14, x23
    b.ge    .Lskel_normalize

    add     x15, x14, #FA_TILE_KV_NEON
    cmp     x15, x23
    csel    x15, x15, x23, lo
    sub     x16, x15, x14

    // QK dot products for tile
    mov     x0, #0
.Lskel_qk_loop:
    cmp     x0, x16
    b.ge    .Lskel_qk_done

    add     x1, x14, x0
    ldr     x2, [sp, #STK_K_NB1]
    mul     x1, x1, x2
    add     x1, x25, x1              // K_row

    // Quantized dot: iterate over K blocks
    movi    d8, #0                    // s8 = dot accumulator
    mov     x2, x24                   // Q ptr
    mov     x3, x1                    // K block ptr
    ldr     x4, [sp, #STK_K_NBLOCKS]

.Lskel_kdot_loop:
    cbz     x4, .Lskel_kdot_done
    // Call type-specific dequant_k_dot_block macro
    // Input: x3 = K block, x2 = Q floats, s8 = accumulator
    dequant_k_dot_block x3, x2, s8
    add     x3, x3, #K_BLOCK_BYTES
    add     x2, x2, #(K_BLOCK_VALUES * 4)
    sub     x4, x4, #1
    b       .Lskel_kdot_loop
.Lskel_kdot_done:

    // s8 = raw dot product, apply scale
    fmul    s8, s8, s29

    // Apply mask
    ldr     x2, [sp, #STK_MASK_PTR_L]
    cbz     x2, .Lskel_no_mask
    ldr     x3, [sp, #STK_IQ_SAVE]
    ldr     x4, [sp, #STK_MASK_NB1]
    add     x5, x14, x0
    ldr     x6, [x19, #CFG_MASK_NB0]
    mul     x3, x3, x6
    mul     x5, x5, x4
    add     x2, x2, x3
    add     x2, x2, x5
    ldr     s16, [x2]
    fadd    s8, s8, s16
.Lskel_no_mask:

    // Softcap
    ldr     w2, [x19, #CFG_MODE]
    tst     w2, #FA_MODE_SOFTCAP
    b.eq    .Lskel_no_softcap
    ldr     s16, [x19, #CFG_LOGIT_SOFTCAP]
    fdiv    s17, s8, s16
    fmul    s18, s17, s17
    mov     w7, #0x41D80000
    fmov    s19, w7
    mov     w7, #0x41100000
    fmov    s20, w7
    fmul    s20, s20, s18
    fadd    s21, s19, s18
    fadd    s20, s19, s20
    fdiv    s21, s21, s20
    fmul    s17, s17, s21
    fmul    s8, s16, s17
.Lskel_no_softcap:

    add     x2, sp, #STK_SCORES
    str     s8, [x2, x0, lsl #2]
    add     x0, x0, #1
    b       .Lskel_qk_loop
.Lskel_qk_done:

    // Find tile max
    add     x0, sp, #STK_SCORES
    mov     w1, #0xFF800000
    fmov    s0, w1
    mov     x1, #0
.Lskel_tmax:
    cmp     x1, x16
    b.ge    .Lskel_tmax_done
    ldr     s16, [x0, x1, lsl #2]
    fmax    s0, s0, s16
    add     x1, x1, #1
    b       .Lskel_tmax
.Lskel_tmax_done:

    // Online softmax correction
    fmax    s1, s31, s0
    fsub    s2, s31, s1
    bl      .Lskel_exp_scalar

    add     x0, sp, #STK_O
    dup     v16.4s, v0.s[0]
    mov     x1, x22
    lsr     x1, x1, #2
.Lskel_correct:
    ld1     {v17.4s}, [x0]
    fmul    v17.4s, v17.4s, v16.4s
    st1     {v17.4s}, [x0], #16
    subs    x1, x1, #1
    b.gt    .Lskel_correct

    fmul    s30, s30, s0
    fmov    s31, s1

    // Softmax probs
    add     x0, sp, #STK_SCORES
    add     x1, sp, #STK_PROBS
    movi    d28, #0
    mov     x2, #0
.Lskel_prob:
    cmp     x2, x16
    b.ge    .Lskel_prob_done
    ldr     s2, [x0, x2, lsl #2]
    fsub    s2, s2, s31
    bl      .Lskel_exp_scalar
    str     s0, [x1, x2, lsl #2]
    fadd    s28, s28, s0
    add     x2, x2, #1
    b       .Lskel_prob
.Lskel_prob_done:
    fadd    s30, s30, s28

    // P * V accumulation
    add     x1, sp, #STK_PROBS
    mov     x2, #0
.Lskel_pv:
    cmp     x2, x16
    b.ge    .Lskel_pv_done
    ldr     s0, [x1, x2, lsl #2]
    fcmp    s0, #0.0
    b.eq    .Lskel_pv_skip

    add     x3, x14, x2
    ldr     x4, [sp, #STK_V_NB1]
    mul     x3, x3, x4
    add     x3, x26, x3

    // Type-specific V accumulation
    add     x4, sp, #STK_O
    ldr     x5, [sp, #STK_V_NBLOCKS]
    dequant_v_accum_block x3, s0, x4, x5

.Lskel_pv_skip:
    add     x2, x2, #1
    b       .Lskel_pv
.Lskel_pv_done:

    add     x14, x14, #FA_TILE_KV_NEON
    b       .Lskel_tile_loop

.Lskel_normalize:
    fcmp    s30, #0.0
    b.le    .Lskel_zero_out

    fmov    s0, #1.0
    fdiv    s0, s0, s30
    dup     v16.4s, v0.s[0]
    add     x0, sp, #STK_O
    mov     x1, x27
    mov     x2, x22
    lsr     x2, x2, #2
.Lskel_norm:
    ld1     {v17.4s}, [x0], #16
    fmul    v17.4s, v17.4s, v16.4s
    st1     {v17.4s}, [x1], #16
    subs    x2, x2, #1
    b.gt    .Lskel_norm
    b       .Lskel_next

.Lskel_zero_out:
    movi    v0.4s, #0
    mov     x1, x27
    mov     x2, x22
    lsr     x2, x2, #2
.Lskel_zero_out_l:
    st1     {v0.4s}, [x1], #16
    subs    x2, x2, #1
    b.gt    .Lskel_zero_out_l

.Lskel_next:
    add     x20, x20, #1
    b       .Lskel_work_loop

.Lskel_epilogue:
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

// Scalar exp subroutine
.Lskel_exp_scalar:
    adrp    x7, .Lskel_exp_clamp_hi
    add     x7, x7, :lo12:.Lskel_exp_clamp_hi
    ldr     s16, [x7]
    fmin    s2, s2, s16
    adrp    x7, .Lskel_exp_clamp_lo
    add     x7, x7, :lo12:.Lskel_exp_clamp_lo
    ldr     s16, [x7]
    fmax    s2, s2, s16

    adrp    x7, .Lskel_exp_log2e
    add     x7, x7, :lo12:.Lskel_exp_log2e
    ldr     s16, [x7]
    fmul    s17, s2, s16
    frintn  s18, s17

    adrp    x7, .Lskel_exp_ln2
    add     x7, x7, :lo12:.Lskel_exp_ln2
    ldr     s16, [x7]
    fnmsub  s19, s18, s16, s2

    adrp    x7, .Lskel_exp_c3
    add     x7, x7, :lo12:.Lskel_exp_c3
    ldr     s0, [x7]
    adrp    x7, .Lskel_exp_c2
    add     x7, x7, :lo12:.Lskel_exp_c2
    ldr     s16, [x7]
    fmadd   s0, s0, s19, s16
    adrp    x7, .Lskel_exp_c1
    add     x7, x7, :lo12:.Lskel_exp_c1
    ldr     s16, [x7]
    fmadd   s0, s0, s19, s16
    adrp    x7, .Lskel_half
    add     x7, x7, :lo12:.Lskel_half
    ldr     s16, [x7]
    fmadd   s0, s0, s19, s16
    adrp    x7, .Lskel_one
    add     x7, x7, :lo12:.Lskel_one
    ldr     s16, [x7]
    fmadd   s0, s0, s19, s16
    fmadd   s0, s0, s19, s16

    fcvtzs  w7, s18
    add     w7, w7, #127
    lsl     w7, w7, #23
    fmov    s16, w7
    fmul    s0, s0, s16
    ret

END_FUNCTION(KERNEL_NAME)
