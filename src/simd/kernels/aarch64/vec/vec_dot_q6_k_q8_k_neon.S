// ===========================================================================
//   vec_dot_q6_k_q8_k_neon.S — Q6_K × Q8_K Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_q6_k_q8_k_neon(int n, float* result,
//                                   const void* vx, const void* vy);
// AAPCS64: w0=n, x1=result, x2=vx(Q6_K), x3=vy(Q8_K)
//
// Q6_K block (210B/256 values):
//   ql[128B] + qh[64B] + scales[16B] + fp16 d
// Q8_K block (292B/256 values):
//   fp32 d(4B) + int8 qs[256] + int16 bsums[16]
//
// 6-bit dequant: q6 = (ql_lo | (qh_bits << 4)) - 32
//   Then val = d * scale * q6
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL simd_vec_dot_q6_k_q8_k_neon

simd_vec_dot_q6_k_q8_k_neon:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8, d9, [sp, #80]

    mov     w19, w0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    lsr     w19, w19, #8                // nb = n/256
    cbz     w19, .Lq6k_zero

    fmov    s15, wzr

.Lq6k_block:
    // Q6_K: ql[0..127], qh[128..191], scales[192..207], d[208..209]
    ldrh    w4, [x21, #208]
    fmov    s0, w4
    fcvt    s0, h0                      // d as fp32

    ldr     s1, [x22]                   // Q8_K d
    fmul    s2, s0, s1                  // combined d

    add     x5, x21, #0                // ql ptr
    add     x6, x21, #128              // qh ptr
    add     x7, x21, #192              // scales ptr (16 int8)
    add     x8, x22, #4                // Q8_K quants

    fmov    s3, wzr                     // sum

    mov     w9, #0                      // sub-block 0..15
.Lq6k_sb:
    cmp     w9, #16
    b.ge    .Lq6k_sb_done

    // Load int8 scale
    ldrsb   w10, [x7, x9]              // scales[sb]

    // Process 16 values per sub-block
    mov     w12, #0
    mov     w23, #0                     // integer dot
.Lq6k_val:
    cmp     w12, #16
    b.ge    .Lq6k_val_done

    mov     w13, w9, lsl #4
    add     w13, w13, w12               // global value index

    // Get ql (low 4 bits contribute to q6)
    lsr     w14, w13, #1                // ql byte index
    ldrb    w24, [x5, w14, uxtw]
    tst     w13, #1
    b.ne    .Lq6k_ql_hi
    and     w24, w24, #0x0F
    b       .Lq6k_ql_done
.Lq6k_ql_hi:
    lsr     w24, w24, #4
.Lq6k_ql_done:

    // Get qh (high 2 bits)
    lsr     w14, w13, #2                // qh byte index
    and     w25, w13, #3                // bit pair
    lsl     w25, w25, #1                // shift amount
    ldrb    w26, [x6, w14, uxtw]
    lsr     w26, w26, w25
    and     w26, w26, #3
    orr     w24, w24, w26, lsl #4       // q6 = ql | (qh << 4)
    sub     w24, w24, #32               // signed: -32..31

    // Q8_K quant
    ldrsb   w4, [x8, w13, uxtw]
    madd    w23, w24, w4, w23

    add     w12, w12, #1
    b       .Lq6k_val
.Lq6k_val_done:

    mul     w23, w10, w23               // scale * dot
    scvtf   s4, w23
    fadd    s3, s3, s4

    add     w9, w9, #1
    b       .Lq6k_sb
.Lq6k_sb_done:

    fmadd   s15, s2, s3, s15

    add     x21, x21, #Q6_K_BLOCK_BYTES
    add     x22, x22, #Q8_K_BLOCK_BYTES
    subs    w19, w19, #1
    b.ne    .Lq6k_block

    str     s15, [x20]
    b       .Lq6k_done

.Lq6k_zero:
    fmov    s15, wzr
    str     s15, [x20]

.Lq6k_done:
    ldp     d8, d9, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

    END_FUNCTION simd_vec_dot_q6_k_q8_k_neon
