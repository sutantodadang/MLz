// ===========================================================================
//   vec_dot_q8_k_q8_k_neon.S — Q8_K × Q8_K Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_q8_k_q8_k_neon(int n, float* result,
//                                   const void* vx, const void* vy);
// AAPCS64: w0=n, x1=result, x2=vx, x3=vy
//
// Q8_K block (292B/256 values):
//   fp32 d(4B) + int8 qs[256](256B) + int16 bsums[16](32B)
//
// Simple: NEON sdot-style accumulate, scale by d0*d1
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_q8_k_q8_k_neon)
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]

    mov     w19, w0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    lsr     w19, w19, #8                // nb = n/256
    cbz     w19, .Lq8kq8k_zero

    fmov    s15, wzr

.Lq8kq8k_block:
    // Load fp32 scales
    ldr     s0, [x21]                   // d0
    ldr     s1, [x22]                   // d1
    fmul    s2, s0, s1                  // scale = d0 * d1

    add     x5, x21, #4                // vx quants
    add     x6, x22, #4                // vy quants

    // Process 256 int8 pairs in groups of 64 (4 x 16-byte loads)
    // Use NEON smull/sadalp for efficient int8 dot product
    movi    v8.4s, #0                   // accumulator 0
    movi    v9.4s, #0                   // accumulator 1

    mov     w10, #0
.Lq8kq8k_inner:
    cmp     w10, #256
    b.ge    .Lq8kq8k_inner_done

    // Load 16 bytes from each
    ld1     {v4.16b}, [x5], #16
    ld1     {v5.16b}, [x6], #16

    smull   v6.8h, v4.8b, v5.8b
    smull2  v7.8h, v4.16b, v5.16b
    sadalp  v8.4s, v6.8h
    sadalp  v9.4s, v7.8h

    add     w10, w10, #16
    b       .Lq8kq8k_inner
.Lq8kq8k_inner_done:

    // Horizontal sum
    add     v8.4s, v8.4s, v9.4s
    addv    s12, v8.4s
    scvtf   s12, s12
    fmadd   s15, s12, s2, s15

    add     x21, x21, #Q8_K_BLOCK_BYTES
    add     x22, x22, #Q8_K_BLOCK_BYTES
    subs    w19, w19, #1
    b.ne    .Lq8kq8k_block

    str     s15, [x20]
    b       .Lq8kq8k_done

.Lq8kq8k_zero:
    fmov    s15, wzr
    str     s15, [x20]

.Lq8kq8k_done:
    ldp     d8, d9, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #64
    ret

    END_FUNCTION(simd_vec_dot_q8_k_q8_k_neon)
