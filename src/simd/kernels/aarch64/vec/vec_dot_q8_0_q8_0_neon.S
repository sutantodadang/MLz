// ===========================================================================
//   vec_dot_q8_0_q8_0_neon.S — Q8_0 × Q8_0 Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_q8_0_q8_0_neon(int n, float* result,
//                                   const void* vx, const void* vy);
// AAPCS64: w0=n, x1=result, x2=vx, x3=vy
//
// Q8_0 block: fp16 d(2B) + 32B quants(32 int8) = 34 bytes/32 values
//
// Algorithm per block:
//   smull/sadalp int8 pairs, horizontal sum, scale by d0*d1
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_q8_0_q8_0_neon)

simd_vec_dot_q8_0_q8_0_neon:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    mov     w19, w0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    lsr     w19, w19, #5                // nb = n/32
    cbz     w19, .Lq8q8_zero

    // Zero accumulator
    fmov    s15, wzr

.Lq8q8_loop:
    // Load scales
    ldr     h2, [x21]                   // d0 fp16
    fcvt    s2, h2
    ldr     h3, [x22]                   // d1 fp16
    fcvt    s3, h3
    fmul    s2, s2, s3                  // scale = d0 * d1

    // Load quants
    add     x4, x21, #2
    add     x5, x22, #2
    ld1     {v4.16b, v5.16b}, [x4]      // 32 int8 from vx
    ld1     {v6.16b, v7.16b}, [x5]      // 32 int8 from vy

    // Signed multiply-accumulate: 4 groups of 8
    smull   v8.8h, v4.8b, v6.8b         // vx[0..7] * vy[0..7]
    smull2  v9.8h, v4.16b, v6.16b       // vx[8..15] * vy[8..15]
    smull   v10.8h, v5.8b, v7.8b        // vx[16..23] * vy[16..23]
    smull2  v11.8h, v5.16b, v7.16b      // vx[24..31] * vy[24..31]

    // Pairwise add to int32
    saddlp  v8.4s, v8.8h
    sadalp  v8.4s, v9.8h
    sadalp  v8.4s, v10.8h
    sadalp  v8.4s, v11.8h

    // Horizontal sum
    addv    s12, v8.4s
    scvtf   s12, s12
    fmadd   s15, s12, s2, s15           // acc += sum * scale

    add     x21, x21, #34
    add     x22, x22, #34
    subs    w19, w19, #1
    b.ne    .Lq8q8_loop

    str     s15, [x20]
    b       .Lq8q8_done

.Lq8q8_zero:
    fmov    s15, wzr
    str     s15, [x20]

.Lq8q8_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

    END_FUNCTION(simd_vec_dot_q8_0_q8_0_neon)
