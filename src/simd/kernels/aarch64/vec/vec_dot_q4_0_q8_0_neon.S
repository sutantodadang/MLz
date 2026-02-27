// ===========================================================================
//   vec_dot_q4_0_q8_0_neon.S — Q4_0 × Q8_0 Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_q4_0_q8_0_neon(int n, float* result,
//                                   const void* vx, const void* vy);
// AAPCS64: w0=n, x1=result, x2=vx(Q4_0), x3=vy(Q8_0)
//
// Q4_0 block: fp16 d(2B) + 16B quants(32 nibbles) = 18 bytes/32 values
// Q8_0 block: fp16 d(2B) + 32B quants(32 int8)    = 34 bytes/32 values
//
// Algorithm per block:
//   Unpack Q4_0 nibbles, subtract 8 -> signed int8
//   Multiply pairs with Q8_0 int8, accumulate
//   Scale by d0 * d1
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_q4_0_q8_0_neon)
    // Prologue
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]

    // w0 = n, x1 = result, x2 = vx (Q4_0), x3 = vy (Q8_0)
    mov     w19, w0                     // n
    mov     x20, x1                     // result ptr
    mov     x21, x2                     // vx
    mov     x22, x3                     // vy

    // nb = n / 32
    lsr     w19, w19, #5
    cbz     w19, .Lq4q8_zero

    // Zero accumulator
    movi    v0.4s, #0                   // fp32 accumulator (4 lanes)
    movi    v1.4s, #0                   // fp32 accumulator (4 lanes)

    // Constant: 8 for unsigned->signed conversion
    movi    v30.16b, #8
    // Nibble mask
    movi    v31.16b, #0x0F

.Lq4q8_loop:
    // Load Q4_0 block: fp16 d at [x21], 16 bytes quants at [x21+2]
    ldr     h2, [x21]                   // d0 as fp16
    fcvt    s2, h2                      // d0 as fp32
    add     x4, x21, #2
    ld1     {v4.16b}, [x4]              // 16 bytes = 32 nibbles

    // Load Q8_0 block: fp16 d at [x22], 32B quants at [x22+2]
    ldr     h3, [x22]                   // d1 as fp16
    fcvt    s3, h3                      // d1 as fp32
    add     x5, x22, #2
    ld1     {v6.16b, v7.16b}, [x5]      // 32 int8 values

    // Scale = d0 * d1
    fmul    s2, s2, s3                  // scale

    // Unpack low nibbles
    and     v8.16b, v4.16b, v31.16b     // low nibbles (0..15)

    // Unpack high nibbles
    ushr    v9.16b, v4.16b, #4          // high nibbles (0..15)

    // Subtract 8 -> signed (-8..7)
    sub     v8.16b, v8.16b, v30.16b     // low nibbles signed
    sub     v9.16b, v9.16b, v30.16b     // high nibbles signed

    // Dot product: low nibbles(16 values) * Q8_0 first 16 bytes
    // Use smull (signed multiply long) + accumulate
    smull   v10.8h, v8.8b, v6.8b       // prod 0-7
    smull2  v11.8h, v8.16b, v6.16b     // prod 8-15
    saddlp  v12.4s, v10.8h             // pairwise add to i32
    sadalp  v12.4s, v11.8h             // accumulate pairs

    // Dot product: high nibbles(16 values) * Q8_0 last 16 bytes
    smull   v10.8h, v9.8b, v7.8b
    smull2  v11.8h, v9.16b, v7.16b
    saddlp  v13.4s, v10.8h
    sadalp  v13.4s, v11.8h

    // Total sum
    add     v12.4s, v12.4s, v13.4s
    addv    s14, v12.4s                 // horizontal sum -> s14

    // Convert to float and scale
    scvtf   s14, s14
    fmadd   s15, s14, s2, s15          // acc += sum * scale

    // Advance pointers
    add     x21, x21, #18              // Q4_0 block size
    add     x22, x22, #34              // Q8_0 block size
    subs    w19, w19, #1
    b.ne    .Lq4q8_loop

    // Store result
    str     s15, [x20]
    b       .Lq4q8_done

.Lq4q8_zero:
    movi    v15.4s, #0
    str     s15, [x20]

.Lq4q8_done:
    // Epilogue
    ldp     d8, d9, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #64
    ret

    END_FUNCTION(simd_vec_dot_q4_0_q8_0_neon)
