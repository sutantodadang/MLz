// ===========================================================================
//   vec_dot_f32_f32_neon.S — F32 × F32 Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_f32_f32_neon(int n, float * r,
//                                 const void * vx, const void * vy);
//
// AAPCS64: w0=n, x1=r (pointer), x2=vx, x3=vy
//
// Algorithm:
//   Parallel FMA in two q-register accumulators (handles 8 f32/iter):
//     v16 += x[0..3] * y[0..3]
//     v17 += x[4..7] * y[4..7]
//   Horizontal reduction via faddp over 8 lanes
//   Scalar tail for n % 8
//
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_f32_f32_neon)
    // Prologue
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Save args
    mov     w19, w0                 // n
    mov     x20, x1                 // r (output pointer)
    mov     x21, x2                 // vx
    mov     x22, x3                 // vy

    // Zero accumulators
    movi    v16.4s, #0              // acc0 (lanes 0-3)
    movi    v17.4s, #0              // acc1 (lanes 4-7)

    // Early out if n == 0
    cbz     w19, .Ldot_zero

    // --- Main loop (8 floats / iter) -----------------------------------------
    lsr     w6, w19, #3             // w6 = n / 8
    cbz     w6, .Ldot_tail

    mov     x7, x21                 // x ptr
    mov     x8, x22                 // y ptr

.Ldot_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32   // x[0..3], x[4..7]
    ld1     {v2.4s, v3.4s}, [x8], #32   // y[0..3], y[4..7]

    fmla    v16.4s, v0.4s, v2.4s    // acc0 += x[0..3] * y[0..3]
    fmla    v17.4s, v1.4s, v3.4s    // acc1 += x[4..7] * y[4..7]

    subs    w6, w6, #1
    b.ne    .Ldot_loop

    // Update x/y ptrs for potential tail
    mov     x7, x7
    mov     x8, x8

.Ldot_tail:
    and     w6, w19, #7             // tail count
    cbz     w6, .Ldot_reduce

.Ldot_tail_loop:
    ldr     s0, [x7], #4
    ldr     s2, [x8], #4
    fmla    s16, s0, s2             // acc into s16 (first lane of v16)

    subs    w6, w6, #1
    b.ne    .Ldot_tail_loop

    ; --- Horizontal sum -------------------------------------------------------
.Ldot_reduce:
    ; v16 = 4 accs (lanes 0-3), v17 = 4 accs (lanes 4-7)
    fadd    v16.4s, v16.4s, v17.4s  ; v16 = combined [s0+s4, s1+s5, s2+s6, s3+s7]
    faddp   v16.4s, v16.4s, v16.4s  ; v16 = [s0+s4+s1+s5, ..., s2+s6+s3+s7, ...]
    faddp   v16.4s, v16.4s, v16.4s  ; v16 = [total, total, total, total]

    str     s16, [x20]              ; store to *r
    b       .Ldot_done

.Ldot_zero:
    fmov    s16, wzr
    str     s16, [x20]

.Ldot_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

    END_FUNCTION(simd_vec_dot_f32_f32_neon)
