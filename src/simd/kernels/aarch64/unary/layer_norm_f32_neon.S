// ===========================================================================
//   layer_norm_f32_neon.S — Layer Norm forward pass, AArch64 NEON
// ===========================================================================
// void simd_layer_norm_f32_neon(int n, float eps,
//                               const float * x,
//                               float       * y);
//
// AAPCS64: w0 = n, s0 = eps, x1 = x, x2 = y
//
// Algorithm (Welford online, mirrors x86 layer_norm_f32_avx2.asm):
//   Pass 1 (F64 reduction for bit-exactness):
//     Simultaneously accumulate sum and sum-of-squares in f64.
//     mean   = sum / n                        (f64)
//     var    = sumsq/n - mean*mean            (f64)
//     scale  = 1.0f / sqrtf(var + eps)        (f32)
//   Pass 2:
//     y[i] = (x[i] - mean) * scale
//
// Vector width: 8 floats per main-loop iteration (two q-regs of 4 lanes
// widened to four d-pairs of 2 f64 lanes).  Tail is scalar.
//
// Validation: U1 (`zig build test-simd`) must be re-run on a real arm64 host;
// this file ships from an x86 dev box and has not been executed locally.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_layer_norm_f32_neon)
    // Prologue
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    // Save argument shadows.
    mov     w19, w0                 // n
    mov     x4,  x1                 // x ptr
    mov     x5,  x2                 // y ptr
    fmov    s31, s0                 // eps in f32 (saved)

    cbz     w19, .Lln_done

    // ----- Pass 1: Welford (sum + sumsq in f64) ---------------------------
    // Four 2-lane f64 accumulators for sum (v16..v19)
    // Four 2-lane f64 accumulators for sumsq (v20..v23)
    movi    v16.2d, #0
    movi    v17.2d, #0
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0

    // nb = n / 8 (number of 8-lane main iterations)
    lsr     w6, w19, #3             // w6 = n / 8
    cbz     w6, .Lln_p1_tail

    mov     x7, x4                  // working src ptr
.Lln_p1_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32 // load 8 floats

    // ---- accumulate sum (widen f32 -> f64) ----
    fcvtl   v24.2d, v0.2s           // lanes 0,1 -> f64
    fcvtl2  v25.2d, v0.4s           // lanes 2,3 -> f64
    fcvtl   v26.2d, v1.2s           // lanes 4,5 -> f64
    fcvtl2  v27.2d, v1.4s           // lanes 6,7 -> f64

    fadd    v16.2d, v16.2d, v24.2d
    fadd    v17.2d, v17.2d, v25.2d
    fadd    v18.2d, v18.2d, v26.2d
    fadd    v19.2d, v19.2d, v27.2d

    // ---- accumulate sum-of-squares (square then widen) ----
    fmul    v0.4s, v0.4s, v0.4s     // x^2 in f32 (lanes 0-3)
    fmul    v1.4s, v1.4s, v1.4s     // x^2 in f32 (lanes 4-7)

    fcvtl   v24.2d, v0.2s           // lanes 0,1 -> f64
    fcvtl2  v25.2d, v0.4s           // lanes 2,3 -> f64
    fcvtl   v26.2d, v1.2s           // lanes 4,5 -> f64
    fcvtl2  v27.2d, v1.4s           // lanes 6,7 -> f64

    fadd    v20.2d, v20.2d, v24.2d
    fadd    v21.2d, v21.2d, v25.2d
    fadd    v22.2d, v22.2d, v26.2d
    fadd    v23.2d, v23.2d, v27.2d

    subs    w6, w6, #1
    b.ne    .Lln_p1_loop

.Lln_p1_tail:
    // Reduce sum accumulators v16..v19 -> scalar f64 in d16.
    fadd    v16.2d, v16.2d, v17.2d
    fadd    v18.2d, v18.2d, v19.2d
    fadd    v16.2d, v16.2d, v18.2d
    faddp   d16, v16.2d             // d16 = sum (f64)

    // Reduce sumsq accumulators v20..v23 -> scalar f64 in d17.
    fadd    v20.2d, v20.2d, v21.2d
    fadd    v22.2d, v22.2d, v23.2d
    fadd    v20.2d, v20.2d, v22.2d
    faddp   d17, v20.2d             // d17 = sumsq (f64)

    // Scalar tail (n & 7 elements) — update both sum and sumsq.
    and     w6, w19, #7             // tail count
    cbz     w6, .Lln_p1_done

    and     w7, w19, #~7
    add     x8, x4, x7, lsl #2      // tail src ptr
.Lln_p1_tail_loop:
    ldr     s0, [x8], #4            // x (f32)
    fcvt    d1, s0                  // -> f64
    fadd    d16, d16, d1            // sum += x

    fmul    s0, s0, s0              // x^2 in f32
    fcvt    d1, s0                  // -> f64
    fadd    d17, d17, d1            // sumsq += x^2

    subs    w6, w6, #1
    b.ne    .Lln_p1_tail_loop

.Lln_p1_done:
    // d16 = sum (f64), d17 = sumsq (f64)
    scvtf   d1, w19                 // d1 = (f64)n

    fdiv    d16, d16, d1            // mean_f64 = sum / n
    fdiv    d17, d17, d1            // avg_sq_f64 = sumsq / n
    fmul    d2, d16, d16            // mean^2 (f64)
    fsub    d17, d17, d2            // var_f64 = avg_sq - mean^2

    // Narrow to f32, compute scale.
    fcvt    s16, d16                // mean_f32
    fcvt    s17, d17                // var_f32
    fadd    s17, s17, s31           // var + eps
    fsqrt   s17, s17                // sqrt(var + eps)
    fmov    s18, #1.0
    fdiv    s17, s18, s17           // scale = 1 / sqrt(...)

    // Broadcast for vectorized pass 2.
    dup     v16.4s, v16.s[0]        // v16.4s = mean (4 lanes)
    dup     v17.4s, v17.s[0]        // v17.4s = scale (4 lanes)

    // ----- Pass 2: y[i] = (x[i] - mean) * scale --------------------------
    lsr     w6, w19, #3             // n / 8
    mov     x7, x4                  // src
    mov     x8, x5                  // dst
    cbz     w6, .Lln_p2_tail

.Lln_p2_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32
    fsub    v0.4s, v0.4s, v16.4s    // x - mean
    fsub    v1.4s, v1.4s, v16.4s
    fmul    v0.4s, v0.4s, v17.4s    // * scale
    fmul    v1.4s, v1.4s, v17.4s
    st1     {v0.4s, v1.4s}, [x8], #32
    subs    w6, w6, #1
    b.ne    .Lln_p2_loop

.Lln_p2_tail:
    and     w6, w19, #7
    cbz     w6, .Lln_done

.Lln_p2_tail_loop:
    ldr     s0, [x7], #4
    fsub    s0, s0, s16             // x - mean
    fmul    s0, s0, s17             // * scale
    str     s0, [x8], #4
    subs    w6, w6, #1
    b.ne    .Lln_p2_tail_loop

.Lln_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
