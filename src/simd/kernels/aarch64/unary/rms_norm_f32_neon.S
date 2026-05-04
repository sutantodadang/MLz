// ===========================================================================
//   rms_norm_f32_neon.S — RMS Norm forward pass, AArch64 NEON
// ===========================================================================
// void simd_rms_norm_f32_neon(int n, float eps,
//                             const float * x,
//                             float       * y);
//
// AAPCS64: w0 = n, s0 = eps, x1 = x, x2 = y
//
// Algorithm (mirrors x86 rms_norm_f32_avx2.asm and upstream
// ggml_compute_forward_rms_norm_f32):
//   Pass 1 (F64 reduction for bit-exactness vs upstream):
//     sum_sq = SUM_i x[i]^2  (squares in f32, accumulator in f64)
//   Mean / scale (matches upstream exactly):
//     mean   = (float)(sum_sq / n)        // f64 div, narrow to f32
//     scale  = 1.0f / sqrtf(mean + eps)   // f32 sqrt, f32 reciprocal
//   Pass 2:
//     y[i] = x[i] * scale
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
    GLOBAL_SYMBOL(simd_rms_norm_f32_neon)
    // Prologue
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    // Save argument shadows.
    mov     w19, w0                 // n
    mov     x4,  x1                 // x ptr
    mov     x5,  x2                 // y ptr
    fmov    s31, s0                 // eps in f32 (saved)

    cbz     w19, .Lrms_done

    // ----- Pass 1: sum of squares in f64 ---------------------------------
    // Four 2-lane f64 accumulators (v16..v19), zeroed.
    movi    v16.2d, #0
    movi    v17.2d, #0
    movi    v18.2d, #0
    movi    v19.2d, #0

    // nb = n / 8 (number of 8-lane main iterations)
    lsr     w6, w19, #3             // w6 = n / 8
    cbz     w6, .Lrms_p1_tail

    mov     x7, x4                  // working src ptr
.Lrms_p1_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32 // load 8 floats
    fmul    v0.4s, v0.4s, v0.4s     // x*x in f32 (lanes 0-3)
    fmul    v1.4s, v1.4s, v1.4s     // x*x in f32 (lanes 4-7)

    // Widen f32 -> f64 (low and high halves) and accumulate.
    fcvtl   v20.2d, v0.2s           // lanes 0,1 -> f64
    fcvtl2  v21.2d, v0.4s           // lanes 2,3 -> f64
    fcvtl   v22.2d, v1.2s           // lanes 4,5 -> f64
    fcvtl2  v23.2d, v1.4s           // lanes 6,7 -> f64

    fadd    v16.2d, v16.2d, v20.2d
    fadd    v17.2d, v17.2d, v21.2d
    fadd    v18.2d, v18.2d, v22.2d
    fadd    v19.2d, v19.2d, v23.2d

    subs    w6, w6, #1
    b.ne    .Lrms_p1_loop

.Lrms_p1_tail:
    // Reduce v16..v19 -> scalar f64 in d16.
    fadd    v16.2d, v16.2d, v17.2d
    fadd    v18.2d, v18.2d, v19.2d
    fadd    v16.2d, v16.2d, v18.2d
    faddp   d16, v16.2d             // d16 = sum_sq (f64)

    // Scalar tail (n & 7 elements).
    and     w6, w19, #7             // tail count
    cbz     w6, .Lrms_p1_done

    and     w7, w19, #~7
    add     x8, x4, x7, lsl #2      // tail src ptr
.Lrms_p1_tail_loop:
    ldr     s0, [x8], #4
    fmul    s0, s0, s0              // x*x in f32
    fcvt    d0, s0                  // -> f64
    fadd    d16, d16, d0
    subs    w6, w6, #1
    b.ne    .Lrms_p1_tail_loop

.Lrms_p1_done:
    // mean = sum_sq / n  (f64 div, then narrow to f32)
    scvtf   d1, w19                 // d1 = (f64)n
    fdiv    d16, d16, d1            // d16 = mean (f64)
    fcvt    s16, d16                // mean -> f32
    fadd    s16, s16, s31           // + eps
    fsqrt   s16, s16                // sqrt(mean + eps)
    fmov    s17, #1.0
    fdiv    s16, s17, s16           // scale = 1 / sqrt(...)

    // Broadcast scale into v17.4s (4 lanes).
    dup     v17.4s, v16.s[0]

    // ----- Pass 2: y[i] = x[i] * scale ----------------------------------
    lsr     w6, w19, #3             // n / 8
    mov     x7, x4                  // src
    mov     x8, x5                  // dst
    cbz     w6, .Lrms_p2_tail

.Lrms_p2_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32
    fmul    v0.4s, v0.4s, v17.4s
    fmul    v1.4s, v1.4s, v17.4s
    st1     {v0.4s, v1.4s}, [x8], #32
    subs    w6, w6, #1
    b.ne    .Lrms_p2_loop

.Lrms_p2_tail:
    and     w6, w19, #7
    cbz     w6, .Lrms_done

.Lrms_p2_tail_loop:
    ldr     s0, [x7], #4
    fmul    s0, s0, s16
    str     s0, [x8], #4
    subs    w6, w6, #1
    b.ne    .Lrms_p2_tail_loop

.Lrms_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
