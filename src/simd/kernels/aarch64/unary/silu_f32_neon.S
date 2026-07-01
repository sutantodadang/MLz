// ===========================================================================
//   silu_f32_neon.S — SiLU activation forward pass, AArch64 NEON
// ===========================================================================
// void simd_silu_f32_neon(int n, const float * x, float * y)//
//
// AAPCS64: w0 = n, x1 = x, x2 = y
//
// Computes:  y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
//
// Algorithm (mirrors x86 silu_f32_avx2.asm):
//   exp(-x) computed via 2^t decomposition with degree-4 Taylor polynomial:
//     t     = -x * log2e
//     n     = frintn(t)               // nearest integer
//     f     = t - n                   // fractional, in [-0.5, 0.5]
//     2^f   = 1 + c1*f + c2*f^2 + c3*f^3 + c4*f^4   (fmla chain)
//     2^n   = construct via (n + 127) << 23 using integer ops
//     exp   = 2^n * 2^f
//   sigmoid = 1 / (1 + exp(-x))
//   silu    = x * sigmoid
//
// Vector width: 8 floats per main-loop iteration (two q-regs of 4 lanes).
// Tail: scalar for n % 8.
//
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_silu_f32_neon)
    // Prologue
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Save args
    mov     w19, w0                 // n
    mov     x20, x1                 // x ptr
    mov     x21, x2                 // y ptr

    cbz     w19, .Lsilu_done

    // Load scalar constants (will broadcast into vectors below)
    adrp    x22, .Lsilu_consts
    add     x22, x22, :lo12:.Lsilu_consts

    // --- Main loop (8 floats / iter) -----------------------------------------
    lsr     w6, w19, #3             // w6 = n / 8
    cbz     w6, .Lsilu_tail

    // Broadcast constants into vector registers
    ldr     s24, [x22, #0]          // log2e
    ldr     s25, [x22, #4]          // sign bit mask
    ldr     s26, [x22, #8]          // 127 (bias, as int32 bits)
    ldr     s27, [x22, #12]         // 1.0f
    dup     v24.4s, v24.s[0]
    dup     v25.4s, v25.s[0]
    dup     v26.4s, v26.s[0]
    dup     v27.4s, v27.s[0]

    // Polynomial coefficients
    ldr     s28, [x22, #16]         // c1
    ldr     s29, [x22, #20]         // c2
    ldr     s30, [x22, #24]         // c3
    ldr     s31, [x22, #28]         // c4
    dup     v28.4s, v28.s[0]
    dup     v29.4s, v29.s[0]
    dup     v30.4s, v30.s[0]
    dup     v31.4s, v31.s[0]

    // Clamp constants
    movi    v20.4s, #0              // clamp lo = 0
    ldr     s21, [x22, #32]         // clamp hi = 254
    dup     v21.4s, v21.s[0]

    mov     x7, x20                 // src
    mov     x8, x21                 // dst

    // We'll use v0-v1 for x, v2-v3 for -x/t, v4-v5 for n, v6-v7 for f
    // and reuse v0-v1 for results
    // Process two q-regs per iteration: v0 (lanes 0-3), v1 (lanes 4-7)
    // When we have non-4-wide instructions, process per-q-reg.

.Lsilu_loop:
    ld1     {v0.4s, v1.4s}, [x7], #32   // load 8 floats

    // ---- Process v0 (lanes 0-3) ----
    // -x
    fneg    v2.4s, v0.4s                // v2 = -x (flip sign bit)
    // t = -x * log2e
    fmul    v2.4s, v2.4s, v24.4s        // v2 = t
    // n = round(t)
    frintn  v4.4s, v2.4s                // v4 = n
    // f = t - n
    fsub    v6.4s, v2.4s, v4.4s         // v6 = f in [-0.5, 0.5]

    // Polynomial 2^f: (((c4*f + c3)*f + c2)*f + c1)*f + 1.0
    fmul    v16.4s, v31.4s, v6.4s       // c4 * f
    fadd    v16.4s, v16.4s, v30.4s      // + c3
    fmul    v16.4s, v16.4s, v6.4s       // * f
    fadd    v16.4s, v16.4s, v29.4s      // + c2
    fmul    v16.4s, v16.4s, v6.4s       // * f
    fadd    v16.4s, v16.4s, v28.4s      // + c1
    fmul    v16.4s, v16.4s, v6.4s       // * f
    fadd    v16.4s, v16.4s, v27.4s      // + 1.0 = 2^f

    // 2^n: (n + 127) << 23, clamped
    fcvtzs  v17.4s, v4.4s               // n as int32
    add     v17.4s, v17.4s, v26.4s      // n + 127
    smax    v17.4s, v17.4s, v20.4s      // clamp lo: max(0, ...)
    smin    v17.4s, v17.4s, v21.4s      // clamp hi: min(254, ...)
    shl     v17.4s, v17.4s, #23         // float bit pattern for 2^n

    // exp(-x) = 2^n * 2^f
    fmul    v2.4s, v17.4s, v16.4s       // v2 = exp(-x)

    // sigmoid = 1 / (1 + exp(-x))
    fadd    v2.4s, v2.4s, v27.4s        // 1 + exp(-x)
    fdiv    v2.4s, v27.4s, v2.4s        // sigmoid

    // silu = x * sigmoid
    fmul    v0.4s, v0.4s, v2.4s

    // ---- Process v1 (lanes 4-7) ----
    fneg    v3.4s, v1.4s
    fmul    v3.4s, v3.4s, v24.4s
    frintn  v5.4s, v3.4s
    fsub    v7.4s, v3.4s, v5.4s

    fmul    v18.4s, v31.4s, v7.4s
    fadd    v18.4s, v18.4s, v30.4s
    fmul    v18.4s, v18.4s, v7.4s
    fadd    v18.4s, v18.4s, v29.4s
    fmul    v18.4s, v18.4s, v7.4s
    fadd    v18.4s, v18.4s, v28.4s
    fmul    v18.4s, v18.4s, v7.4s
    fadd    v18.4s, v18.4s, v27.4s

    fcvtzs  v19.4s, v5.4s
    add     v19.4s, v19.4s, v26.4s
    smax    v19.4s, v19.4s, v20.4s      // clamp lo
    smin    v19.4s, v19.4s, v21.4s      // clamp hi
    shl     v19.4s, v19.4s, #23

    fmul    v3.4s, v19.4s, v18.4s
    fadd    v3.4s, v3.4s, v27.4s
    fdiv    v3.4s, v27.4s, v3.4s
    fmul    v1.4s, v1.4s, v3.4s

    st1     {v0.4s, v1.4s}, [x8], #32

    subs    w6, w6, #1
    b.ne    .Lsilu_loop

.Lsilu_tail:
    and     w6, w19, #7             // tail count
    cbz     w6, .Lsilu_done

    // Reload scalar constants for scalar tail
    ldr     s24, [x22, #0]          // log2e
    ldr     s25, [x22, #4]          // sign mask
    ldr     s26, [x22, #8]          // 127
    ldr     s27, [x22, #12]         // 1.0
    ldr     s28, [x22, #16]         // c1
    ldr     s29, [x22, #20]         // c2
    ldr     s30, [x22, #24]         // c3
    ldr     s31, [x22, #28]         // c4

    mov     x9, x7                  // src (from after loop)
    mov     x10, x8                 // dst

.Lsilu_tail_loop:
    ldr     s0, [x9], #4            // x

    fneg    s2, s0                  // -x
    fmul    s2, s2, s24             // t
    frintn  s4, s2                  // n
    fsub    s6, s2, s4              // f

    fmul    s16, s31, s6            // c4*f
    fadd    s16, s16, s30           // + c3
    fmul    s16, s16, s6            // * f
    fadd    s16, s16, s29           // + c2
    fmul    s16, s16, s6            // * f
    fadd    s16, s16, s28           // + c1
    fmul    s16, s16, s6            // * f
    fadd    s16, s16, s27           // + 1.0

    fcvtzs  w11, s4                 // n as int
    add     w11, w11, #127          // biased exponent
    cmp     w11, #0
    csel    w11, wzr, w11, lt       // clamp lo: max(0, n+127)
    mov     w12, #254
    cmp     w11, w12
    csel    w11, w12, w11, gt       // clamp hi: min(254, n+127)
    lsl     w11, w11, #23
    fmov    s17, w11                // 2^n

    fmul    s2, s17, s16            // exp(-x)
    fadd    s2, s2, s27             // 1 + exp(-x)
    fdiv    s2, s27, s2             // sigmoid
    fmul    s0, s0, s2              // silu

    str     s0, [x10], #4
    subs    w6, w6, #1
    b.ne    .Lsilu_tail_loop

.Lsilu_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

    END_FUNCTION(simd_silu_f32_neon)

// ---------------------------------------------------------------------------
// Read-only constants
// ---------------------------------------------------------------------------
    .section .rodata
    .align 4
.Lsilu_consts:
    .word   0x3FB8AA3B     // log2(e) = 1.4426950f
    .word   0x80000000     // sign-bit mask
    .word   127            // exponent bias
    .word   0x3F800000     // 1.0f
    .word   0x3F317218     // c1 = 0.6931472f (ln 2)
    .word   0x3E75FDF0     // c2 = 0.2402265f
    .word   0x3D635847     // c3 = 0.0555041f
    .word   0x3C1D99AA     // c4 = 0.0096181f
    .word   254            // exp clamp hi
