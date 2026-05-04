// ===========================================================================
//   rope_neox_f32_neon.S — RoPE NEOX rotation (f32), AArch64 NEON
// ===========================================================================
// void simd_rope_neox_f32_neon(int64_t n_pairs,
//                              const float * cache,   // 2*n_pairs floats
//                              const float * src,     // 2*n_pairs floats
//                              float       * dst);    // 2*n_pairs floats
//
// AAPCS64: x0 = n_pairs, x1 = cache, x2 = src, x3 = dst
//
// Layout (matches NEOX/MROPE/IMROPE half-split form):
//   for ic in [0, n_pairs):
//     cos = cache[2*ic + 0]
//     sin = cache[2*ic + 1]
//     x0  = src[ic]
//     x1  = src[ic + n_pairs]
//     dst[ic]            = x0*cos - x1*sin
//     dst[ic + n_pairs]  = x0*sin + x1*cos
//
// NEON main loop processes 4 pairs per iteration via ld2 (deinterleaves
// {cos,sin} pairs into v0=cos[4], v1=sin[4]).  Tail is scalar.
//
// FMA instructions used to mirror x86 fnmadd/fmadd rounding:
//   fmls Vd, Vn, Vm   ->   Vd = Vd - Vn*Vm
//   fmla Vd, Vn, Vm   ->   Vd = Vd + Vn*Vm
//
// Validation: U1 (`zig build test-simd`) must be re-run on a real arm64 host.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_rope_neox_f32_neon)
    // Prologue (no callee-saves needed: only v0..v7 used + scalar GPRs).
    cbz     x0, .Lrope_done

    mov     x4, x0                  // n_pairs
    lsl     x5, x0, #2              // n_pairs * 4 = byte offset to high half
    add     x6, x2, x5              // src_high = src + n_pairs*4
    add     x7, x3, x5              // dst_high = dst + n_pairs*4
    mov     x8, x1                  // cache ptr (advances by 32B per 4 pairs)
    mov     x9, x2                  // src_low ptr
    mov     x10, x6                 // src_high ptr
    mov     x11, x3                 // dst_low ptr
    mov     x12, x7                 // dst_high ptr

    // nv = n_pairs / 4  (vector iterations)
    lsr     x13, x4, #2
    cbz     x13, .Lrope_tail

.Lrope_vec_loop:
    // Load 8 floats (4 cos/sin pairs) deinterleaved:
    //   v0.4s = [c0, c1, c2, c3]
    //   v1.4s = [s0, s1, s2, s3]
    ld2     {v0.4s, v1.4s}, [x8], #32

    ld1     {v2.4s}, [x9], #16      // x0 = src_low[ic..ic+3]
    ld1     {v3.4s}, [x10], #16     // x1 = src_high[ic..ic+3]

    // dst_low = x0*cos - x1*sin
    fmul    v4.4s, v2.4s, v0.4s     // v4 = x0*cos
    fmls    v4.4s, v3.4s, v1.4s     //    -= x1*sin
    st1     {v4.4s}, [x11], #16

    // dst_high = x0*sin + x1*cos
    fmul    v5.4s, v2.4s, v1.4s     // v5 = x0*sin
    fmla    v5.4s, v3.4s, v0.4s     //    += x1*cos
    st1     {v5.4s}, [x12], #16

    subs    x13, x13, #1
    b.ne    .Lrope_vec_loop

.Lrope_tail:
    // Scalar tail: remaining n_pairs & 3 elements.
    and     x13, x4, #3
    cbz     x13, .Lrope_done

.Lrope_tail_loop:
    ldp     s0, s1, [x8], #8        // cos, sin
    ldr     s2, [x9], #4            // x0
    ldr     s3, [x10], #4           // x1

    fmul    s4, s2, s0              // s4 = x0*cos
    // fmsub Sd, Sn, Sm, Sa  =>  Sd = Sa - Sn*Sm
    fmsub   s4, s3, s1, s4          // s4 = (x0*cos) - x1*sin
    str     s4, [x11], #4

    fmul    s5, s2, s1              // s5 = x0*sin
    // fmadd Sd, Sn, Sm, Sa  =>  Sd = Sa + Sn*Sm
    fmadd   s5, s3, s0, s5          // s5 = (x0*sin) + x1*cos
    str     s5, [x12], #4

    subs    x13, x13, #1
    b.ne    .Lrope_tail_loop

.Lrope_done:
    ret
