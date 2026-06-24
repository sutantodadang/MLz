// ===========================================================================
//   rope_standard_f32_neon.S — Standard (interleaved) RoPE rotation, NEON
// ===========================================================================
// void simd_rope_standard_f32_neon(int64_t n_pairs,
//                                  const float * cache,   // 2*n_pairs floats
//                                  const float * src,     // 2*n_pairs floats
//                                  float       * dst);    // 2*n_pairs floats
//
// AAPCS64: x0 = n_pairs, x1 = cache, x2 = src, x3 = dst
//
// Layout (STANDARD / interleaved pairs):
//   for ic in [0, n_pairs):
//     cos = cache[2*ic + 0]
//     sin = cache[2*ic + 1]
//     x0  = src[2*ic + 0]       // EVEN index (adjacent pair)
//     x1  = src[2*ic + 1]       // ODD index (adjacent pair)
//     dst[2*ic + 0] = x0*cos - x1*sin
//     dst[2*ic + 1] = x0*sin + x1*cos
//
// NEON main loop processes 4 pairs (8 floats) per iteration.
// ld2 deinterleaves {cos,sin} and {x0,x1} natively.  st2 interleaves
// results back.  Tail is scalar via ldp/ldr.
//
// FMA instructions mirror x86 fnmadd/fmadd rounding:
//   fmls Vd, Vn, Vm   ->   Vd = Vd - Vn*Vm
//   fmla Vd, Vn, Vm   ->   Vd = Vd + Vn*Vm
//
// Validation: U1 (`zig build test-simd`) must be re-run on a real arm64 host.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_rope_standard_f32_neon)
    // No callee-save NEON regs needed (v0-v7 + scalar GPRs only).
    cbz     x0, .Lrstd_done

    mov     x4, x0                  // n_pairs
    mov     x5, x1                  // cache ptr
    mov     x6, x2                  // src ptr
    mov     x7, x3                  // dst ptr

    // nv = n_pairs / 4  (vector iterations)
    lsr     x8, x4, #2
    cbz     x8, .Lrstd_tail

.Lrstd_vec_loop:
    // Load 4 cos/sin pairs, deinterleaved:
    //   v0.4s = [c0, c1, c2, c3]
    //   v1.4s = [s0, s1, s2, s3]
    ld2     {v0.4s, v1.4s}, [x5], #32

    // Load 4 x0/x1 pairs, deinterleaved:
    //   v2.4s = [x0_0, x0_1, x0_2, x0_3]
    //   v3.4s = [x1_0, x1_1, x1_2, x1_3]
    ld2     {v2.4s, v3.4s}, [x6], #32

    // dst_even = x0*cos - x1*sin
    fmul    v4.4s, v2.4s, v0.4s     // v4 = x0*cos
    fmls    v4.4s, v3.4s, v1.4s     //    -= x1*sin

    // dst_odd = x0*sin + x1*cos
    fmul    v5.4s, v2.4s, v1.4s     // v5 = x0*sin
    fmla    v5.4s, v3.4s, v0.4s     //    += x1*cos

    // Store interleaved: [dst_even0,dst_odd0,dst_even1,dst_odd1,...]
    st2     {v4.4s, v5.4s}, [x7], #32

    subs    x8, x8, #1
    b.ne    .Lrstd_vec_loop

.Lrstd_tail:
    // Scalar tail: remaining n_pairs & 3 elements.
    and     x8, x4, #3
    cbz     x8, .Lrstd_done

.Lrstd_tail_loop:
    ldp     s0, s1, [x5], #8        // cos, sin
    ldp     s2, s3, [x6], #8        // x0, x1

    // dst_even = x0*cos - x1*sin
    fmul    s4, s2, s0              // s4 = x0*cos
    fmsub   s4, s3, s1, s4          // s4 = (x0*cos) - x1*sin
    str     s4, [x7], #4

    // dst_odd = x0*sin + x1*cos
    fmul    s5, s2, s1              // s5 = x0*sin
    fmadd   s5, s3, s0, s5          // s5 = (x0*sin) + x1*cos
    str     s5, [x7], #4

    subs    x8, x8, #1
    b.ne    .Lrstd_tail_loop

.Lrstd_done:
    ret
