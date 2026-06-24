// ===========================================================================
//   quantize_q8_k_f32_neon.S — Q8_K quantization of f32 floats (AArch64 NEON)
// ===========================================================================
// void simd_quantize_q8_k_f32_neon(int n, const float * x, void * y);
//
// AAPCS64: w0 = n, x1 = x, x2 = y
//
// Quantizes groups of 256 floats into block_q8_K structs (292 bytes each).
// Bit-exact with ggml quantize_row_q8_K_ref (ggml-quants.c:2692).
//
// struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
//
// Algorithm per block:
//   1. Per-lane max-abs search across 256 floats (8 lanes, 32 iterations)
//   2. Zero-block check, compute iscale = -127.0f / smax (signed max)
//   3. Quantize: frintn, clamp to 127, sqxtn + sqxtn2 narrow to int8
//   4. Compute bsums via xsaddlv pairwise accumulation
//   5. Store d = 1.0f / iscale
//
// Validation: U1 (`zig build test-simd`) must pass on a real arm64 host.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_quantize_q8_k_f32_neon)

    // --- Prologue: save callee-saved regs, allocate frame ---
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    // sp+48 to sp+79:  temp amax[0..7]  (8 f32 = 32 bytes)
    // sp+80 to sp+111: temp smax[0..7]  (8 f32 = 32 bytes)
    // sp+112 to sp+127: padding

    // --- block count, save args ---
    lsr     w19, w0, #8                     // w19 = n / 256
    cbz     w19, .Ldone
    mov     x20, x1                          // x20 = x (source ptr)
    mov     x21, x2                          // x21 = y (dest ptr)

    // --- pre-load constants ---
    movz    w3, #0xC2FE, lsl #16            // w3 = 0xC2FE0000 = -127.0f
    fmov    s10, w3                          // s10 = -127.0f
    fmov    s11, #1.0                        // s11 = 1.0f
    movi    v13.4s, #127                     // v13 = {127, 127, 127, 127} (int32)

.Lblock_loop:

    // ====================================================================
    // PASS 1: Per-lane max-abs search (8 lanes, 32 iterations of 8 f32)
    //   v16 = amax[0..3],  v17 = amax[4..7]
    //   v18 = smax[0..3],  v19 = smax[4..7]
    // ====================================================================
    movi    v16.4s, #0                       // amax = 0
    movi    v17.4s, #0
    movi    v18.4s, #0                       // smax = 0
    movi    v19.4s, #0
    mov     w22, #32                         // 256 / 8 = 32

.Lp1_loop:
    ld1     {v0.4s, v1.4s}, [x20], #32      // load 8 f32
    fabs    v2.4s, v0.4s                     // abs of first 4
    fabs    v3.4s, v1.4s                     // abs of second 4

    fcmgt   v4.4s, v2.4s, v16.4s             // mask: abs > amax[0..3]?
    fcmgt   v5.4s, v3.4s, v17.4s             // mask: abs > amax[4..7]?

    fmax    v16.4s, v16.4s, v2.4s            // update amax[0..3]
    fmax    v17.4s, v17.4s, v3.4s            // update amax[4..7]

    vbit    v18.4s, v0.4s, v4.4s             // update smax[0..3] where GT
    vbit    v19.4s, v1.4s, v5.4s             // update smax[4..7] where GT

    subs    w22, w22, #1
    b.ne    .Lp1_loop

    // --- store per-lane results to frame for scalar reduction ---
    st1     {v16.4s}, [sp, #48]              // amax[0..3]
    st1     {v17.4s}, [sp, #64]              // amax[4..7]
    st1     {v18.4s}, [sp, #80]              // smax[0..3]
    st1     {v19.4s}, [sp, #96]              // smax[4..7]

    // --- scalar search: find global max amax and its signed value ---
    mov     w4, #0                            // best index
    ldr     s0, [sp, #48]                     // best = amax[0]
    mov     w5, #1                            // current index
.Lred_loop:
    cmp     w5, #8
    b.eq    .Lred_done
    ldr     s1, [sp, #48, w5, uxtw #2]       // amax[w5]
    fcmp    s1, s0
    ble     .Lred_next                         // s1 <= s0: skip
    fmov    s0, s1                             // new best
    mov     w4, w5                             // update best index
.Lred_next:
    add     w5, w5, #1
    b       .Lred_loop
.Lred_done:
    // w4 = index of global amax, s0 = global amax

    // --- zero-block check ---
    fcmp    s0, #0.0
    beq     .Lzero_block

    // --- extract smax at best index ---
    ldr     s7, [sp, #80, w4, uxtw #2]       // s7 = smax[w4]

    // ====================================================================
    // Compute iscale = -127.0f / smax,  d = 1.0f / iscale
    // ====================================================================
    fdiv    s6, s10, s7                       // s6 = iscale = -127.0f / smax
    fdiv    s5, s11, s6                       // s5 = d = 1.0f / iscale  (store later)
    dup     v30.4s, v6.s[0]                    // v30 = iscale broadcast (4 lanes)

    // ====================================================================
    // PASS 2: Quantize 256 floats to int8 (16 batches of 16 floats)
    // ====================================================================
    sub     x20, x20, #1024                    // x20 back to block start
    mov     x22, x20                           // working ptr = block start

    add     x3, x21, #4                        // qs dest = y + 4
    mov     w5, #16                            // 16 batches

.Lq_loop:
    // load 16 f32 (4 q-regs)
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x22], #64

    // multiply by iscale
    fmul    v0.4s, v0.4s, v30.4s
    fmul    v1.4s, v1.4s, v30.4s
    fmul    v2.4s, v2.4s, v30.4s
    fmul    v3.4s, v3.4s, v30.4s

    // round to nearest integer (ties to even)
    frintn  v0.4s, v0.4s
    frintn  v1.4s, v1.4s
    frintn  v2.4s, v2.4s
    frintn  v3.4s, v3.4s

    // convert f32 -> int32
    fcvtzs  v0.4s, v0.4s
    fcvtzs  v1.4s, v1.4s
    fcvtzs  v2.4s, v2.4s
    fcvtzs  v3.4s, v3.4s

    // clamp upper: MIN(v, 127)
    smin    v0.4s, v0.4s, v13.4s
    smin    v1.4s, v1.4s, v13.4s
    smin    v2.4s, v2.4s, v13.4s
    smin    v3.4s, v3.4s, v13.4s

    // narrow int32 -> int16 -> int8 (16 values)
    sqxtn   v4.4h, v0.4s                       // v4[0..3] = int16(v0)
    sqxtn2  v4.8h, v1.4s                       // v4[4..7] = int16(v1)
    sqxtn   v5.4h, v2.4s                       // v5[0..3] = int16(v2)
    sqxtn2  v5.8h, v3.4s                       // v5[4..7] = int16(v3)

    sqxtn   v6.8b, v4.8h                       // v6[0..7] = int8(v4)
    sqxtn2  v6.16b, v5.8h                      // v6[8..15] = int8(v5)

    // store 16 int8
    st1     {v6.16b}, [x3], #16

    subs    w5, w5, #1
    b.ne    .Lq_loop

    // ====================================================================
    // PASS 3: bsums — 16 groups, each sum of 16 consecutive int8
    // ====================================================================
    add     x3, x21, #4                        // qs base
    add     x4, x21, #260                      // bsums base = y + 4 + 256
    mov     w5, #16                            // 16 groups

.Lbsum_loop:
    ld1     {v0.16b}, [x3], #16               // load 16 int8
    sxtl    v1.8h, v0.8b                       // sign-extend low 8 -> 8 int16
    sxtl2   v2.8h, v0.16b                      // sign-extend high 8 -> 8 int16

    saddlv  s6, v1.8h                          // sum of low 8 -> int32 in NEON s6
    saddlv  s7, v2.8h                          // sum of high 8 -> int32 in NEON s7
    fmov    w6, s6                              // transfer NEON s6 -> GPR w6
    fmov    w7, s7                              // transfer NEON s7 -> GPR w7
    add     w6, w6, w7                         // total sum (fits int16: max 2032)

    strh    w6, [x4]                           // store as int16
    add     x4, x4, #2

    subs    w5, w5, #1
    b.ne    .Lbsum_loop

    // ====================================================================
    // Store d = 1.0f / iscale at block offset 0
    // ====================================================================
    str     s5, [x21]

    // advance to next block
    b       .Lblock_advance

.Lzero_block:
    sub     x20, x20, #1024                    // reset to block start (p1 advanced x20)
    // amax == 0: d = 0, memset qs = 0, skip bsums
    str     wzr, [x21]                          // d = 0 (store zero as float)
    movi    v0.16b, #0
    add     x3, x21, #4
    mov     w5, #16
.Lzero_qs_loop:
    st1     {v0.16b}, [x3], #16
    subs    w5, w5, #1
    b.ne    .Lzero_qs_loop

.Lblock_advance:
    add     x20, x20, #1024                    // x += 256 floats (next block start)
    add     x21, x21, #292                     // y += sizeof(block_q8_K)
    subs    w19, w19, #1
    b.ne    .Lblock_loop

.Ldone:
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #128
    ret
END_FUNCTION(simd_quantize_q8_k_f32_neon)
