// ===========================================================================
//  vec_dot_q5_k_q8_k_neon.S — Q5_K × Q8_K Dot Product, AArch64 NEON
// ===========================================================================
//  void simd_vec_dot_q5_k_q8_k_neon(int n, float* result,
//                                    const void* vx, const void* vy);
//  AAPCS64: w0=n, x1=result, x2=vx (Q5_K blocks), x3=vy (Q8_K blocks)
//
//  Bit-for-bit equivalent to upstream
//  ggml_vec_dot_q5_K_q8_K_generic (deps/llama_cpp/ggml/src/ggml-cpu/quants.c).
//
//  BLOCKS
//  ------
//    block_q5_K  (176 B):
//      fp16 d (+0), fp16 dmin (+2), u8 scales[12] (+4),
//      u8 qh[32] (+16), u8 qs[128] (+48)
//
//    block_q8_K  (292 B):
//      f32 d (+0), i8 qs[256] (+4), i16 bsums[16] (+260)
//
//  REGISTER ALLOCATION
//  -------------------
//    w19 = nb         x20 = result   x21 = vx        x22 = vy
//    s15 = sumf
//    v16..v23 = work registers (caller-saved)
//    [sp+96 .. sp+112] = utmp[0..3] = scales[0..7] | mins[0..7]
//
//  MIT License
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_q5_k_q8_k_neon)

    // -------- prologue --------
    // 128-byte frame: 96 B for callee-saved gprs/fpregs + 32 B utmp scratch.
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    // utmp scratch lives at [sp + 96 .. sp + 127]

    mov     w19, w0                      // nb input (n)
    mov     x20, x1                      // result
    mov     x21, x2                      // vx
    mov     x22, x3                      // vy

    lsr     w19, w19, #8                 // nb = n / 256
    fmov    s15, wzr                     // sumf = 0.0f
    cbz     w19, .Lq5k_done

.Lq5k_block:
    // ---- decode d, dmin (fp16) and y_d (f32) ----
    ldrh    w4, [x21, #0]
    fmov    s0, w4
    fcvt    s0, h0                       // s0 = d_x

    ldrh    w4, [x21, #2]
    fmov    s1, w4
    fcvt    s1, h1                       // s1 = dmin_x

    ldr     s2, [x22]                    // y_d
    fmul    s3, s0, s2                   // s3 = d_x * y_d
    fmul    s4, s1, s2                   // s4 = dmin_x * y_d

    // ---- decode 12-byte packed scales/mins -> 16 bytes scratch ----
    // utmp[0..3] are u32; after decode:
    //   utmp[0] = u0 & 0x3f3f3f3f                      -> scales[0..3]
    //   utmp[1] = (u2 & 0x0f0f0f0f) | (((u0>>6)&0x03030303)<<4)
    //                                                   -> scales[4..7]
    //   utmp[2] = u1 & 0x3f3f3f3f                      -> mins[0..3]
    //   utmp[3] = ((u2>>4) & 0x0f0f0f0f) | (((u1>>6)&0x03030303)<<4)
    //                                                   -> mins[4..7]
    ldr     w4, [x21, #4]                // u0 = utmp[0] raw
    ldr     w5, [x21, #8]                // u1 = utmp[1] raw
    ldr     w6, [x21, #12]               // u2 = utmp[2] raw

    movz    w7, #0x3f3f, lsl #16
    movk    w7, #0x3f3f                  // w7 = 0x3f3f3f3f (kmask1)
    movz    w8, #0x0f0f, lsl #16
    movk    w8, #0x0f0f                  // w8 = 0x0f0f0f0f (kmask2)
    movz    w9, #0x0303, lsl #16
    movk    w9, #0x0303                  // w9 = 0x03030303 (kmask3)

    // utmp[3] = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4)
    lsr     w10, w6, #4
    and     w10, w10, w8
    lsr     w11, w5, #6
    and     w11, w11, w9
    lsl     w11, w11, #4
    orr     w11, w10, w11                // w11 = utmp[3]

    // utmp[2] = u1 & kmask1
    and     w10, w5, w7                  // w10 = utmp[2]

    // utmp[1] = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4)
    and     w12, w6, w8
    lsr     w13, w4, #6
    and     w13, w13, w9
    lsl     w13, w13, #4
    orr     w12, w12, w13                // w12 = utmp[1]

    // utmp[0] = u0 & kmask1
    and     w14, w4, w7                  // w14 = utmp[0]

    str     w14, [sp, #96]               // scales[0..3]
    str     w12, [sp, #100]              // scales[4..7]
    str     w10, [sp, #104]              // mins[0..3]
    str     w11, [sp, #108]              // mins[4..7]

    // ---- sumi_mins = sum_{j=0..15} bsums[j] * mins[j/2] ----
    // ldp on q regs requires 16-aligned imm; 260 isn't, so build the address.
    add     x4, x22, #260                // x4 = &bsums[0]
    ldp     q16, q17, [x4]               // 32 B = 16 i16

    ldr     d18, [sp, #104]              // 8 mins bytes -> d18 (low 8 B of v18)
    uxtl    v18.8h, v18.8b               // 8 i16 (zero-extended)
    // Duplicate each mins lane: we need m0,m0,m1,m1,...,m7,m7
    zip1    v19.8h, v18.8h, v18.8h       // m0,m0,m1,m1,m2,m2,m3,m3
    zip2    v20.8h, v18.8h, v18.8h       // m4,m4,m5,m5,m6,m6,m7,m7

    smull   v21.4s, v16.4h, v19.4h
    smlal2  v21.4s, v16.8h, v19.8h
    smlal   v21.4s, v17.4h, v20.4h
    smlal2  v21.4s, v17.8h, v20.8h
    addv    s21, v21.4s
    fmov    w23, s21                     // w23 = sumi_mins (i32)

    // ---- accumulate 8 sub-blocks into w24 (acc_total) ----
    mov     w24, wzr

    add     x25, x21, #16                // qh base (32 B)
    add     x26, x21, #48                // qs base (128 B in 4 chunks of 32)
    add     x4,  x22, #4                 // y_qs base

    // Load qh once into v22 (32 B = q22 + q23 by alignment is two q regs)
    ldp     q22, q23, [x25]              // v22 lo16, v23 hi16 of qh
    // For uniform code we keep qh in v22:v23 as 32 B in i8 form.

    // We'll use w7 as the bit mask (1 << S) and w8 as the sub-block index S.
    mov     w8, #0
.Lq5k_subblock:
    // Determine which 32-byte qs chunk we're in and whether we're lo or hi nibble.
    // S=0,1 -> qs[0..31];  S=2,3 -> qs[32..63];  S=4,5 -> qs[64..95];  S=6,7 -> qs[96..127]
    // S even -> low nibble.  S odd -> high nibble.
    lsr     w9,  w8, #1
    lsl     w9,  w9, #5                  // chunk offset = (S/2)*32
    add     x10, x26, x9                 // -> qs chunk base
    ldp     q16, q17, [x10]              // 32 B raw nibble bytes

    tst     w8, #1
    b.ne    .Lq5k_hi_nib
    // low nibble: x_u8 = raw & 0x0F
    movi    v18.16b, #0x0F
    and     v16.16b, v16.16b, v18.16b
    and     v17.16b, v17.16b, v18.16b
    b       .Lq5k_nib_done
.Lq5k_hi_nib:
    ushr    v16.16b, v16.16b, #4
    ushr    v17.16b, v17.16b, #4
.Lq5k_nib_done:

    // High-bit injection: where qh & (1<<S) is set, add 16.
    mov     w7, #1
    lsl     w7, w7, w8                   // mask byte = 1 << S
    dup     v18.16b, w7                  // broadcast
    and     v19.16b, v22.16b, v18.16b
    cmeq    v19.16b, v19.16b, v18.16b    // 0xFF where bit set
    movi    v20.16b, #16
    and     v19.16b, v19.16b, v20.16b    // 16 where set, 0 elsewhere
    add     v16.16b, v16.16b, v19.16b    // first 16 bytes of x_u8

    and     v19.16b, v23.16b, v18.16b
    cmeq    v19.16b, v19.16b, v18.16b
    and     v19.16b, v19.16b, v20.16b
    add     v17.16b, v17.16b, v19.16b    // next 16 bytes of x_u8

    // Load 32 i8 of y for sub-block S
    lsl     w12, w8, #5                  // S * 32
    add     x11, x4, x12
    ldp     q20, q21, [x11]              // y bytes

    // smlal of u8 by i8:  Arm gives us umull/umlal (u8*u8) and smull/smlal (i8*i8).
    // We need u8 * i8. Trick: cast x_u8 (range 0..31) to i8 — the high bit is
    // never set so the i8 reinterpretation has the same value.
    // -> smull.
    smull   v18.8h, v16.8b,  v20.8b
    smlal2  v18.8h, v16.16b, v20.16b
    smlal   v18.8h, v17.8b,  v21.8b
    smlal2  v18.8h, v17.16b, v21.16b
    // v18 now holds 8 i16 partial sums for 32 i8*i8 products.  Sum to one i32.
    saddlv  s18, v18.8h
    fmov    w11, s18                     // subdot (i32)

    // scale = scales[S]  (u8 at [sp + 96 + S])
    add     x12, sp, #96
    ldrb    w13, [x12, x8]
    mul     w11, w11, w13
    add     w24, w24, w11                // acc_total += scale * subdot

    add     w8, w8, #1
    cmp     w8, #8
    b.lt    .Lq5k_subblock

    // ---- sumf += d_x * y_d * acc_total - dmin_x * y_d * sumi_mins ----
    scvtf   s5, w24
    fmul    s5, s5, s3                   // * (d_x * y_d)
    scvtf   s6, w23
    fmul    s6, s6, s4                   // * (dmin_x * y_d)
    fsub    s5, s5, s6
    fadd    s15, s15, s5

    // advance and loop
    add     x21, x21, #176
    add     x22, x22, #292
    subs    w19, w19, #1
    b.ne    .Lq5k_block

.Lq5k_done:
    str     s15, [x20]

    // -------- epilogue --------
    ldp     d8,  d9,  [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #128
    ret
