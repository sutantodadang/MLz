// ===========================================================================
//   vec_dot_q2_k_q8_k_neon.S — Q2_K × Q8_K Dot Product, AArch64 NEON
// ===========================================================================
// void simd_vec_dot_q2_k_q8_k_neon(int n, float* result,
//                                   const void* vx, const void* vy);
// AAPCS64: w0=n, x1=result, x2=vx(Q2_K), x3=vy(Q8_K)
//
// Q2_K block (84B/256 values):
//   scales[16] + quants[64] + fp16 d + fp16 dmin
// Q8_K block (292B/256 values):
//   fp32 d + int8 qs[256] + int16 bsums[16]
//
// For this scalar kernel, we dequantize Q2_K and compute dot with Q8_K
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_vec_dot_q2_k_q8_k_neon)

simd_vec_dot_q2_k_q8_k_neon:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8, d9, [sp, #80]

    mov     w19, w0                     // n
    mov     x20, x1                     // result ptr
    mov     x21, x2                     // vx (Q2_K)
    mov     x22, x3                     // vy (Q8_K)

    // nb = n / 256
    lsr     w19, w19, #8
    cbz     w19, .Lq2k_zero

    fmov    s15, wzr                    // accumulator

.Lq2k_block:
    // Q2_K layout: scales[16B] + quants[64B] + d(fp16) + dmin(fp16)
    // offsets: scales=0, qs=16, d=80, dmin=82
    ldrh    w4, [x21, #80]              // d as fp16 bits
    fmov    s0, w4
    fcvt    s0, h0                      // d as fp32

    ldrh    w4, [x21, #82]              // dmin as fp16 bits
    fmov    s1, w4
    fcvt    s1, h1                      // dmin as fp32

    // Q8_K: d at offset 0 (fp32)
    ldr     s2, [x22]                   // Q8_K d
    fmul    s3, s0, s2                  // d * q8k_d
    fmul    s4, s1, s2                  // dmin * q8k_d
    fneg    s4, s4                      // -dmin * q8k_d

    // Process 16 sub-blocks of 16 values each
    add     x5, x21, #0                // scales ptr
    add     x6, x21, #16               // quants ptr
    add     x7, x22, #4                // Q8_K quants
    add     x8, x22, #260              // Q8_K bsums (int16)

    fmov    s5, wzr                     // sumf
    fmov    s6, wzr                     // sums (for bsums * m)

    mov     w9, #0                      // sub-block counter (0..15)
.Lq2k_sb:
    cmp     w9, #16
    b.ge    .Lq2k_sb_done

    // Load scale byte
    ldrb    w10, [x5, x9]
    and     w11, w10, #0x0F             // sc = scales[i] & 0x0F
    lsr     w12, w10, #4                // m  = scales[i] >> 4

    // bsum contribution: m * bsums[i] -> sums
    ldrsh   w13, [x8, x9, lsl #1]      // bsums[i] as int16
    mul     w14, w12, w13               // m * bsums[i]
    scvtf   s7, w14
    fadd    s6, s6, s7

    // Process 16 Q2 values in this sub-block
    // Each quant byte holds 4 x 2-bit values
    mov     w14, #0                     // value counter
    mov     w23, #0                     // dot accumulator (integer)
    // 16 values = 4 quant bytes
    lsl     w24, w9, #2            // byte offset = sb * 4
.Lq2k_val:
    cmp     w14, #16
    b.ge    .Lq2k_val_done

    // Compute byte and shift for this 2-bit value
    lsr     w25, w14, #2                // byte index within sub-block group
    and     w26, w14, #3                // bit pair index (0..3)
    lsl     w26, w26, #1                // shift amount (0,2,4,6)
    add     w25, w25, w24               // actual byte offset in quants
    ldrb    w4, [x6, x25]
    lsr     w4, w4, w26
    and     w4, w4, #3                  // 2-bit quant value

    // Load Q8_K quant
    // Q8_K quant index = sb * 16 + val
    lsl     w25, w9, #4
    add     w25, w25, w14
    ldrsb   w26, [x7, x25]

    // dot += sc * q2 * q8
    mul     w4, w4, w26
    madd    w23, w11, w4, w23           // Not right - should be sc*q2*q8...
                                        // Actually: dot += q2 * q8, then scale at end

    add     w14, w14, #1
    b       .Lq2k_val
.Lq2k_val_done:
    // sumf += sc * dot (integer dot = sum of q2*q8 for this sub-block)
    // Well actually we accumulated sc*q2*q8 already. Let's just add.
    scvtf   s7, w23
    fadd    s5, s5, s7

    add     w9, w9, #1
    b       .Lq2k_sb
.Lq2k_sb_done:

    // result += d*q8d * sumf - dmin*q8d * sums_contribution
    fmadd   s15, s3, s5, s15           // + d * q8_d * sumf
    fmadd   s15, s4, s6, s15           // - dmin * q8_d * sums

    // Advance to next super-block
    add     x21, x21, #Q2_K_BLOCK_BYTES    // 84
    add     x22, x22, #Q8_K_BLOCK_BYTES    // 292
    subs    w19, w19, #1
    b.ne    .Lq2k_block

    str     s15, [x20]
    b       .Lq2k_done

.Lq2k_zero:
    fmov    s15, wzr
    str     s15, [x20]

.Lq2k_done:
    ldp     d8, d9, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

    END_FUNCTION(simd_vec_dot_q2_k_q8_k_neon)
