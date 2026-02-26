// ===========================================================================
//       FLASH ATTENTION 2 - Q5_1 AArch64 NEON KERNEL
// ===========================================================================
// Q5_1: fp16 d(2B) + fp16 m(2B) + 4B high-bits + 16B nibbles = 24B / 32 vals
// Dequant: val = d * (nibble | (hbit<<4)) + m  (unsigned 5-bit, 0..31)
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_q5_1_neon
#define K_BLOCK_BYTES   Q5_1_BLOCK_BYTES
#define V_BLOCK_BYTES   Q5_1_BLOCK_BYTES
#define K_BLOCK_VALUES  Q5_1_BLOCK_VALUES
#define V_BLOCK_VALUES  Q5_1_BLOCK_VALUES

// Q5_1 block layout:
//   [0..1]   fp16 d (scale)
//   [2..3]   fp16 m (min)
//   [4..7]   4 bytes high-bit mask (1 bit per value)
//   [8..23]  16 bytes packed nibbles

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]
    fcvt    s10, h10                    // d
    ldr     h11, [\k_ptr, #2]
    fcvt    s11, h11                    // m
    ldr     w13, [\k_ptr, #4]          // high-bit mask
    add     x9, \k_ptr, #8            // nibble data

    mov     x14, #0

.Lq51_kelem_\@:
    cmp     x14, #32
    b.ge    .Lq51_kdone_\@

    lsr     x15, x14, #1
    ldrb    w12, [x9, x15]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    lsr     w15, w13, w14
    and     w15, w15, #1
    orr     w12, w12, w15, lsl #4      // 5-bit unsigned (0..31)

    ucvtf   s16, w12
    fmadd   s16, s16, s10, s11         // val = d * q + m

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq51_kelem_\@
.Lq51_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq51_vb_\@:
    cbz     x11, .Lq51_ve_\@

    ldr     h10, [x9]; fcvt s10, h10
    ldr     h11, [x9, #2]; fcvt s11, h11
    ldr     w13, [x9, #4]
    add     x12, x9, #8

    mov     x14, #0
.Lq51_velem_\@:
    cmp     x14, #32
    b.ge    .Lq51_vdone_\@

    lsr     x15, x14, #1
    ldrb    w15, [x12, x15]
    tst     x14, #1
    b.eq    3f; lsr w15, w15, #4; b 4f
3:  and     w15, w15, #0xF
4:
    lsr     w16, w13, w14; and w16, w16, #1
    orr     w15, w15, w16, lsl #4
    ucvtf   s16, w15
    fmadd   s16, s16, s10, s11         // val = d*q + m
    fmul    s16, s16, s0               // * prob
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq51_velem_\@
.Lq51_vdone_\@:
    add     x9, x9, #Q5_1_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq51_vb_\@
.Lq51_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
