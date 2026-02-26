// ===========================================================================
//       FLASH ATTENTION 2 - Q6_K AArch64 NEON KERNEL
// ===========================================================================
// Q6_K: 6-bit K-quant. 256 values per super-block.
// Layout: ql(128B) + qh(64B) + scales(16B) + fp16 d(2B) = 210 bytes
//   [0..127]:   128 bytes ql (low 4 bits per value, packed nibbles)
//   [128..191]: 64 bytes qh (high 2 bits per value, packed 2-bit)
//   [192..207]: 16 bytes scales (int8 per sub-block of 16 values)
//   [208..209]: fp16 d (super-block scale)
// Dequant: val = d * scale * ((ql_nib | (qh_2bit << 4)) - 32)
// License: MIT
// ===========================================================================

#define KERNEL_NAME     simd_flash_attn_q6_k_neon
#define K_BLOCK_BYTES   Q6_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q6_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q6_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q6_K_BLOCK_VALUES

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr, #208]        // d
    fcvt    s10, h10

    mov     x14, #0

.Lq6k_kelem_\@:
    cmp     x14, #256
    b.ge    .Lq6k_kdone_\@

    // Scale index: 16 groups of 16 values -> scale[i/16]
    lsr     x15, x14, #4

    // Get low 4-bit nibble from ql[i/2]
    lsr     x12, x14, #1
    ldrb    w12, [\k_ptr, x12]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    // Get high 2-bit from qh[i/4]
    add     x9, \k_ptr, #128
    lsr     x13, x14, #2
    ldrb    w13, [x9, x13]
    and     x16, x14, #3
    lsl     x16, x16, #1              // shift = (i%4)*2
    lsr     w13, w13, w16
    and     w13, w13, #3              // 2-bit high

    // 6-bit value: ql | (qh << 4), then subtract 32 for signed
    orr     w12, w12, w13, lsl #4      // 0..63
    sub     w12, w12, #32              // -32..31

    // Scale (int8)
    add     x9, \k_ptr, #192
    ldrsb   w13, [x9, x15]

    mul     w12, w12, w13              // scale * q6_signed
    scvtf   s16, w12
    fmul    s16, s16, s10              // * d

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq6k_kelem_\@
.Lq6k_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq6k_vb_\@:
    cbz     x11, .Lq6k_ve_\@

    ldr     h10, [x9, #208]; fcvt s10, h10

    mov     x14, #0
.Lq6k_velem_\@:
    cmp     x14, #256
    b.ge    .Lq6k_vdone_\@

    lsr     x15, x14, #4

    lsr     x12, x14, #1; ldrb w12, [x9, x12]
    tst     x14, #1; b.eq 3f; lsr w12, w12, #4; b 4f
3:  and     w12, w12, #0xF
4:
    add     x13, x9, #128
    lsr     x16, x14, #2; ldrb w16, [x13, x16]
    and     x17, x14, #3; lsl x17, x17, #1
    lsr     w16, w16, w17; and w16, w16, #3

    orr     w12, w12, w16, lsl #4; sub w12, w12, #32

    add     x13, x9, #192; ldrsb w13, [x13, x15]
    mul     w12, w12, w13
    scvtf   s16, w12; fmul s16, s16, s10
    fmul    s16, s16, s0
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq6k_velem_\@
.Lq6k_vdone_\@:
    add     x9, x9, #Q6_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq6k_vb_\@
.Lq6k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
