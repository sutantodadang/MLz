// ===========================================================================
//       FLASH ATTENTION 2 - Q2_K AArch64 NEON KERNEL
// ===========================================================================
// Q2_K: 2-bit K-quant. 256 values per super-block.
// Layout: scales(16B) + quants(64B) + fp16 d(2B) + fp16 dmin(2B) = 84 bytes
//   [0..15]:   16 bytes scale nibble pairs (low=sc, high=m)
//   [16..79]:  64 bytes quants (2 bits per value, 4 values per byte)
//   [80..81]:  fp16 d (super-block scale)
//   [82..83]:  fp16 dmin (super-block min)
// Dequant: val = d * sc * q - dmin * m
//   where sc = scales[sb] & 0xF, m = scales[sb] >> 4, q = 2-bit quant
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_q2_k_neon
#define K_BLOCK_BYTES   Q2_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q2_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q2_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q2_K_BLOCK_VALUES

// Scalar dequant — K-quant layouts have nested sub-block structures
// that are non-trivial to vectorize on first pass.
.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr, #80]         // d
    fcvt    s10, h10
    ldr     h11, [\k_ptr, #82]         // dmin
    fcvt    s11, h11

    mov     x14, #0                    // element index (0..255)

.Lq2k_kelem_\@:
    cmp     x14, #256
    b.ge    .Lq2k_kdone_\@

    // Sub-block index: sb = i / 32
    lsr     x15, x14, #5
    // Get scale nibble pair: scales[sb]
    ldrb    w12, [\k_ptr, x15]
    and     w16, w12, #0xF             // sc (low nibble)
    lsr     w17, w12, #4               // m (high nibble)

    // Get 2-bit quant: quants[(i/4)] >> ((i%4)*2) & 3
    add     x9, \k_ptr, #16
    lsr     x12, x14, #2
    ldrb    w12, [x9, x12]
    and     x15, x14, #3
    lsl     x15, x15, #1               // shift = (i%4)*2
    lsr     w12, w12, w15
    and     w12, w12, #3               // q (0..3)

    // dequant: d * sc * q - dmin * m
    mul     w12, w12, w16              // sc * q
    scvtf   s16, w12
    fmul    s16, s16, s10              // d * sc * q
    scvtf   s17, w17
    fmul    s17, s17, s11              // dmin * m
    fsub    s16, s16, s17              // final value

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq2k_kelem_\@
.Lq2k_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq2k_vb_\@:
    cbz     x11, .Lq2k_ve_\@

    ldr     h10, [x9, #80]; fcvt s10, h10  // d
    ldr     h11, [x9, #82]; fcvt s11, h11  // dmin

    mov     x14, #0
.Lq2k_velem_\@:
    cmp     x14, #256
    b.ge    .Lq2k_vdone_\@

    lsr     x15, x14, #5
    ldrb    w12, [x9, x15]
    and     w16, w12, #0xF
    lsr     w17, w12, #4

    add     x12, x9, #16
    lsr     x13, x14, #2
    ldrb    w13, [x12, x13]
    and     x15, x14, #3; lsl x15, x15, #1
    lsr     w13, w13, w15; and w13, w13, #3

    mul     w13, w13, w16
    scvtf   s16, w13; fmul s16, s16, s10
    scvtf   s17, w17; fmul s17, s17, s11
    fsub    s16, s16, s17
    fmul    s16, s16, s0               // * prob
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq2k_velem_\@
.Lq2k_vdone_\@:
    add     x9, x9, #Q2_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq2k_vb_\@
.Lq2k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
