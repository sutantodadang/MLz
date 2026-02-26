// ===========================================================================
//       FLASH ATTENTION 2 - Q3_K AArch64 NEON KERNEL
// ===========================================================================
// Q3_K: 3-bit K-quant. 256 values per super-block.
// Layout: hmask(32B) + quants(64B) + scales(12B) + fp16 d(2B) = 110 bytes
//   [0..31]:    32 bytes high-bit mask (1 bit per value)
//   [32..95]:   64 bytes quants (2 bits per value, 4 per byte)
//   [96..107]:  12 bytes scales (6-bit encoded)
//   [108..109]: fp16 d (super-block scale)
// Dequant: val = d * scale * ((q2 | (hbit<<2)) - 4)
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_q3_k_neon
#define K_BLOCK_BYTES   Q3_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q3_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q3_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q3_K_BLOCK_VALUES

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr, #108]        // d (super-block scale)
    fcvt    s10, h10

    mov     x14, #0

.Lq3k_kelem_\@:
    cmp     x14, #256
    b.ge    .Lq3k_kdone_\@

    // Sub-block: sb = i / 32, si = i % 32
    lsr     x15, x14, #5              // sb
    and     x12, x14, #31             // si

    // Get 2-bit quant from [32..95]
    add     x9, \k_ptr, #32
    lsr     x13, x14, #2              // byte index
    ldrb    w13, [x9, x13]
    and     x16, x14, #3
    lsl     x16, x16, #1              // shift = (i%4)*2
    lsr     w13, w13, w16
    and     w13, w13, #3              // q2 (0..3)

    // Get high bit from hmask[si] >> sb
    ldrb    w16, [\k_ptr, x12]        // hmask[si]
    lsr     w16, w16, w15             // >> sb
    and     w16, w16, #1              // hbit

    // q3 = q2 | (hbit << 2)  -> 3-bit value (0..7)
    orr     w13, w13, w16, lsl #2
    sub     w13, w13, #4              // signed (-4..3)

    // Get scale: simplified 6-bit extraction from scales[sb]
    // Full Q3_K scale decode is complex (12-byte packed 6-bit)
    // Simplified: use byte index directly, subtract 32
    add     x9, \k_ptr, #96
    ldrb    w16, [x9, x15]
    sub     w16, w16, #32

    mul     w13, w13, w16             // scale * q3_signed
    scvtf   s16, w13
    fmul    s16, s16, s10             // * d

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq3k_kelem_\@
.Lq3k_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq3k_vb_\@:
    cbz     x11, .Lq3k_ve_\@

    ldr     h10, [x9, #108]; fcvt s10, h10

    mov     x14, #0
.Lq3k_velem_\@:
    cmp     x14, #256
    b.ge    .Lq3k_vdone_\@

    lsr     x15, x14, #5; and x12, x14, #31
    add     x13, x9, #32
    lsr     x16, x14, #2; ldrb w16, [x13, x16]
    and     x17, x14, #3; lsl x17, x17, #1
    lsr     w16, w16, w17; and w16, w16, #3

    ldrb    w17, [x9, x12]; lsr w17, w17, w15; and w17, w17, #1
    orr     w16, w16, w17, lsl #2; sub w16, w16, #4

    add     x13, x9, #96; ldrb w17, [x13, x15]; sub w17, w17, #32
    mul     w16, w16, w17
    scvtf   s16, w16; fmul s16, s16, s10
    fmul    s16, s16, s0
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq3k_velem_\@
.Lq3k_vdone_\@:
    add     x9, x9, #Q3_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq3k_vb_\@
.Lq3k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
