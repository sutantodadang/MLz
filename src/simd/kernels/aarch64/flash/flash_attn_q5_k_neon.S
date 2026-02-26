// ===========================================================================
//       FLASH ATTENTION 2 - Q5_K AArch64 NEON KERNEL
// ===========================================================================
// Q5_K: 5-bit K-quant. 256 values per super-block.
// Layout: fp16 d(2B) + fp16 dmin(2B) + scales(12B) + qh(32B) + qs(128B) = 176B
//   [0..1]:     fp16 d
//   [2..3]:     fp16 dmin
//   [4..15]:    12 bytes 6-bit packed scales
//   [16..47]:   32 bytes high-bit mask (qh, 1 bit per value)
//   [48..175]:  128 bytes quants (4-bit low nibbles, qs)
// Dequant: val = d * sc * (q4 | (hbit<<4)) - dmin * m
// License: MIT
// ===========================================================================

#define KERNEL_NAME     simd_flash_attn_q5_k_neon
#define K_BLOCK_BYTES   Q5_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q5_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q5_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q5_K_BLOCK_VALUES

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]              // d
    fcvt    s10, h10
    ldr     h11, [\k_ptr, #2]          // dmin
    fcvt    s11, h11

    mov     x14, #0

.Lq5k_kelem_\@:
    cmp     x14, #256
    b.ge    .Lq5k_kdone_\@

    lsr     x15, x14, #5              // sub-block j

    // Get 4-bit low quant from qs[i/2]
    add     x9, \k_ptr, #48
    lsr     x12, x14, #1
    ldrb    w12, [x9, x12]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    // Get high bit from qh: qh[i%32] >> (i/32)
    and     x13, x14, #31
    add     x9, \k_ptr, #16
    ldrb    w13, [x9, x13]
    lsr     w13, w13, w15
    and     w13, w13, #1
    orr     w12, w12, w13, lsl #4      // 5-bit value (0..31)

    // Get scale pair
    add     x9, \k_ptr, #4
    ldrb    w16, [x9, x15]
    and     w17, w16, #0x3F           // sc
    add     x13, x15, #8; cmp x13, #12; b.lt 3f; sub x13, x13, #12
3:  ldrb    w16, [x9, x13]
    and     w16, w16, #0x3F           // m

    mul     w12, w12, w17             // sc * q5
    scvtf   s16, w12; fmul s16, s16, s10
    scvtf   s17, w16; fmul s17, s17, s11
    fsub    s16, s16, s17

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq5k_kelem_\@
.Lq5k_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq5k_vb_\@:
    cbz     x11, .Lq5k_ve_\@

    ldr     h10, [x9]; fcvt s10, h10
    ldr     h11, [x9, #2]; fcvt s11, h11

    mov     x14, #0
.Lq5k_velem_\@:
    cmp     x14, #256
    b.ge    .Lq5k_vdone_\@

    lsr     x15, x14, #5

    add     x12, x9, #48
    lsr     x13, x14, #1; ldrb w13, [x12, x13]
    tst     x14, #1; b.eq 5f; lsr w13, w13, #4; b 6f
5:  and     w13, w13, #0xF
6:
    and     x16, x14, #31; add x12, x9, #16; ldrb w16, [x12, x16]
    lsr     w16, w16, w15; and w16, w16, #1
    orr     w13, w13, w16, lsl #4

    add     x12, x9, #4; ldrb w16, [x12, x15]; and w17, w16, #0x3F
    add     x16, x15, #8; cmp x16, #12; b.lt 7f; sub x16, x16, #12
7:  ldrb    w16, [x12, x16]; and w16, w16, #0x3F

    mul     w13, w13, w17; scvtf s16, w13; fmul s16, s16, s10
    scvtf   s17, w16; fmul s17, s17, s11; fsub s16, s16, s17
    fmul    s16, s16, s0
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq5k_velem_\@
.Lq5k_vdone_\@:
    add     x9, x9, #Q5_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq5k_vb_\@
.Lq5k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
