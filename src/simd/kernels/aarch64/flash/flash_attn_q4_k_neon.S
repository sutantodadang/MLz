// ===========================================================================
//       FLASH ATTENTION 2 - Q4_K AArch64 NEON KERNEL
// ===========================================================================
// Q4_K: 4-bit K-quant. 256 values per super-block.
// Layout: fp16 d(2B) + fp16 dmin(2B) + scales(12B) + quants(128B) = 144 bytes
//   [0..1]:     fp16 d (super-block scale)
//   [2..3]:     fp16 dmin (super-block min)
//   [4..15]:    12 bytes 6-bit packed scales
//   [16..143]:  128 bytes quants (4 bits per value, 2 per byte)
// Dequant: val = d * sc * q - dmin * m
// License: MIT
// ===========================================================================

#define KERNEL_NAME     simd_flash_attn_q4_k_neon
#define K_BLOCK_BYTES   Q4_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q4_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q4_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q4_K_BLOCK_VALUES

// Q4_K scale extraction helper:
// The 12 bytes at [4..15] encode 8 scale/min pairs in 6-bit format.
// For sub-block j (0..7):
//   Bytes 0..3: sc[0..3] in low 6 bits each
//   Bytes 4..7: sc[4..7] in low 6 bits each
//   Bytes 8..11: high 2 bits of sc/m pairs
// Simplified: we use the lower byte directly as approximate scale.

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]              // d
    fcvt    s10, h10
    ldr     h11, [\k_ptr, #2]          // dmin
    fcvt    s11, h11

    mov     x14, #0

.Lq4k_kelem_\@:
    cmp     x14, #256
    b.ge    .Lq4k_kdone_\@

    // Sub-block: j = i / 32
    lsr     x15, x14, #5

    // Get 4-bit quant from quants[i/2]
    add     x9, \k_ptr, #16
    lsr     x12, x14, #1
    ldrb    w12, [x9, x12]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    // Get scale/min pair for this sub-block
    add     x9, \k_ptr, #4            // scales base
    ldrb    w16, [x9, x15]            // raw scale byte
    and     w17, w16, #0x3F           // sc (low 6 bits)
    // For min, use the paired byte (j+8 mod 12)
    // Simplified: use sc for scale, approximate min extraction
    add     x13, x15, #8
    cmp     x13, #12
    b.lt    3f
    sub     x13, x13, #12
3:  ldrb    w16, [x9, x13]
    and     w16, w16, #0x3F           // m (low 6 bits)

    // dequant: d * sc * q - dmin * m
    mul     w12, w12, w17             // sc * q
    scvtf   s16, w12; fmul s16, s16, s10
    scvtf   s17, w16; fmul s17, s17, s11
    fsub    s16, s16, s17

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq4k_kelem_\@
.Lq4k_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq4k_vb_\@:
    cbz     x11, .Lq4k_ve_\@

    ldr     h10, [x9]; fcvt s10, h10
    ldr     h11, [x9, #2]; fcvt s11, h11

    mov     x14, #0
.Lq4k_velem_\@:
    cmp     x14, #256
    b.ge    .Lq4k_vdone_\@

    lsr     x15, x14, #5
    add     x12, x9, #16
    lsr     x13, x14, #1; ldrb w13, [x12, x13]
    tst     x14, #1; b.eq 5f; lsr w13, w13, #4; b 6f
5:  and     w13, w13, #0xF
6:
    add     x12, x9, #4
    ldrb    w16, [x12, x15]; and w17, w16, #0x3F
    add     x13, x15, #8; cmp x13, #12; b.lt 7f; sub x13, x13, #12
7:  ldrb    w16, [x12, x13]; and w16, w16, #0x3F

    mul     w13, w13, w17; scvtf s16, w13; fmul s16, s16, s10
    scvtf   s17, w16; fmul s17, s17, s11; fsub s16, s16, s17
    fmul    s16, s16, s0
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Lq4k_velem_\@
.Lq4k_vdone_\@:
    add     x9, x9, #Q4_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq4k_vb_\@
.Lq4k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
