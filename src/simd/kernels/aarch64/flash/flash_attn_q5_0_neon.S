// ===========================================================================
//       FLASH ATTENTION 2 - Q5_0 AArch64 NEON KERNEL
// ===========================================================================
// Q5_0: fp16 scale(2B) + 4B high-bits + 16B nibbles = 22 bytes / 32 vals
// Dequant: val = d * ((nibble | (hbit<<4)) - 16)  (signed 5-bit, -16..15)
// License: MIT
// ===========================================================================

#define KERNEL_NAME     simd_flash_attn_q5_0_neon
#define K_BLOCK_BYTES   Q5_0_BLOCK_BYTES
#define V_BLOCK_BYTES   Q5_0_BLOCK_BYTES
#define K_BLOCK_VALUES  Q5_0_BLOCK_VALUES
#define V_BLOCK_VALUES  Q5_0_BLOCK_VALUES

// ---------------------------------------------------------------------------
// Q5_0 block layout:
//   [0..1]   fp16 d (scale)
//   [2..5]   4 bytes high-bit mask (1 bit per value, 32 bits)
//   [6..21]  16 bytes packed nibbles (2 nibbles per byte, 32 values)
//
// Scalar dequant for correctness. Q5_0 bit extraction is complex.
// ---------------------------------------------------------------------------

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]              // d (scale) as fp16
    fcvt    s10, h10                    // convert to fp32

    // Load high-bit mask as 32-bit word
    ldr     w13, [\k_ptr, #2]          // 32 high bits packed

    add     x9, \k_ptr, #6             // nibble data pointer
    mov     x14, #0                    // element index
    movi    v6.4s, #0                   // accumulator

.Lq50_kelem_\@:
    cmp     x14, #32
    b.ge    .Lq50_kdone_\@

    // Get nibble: byte = nibbles[i/2], nibble = (i&1) ? byte>>4 : byte&0xF
    lsr     x15, x14, #1
    ldrb    w12, [x9, x15]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    // Get high bit: (hmask >> i) & 1
    lsr     w15, w13, w14
    and     w15, w15, #1
    orr     w12, w12, w15, lsl #4      // 5-bit value (0..31)
    sub     w12, w12, #16              // signed (-16..15)

    scvtf   s16, w12
    fmul    s16, s16, s10              // val = d * signed_5bit

    ldr     s17, [\q_ptr, x14, lsl #2] // Q[i]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Lq50_kelem_\@
.Lq50_kdone_\@:

    // No pointer fixup needed - skeleton handles advancement
    // But we must NOT have advanced k_ptr/q_ptr ourselves
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    dup     v24.4s, v0.s[0]            // broadcast prob
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq50_vb_\@:
    cbz     x11, .Lq50_ve_\@

    ldr     h10, [x9]
    fcvt    s10, h10                    // d
    ldr     w13, [x9, #2]              // high-bit mask
    add     x12, x9, #6               // nibble data

    mov     x14, #0
.Lq50_velem_\@:
    cmp     x14, #32
    b.ge    .Lq50_vdone_\@

    lsr     x15, x14, #1
    ldrb    w15, [x12, x15]
    tst     x14, #1
    b.eq    3f
    lsr     w15, w15, #4
    b       4f
3:  and     w15, w15, #0xF
4:
    lsr     w16, w13, w14
    and     w16, w16, #1
    orr     w15, w15, w16, lsl #4
    sub     w15, w15, #16

    scvtf   s16, w15
    fmul    s16, s16, s10              // val = d * q
    // O[i] += prob * val
    ldr     s17, [x10]
    fmul    s18, s16, s0               // prob * val (s0 = prob)
    fadd    s17, s17, s18
    str     s17, [x10], #4

    add     x14, x14, #1
    b       .Lq50_velem_\@
.Lq50_vdone_\@:

    add     x9, x9, #Q5_0_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq50_vb_\@
.Lq50_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
