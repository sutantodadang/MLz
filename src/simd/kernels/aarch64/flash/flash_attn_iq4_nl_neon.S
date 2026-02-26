// ===========================================================================
//       FLASH ATTENTION 2 - IQ4_NL AArch64 NEON KERNEL
// ===========================================================================
// IQ4_NL: fp16 scale(2B) + 16B nibbles = 18 bytes / 32 vals
// Non-linear 4-bit quantization with lookup table dequant.
// Dequant: val = d * LUT[nibble]
// LUT: {-31,-21,-15,-11,-7,-4,-2,0,2,4,6,9,13,19,28,43}
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_iq4_nl_neon
#define K_BLOCK_BYTES   IQ4_NL_BLOCK_BYTES
#define V_BLOCK_BYTES   IQ4_NL_BLOCK_BYTES
#define K_BLOCK_VALUES  IQ4_NL_BLOCK_VALUES
#define V_BLOCK_VALUES  IQ4_NL_BLOCK_VALUES

// IQ4_NL lookup table (int8 values, 16 entries)
    .section .rodata
    .align 4
.Liq4nl_lut:
    .byte   0xE1    // -31
    .byte   0xEB    // -21
    .byte   0xF1    // -15
    .byte   0xF5    // -11
    .byte   0xF9    //  -7
    .byte   0xFC    //  -4
    .byte   0xFE    //  -2
    .byte   0x00    //   0
    .byte   0x02    //   2
    .byte   0x04    //   4
    .byte   0x06    //   6
    .byte   0x09    //   9
    .byte   0x0D    //  13
    .byte   0x13    //  19
    .byte   0x1C    //  28
    .byte   0x2B    //  43

    .text

.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]
    fcvt    s10, h10                    // d (scale)
    add     x9, \k_ptr, #2            // nibble data

    // Load LUT address
    adrp    x15, .Liq4nl_lut
    add     x15, x15, :lo12:.Liq4nl_lut

    mov     x14, #0

.Liq4_kelem_\@:
    cmp     x14, #32
    b.ge    .Liq4_kdone_\@

    // Get nibble
    lsr     x12, x14, #1
    ldrb    w12, [x9, x12]
    tst     x14, #1
    b.eq    1f
    lsr     w12, w12, #4
    b       2f
1:  and     w12, w12, #0xF
2:
    // LUT lookup: signed byte
    ldrsb   w12, [x15, w12, uxtw]
    scvtf   s16, w12
    fmul    s16, s16, s10              // val = d * LUT[nibble]

    ldr     s17, [\q_ptr, x14, lsl #2]
    fmadd   \s_acc, s16, s17, \s_acc

    add     x14, x14, #1
    b       .Liq4_kelem_\@
.Liq4_kdone_\@:
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

    adrp    x15, .Liq4nl_lut
    add     x15, x15, :lo12:.Liq4nl_lut

.Liq4_vb_\@:
    cbz     x11, .Liq4_ve_\@

    ldr     h10, [x9]; fcvt s10, h10
    add     x12, x9, #2

    mov     x14, #0
.Liq4_velem_\@:
    cmp     x14, #32
    b.ge    .Liq4_vdone_\@

    lsr     x13, x14, #1; ldrb w13, [x12, x13]
    tst     x14, #1; b.eq 3f; lsr w13, w13, #4; b 4f
3:  and     w13, w13, #0xF
4:
    ldrsb   w13, [x15, w13, uxtw]
    scvtf   s16, w13; fmul s16, s16, s10
    fmul    s16, s16, s0               // * prob
    ldr     s17, [x10]; fadd s17, s17, s16; str s17, [x10], #4

    add     x14, x14, #1
    b       .Liq4_velem_\@
.Liq4_vdone_\@:
    add     x9, x9, #IQ4_NL_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Liq4_vb_\@
.Liq4_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
