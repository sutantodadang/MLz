// ===========================================================================
//       FLASH ATTENTION 2 - Q4_1 AArch64 NEON KERNEL
// ===========================================================================
// Q4_1: fp16 scale(2B) + fp16 min(2B) + 16B nibbles = 20 bytes / 32 vals
// Dequant: val = d * nibble + m  (unsigned nibble 0..15)
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_q4_1_neon
#define K_BLOCK_BYTES   Q4_1_BLOCK_BYTES
#define V_BLOCK_BYTES   Q4_1_BLOCK_BYTES
#define K_BLOCK_VALUES  Q4_1_BLOCK_VALUES
#define V_BLOCK_VALUES  Q4_1_BLOCK_VALUES

// ---------------------------------------------------------------------------
// dequant_k_dot_block: dot(Q_f32[32], dequant(K_q4_1[32]))
// Q4_1 layout: [0] fp16 d, [2] fp16 m, [4..19] 16 packed nibble bytes
// ---------------------------------------------------------------------------
.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     h10, [\k_ptr]              // d (scale)
    fcvt    s10, h10
    ldr     h11, [\k_ptr, #2]          // m (min)
    fcvt    s11, h11
    add     x9, \k_ptr, #4             // nibble data

    movi    v6.4s, #0                   // accum0
    movi    v7.4s, #0                   // accum1
    movi    v22.16b, #0x0F             // nibble mask

    // Load 16 packed bytes = 32 nibbles
    ld1     {v16.16b}, [x9]

    // Extract low nibbles (even indices) and high nibbles (odd indices)
    and     v18.16b, v16.16b, v22.16b  // low nibbles [0,2,4,...,30]
    ushr    v19.16b, v16.16b, #4       // high nibbles [1,3,5,...,31]

    // Interleave: low0,high0,low1,high1,... to get sequential order
    zip1    v20.16b, v18.16b, v19.16b  // bytes 0-15: vals 0..15
    zip2    v21.16b, v18.16b, v19.16b  // bytes 0-15: vals 16..31

    // Process vals 0..3
    uxtl    v23.8h, v20.8b             // u8->u16 for first 8
    uxtl    v2.4s, v23.4h              // u16->u32 for first 4
    ucvtf   v2.4s, v2.4s               // u32->f32
    dup     v3.4s, v10.s[0]
    fmul    v2.4s, v2.4s, v3.4s        // * d
    dup     v3.4s, v11.s[0]
    fadd    v2.4s, v2.4s, v3.4s        // + m
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v2.4s, v4.4s

    // Process vals 4..7
    uxtl2   v2.4s, v23.8h
    ucvtf   v2.4s, v2.4s
    fmul    v2.4s, v2.4s, v3.4s        // reuse v3=dup(m) -- wrong, need d
    dup     v3.4s, v10.s[0]
    fmul    v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]
    fadd    v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v2.4s, v4.4s

    // Process vals 8..11
    uxtl2   v23.8h, v20.16b
    uxtl    v2.4s, v23.4h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v2.4s, v4.4s

    // Process vals 12..15
    uxtl2   v2.4s, v23.8h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v2.4s, v4.4s

    // Process vals 16..19
    uxtl    v23.8h, v21.8b
    uxtl    v2.4s, v23.4h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v2.4s, v4.4s

    // Process vals 20..23
    uxtl2   v2.4s, v23.8h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v2.4s, v4.4s

    // Process vals 24..27
    uxtl2   v23.8h, v21.16b
    uxtl    v2.4s, v23.4h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v2.4s, v4.4s

    // Process vals 28..31
    uxtl2   v2.4s, v23.8h
    ucvtf   v2.4s, v2.4s
    dup     v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup     v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v2.4s, v4.4s

    fadd    v6.4s, v6.4s, v7.4s
    faddp   v6.4s, v6.4s, v6.4s
    faddp   s16, v6.2s
    fadd    \s_acc, \s_acc, s16

    sub     \k_ptr, \k_ptr, #Q4_1_BLOCK_BYTES
    sub     \q_ptr, \q_ptr, #(Q4_1_BLOCK_VALUES * 4)
.endm

// ---------------------------------------------------------------------------
// dequant_v_accum_block
// ---------------------------------------------------------------------------
.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    dup     v24.4s, v0.s[0]
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk
    movi    v22.16b, #0x0F

.Lq41_vb_\@:
    cbz     x11, .Lq41_ve_\@

    ldr     h10, [x9]; fcvt s10, h10
    ldr     h11, [x9, #2]; fcvt s11, h11
    add     x12, x9, #4
    ld1     {v16.16b}, [x12]
    and     v18.16b, v16.16b, v22.16b
    ushr    v19.16b, v16.16b, #4
    zip1    v20.16b, v18.16b, v19.16b
    zip2    v21.16b, v18.16b, v19.16b

    // Macro to dequant 4 vals and accumulate
    // vals 0..3
    uxtl    v23.8h, v20.8b
    uxtl    v2.4s, v23.4h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    // vals 4..7
    uxtl2 v2.4s, v23.8h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    // vals 8..15
    uxtl2 v23.8h, v20.16b
    uxtl v2.4s, v23.4h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    uxtl2 v2.4s, v23.8h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    // vals 16..23
    uxtl v23.8h, v21.8b
    uxtl v2.4s, v23.4h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    uxtl2 v2.4s, v23.8h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    // vals 24..31
    uxtl2 v23.8h, v21.16b
    uxtl v2.4s, v23.4h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16
    uxtl2 v2.4s, v23.8h; ucvtf v2.4s, v2.4s
    dup v3.4s, v10.s[0]; fmul v2.4s, v2.4s, v3.4s
    dup v3.4s, v11.s[0]; fadd v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v24.4s
    ld1 {v5.4s}, [x10]; fadd v5.4s, v5.4s, v2.4s; st1 {v5.4s}, [x10], #16

    add     x9, x9, #Q4_1_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq41_vb_\@
.Lq41_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
