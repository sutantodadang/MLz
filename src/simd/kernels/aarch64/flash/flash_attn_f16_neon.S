// ===========================================================================
//       FLASH ATTENTION 2 - F16 AArch64 NEON KERNEL
// ===========================================================================
// F16: 32 fp16 values = 64 bytes per block of 32 values
// Each value is a native IEEE 754 half-precision float.
// Dequant: fcvtl (fp16 -> fp32)
// License: MIT
// ===========================================================================

#define KERNEL_NAME     simd_flash_attn_f16_neon
#define K_BLOCK_BYTES   F16_BLOCK_BYTES
#define V_BLOCK_BYTES   F16_BLOCK_BYTES
#define K_BLOCK_VALUES  F16_BLOCK_VALUES
#define V_BLOCK_VALUES  F16_BLOCK_VALUES

// ---------------------------------------------------------------------------
// dequant_k_dot_block: dot(Q_f32[32], dequant(K_f16[32]))
//   \k_ptr (x3): pointer to 64 bytes of fp16 K values
//   \q_ptr (x2): pointer to 32 fp32 Q values
//   \s_acc (s8): scalar float accumulator (fmadd into)
// Processes 32 values = 4 groups of 8 fp16
// ---------------------------------------------------------------------------
.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    // Group 0: 8 fp16 values -> 2x4 fp32
    ld1     {v16.8h}, [\k_ptr], #16
    fcvtl   v18.4s, v16.4h              // low 4 fp16 -> fp32
    fcvtl2  v19.4s, v16.8h              // high 4 fp16 -> fp32
    ld1     {v4.4s}, [\q_ptr], #16
    ld1     {v5.4s}, [\q_ptr], #16
    fmla    v18.4s, v18.4s, v4.4s       // reuse v18 as temp - wrong, need accum
    // Actually: accumulate into separate vectors
    movi    v6.4s, #0                   // accum0
    movi    v7.4s, #0                   // accum1
    // Redo properly:
    sub     \q_ptr, \q_ptr, #32         // rewind
    sub     \k_ptr, \k_ptr, #16         // rewind

    // Process all 32 values with 4 groups of 8
    ld1     {v16.8h}, [\k_ptr], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    ld1     {v4.4s}, [\q_ptr], #16
    fmul    v6.4s, v18.4s, v4.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmul    v7.4s, v19.4s, v4.4s

    ld1     {v16.8h}, [\k_ptr], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v18.4s, v4.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v19.4s, v4.4s

    ld1     {v16.8h}, [\k_ptr], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v18.4s, v4.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v19.4s, v4.4s

    ld1     {v16.8h}, [\k_ptr], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v18.4s, v4.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v19.4s, v4.4s

    // Horizontal sum -> s_acc
    fadd    v6.4s, v6.4s, v7.4s
    faddp   v6.4s, v6.4s, v6.4s
    faddp   s16, v6.2s
    fadd    \s_acc, \s_acc, s16

    // Rewind pointers (skeleton advances by K_BLOCK_BYTES/K_BLOCK_VALUES*4)
    sub     \k_ptr, \k_ptr, #F16_BLOCK_BYTES
    sub     \q_ptr, \q_ptr, #(F16_BLOCK_VALUES * 4)
.endm

// ---------------------------------------------------------------------------
// dequant_v_accum_block: O[i] += prob * dequant(V_f16[i]) for all V blocks
//   \v_ptr (x3): V row base pointer
//   \prob  (s0): probability scalar
//   \o_ptr (x4): O accumulator base pointer
//   \nblk  (x5): number of V blocks
// ---------------------------------------------------------------------------
.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    dup     v24.4s, v0.s[0]            // broadcast prob
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk
.Lf16_vb_\@:
    cbz     x11, .Lf16_ve_\@

    // 32 fp16 values = 4 groups of 8
    ld1     {v16.8h}, [x9], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    fmul    v18.4s, v18.4s, v24.4s
    fmul    v19.4s, v19.4s, v24.4s
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v18.4s
    st1     {v5.4s}, [x10], #16
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v19.4s
    st1     {v5.4s}, [x10], #16

    ld1     {v16.8h}, [x9], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    fmul    v18.4s, v18.4s, v24.4s
    fmul    v19.4s, v19.4s, v24.4s
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v18.4s
    st1     {v5.4s}, [x10], #16
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v19.4s
    st1     {v5.4s}, [x10], #16

    ld1     {v16.8h}, [x9], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    fmul    v18.4s, v18.4s, v24.4s
    fmul    v19.4s, v19.4s, v24.4s
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v18.4s
    st1     {v5.4s}, [x10], #16
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v19.4s
    st1     {v5.4s}, [x10], #16

    ld1     {v16.8h}, [x9], #16
    fcvtl   v18.4s, v16.4h
    fcvtl2  v19.4s, v16.8h
    fmul    v18.4s, v18.4s, v24.4s
    fmul    v19.4s, v19.4s, v24.4s
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v18.4s
    st1     {v5.4s}, [x10], #16
    ld1     {v5.4s}, [x10]
    fadd    v5.4s, v5.4s, v19.4s
    st1     {v5.4s}, [x10], #16

    sub     x11, x11, #1
    b       .Lf16_vb_\@
.Lf16_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
