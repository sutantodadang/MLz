// ===========================================================================
//       FLASH ATTENTION 2 - Q8_K AArch64 NEON KERNEL
// ===========================================================================
// Q8_K: 8-bit K-quant. 256 values per super-block.
// Layout: fp32 d(4B) + int8 quants(256B) + int16 bsums(32B) = 292 bytes
//   [0..3]:     fp32 d (super-block scale, already f32)
//   [4..259]:   256 int8 quant values
//   [260..291]: 16 int16 block sums (unused in FA)
// Dequant: val = d * (int8)q  — simplest K-quant type
// License: MIT
// ===========================================================================

#include "neon_common.h"

#define KERNEL_NAME     simd_flash_attn_q8_k_neon
#define K_BLOCK_BYTES   Q8_K_BLOCK_BYTES
#define V_BLOCK_BYTES   Q8_K_BLOCK_BYTES
#define K_BLOCK_VALUES  Q8_K_BLOCK_VALUES
#define V_BLOCK_VALUES  Q8_K_BLOCK_VALUES

// ---------------------------------------------------------------------------
// Q8_K dequant is simple enough for full NEON vectorization.
// Process 16 int8 values per iteration (4 iterations = 64 values ...
// but we have 256 values, so 16 iterations of 16 values).
// ---------------------------------------------------------------------------
.macro dequant_k_dot_block k_ptr, q_ptr, s_acc
    ldr     s10, [\k_ptr]              // d (already fp32!)
    add     x9, \k_ptr, #4            // quant data

    movi    v6.4s, #0                   // accum0
    movi    v7.4s, #0                   // accum1
    mov     x14, #0                    // processed count

.Lq8k_kv_\@:
    cmp     x14, #256
    b.ge    .Lq8k_kdone_\@

    // Load 16 int8 values
    ld1     {v16.16b}, [x9], #16

    // Sign-extend to 4x4 int32 and convert to fp32
    sxtl    v18.8h, v16.8b             // bytes 0-7 -> halfwords
    sxtl2   v19.8h, v16.16b            // bytes 8-15 -> halfwords

    sxtl    v20.4s, v18.4h             // hw 0-3 -> words
    scvtf   v20.4s, v20.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v18.8h             // hw 4-7 -> words
    scvtf   v21.4s, v21.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v21.4s, v4.4s

    sxtl    v20.4s, v19.4h             // hw 8-11 -> words
    scvtf   v20.4s, v20.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v6.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v19.8h             // hw 12-15 -> words
    scvtf   v21.4s, v21.4s
    ld1     {v4.4s}, [\q_ptr], #16
    fmla    v7.4s, v21.4s, v4.4s

    add     x14, x14, #16
    b       .Lq8k_kv_\@
.Lq8k_kdone_\@:

    // Horizontal sum * d -> accumulate into s_acc
    fadd    v6.4s, v6.4s, v7.4s
    faddp   v6.4s, v6.4s, v6.4s
    faddp   s16, v6.2s
    fmul    s16, s16, s10              // * d
    fadd    \s_acc, \s_acc, s16

    // Rewind pointers (skeleton advances by K_BLOCK_BYTES/VALUES*4)
    sub     x9, x9, #256              // not needed, x9 is local
    sub     \q_ptr, \q_ptr, #(Q8_K_BLOCK_VALUES * 4)
.endm

.macro dequant_v_accum_block v_ptr, prob, o_ptr, nblk
    dup     v24.4s, v0.s[0]            // broadcast prob
    mov     x9, \v_ptr
    mov     x10, \o_ptr
    mov     x11, \nblk

.Lq8k_vb_\@:
    cbz     x11, .Lq8k_ve_\@

    ldr     s10, [x9]                   // d (fp32)
    add     x12, x9, #4

    mov     x14, #0
.Lq8k_vv_\@:
    cmp     x14, #256
    b.ge    .Lq8k_vdone_\@

    ld1     {v16.16b}, [x12], #16

    sxtl    v18.8h, v16.8b
    sxtl2   v19.8h, v16.16b

    // Process 4 values at a time, multiply by d and prob
    sxtl    v20.4s, v18.4h; scvtf v20.4s, v20.4s
    fmul    v20.4s, v20.4s, v10.s[0]
    fmul    v20.4s, v20.4s, v24.4s
    ld1     {v5.4s}, [x10]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x10], #16

    sxtl2   v21.4s, v18.8h; scvtf v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v10.s[0]
    fmul    v21.4s, v21.4s, v24.4s
    ld1     {v5.4s}, [x10]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x10], #16

    sxtl    v20.4s, v19.4h; scvtf v20.4s, v20.4s
    fmul    v20.4s, v20.4s, v10.s[0]
    fmul    v20.4s, v20.4s, v24.4s
    ld1     {v5.4s}, [x10]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x10], #16

    sxtl2   v21.4s, v19.8h; scvtf v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v10.s[0]
    fmul    v21.4s, v21.4s, v24.4s
    ld1     {v5.4s}, [x10]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x10], #16

    add     x14, x14, #16
    b       .Lq8k_vv_\@
.Lq8k_vdone_\@:

    add     x9, x9, #Q8_K_BLOCK_BYTES
    sub     x11, x11, #1
    b       .Lq8k_vb_\@
.Lq8k_ve_\@:
.endm

#include "flash_attn_fa2_skeleton_neon.S"
