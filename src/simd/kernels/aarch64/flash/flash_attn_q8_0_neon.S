// ===========================================================================
//              FLASH ATTENTION 2 - Q8_0 AArch64 NEON KERNEL
// ===========================================================================
// Q8_0: 2 bytes (fp16 scale) + 32 bytes (int8 values) = 34 bytes/32 vals
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .arch armv8-a+fp16
    .text
    .align 4

    RODATA_SECTION
    .align 4
q8_0_neon_exp_clamp_hi: .float 88.7228
q8_0_neon_exp_clamp_lo: .float -103.278
q8_0_neon_exp_log2e:    .float 1.44269504
q8_0_neon_exp_ln2:      .float 0.693147180
q8_0_neon_exp_c1:       .float 0.166666667
q8_0_neon_exp_c2:       .float 0.041666668
q8_0_neon_exp_c3:       .float 0.008333334
q8_0_neon_one:          .float 1.0
q8_0_neon_half:         .float 0.5

    .text
    .align 4

GLOBAL_SYMBOL(simd_flash_attn_q8_0_neon)
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    mov     x19, x0
    sub     sp, sp, #FRAME_SIZE

    ldr     x21, [x19, #CFG_HEAD_DIM_K]
    ldr     x22, [x19, #CFG_HEAD_DIM_V]
    ldr     x23, [x19, #CFG_N_KV]
    ldr     x3,  [x19, #CFG_N_QUERIES]
    ldr     x4,  [x19, #CFG_N_HEAD_Q]
    ldr     x5,  [x19, #CFG_N_HEAD_KV]
    ldr     x6,  [x19, #CFG_BATCH_SIZE]
    str     x21, [sp, #STK_HEAD_DIM_K]; str x22, [sp, #STK_HEAD_DIM_V]
    str     x23, [sp, #STK_N_KV_L]; str x3, [sp, #STK_N_QUERIES]
    str     x4, [sp, #STK_N_HEAD_Q]; str x5, [sp, #STK_N_HEAD_KV]
    str     x6, [sp, #STK_BATCH_SIZE]; str x19, [sp, #STK_CONFIG]
    ldr     s29, [x19, #CFG_SCALE]; str s29, [sp, #STK_SCALE_VAL]
    ldr     x7, [x19, #CFG_MASK_PTR]; str x7, [sp, #STK_MASK_PTR_L]

    ldr     x7, [x19, #CFG_Q_NB1];   str x7, [sp, #STK_Q_NB1]
    ldr     x7, [x19, #CFG_K_NB1];   str x7, [sp, #STK_K_NB1]
    ldr     x7, [x19, #CFG_V_NB1];   str x7, [sp, #STK_V_NB1]
    ldr     x7, [x19, #CFG_DST_NB1]; str x7, [sp, #STK_DST_NB1]
    ldr     x7, [x19, #CFG_MASK_NB1]; str x7, [sp, #STK_MASK_NB1]
    ldr     x7, [x19, #CFG_Q_NB2];   str x7, [sp, #STK_Q_NB2]
    ldr     x7, [x19, #CFG_K_NB2];   str x7, [sp, #STK_K_NB2]
    ldr     x7, [x19, #CFG_V_NB2];   str x7, [sp, #STK_V_NB2]
    ldr     x7, [x19, #CFG_DST_NB2]; str x7, [sp, #STK_DST_NB2]
    ldr     x7, [x19, #CFG_Q_NB3];   str x7, [sp, #STK_Q_NB3]
    ldr     x7, [x19, #CFG_K_NB3];   str x7, [sp, #STK_K_NB3]
    ldr     x7, [x19, #CFG_V_NB3];   str x7, [sp, #STK_V_NB3]
    ldr     x7, [x19, #CFG_DST_NB3]; str x7, [sp, #STK_DST_NB3]

    lsr     x8, x21, #5; str x8, [sp, #STK_K_NBLOCKS]
    lsr     x8, x22, #5; str x8, [sp, #STK_V_NBLOCKS]

    mul     x7, x3, x4; mul x7, x7, x6
    ldrsw   x8, [x19, #CFG_ITH]; ldrsw x9, [x19, #CFG_NTH]
    add     x10, x7, x9; sub x10, x10, #1; udiv x10, x10, x9
    mul     x11, x8, x10; cmp x11, x7; csel x11, x11, x7, lo
    add     x12, x11, x10; cmp x12, x7; csel x12, x12, x7, lo
    str     x11, [sp, #STK_WORK_START]; str x12, [sp, #STK_WORK_END]
    cmp     x11, x12; b.ge .Lq8_0_epi
    mov     x20, x11

.Lq8_0_wl:
    ldr     x12, [sp, #STK_WORK_END]; cmp x20, x12; b.ge .Lq8_0_epi

    ldr     x3, [sp, #STK_N_QUERIES]; ldr x4, [sp, #STK_N_HEAD_Q]
    mul     x5, x3, x4; udiv x6, x20, x5; msub x7, x6, x5, x20
    udiv    x8, x7, x3; msub x9, x8, x3, x7; str x9, [sp, #STK_IQ_SAVE]
    ldr     x10, [sp, #STK_N_HEAD_KV]; mul x11, x8, x10; udiv x11, x11, x4

    ldr     x24, [x19, #CFG_Q_PTR]
    ldr     x12, [sp, #STK_Q_NB1]; madd x24, x9, x12, x24
    ldr     x12, [sp, #STK_Q_NB2]; madd x24, x8, x12, x24
    ldr     x12, [sp, #STK_Q_NB3]; madd x24, x6, x12, x24
    ldr     x25, [x19, #CFG_K_PTR]
    ldr     x12, [sp, #STK_K_NB2]; madd x25, x11, x12, x25
    ldr     x12, [sp, #STK_K_NB3]; madd x25, x6, x12, x25
    ldr     x26, [x19, #CFG_V_PTR]
    ldr     x12, [sp, #STK_V_NB2]; madd x26, x11, x12, x26
    ldr     x12, [sp, #STK_V_NB3]; madd x26, x6, x12, x26
    ldr     x27, [x19, #CFG_DST_PTR]
    ldr     x12, [sp, #STK_DST_NB1]; madd x27, x9, x12, x27
    ldr     x12, [sp, #STK_DST_NB2]; madd x27, x8, x12, x27
    ldr     x12, [sp, #STK_DST_NB3]; madd x27, x6, x12, x27

    add     x0, sp, #STK_O; movi v0.4s, #0
    mov     x1, x22; lsr x1, x1, #2
.Lq8_0_zo: st1 {v0.4s}, [x0], #16; subs x1, x1, #1; b.gt .Lq8_0_zo

    mov     w0, #0xFF800000; fmov s31, w0; movi d30, #0
    ldr     s29, [sp, #STK_SCALE_VAL]
    mov     x14, #0

.Lq8_0_tl:
    cmp     x14, x23; b.ge .Lq8_0_norm
    add     x15, x14, #FA_TILE_KV_NEON; cmp x15, x23; csel x15, x15, x23, lo
    sub     x16, x15, x14

    mov     x0, #0
.Lq8_0_qk:
    cmp     x0, x16; b.ge .Lq8_0_qk_d

    add     x1, x14, x0; ldr x2, [sp, #STK_K_NB1]; mul x1, x1, x2
    add     x1, x25, x1

    movi    v0.4s, #0; movi v1.4s, #0
    mov     x2, x24; mov x3, x1; ldr x4, [sp, #STK_K_NBLOCKS]

.Lq8_0_kb:
    cbz     x4, .Lq8_0_kd

    // Q8_0: load scale (fp16), then 32 int8 values
    ldr     h8, [x3]; fcvt s8, h8
    add     x9, x3, #2

    // Load 32 int8 values as 4x v16b
    ld1     {v16.16b, v17.16b}, [x9]

    // Convert int8 to float32 (4 values at a time)
    // Bytes 0-7 -> v16 low
    sxtl    v18.8h, v16.8b          // sign-extend 8 bytes to 8 halfwords
    sxtl    v20.4s, v18.4h          // sign-extend 4 hw to 4 words
    scvtf   v20.4s, v20.4s          // int32 to float
    fmul    v20.4s, v20.4s, v8.s[0] // * scale
    ld1     {v4.4s}, [x2], #16
    fmla    v0.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v18.8h
    scvtf   v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v0.4s, v21.4s, v4.4s

    // Bytes 8-15
    sxtl2   v18.8h, v16.16b
    sxtl    v20.4s, v18.4h
    scvtf   v20.4s, v20.4s
    fmul    v20.4s, v20.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v1.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v18.8h
    scvtf   v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v1.4s, v21.4s, v4.4s

    // Bytes 16-23
    sxtl    v18.8h, v17.8b
    sxtl    v20.4s, v18.4h
    scvtf   v20.4s, v20.4s
    fmul    v20.4s, v20.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v0.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v18.8h
    scvtf   v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v0.4s, v21.4s, v4.4s

    // Bytes 24-31
    sxtl2   v18.8h, v17.16b
    sxtl    v20.4s, v18.4h
    scvtf   v20.4s, v20.4s
    fmul    v20.4s, v20.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v1.4s, v20.4s, v4.4s

    sxtl2   v21.4s, v18.8h
    scvtf   v21.4s, v21.4s
    fmul    v21.4s, v21.4s, v8.s[0]
    ld1     {v4.4s}, [x2], #16
    fmla    v1.4s, v21.4s, v4.4s

    add     x3, x3, #Q8_0_BLOCK_BYTES
    sub     x4, x4, #1
    b       .Lq8_0_kb

.Lq8_0_kd:
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s; faddp s8, v0.2s
    fmul    s8, s8, s29

    ldr     x2, [sp, #STK_MASK_PTR_L]; cbz x2, .Lq8_0_nm
    ldr     x3, [sp, #STK_IQ_SAVE]; ldr x4, [sp, #STK_MASK_NB1]
    add     x5, x14, x0; ldr x6, [x19, #CFG_MASK_NB0]
    mul     x3, x3, x6; mul x5, x5, x4
    add     x2, x2, x3; add x2, x2, x5; ldr s16, [x2]; fadd s8, s8, s16
.Lq8_0_nm:
    ldr     w2, [x19, #CFG_MODE]; tst w2, #FA_MODE_SOFTCAP; b.eq .Lq8_0_ns
    ldr     s16, [x19, #CFG_LOGIT_SOFTCAP]; fdiv s17, s8, s16
    fmul    s18, s17, s17; mov w7, #0x41D80000; fmov s19, w7
    mov     w7, #0x41100000; fmov s20, w7; fmul s20, s20, s18
    fadd    s21, s19, s18; fadd s20, s19, s20; fdiv s21, s21, s20
    fmul    s17, s17, s21; fmul s8, s16, s17
.Lq8_0_ns:
    add     x2, sp, #STK_SCORES; str s8, [x2, x0, lsl #2]
    add     x0, x0, #1; b .Lq8_0_qk
.Lq8_0_qk_d:

    // Tile max, correction, probs, P*V (same structure as Q4_0)
    add     x0, sp, #STK_SCORES; mov w1, #0xFF800000; fmov s0, w1
    mov     x1, #0
.Lq8_0_tm: cmp x1, x16; b.ge .Lq8_0_tmd; ldr s16, [x0, x1, lsl #2]; fmax s0, s0, s16; add x1, x1, #1; b .Lq8_0_tm
.Lq8_0_tmd:
    fmax    s1, s31, s0; fsub s2, s31, s1; bl .Lq8_0_exp
    add     x0, sp, #STK_O; dup v16.4s, v0.s[0]; mov x1, x22; lsr x1, x1, #2
.Lq8_0_co: ld1 {v17.4s}, [x0]; fmul v17.4s, v17.4s, v16.4s; st1 {v17.4s}, [x0], #16; subs x1, x1, #1; b.gt .Lq8_0_co
    fmul    s30, s30, s0; fmov s31, s1

    add     x0, sp, #STK_SCORES; add x1, sp, #STK_PROBS; movi d28, #0; mov x2, #0
.Lq8_0_pr: cmp x2, x16; b.ge .Lq8_0_pd; ldr s2, [x0, x2, lsl #2]; fsub s2, s2, s31
    bl .Lq8_0_exp; str s0, [x1, x2, lsl #2]; fadd s28, s28, s0; add x2, x2, #1; b .Lq8_0_pr
.Lq8_0_pd: fadd s30, s30, s28

    // P * V accumulation (Q8_0 V dequant)
    add     x1, sp, #STK_PROBS; mov x2, #0
.Lq8_0_pv: cmp x2, x16; b.ge .Lq8_0_pvd
    ldr     s0, [x1, x2, lsl #2]; fcmp s0, #0.0; b.eq .Lq8_0_pvs
    add     x3, x14, x2; ldr x4, [sp, #STK_V_NB1]; mul x3, x3, x4; add x3, x26, x3
    dup     v24.4s, v0.s[0]; add x4, sp, #STK_O; ldr x5, [sp, #STK_V_NBLOCKS]
.Lq8_0_pvb: cbz x5, .Lq8_0_pvs
    ldr     h8, [x3]; fcvt s8, h8; add x9, x3, #2
    ld1     {v16.16b, v17.16b}, [x9]
    // Dequant & accumulate (8 groups of 4)
    sxtl v18.8h, v16.8b; sxtl v20.4s, v18.4h; scvtf v20.4s, v20.4s; fmul v20.4s, v20.4s, v8.s[0]; fmul v20.4s, v20.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v21.4s, v18.8h; scvtf v21.4s, v21.4s; fmul v21.4s, v21.4s, v8.s[0]; fmul v21.4s, v21.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v18.8h, v16.16b; sxtl v20.4s, v18.4h; scvtf v20.4s, v20.4s; fmul v20.4s, v20.4s, v8.s[0]; fmul v20.4s, v20.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v21.4s, v18.8h; scvtf v21.4s, v21.4s; fmul v21.4s, v21.4s, v8.s[0]; fmul v21.4s, v21.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x4], #16
    sxtl v18.8h, v17.8b; sxtl v20.4s, v18.4h; scvtf v20.4s, v20.4s; fmul v20.4s, v20.4s, v8.s[0]; fmul v20.4s, v20.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v21.4s, v18.8h; scvtf v21.4s, v21.4s; fmul v21.4s, v21.4s, v8.s[0]; fmul v21.4s, v21.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v18.8h, v17.16b; sxtl v20.4s, v18.4h; scvtf v20.4s, v20.4s; fmul v20.4s, v20.4s, v8.s[0]; fmul v20.4s, v20.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v20.4s; st1 {v5.4s}, [x4], #16
    sxtl2 v21.4s, v18.8h; scvtf v21.4s, v21.4s; fmul v21.4s, v21.4s, v8.s[0]; fmul v21.4s, v21.4s, v24.4s
    ld1 {v5.4s}, [x4]; fadd v5.4s, v5.4s, v21.4s; st1 {v5.4s}, [x4], #16
    add x3, x3, #Q8_0_BLOCK_BYTES; sub x5, x5, #1; b .Lq8_0_pvb
.Lq8_0_pvs: add x2, x2, #1; b .Lq8_0_pv
.Lq8_0_pvd:
    add     x14, x14, #FA_TILE_KV_NEON; b .Lq8_0_tl

.Lq8_0_norm:
    fcmp    s30, #0.0; b.le .Lq8_0_zout
    fmov    s0, #1.0; fdiv s0, s0, s30; dup v16.4s, v0.s[0]
    add     x0, sp, #STK_O; mov x1, x27; mov x2, x22; lsr x2, x2, #2
.Lq8_0_nl: ld1 {v17.4s}, [x0], #16; fmul v17.4s, v17.4s, v16.4s; st1 {v17.4s}, [x1], #16; subs x2, x2, #1; b.gt .Lq8_0_nl
    b .Lq8_0_nx
.Lq8_0_zout: movi v0.4s, #0; mov x1, x27; mov x2, x22; lsr x2, x2, #2
.Lq8_0_zl: st1 {v0.4s}, [x1], #16; subs x2, x2, #1; b.gt .Lq8_0_zl
.Lq8_0_nx: add x20, x20, #1; b .Lq8_0_wl

.Lq8_0_epi:
    add     sp, sp, #FRAME_SIZE
    ldp     d14, d15, [sp], #16; ldp d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16; ldp d8, d9, [sp], #16
    ldp     x27, x28, [sp], #16; ldp x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16; ldp x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16; ldp x29, x30, [sp], #16
    ret

.Lq8_0_exp:
    adrp x7, q8_0_neon_exp_clamp_hi; add x7, x7, :lo12:q8_0_neon_exp_clamp_hi; ldr s16, [x7]; fmin s2, s2, s16
    adrp x7, q8_0_neon_exp_clamp_lo; add x7, x7, :lo12:q8_0_neon_exp_clamp_lo; ldr s16, [x7]; fmax s2, s2, s16
    adrp x7, q8_0_neon_exp_log2e; add x7, x7, :lo12:q8_0_neon_exp_log2e; ldr s16, [x7]; fmul s17, s2, s16; frintn s18, s17
    adrp x7, q8_0_neon_exp_ln2; add x7, x7, :lo12:q8_0_neon_exp_ln2; ldr s16, [x7]; fnmsub s19, s18, s16, s2
    adrp x7, q8_0_neon_exp_c3; add x7, x7, :lo12:q8_0_neon_exp_c3; ldr s0, [x7]
    adrp x7, q8_0_neon_exp_c2; add x7, x7, :lo12:q8_0_neon_exp_c2; ldr s16, [x7]; fmadd s0, s0, s19, s16
    adrp x7, q8_0_neon_exp_c1; add x7, x7, :lo12:q8_0_neon_exp_c1; ldr s16, [x7]; fmadd s0, s0, s19, s16
    adrp x7, q8_0_neon_half; add x7, x7, :lo12:q8_0_neon_half; ldr s16, [x7]; fmadd s0, s0, s19, s16
    adrp x7, q8_0_neon_one; add x7, x7, :lo12:q8_0_neon_one; ldr s16, [x7]; fmadd s0, s0, s19, s16; fmadd s0, s0, s19, s16
    fcvtzs w7, s18; add w7, w7, #127; lsl w7, w7, #23; fmov s16, w7; fmul s0, s0, s16
    ret

END_FUNCTION(simd_flash_attn_q8_0_neon)
