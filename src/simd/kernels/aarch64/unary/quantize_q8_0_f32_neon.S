// ===========================================================================
//   quantize_q8_0_f32_neon.S — Q8_0 quantization of f32 floats, AArch64 NEON
// ===========================================================================
// void simd_quantize_q8_0_f32_neon(int n, const float * x, void * y);
//
// AAPCS64: w0 = n, x1 = x, x2 = y
//
// Quantizes groups of 32 floats into block_q8_0 structs (34 bytes each):
//   typedef struct {
//       ggml_half d;       // f16 scale — amax / 127.0f  (2 bytes)
//       int8_t    qs[32];  // 32 quantised int8 values
//   } block_q8_0;
//
// Algorithm — matches ggml's quantize_row_q8_0 (arch/arm/quants.c):
//   For each block of 32 floats:
//     1. Load 8 × q-reg (32 floats, 4 per reg)
//     2. Compute amax = max(|x|) via fabs + pairwise fmax + fmaxv
//     3. d = amax / 127.0f  → fcvt to f16 → strh at y[i].d (offset 0)
//     4. id = (amax != 0) ? 127.0f / amax : 0.0f
//     5. Multiply all 32 floats by id (fmul, scalar broadcast)
//     6. Round to nearest integer + convert to i32 (fcvtns)
//     7. Pack i32 → i16 → i8 (sqxtn + sqxtn2 × 2 stages)
//     8. Store 32 int8 values to y[i].qs (offset 2)
//
// Vector width: 8 iterations of 4 floats each (two q-regs per 8 floats).
// Narrowing chain: sqxtn (4×i32→4×i16) + sqxtn2 → then sqxtn (8×i16→8×i8)
// + sqxtn2 → 16 int8 per q-reg.  Final two q-regs stored at offsets 2 and 18.
//
// Validation: U1 (`zig build test-simd`) must be re-run on a real arm64 host;
// this file ships from an x86 dev box and has not been executed locally.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(simd_quantize_q8_0_f32_neon)

    // Prologue
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    // Save args
    mov     w19, w0                 // w19 = n
    mov     x3,  x1                 // x3 = x (src)
    mov     x4,  x2                 // x4 = y (dst)

    // ------------------------------------------------------------------
    // nb = n / 32   (number of blocks)
    // ------------------------------------------------------------------
    lsr     w20, w19, #5            // w20 = n / 32
    cbz     w20, .Ldone

    // Pre-load 127.0f constant into s16
    ldr     s16, .Lconst_127

.Lblock_loop:
    // ---------------------------------------------------------------
    // 1. Load 32 floats (8 × q-reg of 4 floats each)
    // ---------------------------------------------------------------
    ldp     q0, q1, [x3, #0]
    ldp     q2, q3, [x3, #32]
    ldp     q4, q5, [x3, #64]
    ldp     q6, q7, [x3, #96]
    add     x3, x3, #128            // advance src past 32 floats

    // ---------------------------------------------------------------
    // 2. Compute amax = max(|x[i]|)
    //    fabs on all 8 regs into temps v24..v31, then pairwise max.
    // ---------------------------------------------------------------
    fabs    v24.4s, v0.4s
    fabs    v25.4s, v1.4s
    fabs    v26.4s, v2.4s
    fabs    v27.4s, v3.4s
    fabs    v28.4s, v4.4s
    fabs    v29.4s, v5.4s
    fabs    v30.4s, v6.4s
    fabs    v31.4s, v7.4s

    // Pairwise max: 8 → 4 regs
    fmax    v24.4s, v24.4s, v25.4s
    fmax    v26.4s, v26.4s, v27.4s
    fmax    v28.4s, v28.4s, v29.4s
    fmax    v30.4s, v30.4s, v31.4s

    // Pairwise max: 4 → 2 regs
    fmax    v24.4s, v24.4s, v26.4s
    fmax    v28.4s, v28.4s, v30.4s

    // Pairwise max: 2 → 1 reg
    fmax    v24.4s, v24.4s, v28.4s

    // Full horizontal max: 4 lanes → scalar (s5)
    fmaxv   s5, v24.4s              // s5 = amax

    // ---------------------------------------------------------------
    // 3. d = amax / 127.0f  → store as f16 at y[i].d (offset 0)
    // ---------------------------------------------------------------
    fdiv    s6, s5, s16             // s6 = d = amax / 127.0f
    fcvt    h6, s6                  // h6 = f16(d)
    strh    h6, [x4]                // store at offset 0

    // ---------------------------------------------------------------
    // 4. id = (amax != 0) ? 127.0f / amax : 0.0f
    // ---------------------------------------------------------------
    fmov    s7, wzr                 // s7 = 0.0f
    fcmp    s5, s7
    beq     .Lzero_id

    fdiv    s7, s16, s5             // s7 = 127.0f / amax
    b       .Lapply_id

.Lzero_id:
    fmov    s7, wzr                 // id = 0.0f

.Lapply_id:
    // ---------------------------------------------------------------
    // 5. Multiply all 32 floats by id
    //    Broadcast id (s7) to q-reg lanes then fmul.
    // ---------------------------------------------------------------
    dup     v8.4s, v7.s[0]          // v8 = {id, id, id, id}

    fmul    v0.4s, v0.4s, v8.4s
    fmul    v1.4s, v1.4s, v8.4s
    fmul    v2.4s, v2.4s, v8.4s
    fmul    v3.4s, v3.4s, v8.4s
    fmul    v4.4s, v4.4s, v8.4s
    fmul    v5.4s, v5.4s, v8.4s
    fmul    v6.4s, v6.4s, v8.4s
    fmul    v7.4s, v7.4s, v8.4s

    // ---------------------------------------------------------------
    // 6. Round to nearest (ties to even) + convert to int32
    //    fcvtns = FRINTN + FCVTZS in one instruction.
    //    Matches vcvtnq_s32_f32 from the C intrinsics.
    // ---------------------------------------------------------------
    fcvtns  v0.4s, v0.4s
    fcvtns  v1.4s, v1.4s
    fcvtns  v2.4s, v2.4s
    fcvtns  v3.4s, v3.4s
    fcvtns  v4.4s, v4.4s
    fcvtns  v5.4s, v5.4s
    fcvtns  v6.4s, v6.4s
    fcvtns  v7.4s, v7.4s

    // ---------------------------------------------------------------
    // 7. Narrow: i32 → i16 → i8 (signed saturating)
    //
    //    Stage 1: 4×i32 → 8×i16 (sqxtn + sqxtn2 per pair of q-regs)
    //      v16.8h = [narrow(v0), narrow(v1)]
    //      v17.8h = [narrow(v2), narrow(v3)]
    //      v18.8h = [narrow(v4), narrow(v5)]
    //      v19.8h = [narrow(v6), narrow(v7)]
    //
    //    Stage 2: 8×i16 → 16×i8 (sqxtn + sqxtn2 per pair)
    //      v20.16b = [narrow(v16), narrow(v17)]
    //      v21.16b = [narrow(v18), narrow(v19)]
    // ---------------------------------------------------------------

    // Stage 1 — i32 → i16
    sqxtn   v16.4h, v0.4s
    sqxtn2  v16.8h, v1.4s

    sqxtn   v17.4h, v2.4s
    sqxtn2  v17.8h, v3.4s

    sqxtn   v18.4h, v4.4s
    sqxtn2  v18.8h, v5.4s

    sqxtn   v19.4h, v6.4s           ; v6.4s still contains q6's int32
    sqxtn2  v19.8h, v7.4s

    // Stage 2 — i16 → i8
    sqxtn   v0.8b, v16.8h           ; v0[0..7] = narrow(v16)
    sqxtn2  v0.16b, v17.8h          ; v0[8..15] = narrow(v17)
    ; v0.16b = [v0_i8, v1_i8, v2_i8, v3_i8] — 16 bytes

    sqxtn   v1.8b, v18.8h           ; v1[0..7] = narrow(v18)
    sqxtn2  v1.16b, v19.8h          ; v1[8..15] = narrow(v19)
    ; v1.16b = [v4_i8, v5_i8, v6_i8, v7_i8] — 16 bytes

    // ---------------------------------------------------------------
    // 8. Store 32 int8 values at y[i].qs (offset 2)
    //    Two 16-byte stores at offsets 2 and 18.
    // ---------------------------------------------------------------
    str     q0, [x4, #2]            // qs[ 0..15] — first 16 int8
    str     q1, [x4, #18]           // qs[16..31] — second 16 int8

    add     x4, x4, #34             // advance dst by 34 bytes (one block)

    sub     w20, w20, #1
    cbnz    w20, .Lblock_loop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

    // ------------------------------------------------------------------
    // Constant pool
    // ------------------------------------------------------------------
    .align 4
.Lconst_127:
    .word   0x42fe0000              // 127.0f
