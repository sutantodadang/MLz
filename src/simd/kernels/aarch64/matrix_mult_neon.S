// ===========================================================================
//   matrix_mult_neon.S — F32 Matrix Multiplication, AArch64 NEON
// ===========================================================================
// void matrix_mult_neon(const float* A, const float* B, float* C,
//                       size_t M, size_t N, size_t K);
// AAPCS64: x0=A, x1=B, x2=C, x3=M, x4=N, x5=K
//
// Computes C = A × B for single-precision fp32 row-major matrices.
// Uses 4×4 register blocking with NEON FMLA for throughput.
// Handles edge cases for non-multiple-of-4 dimensions.
// License: MIT
// ===========================================================================

#include "neon_common.h"

    .text
    .align 4
    GLOBAL_SYMBOL(matrix_mult_neon)
    // Prologue: save callee-saved registers
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]

    // Save parameters
    mov     x19, x0                     // A
    mov     x20, x1                     // B
    mov     x21, x2                     // C
    mov     x22, x3                     // M
    mov     x23, x4                     // N
    mov     x24, x5                     // K

    // Strides (in bytes)
    lsl     x25, x24, #2               // A row stride = K * 4
    lsl     x26, x23, #2               // B row stride = N * 4 = C row stride

    // i = 0 (row loop)
    mov     x27, #0

.Lmm_row:
    cmp     x27, x22                    // i < M?
    b.ge    .Lmm_done

    // A_row = A + i * K * 4
    mul     x0, x27, x25
    add     x0, x19, x0                // x0 = A_row

    // C_row = C + i * N * 4
    mul     x1, x27, x26
    add     x1, x21, x1                // x1 = C_row

    // j = 0 (column tile loop, 4 columns per tile)
    mov     x28, #0

.Lmm_col4:
    add     x2, x28, #4
    cmp     x2, x23                     // j+4 <= N?
    b.gt    .Lmm_col1                   // handle remainder

    // Initialize 4 accumulators to zero
    movi    v0.4s, #0                   // C[i][j:j+4]

    // K loop
    mov     x3, x0                      // A_row ptr
    // B_col = B + j * 4
    mov     x4, x28
    lsl     x4, x4, #2
    add     x4, x20, x4                // x4 = &B[0][j]
    mov     x5, x24                     // k counter

.Lmm_k4:
    cmp     x5, #4
    b.lt    .Lmm_k1

    // Load 4 A values (broadcast each)
    ldr     s4, [x3], #4               // A[i][k]
    ldr     s5, [x3], #4               // A[i][k+1]
    ldr     s6, [x3], #4               // A[i][k+2]
    ldr     s7, [x3], #4               // A[i][k+3]

    // Load B[k][j:j+4], B[k+1][j:j+4], etc.
    ldr     q8,  [x4]                   // B[k][j:j+4]
    add     x4, x4, x26
    ldr     q9,  [x4]                   // B[k+1][j:j+4]
    add     x4, x4, x26
    ldr     q10, [x4]                   // B[k+2][j:j+4]
    add     x4, x4, x26
    ldr     q11, [x4]                   // B[k+3][j:j+4]
    add     x4, x4, x26

    // FMA
    fmla    v0.4s, v8.4s, v4.s[0]
    fmla    v0.4s, v9.4s, v5.s[0]
    fmla    v0.4s, v10.4s, v6.s[0]
    fmla    v0.4s, v11.4s, v7.s[0]

    sub     x5, x5, #4
    b       .Lmm_k4

.Lmm_k1:
    cbz     x5, .Lmm_k_done

    ldr     s4, [x3], #4
    ldr     q8, [x4]
    add     x4, x4, x26
    fmla    v0.4s, v8.4s, v4.s[0]

    sub     x5, x5, #1
    b       .Lmm_k1

.Lmm_k_done:
    // Store 4 C values
    mov     x4, x28
    lsl     x4, x4, #2
    add     x4, x1, x4                 // &C[i][j]
    str     q0, [x4]

    add     x28, x28, #4
    b       .Lmm_col4

.Lmm_col1:
    // Handle remaining columns (1-3)
    cmp     x28, x23
    b.ge    .Lmm_row_next

    fmov    s0, wzr                     // accumulator

    mov     x3, x0                      // A_row ptr
    // B_col = B + j * 4
    mov     x4, x28
    lsl     x4, x4, #2
    add     x4, x20, x4
    mov     x5, x24                     // k counter

.Lmm_k1_scalar:
    cbz     x5, .Lmm_k1_scalar_done

    ldr     s4, [x3], #4               // A[i][k]
    ldr     s5, [x4]                    // B[k][j]
    add     x4, x4, x26                // next B row
    fmadd   s0, s4, s5, s0

    sub     x5, x5, #1
    b       .Lmm_k1_scalar

.Lmm_k1_scalar_done:
    mov     x4, x28
    lsl     x4, x4, #2
    add     x4, x1, x4
    str     s0, [x4]

    add     x28, x28, #1
    b       .Lmm_col1

.Lmm_row_next:
    add     x27, x27, #1
    b       .Lmm_row

.Lmm_done:
    // Epilogue
    ldp     d8, d9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #112
    ret

    END_FUNCTION(matrix_mult_neon)
