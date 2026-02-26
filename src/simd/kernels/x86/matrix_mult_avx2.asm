;==============================================================================
;                    SIMD MATRIX MULTIPLICATION - AVX2
;==============================================================================
;
; High-performance matrix multiplication using AVX2 (256-bit) SIMD instructions.
; Computes C = A × B for single-precision floating-point matrices.
;
; This implementation uses:
;   - FMA (Fused Multiply-Add) instructions for maximum throughput
;   - Register blocking to minimize memory access
;   - Row-wise iteration for optimal cache performance
;   - Edge handling for non-multiple-of-8 dimensions
;
; Supported calling conventions:
;   - System V AMD64 (Linux, macOS, BSD)
;   - Microsoft x64 (Windows) - controlled by WIN64 define
;
; Based on: https://github.com/sutantodadang/assembly-simd
; License: MIT
;
;==============================================================================

; Detect platform
%ifidn __OUTPUT_FORMAT__, win64
    %define WIN64
%endif

;------------------------------------------------------------------------------
; Section: Read-only data
;------------------------------------------------------------------------------
section .rodata
    align 32
    ; Mask for handling edge cases (last 1-7 floats)
    edge_mask:
        dd 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        dd 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        dd 0x00000000, 0x00000000, 0x00000000, 0x00000000
        dd 0x00000000, 0x00000000, 0x00000000, 0x00000000

;------------------------------------------------------------------------------
; Section: Code
;------------------------------------------------------------------------------
section .text

;==============================================================================
; Function: matrix_mult_avx2
;==============================================================================
;
; Computes C = A × B for float matrices.
;
; Parameters (System V AMD64 ABI):
;   rdi = A    - Pointer to matrix A (M × K), row-major order
;   rsi = B    - Pointer to matrix B (K × N), row-major order
;   rdx = C    - Pointer to result matrix C (M × N), row-major order
;   rcx = M    - Number of rows in A and C
;   r8  = N    - Number of columns in B and C
;   r9  = K    - Number of columns in A / rows in B
;
; Parameters (Microsoft x64 ABI):
;   rcx = A    - Pointer to matrix A
;   rdx = B    - Pointer to matrix B
;   r8  = C    - Pointer to result matrix C
;   r9  = M    - Number of rows in A and C
;   [rsp+40] = N   - Number of columns in B and C
;   [rsp+48] = K   - Number of columns in A / rows in B
;
; Returns: Nothing (C is modified in place)
;
;==============================================================================

global matrix_mult_avx2
matrix_mult_avx2:
    ;--------------------------------------------------------------------------
    ; PROLOGUE: Save callee-saved registers and set up stack frame
    ;--------------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WIN64
    ; Windows: Allocate space and preserve RDI/RSI (callee-saved)
    push    rdi
    push    rsi
    sub     rsp, 248
    
    ; Save XMM6-XMM15
    vmovaps [rsp],      xmm6
    vmovaps [rsp + 16], xmm7
    vmovaps [rsp + 32], xmm8
    vmovaps [rsp + 48], xmm9
    vmovaps [rsp + 64], xmm10
    vmovaps [rsp + 80], xmm11
    vmovaps [rsp + 96], xmm12
    vmovaps [rsp + 112], xmm13
    vmovaps [rsp + 128], xmm14
    vmovaps [rsp + 144], xmm15
    
    ; Remap Windows x64 parameters to match System V layout
    mov     rdi, rcx                    ; A
    mov     rsi, rdx                    ; B
    mov     rdx, r8                     ; C
    mov     rcx, r9                     ; M
    mov     r8,  [rbp + 48]             ; N
    mov     r9,  [rbp + 56]             ; K
%else
    ; Linux/macOS: Just allocate space for local variables
    sub     rsp, 80
%endif

    ;--------------------------------------------------------------------------
    ; PARAMETER SETUP
    ;--------------------------------------------------------------------------
    mov     [rbp - 64], rdi             ; Save A base
    mov     [rbp - 72], rsi             ; Save B base
    mov     [rbp - 80], rdx             ; Save C base
    mov     [rbp - 88], rcx             ; Save M
    mov     [rbp - 96], r8              ; Save N
    mov     [rbp - 104], r9             ; Save K
    
    ; Calculate row strides (in bytes)
    mov     r10, r9                      
    shl     r10, 2                       ; r10 = K * 4 = A row stride
    mov     r11, r8
    shl     r11, 2                       ; r11 = N * 4 = B row stride
    mov     r12, r11                     ; r12 = N * 4 = C row stride
    
    ; Calculate N / 32 (number of full 32-column tiles)
    mov     rax, r8
    shr     rax, 5
    mov     [rbp - 112], rax
    
    ;--------------------------------------------------------------------------
    ; MAIN COMPUTATION: 4×32 register-blocked tiled matrix multiplication
    ;--------------------------------------------------------------------------
    xor     r13, r13                     ; i = 0 (row counter)
    
.row_loop:
    cmp     r13, [rbp - 88]              ; Compare i with M
    jge     .done
    
    ; Calculate A row pointer: A_row = A + i * K * 4
    mov     rax, r13
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax
    mov     [rbp - 128], rbx             ; Save A row pointer
    
    ; Calculate C row pointer: C_row = C + i * N * 4
    mov     rax, r13
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax
    
    ; Calculate number of 32-column tiles
    mov     rax, [rbp - 96]
    shr     rax, 5
    mov     [rbp - 136], rax
    
    xor     r15, r15                     ; j_tile = 0
    
.tile_loop:
    mov     rax, r15
    shr     rax, 5
    cmp     rax, [rbp - 136]
    jge     .tile_remainder
    
    ; Initialize 4 YMM accumulators to zero
    vxorps  ymm0, ymm0, ymm0             ; C[i][j:j+8]
    vxorps  ymm1, ymm1, ymm1             ; C[i][j+8:j+16]
    vxorps  ymm2, ymm2, ymm2             ; C[i][j+16:j+24]
    vxorps  ymm3, ymm3, ymm3             ; C[i][j+24:j+32]
    
    ; K-loop: Iterate over inner dimension
    mov     rbx, [rbp - 128]             ; A_row pointer
    mov     r9, [rbp - 104]              ; K
    
    ; Calculate B column pointer: B[0][j]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.k_loop_tiled:
    test    r9, r9
    jz      .k_done_tiled
    
    ; Broadcast A[i][k] to ymm4
    vbroadcastss ymm4, [rbx]
    
    ; FMA for 4 column blocks
    vfmadd231ps ymm0, ymm4, [rcx]        ; C[i][j:j+8] += A * B
    vfmadd231ps ymm1, ymm4, [rcx + 32]   ; C[i][j+8:j+16] += A * B
    vfmadd231ps ymm2, ymm4, [rcx + 64]   ; C[i][j+16:j+24] += A * B
    vfmadd231ps ymm3, ymm4, [rcx + 96]   ; C[i][j+24:j+32] += A * B
    
    ; Advance pointers
    add     rbx, 4                       ; A pointer += 1 element
    add     rcx, r11                     ; B pointer += N elements (next row)
    dec     r9
    jnz     .k_loop_tiled
    
.k_done_tiled:
    ; Store 4 YMM accumulators to C
    vmovups [r14], ymm0
    vmovups [r14 + 32], ymm1
    vmovups [r14 + 64], ymm2
    vmovups [r14 + 96], ymm3
    
    add     r14, 128                     ; C pointer += 32 floats
    add     r15, 32                      ; j += 32
    jmp     .tile_loop
    
.tile_remainder:
    ; Handle remaining columns (N mod 32)
    mov     rax, [rbp - 96]
    and     rax, 31
    test    rax, rax
    jz      .row_next
    
    ; Process remaining full 8-column blocks
    mov     rdx, rax
    shr     rdx, 3                       ; remainder / 8
    mov     r8, rax
    and     r8, 7                        ; remainder mod 8
    
    test    rdx, rdx
    jz      .final_edge
    
.rem_8_loop:
    vxorps  ymm0, ymm0, ymm0
    mov     rbx, [rbp - 128]
    mov     r9, [rbp - 104]
    
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.rem_8_k_loop:
    test    r9, r9
    jz      .rem_8_k_done
    
    vbroadcastss ymm4, [rbx]
    vfmadd231ps ymm0, ymm4, [rcx]
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .rem_8_k_loop
    
.rem_8_k_done:
    vmovups [r14], ymm0
    add     r14, 32
    add     r15, 8
    dec     rdx
    jnz     .rem_8_loop
    
.final_edge:
    ; Handle final 1-7 columns with masking
    test    r8, r8
    jz      .row_next
    
    lea     rsi, [rel edge_mask]
    mov     rax, 8
    sub     rax, r8
    shl     rax, 2
    vmovups ymm15, [rsi + rax]
    
    vxorps  ymm0, ymm0, ymm0
    mov     rbx, [rbp - 128]
    mov     r9, [rbp - 104]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.final_k_loop:
    test    r9, r9
    jz      .final_k_done
    
    vbroadcastss ymm4, [rbx]
    vmaskmovps ymm5, ymm15, [rcx]
    vfmadd231ps ymm0, ymm4, ymm5
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .final_k_loop
    
.final_k_done:
    vmaskmovps [r14], ymm15, ymm0
    
.row_next:
    inc     r13
    jmp     .row_loop
    
.done:
    ;--------------------------------------------------------------------------
    ; EPILOGUE: Restore callee-saved registers and return
    ;--------------------------------------------------------------------------
    vzeroupper
    
%ifdef WIN64
    ; Windows: Restore XMM registers
    vmovaps xmm15, [rsp + 144]
    vmovaps xmm14, [rsp + 128]
    vmovaps xmm13, [rsp + 112]
    vmovaps xmm12, [rsp + 96]
    vmovaps xmm11, [rsp + 80]
    vmovaps xmm10, [rsp + 64]
    vmovaps xmm9,  [rsp + 48]
    vmovaps xmm8,  [rsp + 32]
    vmovaps xmm7,  [rsp + 16]
    vmovaps xmm6,  [rsp]
    add     rsp, 248
    pop     rsi
    pop     rdi
%else
    add     rsp, 80
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
