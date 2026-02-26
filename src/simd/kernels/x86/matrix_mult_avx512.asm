;==============================================================================
;                    SIMD MATRIX MULTIPLICATION - AVX512
;==============================================================================
;
; High-performance matrix multiplication using AVX-512 (512-bit) SIMD.
; Computes C = A × B for single-precision floating-point matrices.
;
; This implementation uses:
;   - AVX-512F FMA instructions for maximum throughput (16 floats at once)
;   - Register blocking with 64-wide column tiles
;   - Optimal cache access patterns
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
    align 64
    ; Mask table for edge handling (0-15 remaining floats)
    edge_mask_16:
        times 16 dd 0xFFFFFFFF
        times 16 dd 0x00000000

;------------------------------------------------------------------------------
; Section: Code
;------------------------------------------------------------------------------
section .text

;==============================================================================
; Function: matrix_mult_avx512
;==============================================================================
;
; Computes C = A × B for float matrices using AVX-512.
;
; Parameters (System V AMD64 ABI):
;   rdi = A    - Pointer to matrix A (M × K), row-major, 64-byte aligned
;   rsi = B    - Pointer to matrix B (K × N), row-major, 64-byte aligned
;   rdx = C    - Pointer to result matrix C (M × N), row-major, 64-byte aligned
;   rcx = M    - Number of rows in A and C
;   r8  = N    - Number of columns in B and C
;   r9  = K    - Number of columns in A / rows in B
;
; Parameters (Microsoft x64 ABI):
;   rcx = A, rdx = B, r8 = C, r9 = M, [rsp+40] = N, [rsp+48] = K
;
;==============================================================================

global matrix_mult_avx512
matrix_mult_avx512:
    ;--------------------------------------------------------------------------
    ; PROLOGUE
    ;--------------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WIN64
    push    rdi
    push    rsi
    sub     rsp, 312                    ; Space for locals + XMM registers
    
    ; Save XMM6-XMM15 (Windows callee-saved)
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
    
    ; Remap Windows x64 parameters
    mov     rdi, rcx                    ; A
    mov     rsi, rdx                    ; B
    mov     rdx, r8                     ; C
    mov     rcx, r9                     ; M
    mov     r8,  [rbp + 48]             ; N
    mov     r9,  [rbp + 56]             ; K
%else
    sub     rsp, 128
%endif

    ;--------------------------------------------------------------------------
    ; PARAMETER SETUP
    ;--------------------------------------------------------------------------
    mov     [rbp - 64], rdi             ; A
    mov     [rbp - 72], rsi             ; B
    mov     [rbp - 80], rdx             ; C
    mov     [rbp - 88], rcx             ; M
    mov     [rbp - 96], r8              ; N
    mov     [rbp - 104], r9             ; K
    
    ; Calculate strides
    mov     r10, r9
    shl     r10, 2                       ; A stride = K * 4
    mov     r11, r8
    shl     r11, 2                       ; B stride = N * 4
    mov     r12, r11                     ; C stride = N * 4
    
    ;--------------------------------------------------------------------------
    ; MAIN COMPUTATION: Process 64 columns at a time (4 ZMM registers)
    ;--------------------------------------------------------------------------
    xor     r13, r13                     ; i = 0
    
.row_loop:
    cmp     r13, [rbp - 88]
    jge     .done
    
    ; A row pointer
    mov     rax, r13
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax
    mov     [rbp - 128], rbx
    
    ; C row pointer
    mov     rax, r13
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax
    
    ; Number of 64-column tiles
    mov     rax, [rbp - 96]
    shr     rax, 6                       ; N / 64
    mov     [rbp - 136], rax
    
    xor     r15, r15                     ; j = 0
    
.tile_loop:
    mov     rax, r15
    shr     rax, 6
    cmp     rax, [rbp - 136]
    jge     .tile_remainder
    
    ; Initialize 4 ZMM accumulators
    vxorps  zmm0, zmm0, zmm0             ; C[i][j:j+16]
    vxorps  zmm1, zmm1, zmm1             ; C[i][j+16:j+32]
    vxorps  zmm2, zmm2, zmm2             ; C[i][j+32:j+48]
    vxorps  zmm3, zmm3, zmm3             ; C[i][j+48:j+64]
    
    mov     rbx, [rbp - 128]             ; A row
    mov     r9, [rbp - 104]              ; K
    
    ; B column pointer
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.k_loop:
    test    r9, r9
    jz      .k_done
    
    ; Broadcast A[i][k]
    vbroadcastss zmm4, [rbx]
    
    ; FMA for 4 column blocks (64 floats total)
    vfmadd231ps zmm0, zmm4, [rcx]
    vfmadd231ps zmm1, zmm4, [rcx + 64]
    vfmadd231ps zmm2, zmm4, [rcx + 128]
    vfmadd231ps zmm3, zmm4, [rcx + 192]
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .k_loop
    
.k_done:
    ; Store results
    vmovups [r14], zmm0
    vmovups [r14 + 64], zmm1
    vmovups [r14 + 128], zmm2
    vmovups [r14 + 192], zmm3
    
    add     r14, 256                     ; C += 64 floats
    add     r15, 64
    jmp     .tile_loop
    
.tile_remainder:
    ; Handle remaining columns (N mod 64)
    mov     rax, [rbp - 96]
    and     rax, 63
    test    rax, rax
    jz      .row_next
    
    ; Process remaining 16-column blocks
    mov     rdx, rax
    shr     rdx, 4                       ; remainder / 16
    mov     r8, rax
    and     r8, 15                       ; remainder mod 16
    
    test    rdx, rdx
    jz      .final_edge
    
.rem_16_loop:
    vxorps  zmm0, zmm0, zmm0
    mov     rbx, [rbp - 128]
    mov     r9, [rbp - 104]
    
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.rem_16_k_loop:
    test    r9, r9
    jz      .rem_16_k_done
    
    vbroadcastss zmm4, [rbx]
    vfmadd231ps zmm0, zmm4, [rcx]
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .rem_16_k_loop
    
.rem_16_k_done:
    vmovups [r14], zmm0
    add     r14, 64
    add     r15, 16
    dec     rdx
    jnz     .rem_16_loop
    
.final_edge:
    ; Handle final 1-15 columns with mask
    test    r8, r8
    jz      .row_next
    
    ; Create mask - shift count must be in cl register
    mov     rax, 16
    sub     rax, r8
    mov     rcx, 1
    ; Save r8 to use cl for shift
    push    r8
    mov     cl, r8b                      ; Move low byte of r8 to cl
    shl     rcx, cl                      ; Shift by cl (variable count requires cl)
    pop     r8
    dec     rcx                          ; k1 = (1 << r8) - 1
    kmovd   k1, ecx                      ; Use kmovd for 32-bit operand
    
    vxorps  zmm0, zmm0, zmm0
    mov     rbx, [rbp - 128]
    mov     r9, [rbp - 104]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.final_k_loop:
    test    r9, r9
    jz      .final_k_done
    
    vbroadcastss zmm4, [rbx]
    vmovups zmm5{k1}{z}, [rcx]
    vfmadd231ps zmm0, zmm4, zmm5
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .final_k_loop
    
.final_k_done:
    vmovups [r14]{k1}, zmm0
    
.row_next:
    inc     r13
    jmp     .row_loop
    
.done:
    ;--------------------------------------------------------------------------
    ; EPILOGUE
    ;--------------------------------------------------------------------------
    vzeroupper
    
%ifdef WIN64
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
    add     rsp, 312
    pop     rsi
    pop     rdi
%else
    add     rsp, 128
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
