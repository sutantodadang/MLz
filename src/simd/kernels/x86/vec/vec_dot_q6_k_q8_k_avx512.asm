;; =============================================================================
;; vec_dot_q6_k_q8_k_avx512.asm - Q6_K x Q8_K dot product kernel (AVX-512)
;; =============================================================================

section .data
    align 64
    mask_low4: times 64 db 0x0F
    val_32: times 64 db 32
    
    ; Shuffle masks for scales (same as Q2_K)
    mask_scale_0123:
        times 16 db 0
        times 16 db 1
        times 16 db 2
        times 16 db 3
    
    mask_scale_4567:
        times 16 db 4
        times 16 db 5
        times 16 db 6
        times 16 db 7
        
    mask_scale_89AB:
        times 16 db 8
        times 16 db 9
        times 16 db 10
        times 16 db 11
        
    mask_scale_CDEF:
        times 16 db 12
        times 16 db 13
        times 16 db 14
        times 16 db 15
        
    val_neg_32_f: times 16 dd -32.0
    ones_16: times 32 dw 1

section .text

%ifdef WINDOWS
    %define ARG1   rcx
    %define ARG1_32 ecx
    %define ARG2   rdx
    %define ARG3   r8
    %define ARG4   r9
%else
    %define ARG1   rdi
    %define ARG1_32 edi
    %define ARG2   rsi
    %define ARG3   rdx
    %define ARG4   rcx
%endif

global simd_vec_dot_q6_k_q8_k_avx512

simd_vec_dot_q6_k_q8_k_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    sub     rsp, 168
    vmovups [rsp+0],   xmm6
    vmovups [rsp+16],  xmm7
    vmovups [rsp+32],  xmm8
    vmovups [rsp+48],  xmm9
    vmovups [rsp+64],  xmm10
    vmovups [rsp+80],  xmm11
    vmovups [rsp+96],  xmm12
    vmovups [rsp+112], xmm13
    vmovups [rsp+128], xmm14
    vmovups [rsp+144], xmm15
%endif

    mov     r10d, ARG1_32
    shr     r10d, 8
    test    r10d, r10d
    jz      .done_zero

    mov     r11, ARG3
    mov     r12, ARG4
    mov     r13, ARG2

    vxorps  zmm31, zmm31, zmm31     ; Accumulator

    ; Load Constants
    vmovdqa64 zmm16, [rel mask_low4]
    vmovdqa64 zmm17, [rel val_32]
    vmovdqa64 zmm19, [rel ones_16]
    
    ; Load shuffle masks
    vmovdqa64 zmm20, [rel mask_scale_0123]
    vmovdqa64 zmm21, [rel mask_scale_4567]
    vmovdqa64 zmm22, [rel mask_scale_89AB]
    vmovdqa64 zmm23, [rel mask_scale_CDEF]

.main_loop:
    ; --------------------------------------------------------------------------
    ; STEP 1: SCALES (d * d_y)
    ; --------------------------------------------------------------------------
    vmovd   xmm0, [r11 + 208]
    vcvtph2ps xmm0, xmm0
    vmovss  xmm1, [r12]
    vbroadcastss zmm1, xmm1
    
    vbroadcastss zmm2, xmm0         ; d (expanded)
    vmulps zmm2, zmm2, zmm1         ; Scale
    
    ; --------------------------------------------------------------------------
    ; STEP 2: MINS (-32 * s * bsum)
    ; --------------------------------------------------------------------------
    vmovdqu xmm4, [r11 + 192]       ; scales (16 bytes)
    vpmovsxbw ymm4, xmm4            ; 16x i16
    
    vmovdqu ymm5, [r12 + 260]       ; bsums (16x i16)
    vpmaddwd ymm6, ymm4, ymm5       ; i32 sum
    vcvtdq2ps ymm6, ymm6
    
    vbroadcastss zmm3, [rel val_neg_32_f]
    ; Fix vmulps size mismatch
    vmulps ymm6, ymm6, ymm3         ; * -32 (using ymm3 low part)
    
    ; Add to accumulator (low 256)
    vextractf64x4 ymm30, zmm31, 0
    vaddps ymm30, ymm30, ymm6
    vinsertf64x4 zmm31, zmm31, ymm30, 0
    
    ; --------------------------------------------------------------------------
    ; STEP 3: WEIGHTS (s * q * y)
    ; --------------------------------------------------------------------------
    xor r14d, r14d
    
.quarter_loop:
    ; Load ql (32 bytes)
    mov rax, r14
    shr rax, 1
    vmovdqu ymm0, [r11 + rax]   ; ql (32 bytes)
    
    ; Expand 32 bytes to 64 bytes (low/high nibbles)
    vpmovzxbw zmm0, ymm0    ; 32 bytes -> 64 bytes (00XX)
    
    vpandq zmm4, zmm0, zmm16 ; Low nibbles (mask 0x0F) - Use vpandq
    
    vpsrlw zmm5, zmm0, 4
    vpandq zmm5, zmm5, zmm16 ; High nibbles - Use vpandq
    
    vpsllw zmm5, zmm5, 8     ; Move high nibbles to high byte
    vporq zmm4, zmm4, zmm5   ; Combine [0H 0L]
    
    vpsubb zmm4, zmm4, zmm17 ; q - 32
    
    ; Load y (64 bytes)
    lea rbx, [r12 + 4]
    add rbx, r14
    vmovdqu64 zmm8, [rbx]
    
    ; Dot product
    vpmaddubsw zmm4, zmm4, zmm8
    
    ; Scales
    lea rbx, [rel mask_scale_0123]
    vmovdqa64 zmm12, [rbx + r14]
    
    ; Load scales
    vmovdqu xmm10, [r11 + 192]
    ; Broadcast scales using vshuff32x4
    vshuff32x4 zmm10, zmm10, zmm10, 0x00
    
    vpshufb zmm13, zmm10, zmm12     ; Expanded scales
    vpmovsxbw zmm13, ymm13          ; Scales are signed i8
    
    vpmullw zmm4, zmm4, zmm13
    
    vpmaddwd zmm4, zmm4, zmm19
    vcvtdq2ps zmm4, zmm4
    
    vaddps zmm31, zmm31, zmm4
    
    add r14, 64
    cmp r14, 256
    jl .quarter_loop

    vmulps zmm31, zmm31, zmm2       ; Acc *= Scale
    
    add r11, 210
    add r12, 292
    dec r10d
    jnz .main_loop
    
.horizontal_sum:
    vextractf64x4 ymm0, zmm31, 1
    vaddps    ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps    xmm0, xmm0, xmm1
    vhaddps   xmm0, xmm0, xmm0
    vhaddps   xmm0, xmm0, xmm0
    vmovss    [r13], xmm0

.done_zero:
    vzeroupper
%ifdef WINDOWS
    vmovups xmm6,  [rsp+0]
    vmovups xmm7,  [rsp+16]
    vmovups xmm8,  [rsp+32]
    vmovups xmm9,  [rsp+48]
    vmovups xmm10, [rsp+64]
    vmovups xmm11, [rsp+80]
    vmovups xmm12, [rsp+96]
    vmovups xmm13, [rsp+112]
    vmovups xmm14, [rsp+128]
    vmovups xmm15, [rsp+144]
    add     rsp, 168
%endif
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
