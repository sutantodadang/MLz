;; =============================================================================
;; vec_dot_q2_k_q8_k_avx512.asm - Optimized Q2_K x Q8_K dot product kernel (2-Block Unroll)
;; =============================================================================

section .data
    align 64
    mask_2bit:   times 64 db 0x03
    mask_low4:   times 64 db 0x0F
    ones_16:     times 32 dw 1
    
    align 16
    shuf_mask_scale:
        db 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3

section .text

%ifdef AVX512_ENABLED
global simd_vec_dot_q2_k_q8_k_avx512

;; Calling Convention
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

simd_vec_dot_q2_k_q8_k_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    sub     rsp, 168
    vmovdqu64 [rsp+0],   xmm6
    vmovdqu64 [rsp+16],  xmm7
    vmovdqu64 [rsp+32],  xmm8
    vmovdqu64 [rsp+48],  xmm9
    vmovdqu64 [rsp+64],  xmm10
    vmovdqu64 [rsp+80],  xmm11
    vmovdqu64 [rsp+96],  xmm12
    vmovdqu64 [rsp+112], xmm13
    vmovdqu64 [rsp+128], xmm14
    vmovdqu64 [rsp+144], xmm15
%endif

    ; Setup
    mov     r10d, ARG1_32
    shr     r10d, 8                 ; n / 256 = num blocks
    test    r10d, r10d
    jz      .done_zero

    mov     r11, ARG3               ; vx (Q2_K)
    mov     r12, ARG4               ; vy (Q8_K)
    mov     r13, ARG2               ; result

    vxorps  zmm31, zmm31, zmm31     ; Main Accumulator (Integer)
    vxorps  zmm26, zmm26, zmm26     ; M_term Accumulator (Float)

    ; Constants
    vmovdqa64 zmm30, [rel mask_2bit]
    vmovdqa64 zmm29, [rel mask_low4]
    vmovdqa64 zmm28, [rel ones_16]
    vmovdqa64 zmm27, [rel shuf_mask_scale] ; Load shuffle mask to ZMM (we use xmm part)

    ; Loop
.main_loop:
    ; ==========================================================================
    ; STEP 1: PREPARE SCALES (D, Dmin, dy)
    ; ==========================================================================
    ; Load d, dmin (Offset 80 in Q2)
    vmovd   xmm0, [r11 + 80]
    vcvtph2ps xmm0, xmm0            ; [?, ?, dmin, d]
    
    ; Load dy (Offset 0 in Q8)
    vmovss  xmm1, [r12]
    vbroadcastss zmm1, xmm1
    
    ; Broadcast D and Dmin
    vshufps xmm2, xmm0, xmm0, 0x00  ; d
    vshufps xmm3, xmm0, xmm0, 0x55  ; dmin
    vbroadcastss zmm2, xmm2         ; D
    vbroadcastss zmm3, xmm3         ; Dmin
    
    vmulps  zmm2, zmm2, zmm1        ; Scale = D * dy
    vmulps  zmm3, zmm3, zmm1        ; MinScale = Dmin * dy

    ; ==========================================================================
    ; STEP 2: MINS TERM (m * bsums)
    ; ==========================================================================
    ; Load scales (16 bytes)
    vmovdqu xmm4, [r11]
    
    ; m = scales >> 4
    vpsrlw  xmm5, xmm4, 4
    vpandd  xmm5, xmm5, xmm29       ; m (16x u8) - Use XMM29 (low 128 of ZMM29)
    vpmovzxbw ymm5, xmm5            ; m (16x i16) - Use YMM
    
    vmovdqu ymm6, [r12 + 260]       ; bsums (16x i16)
    vpmaddwd ymm5, ymm5, ymm6       ; [sum0..sum7] (8x i32)
    vcvtdq2ps ymm5, ymm5
    
    ; Sum the partials in YMM5
    vextractf128 xmm6, ymm5, 1
    vaddps    xmm5, xmm5, xmm6
    vhaddps   xmm5, xmm5, xmm5
    vhaddps   xmm5, xmm5, xmm5      ; xmm5[0] = Total M Sum
    
    vbroadcastss zmm5, xmm5
    vmulps    zmm5, zmm5, zmm3      ; M_term = Sum * MinScale
    
    vaddps    zmm26, zmm26, zmm5    ; Accumulate M_term (Float)

    ; ==========================================================================
    ; STEP 3: WEIGHTS TERM (s * q * y)
    ; ==========================================================================
    ; scales (s) = scales & 0x0F
    vpandd  xmm4, xmm4, xmm29       ; s (16x u8)
    
    ; Prepare Scales Broadcast
    ; xmm4 has [s0..s15]
    ; xmm27 has shuffle mask
    
    ; S_Stream0 (s0..s3)
    vpshufb   xmm10, xmm4, xmm27    ; Broadcast s0..s3
    vpmovzxbw ymm10, xmm10          ; Expand to i16 (YMM)
    vpmovzxwd zmm10, ymm10          ; Expand to i32 (ZMM)
    
    ; S_Stream1 (s4..s7)
    vpsrldq   xmm5, xmm4, 4
    vpshufb   xmm11, xmm5, xmm27
    vpmovzxbw ymm11, xmm11
    vpmovzxwd zmm11, ymm11
    
    ; S_Stream2 (s8..s11)
    vpsrldq   xmm5, xmm5, 4
    vpshufb   xmm12, xmm5, xmm27
    vpmovzxbw ymm12, xmm12
    vpmovzxwd zmm12, ymm12
    
    ; S_Stream3 (s12..s15)
    vpsrldq   xmm5, xmm5, 4
    vpshufb   xmm13, xmm5, xmm27
    vpmovzxbw ymm13, xmm13
    vpmovzxwd zmm13, ymm13
    
    ; Load QS (64 bytes)
    vmovdqu64 zmm0, [r11 + 16]      ; qs (64 bytes)
    
    ; Extract Streams
    vpandd    zmm1, zmm0, zmm30     ; Stream 0
    vpsrlw    zmm2, zmm0, 2
    vpandd    zmm2, zmm2, zmm30     ; Stream 1
    vpsrlw    zmm3, zmm0, 4
    vpandd    zmm3, zmm3, zmm30     ; Stream 2
    vpsrlw    zmm4, zmm0, 6
    vpandd    zmm4, zmm4, zmm30     ; Stream 3
    
    ; Load Y (256 bytes)
    vmovdqu64 zmm5, [r12 + 4]       ; y[0..63]
    vmovdqu64 zmm6, [r12 + 68]      ; y[64..127]
    vmovdqu64 zmm7, [r12 + 132]     ; y[128..191]
    vmovdqu64 zmm8, [r12 + 196]     ; y[192..255]
    
    ; Dot Products (u8 * i8)
    vpmaddubsw zmm1, zmm1, zmm5     ; 32x i16
    vpmaddubsw zmm2, zmm2, zmm6
    vpmaddubsw zmm3, zmm3, zmm7
    vpmaddubsw zmm4, zmm4, zmm8
    
    ; Horizontal Sum (i16 -> i32)
    vpmaddwd  zmm1, zmm1, zmm28     ; 16x i32
    vpmaddwd  zmm2, zmm2, zmm28
    vpmaddwd  zmm3, zmm3, zmm28
    vpmaddwd  zmm4, zmm4, zmm28
    
    ; Apply Scales (Multiply i32 sums by i32 scales)
    vpmulld   zmm1, zmm1, zmm10
    vpmulld   zmm2, zmm2, zmm11
    vpmulld   zmm3, zmm3, zmm12
    vpmulld   zmm4, zmm4, zmm13
    
    ; Accumulate (Integer)
    vpaddd    zmm31, zmm31, zmm1
    vpaddd    zmm31, zmm31, zmm2
    vpaddd    zmm31, zmm31, zmm3
    vpaddd    zmm31, zmm31, zmm4
    
    ; Loop End
    add     r11, 84
    add     r12, 292
    dec     r10d
    jnz     .main_loop

    ; Convert Integer Sums to Float
    vcvtdq2ps zmm31, zmm31

    ; Scale by D
    vmulps  zmm31, zmm31, zmm2      ; Total * D
    
    ; Add M Term
    vaddps  zmm31, zmm31, zmm26     ; Total + M_term
    
    ; Horizontal Reduce
    vextractf64x4 ymm0, zmm31, 1
    vaddps    ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps    xmm0, xmm0, xmm1
    vhaddps   xmm0, xmm0, xmm0
    vhaddps   xmm0, xmm0, xmm0
    vmovss    [r13], xmm0

.epilogue:
    vzeroupper
%ifdef WINDOWS
    vmovdqu64 xmm6,  [rsp+0]
    vmovdqu64 xmm7,  [rsp+16]
    vmovdqu64 xmm8,  [rsp+32]
    vmovdqu64 xmm9,  [rsp+48]
    vmovdqu64 xmm10, [rsp+64]
    vmovdqu64 xmm11, [rsp+80]
    vmovdqu64 xmm12, [rsp+96]
    vmovdqu64 xmm13, [rsp+112]
    vmovdqu64 xmm14, [rsp+128]
    vmovdqu64 xmm15, [rsp+144]
    add     rsp, 168
%endif
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

.done_zero:
    vxorps xmm0, xmm0, xmm0
    vmovss [r13], xmm0
    jmp .epilogue

%endif
