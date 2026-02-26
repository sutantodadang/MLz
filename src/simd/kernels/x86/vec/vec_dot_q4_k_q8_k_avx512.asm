;; =============================================================================
;; vec_dot_q4_k_q8_k_avx512.asm - Deeply Optimized Q4_K x Q8_K (AVX-512)
;; =============================================================================

bits 64
default rel

section .data
    align 64
    mask_low4_64:   times 64 db 0x0F
    ones_16_64:     times 32 dw 1
    
    ; Shuffles for unpacking scales (128-bit)
    shuf_d_base:    db 0,1,2,3, 8,9,10,11, 0,0,0,0,0,0,0,0
    shuf_d_high:    db 0,0,0,0, 0,1,2,3,   0,0,0,0,0,0,0,0
    shuf_m_base:    db 4,5,6,7, 8,9,10,11, 0,0,0,0,0,0,0,0
    shuf_m_high:    db 4,5,6,7, 4,5,6,7,   0,0,0,0,0,0,0,0
    
    mask_6bit:      times 8 dw 0x3F
    mask_4bit:      times 8 dw 0x0F
    
    ; 128-bit Constants
    mask_0F_128:    dq 0x000F000F000F000F, 0
    mask_3F_128:    dq 0x003F003F003F003F, 0
    mask_FF_128:    dq 0xFFFFFFFFFFFFFFFF, 0
    
    ; Permutation indices (512-bit full)
    perm_0_1:       dd 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
    perm_2_3:       dd 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
    perm_4_5:       dd 4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
    perm_6_7:       dd 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0

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

global simd_vec_dot_q4_k_q8_k_avx512

simd_vec_dot_q4_k_q8_k_avx512:
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
    shr     r10d, 8                 ; n / 256
    mov     r11, ARG3               ; vx (Q4_K)
    mov     r12, ARG4               ; vy (Q8_K)
    mov     r13, ARG2               ; result

    vxorps  zmm15, zmm15, zmm15     ; Accumulator
    
    ; Load Constants
    vmovdqu64 zmm14, [rel mask_low4_64]
    vmovdqu64 zmm13, [rel ones_16_64]
    
    align 64
.main_loop:
    ; ==========================================================================
    ; STEP 1: VECTORIZED SCALE UNPACKING
    ; ==========================================================================
    vmovdqu xmm0, [r11 + 4]
    
    ; --- Unpack D ---
    vmovdqu    xmm8, [rel shuf_d_base] ; Use low xmm8
    vpshufb    xmm1, xmm0, xmm8
    vpmovzxbw  xmm1, xmm1
    
    vmovdqu    xmm8, [rel shuf_d_high]
    vpshufb    xmm2, xmm0, xmm8
    vpmovzxbw  xmm2, xmm2
    
    vmovdqu     xmm4, [rel mask_0F_128]
    vmovdqu     xmm5, [rel mask_3F_128]
    vpunpcklqdq xmm4, xmm5, xmm4
    
    vpand      xmm1, xmm1, xmm4
    vpsrlw     xmm2, xmm2, 6
    vpsllw     xmm2, xmm2, 4
    
    vmovdqu    xmm5, [rel mask_FF_128]
    vpslldq    xmm5, xmm5, 8
    vpand      xmm2, xmm2, xmm5
    vpor       xmm1, xmm1, xmm2     ; D_combined
    
    ; --- Unpack M ---
    vmovdqu    xmm8, [rel shuf_m_base]
    vpshufb    xmm2, xmm0, xmm8
    vpmovzxbw  xmm2, xmm2
    
    vmovdqu    xmm8, [rel shuf_m_high]
    vpshufb    xmm3, xmm0, xmm8
    vpmovzxbw  xmm3, xmm3
    
    vmovdqu    xmm8, [rel mask_6bit]
    vpand      xmm5, xmm2, xmm8
    vpsrlw     xmm6, xmm2, 4
    vpblendd   xmm2, xmm5, xmm6, 0xF0
    
    vpsrlw     xmm3, xmm3, 6
    vpsllw     xmm3, xmm3, 4
    vmovdqu    xmm5, [rel mask_FF_128]
    vpslldq    xmm5, xmm5, 8
    vpand      xmm3, xmm3, xmm5
    vpor       xmm2, xmm2, xmm3     ; M_combined
    
    ; Convert to floats
    vpmovzxwd ymm10, xmm1
    vcvtdq2ps ymm10, ymm10
    
    vpmovzxwd ymm11, xmm2
    vcvtdq2ps ymm11, ymm11
    
    ; Apply Superblock Scales
    vmovd   xmm0, [r11]
    vcvtph2ps xmm0, xmm0
    vshufps xmm1, xmm0, xmm0, 0x00
    vshufps xmm2, xmm0, xmm0, 0x55
    vpbroadcastd ymm1, xmm1
    vpbroadcastd ymm2, xmm2
    
    vmulps ymm10, ymm10, ymm1       ; Final D (8 floats)
    vmulps ymm11, ymm11, ymm2       ; Final M (8 floats)
    
    ; --- Pre-calculate Sums ---
    vmovdqu ymm0, [r12 + 260]
    vphaddw ymm0, ymm0, ymm0
    vpmaddwd ymm0, ymm0, ymm13
    vcvtdq2ps ymm12, ymm0           ; Sums (8 floats)
    
    ; ==========================================================================
    ; STEP 2: LOOP (4 Iterations)
    ; ==========================================================================
    
    ; Iteration 0
    vmovdqu64 zmm5, [rel perm_0_1]
    vpermd    zmm1, zmm5, zmm10     ; Use vpermd
    vpermd    zmm2, zmm5, zmm11     ; Use vpermd
    vpermd    zmm3, zmm5, zmm12     ; Use vpermd
    
    vmovdqu   ymm6, [r11 + 16]      ; Load 64 weights
    vpsrlw    ymm7, ymm6, 4
    vpand     ymm6, ymm6, ymm14
    vpunpcklbw ymm8, ymm6, ymm7
    vpunpckhbw ymm9, ymm6, ymm7
    
    ; USER FIX START
    ; 1. Chunk0 -> zmm6[0] (xmm8 is low 128 of ymm8)
    vinserti32x4 zmm6, zmm6, xmm8, 0
    ; 2. Chunk1 -> zmm6[1] (xmm9 is low 128 of ymm9)
    vinserti32x4 zmm6, zmm6, xmm9, 1
    ; 3. Chunk2 -> zmm6[2]
    vextracti32x4 xmm30, ymm8, 1
    vinserti32x4 zmm6, zmm6, xmm30, 2
    ; 4. Chunk3 -> zmm6[3]
    vextracti32x4 xmm30, ymm9, 1
    vinserti32x4 zmm6, zmm6, xmm30, 3
    ; USER FIX END
    
    vmovdqu64 zmm7, [r12 + 4]
    vpmaddubsw zmm6, zmm6, zmm7
    vpmaddwd   zmm6, zmm6, zmm13
    vcvtdq2ps  zmm6, zmm6
    
    vfmadd231ps zmm15, zmm6, zmm1
    vfnmadd231ps zmm15, zmm3, zmm2
    
    ; Iteration 1
    vmovdqu64 zmm5, [rel perm_2_3]
    vpermd    zmm1, zmm5, zmm10
    vpermd    zmm2, zmm5, zmm11
    vpermd    zmm3, zmm5, zmm12
    
    vmovdqu   ymm6, [r11 + 16 + 32]
    vpsrlw    ymm7, ymm6, 4
    vpand     ymm6, ymm6, ymm14
    vpunpcklbw ymm8, ymm6, ymm7
    vpunpckhbw ymm9, ymm6, ymm7
    
    vinserti32x4 zmm6, zmm6, xmm8, 0
    vinserti32x4 zmm6, zmm6, xmm9, 1
    vextracti32x4 xmm30, ymm8, 1
    vinserti32x4 zmm6, zmm6, xmm30, 2
    vextracti32x4 xmm30, ymm9, 1
    vinserti32x4 zmm6, zmm6, xmm30, 3
    
    vmovdqu64 zmm7, [r12 + 4 + 64]
    vpmaddubsw zmm6, zmm6, zmm7
    vpmaddwd   zmm6, zmm6, zmm13
    vcvtdq2ps  zmm6, zmm6
    
    vfmadd231ps zmm15, zmm6, zmm1
    vfnmadd231ps zmm15, zmm3, zmm2
    
    ; Iteration 2
    vmovdqu64 zmm5, [rel perm_4_5]
    vpermd    zmm1, zmm5, zmm10
    vpermd    zmm2, zmm5, zmm11
    vpermd    zmm3, zmm5, zmm12
    
    vmovdqu   ymm6, [r11 + 16 + 64]
    vpsrlw    ymm7, ymm6, 4
    vpand     ymm6, ymm6, ymm14
    vpunpcklbw ymm8, ymm6, ymm7
    vpunpckhbw ymm9, ymm6, ymm7
    
    vinserti32x4 zmm6, zmm6, xmm8, 0
    vinserti32x4 zmm6, zmm6, xmm9, 1
    vextracti32x4 xmm30, ymm8, 1
    vinserti32x4 zmm6, zmm6, xmm30, 2
    vextracti32x4 xmm30, ymm9, 1
    vinserti32x4 zmm6, zmm6, xmm30, 3
    
    vmovdqu64 zmm7, [r12 + 4 + 128]
    vpmaddubsw zmm6, zmm6, zmm7
    vpmaddwd   zmm6, zmm6, zmm13
    vcvtdq2ps  zmm6, zmm6
    
    vfmadd231ps zmm15, zmm6, zmm1
    vfnmadd231ps zmm15, zmm3, zmm2
    
    ; Iteration 3
    vmovdqu64 zmm5, [rel perm_6_7]
    vpermd    zmm1, zmm5, zmm10
    vpermd    zmm2, zmm5, zmm11
    vpermd    zmm3, zmm5, zmm12
    
    vmovdqu   ymm6, [r11 + 16 + 96]
    vpsrlw    ymm7, ymm6, 4
    vpand     ymm6, ymm6, ymm14
    vpunpcklbw ymm8, ymm6, ymm7
    vpunpckhbw ymm9, ymm6, ymm7
    
    vinserti32x4 zmm6, zmm6, xmm8, 0
    vinserti32x4 zmm6, zmm6, xmm9, 1
    vextracti32x4 xmm30, ymm8, 1
    vinserti32x4 zmm6, zmm6, xmm30, 2
    vextracti32x4 xmm30, ymm9, 1
    vinserti32x4 zmm6, zmm6, xmm30, 3
    
    vmovdqu64 zmm7, [r12 + 4 + 192]
    vpmaddubsw zmm6, zmm6, zmm7
    vpmaddwd   zmm6, zmm6, zmm13
    vcvtdq2ps  zmm6, zmm6
    
    vfmadd231ps zmm15, zmm6, zmm1
    vfnmadd231ps zmm15, zmm3, zmm2
    
    ; Next block
    add r11, 144
    add r12, 292
    dec r10d
    jnz .main_loop
    
.horizontal_sum:
    vextractf32x8 ymm0, zmm15, 1
    vaddps    ymm15, ymm15, ymm0
    vextractf128 xmm0, ymm15, 1
    vaddps    xmm0, xmm0, xmm15
    vhaddps   xmm0, xmm0, xmm0
    vhaddps   xmm0, xmm0, xmm0
    vmovss    [r13], xmm0

.epilogue:
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
