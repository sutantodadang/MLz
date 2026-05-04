; ----------------------------------------------------------------------------
; rms_norm_f32_avx512.asm  --  Single-row RMS norm (AVX-512F)
; ----------------------------------------------------------------------------
;
; Same contract as the AVX2 variant; processes 16 f32 / iter.
; Numerical strategy: parallel f64 accumulator (8 lanes) for the sum-of-squares
; using vcvtps2pd zmm, ymm halves; matches AVX2 ULP envelope (<= 2 ULP).
; ----------------------------------------------------------------------------

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_EPS  xmm1
    %define ARG_X    r8
    %define ARG_Y    r9
%else
    %define ARG_N    rdi
    %define ARG_EPS  xmm0
    %define ARG_X    rsi
    %define ARG_Y    rdx
%endif

section .text
global simd_rms_norm_f32_avx512

simd_rms_norm_f32_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168
    vmovdqu [rsp+ 0],  xmm6
    vmovdqu [rsp+16],  xmm7
    vmovdqu [rsp+32],  xmm8
    vmovdqu [rsp+48],  xmm9
    vmovdqu [rsp+64],  xmm10
    vmovdqu [rsp+80],  xmm11
    vmovdqu [rsp+96],  xmm12
    vmovdqu [rsp+112], xmm13
    vmovdqu [rsp+128], xmm14
    vmovdqu [rsp+144], xmm15

    vmovss   xmm15, xmm15, ARG_EPS
    mov      r14, ARG_N
    mov      r12, ARG_X
    mov      r13, ARG_Y

    ; --- Pass 1: sum of squares in 8 parallel f64 lanes (zmm0/zmm1) ---
    vpxorq   zmm0, zmm0, zmm0
    vpxorq   zmm1, zmm1, zmm1

    mov      rbx, r14
    shr      rbx, 4                 ; n / 16
    test     rbx, rbx
    jz       .pass1_tail

    mov      rax, r12

.pass1_vec:
    vmovups  zmm2, [rax]            ; 16 f32
    vmulps   zmm2, zmm2, zmm2       ; squares
    vextractf32x8 ymm3, zmm2, 1     ; hi 8
    vcvtps2pd zmm4, ymm2            ; lo 8 -> f64
    vcvtps2pd zmm5, ymm3            ; hi 8 -> f64
    vaddpd   zmm0, zmm0, zmm4
    vaddpd   zmm1, zmm1, zmm5
    add      rax, 64
    dec      rbx
    jnz      .pass1_vec

.pass1_tail:
    vaddpd   zmm0, zmm0, zmm1       ; combine
    ; horizontal-sum 8 f64 lanes -> xmm6
    vextractf64x4 ymm1, zmm0, 1
    vaddpd   ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddpd   xmm0, xmm0, xmm1
    vhaddpd  xmm0, xmm0, xmm0
    vmovapd  xmm6, xmm0

    ; Scalar tail (n % 16)
    mov      rax, r14
    and      rax, 15
    jz       .pass1_done

    mov      rcx, r14
    and      rcx, ~15
    lea      rdx, [r12 + rcx*4]

.pass1_tail_loop:
    vmovss   xmm2, [rdx]
    vmulss   xmm2, xmm2, xmm2
    vcvtss2sd xmm3, xmm3, xmm2
    vaddsd   xmm6, xmm6, xmm3
    add      rdx, 4
    dec      rax
    jnz      .pass1_tail_loop

.pass1_done:
    vcvtsi2sd xmm7, xmm7, r14
    vdivsd   xmm6, xmm6, xmm7
    vcvtsd2ss xmm6, xmm6, xmm6
    vaddss   xmm6, xmm6, xmm15
    vsqrtss  xmm6, xmm6, xmm6
    vmovss   xmm7, [rel rms_const_one_z]
    vdivss   xmm6, xmm7, xmm6
    vbroadcastss zmm7, xmm6

    ; --- Pass 2: y = x * scale (16-wide) ---
    mov      rbx, r14
    shr      rbx, 4
    mov      rax, r12
    mov      rdx, r13
    test     rbx, rbx
    jz       .pass2_tail

.pass2_vec:
    vmovups  zmm0, [rax]
    vmulps   zmm0, zmm0, zmm7
    vmovups  [rdx], zmm0
    add      rax, 64
    add      rdx, 64
    dec      rbx
    jnz      .pass2_vec

.pass2_tail:
    mov      rcx, r14
    and      rcx, 15
    jz       .epilogue

.pass2_tail_loop:
    vmovss   xmm0, [rax]
    vmulss   xmm0, xmm0, xmm6
    vmovss   [rdx], xmm0
    add      rax, 4
    add      rdx, 4
    dec      rcx
    jnz      .pass2_tail_loop

.epilogue:
    vmovdqu  xmm6,  [rsp+ 0]
    vmovdqu  xmm7,  [rsp+16]
    vmovdqu  xmm8,  [rsp+32]
    vmovdqu  xmm9,  [rsp+48]
    vmovdqu  xmm10, [rsp+64]
    vmovdqu  xmm11, [rsp+80]
    vmovdqu  xmm12, [rsp+96]
    vmovdqu  xmm13, [rsp+112]
    vmovdqu  xmm14, [rsp+128]
    vmovdqu  xmm15, [rsp+144]
    add      rsp, 168
    pop      r15
    pop      r14
    pop      r13
    pop      r12
    pop      rbx
    pop      rbp
    vzeroupper
    ret

section .rodata align=16
rms_const_one_z:  dd 0x3F800000
