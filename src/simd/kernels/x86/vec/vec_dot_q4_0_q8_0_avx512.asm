;; =============================================================================
;; vec_dot_q4_0_q8_0_avx512.asm — Conservative AVX-512 port; per-block algorithm
;; identical to vec_dot_q4_0_q8_0_avx2.asm.  Constants live in zmm regs.
;; =============================================================================

section .data
    align 64
    q40_mask_lo4_z: times 64 db 0x0F
    q40_const_8_z:  times 64 db 8

section .text

%ifdef WINDOWS
    %define ARG1    rcx
    %define ARG1_32 ecx
    %define ARG2    rdx
    %define ARG3    r8
    %define ARG4    r9
%else
    %define ARG1    rdi
    %define ARG1_32 edi
    %define ARG2    rsi
    %define ARG3    rdx
    %define ARG4    rcx
%endif

%define BS_Q4_0 18
%define BS_Q8_0 34

global simd_vec_dot_q4_0_q8_0_avx512

simd_vec_dot_q4_0_q8_0_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    sub     rsp, 184
    vmovdqu [rsp +   0], xmm6
    vmovdqu [rsp +  16], xmm7
    vmovdqu [rsp +  32], xmm8
    vmovdqu [rsp +  48], xmm9
    vmovdqu [rsp +  64], xmm10
    vmovdqu [rsp +  80], xmm11
    vmovdqu [rsp +  96], xmm12
    vmovdqu [rsp + 112], xmm13
    vmovdqu [rsp + 128], xmm14
    vmovdqu [rsp + 144], xmm15
%else
    sub     rsp, 184
%endif

    mov     r10d, ARG1_32
    shr     r10d, 5
    mov     r13, ARG2
    vxorps  xmm15, xmm15, xmm15
    test    r10d, r10d
    jz      .write_result

    mov     r11, ARG3
    mov     r12, ARG4

    vmovdqa64 zmm10, [rel q40_mask_lo4_z]
    vmovdqa64 zmm11, [rel q40_const_8_z]

.main_loop:
    vmovdqu      xmm0, [r11 + 2]
    vpand        xmm1, xmm0, xmm10
    vpsrlw       xmm2, xmm0, 4
    vpand        xmm2, xmm2, xmm10
    vinserti128  ymm6, ymm1, xmm2, 1
    vpsubb       ymm6, ymm6, ymm11

    vmovdqu      ymm7, [r12 + 2]

    vpmovsxbw    ymm0, xmm6
    vextracti128 xmm1, ymm6, 1
    vpmovsxbw    ymm1, xmm1

    vpmovsxbw    ymm2, xmm7
    vextracti128 xmm3, ymm7, 1
    vpmovsxbw    ymm3, xmm3

    vpmaddwd     ymm0, ymm0, ymm2
    vpmaddwd     ymm1, ymm1, ymm3
    vpaddd       ymm0, ymm0, ymm1

    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1
    vphaddd      xmm0, xmm0, xmm0
    vphaddd      xmm0, xmm0, xmm0
    vmovd        eax, xmm0

    vmovd        xmm2, dword [r11]
    vcvtph2ps    xmm2, xmm2
    vmovd        xmm3, dword [r12]
    vcvtph2ps    xmm3, xmm3
    vmulss       xmm2, xmm2, xmm3

    vcvtsi2ss    xmm0, xmm0, eax
    vmulss       xmm0, xmm0, xmm2
    vaddss       xmm15, xmm15, xmm0

    add          r11, BS_Q4_0
    add          r12, BS_Q8_0
    dec          r10d
    jnz          .main_loop

.write_result:
    vmovss       [r13], xmm15

.done:
    vzeroupper
%ifdef WINDOWS
    vmovdqu xmm6,  [rsp +   0]
    vmovdqu xmm7,  [rsp +  16]
    vmovdqu xmm8,  [rsp +  32]
    vmovdqu xmm9,  [rsp +  48]
    vmovdqu xmm10, [rsp +  64]
    vmovdqu xmm11, [rsp +  80]
    vmovdqu xmm12, [rsp +  96]
    vmovdqu xmm13, [rsp + 112]
    vmovdqu xmm14, [rsp + 128]
    vmovdqu xmm15, [rsp + 144]
%endif
    add     rsp, 184
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    pop     rbp
    ret
