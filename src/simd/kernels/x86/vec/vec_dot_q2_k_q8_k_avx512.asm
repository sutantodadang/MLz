;; =============================================================================
;; vec_dot_q2_k_q8_k_avx512.asm — Conservative AVX-512 port; per-block algorithm
;; identical to vec_dot_q2_k_q8_k_avx2.asm.  Constants live in zmm regs.
;; =============================================================================

section .data
    align 64
    q2k_mask_03_z:  times 64 db 0x03
    q2k_mask_lo4_z: times 64 db 0x0F

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

%define BS_Q2_K 84
%define BS_Q8_K 292
%define D_OFF       80
%define DMIN_OFF    82
%define BSUMS_OFF   260

%macro Q2K_SUBDOT 3
    vpmovzxbw    ymm0, %1
    vpmovsxbw    ymm1, %2
    vpmaddwd     ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1
    vphaddd      xmm0, xmm0, xmm0
    vphaddd      xmm0, xmm0, xmm0
    vmovd        eax, xmm0
    imul         eax, %3
    add          r14d, eax
%endmacro

%macro Q2K_SHIFT 3
    %if %1 == 0
        vmovdqa  ymm7, ymm6
    %else
        vpsrlw   ymm7, ymm6, %1
    %endif
    vpand        ymm7, ymm7, ymm10

    vmovdqu      ymm8, [r12 + 4 + (%2)]

    movzx        ebx, byte [r11 + (%3)]
    and          ebx, 0x0F
    movzx        ecx, byte [r11 + (%3) + 1]
    and          ecx, 0x0F

    Q2K_SUBDOT   xmm7, xmm8, ebx

    vextracti128 xmm12, ymm7, 1
    vextracti128 xmm13, ymm8, 1
    Q2K_SUBDOT   xmm12, xmm13, ecx
%endmacro

global simd_vec_dot_q2_k_q8_k_avx512

simd_vec_dot_q2_k_q8_k_avx512:
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
    shr     r10d, 8
    mov     r13, ARG2
    vxorps  xmm15, xmm15, xmm15
    test    r10d, r10d
    jz      .write_result

    mov     r11, ARG3
    mov     r12, ARG4

    vmovdqa64 zmm10, [rel q2k_mask_03_z]
    vmovdqa64 zmm11, [rel q2k_mask_lo4_z]

.main_loop:
    vmovss      xmm14, [r12]

    vmovdqu     xmm0, [r11]
    vpsrlw      xmm0, xmm0, 4
    vpand       xmm0, xmm0, xmm11
    vpmovzxbw   ymm0, xmm0
    vmovdqu     ymm1, [r12 + BSUMS_OFF]
    vpmaddwd    ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddd      xmm0, xmm0, xmm1
    vphaddd     xmm0, xmm0, xmm0
    vphaddd     xmm0, xmm0, xmm0
    vmovd       r15d, xmm0

    xor         r14d, r14d

    vmovdqu     ymm6, [r11 + 16 + 0*32]
    Q2K_SHIFT   0,   0, 0
    Q2K_SHIFT   2,  32, 2
    Q2K_SHIFT   4,  64, 4
    Q2K_SHIFT   6,  96, 6

    vmovdqu     ymm6, [r11 + 16 + 1*32]
    Q2K_SHIFT   0, 128,  8
    Q2K_SHIFT   2, 160, 10
    Q2K_SHIFT   4, 192, 12
    Q2K_SHIFT   6, 224, 14

    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0
    vmulss      xmm0, xmm0, xmm14
    vcvtsi2ss   xmm1, xmm1, r14d
    vmulss      xmm0, xmm0, xmm1

    vmovd       xmm2, dword [r11 + DMIN_OFF]
    vcvtph2ps   xmm2, xmm2
    vmulss      xmm2, xmm2, xmm14
    vcvtsi2ss   xmm3, xmm3, r15d
    vmulss      xmm2, xmm2, xmm3

    vsubss      xmm0, xmm0, xmm2
    vaddss      xmm15, xmm15, xmm0

    add         r11, BS_Q2_K
    add         r12, BS_Q8_K
    dec         r10d
    jnz         .main_loop

.write_result:
    vmovss      [r13], xmm15

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
