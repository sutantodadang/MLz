;; =============================================================================
;; vec_dot_q6_k_q8_k_avx512.asm — Conservative AVX-512 port.  Per-sub-block
;; algorithm identical to vec_dot_q6_k_q8_k_avx2.asm; constants live in
;; zmm regs.  Validated by U1 against scalar reference.
;; =============================================================================

section .data
    align 64
    q6k_mask_lo4_z:  times 64 db 0x0F
    q6k_mask_03_z:   times 64 db 0x03
    q6k_const_32_z:  times 64 db 32

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

%define BS_Q6_K 210
%define BS_Q8_K 292
%define SCALES_OFF 192
%define D_OFF      208

%macro DECODE_Q6_QUARTER 4
    %ifidn %3, lo
        vpand    %1, %2, ymm10
    %else
        vpsrlw   %1, %2, 4
        vpand    %1, %1, ymm10
    %endif

    %if %4 == 0
        vmovdqa  ymm0, ymm8
    %else
        vpsrlw   ymm0, ymm8, %4
    %endif
    vpand    ymm0, ymm0, ymm11
    vpsllw   ymm0, ymm0, 4
    vpor     %1, %1, ymm0
    vpsubb   %1, %1, ymm12
%endmacro

%macro Q6K_SUBDOT 3
    vpmovsxbw    ymm0, %1
    vpmovsxbw    ymm1, [r12 + 4 + (%2)]
    vpmaddwd     ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1
    vphaddd      xmm0, xmm0, xmm0
    vphaddd      xmm0, xmm0, xmm0
    vmovd        eax, xmm0
    movsx        ebx, byte [r11 + SCALES_OFF + (%3)]
    imul         eax, ebx
    add          r14d, eax
%endmacro

%macro Q6K_QUARTER 5
    DECODE_Q6_QUARTER ymm9, %1, %2, %3
    Q6K_SUBDOT        xmm9, (%4),       (%5)
    vextracti128      xmm2, ymm9, 1
    Q6K_SUBDOT        xmm2, ((%4) + 16), ((%5) + 1)
%endmacro

global simd_vec_dot_q6_k_q8_k_avx512

simd_vec_dot_q6_k_q8_k_avx512:
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

    ;; broadcast 32 B from zmm constants into ymm working set
    vmovdqa64 zmm10, [rel q6k_mask_lo4_z]
    vmovdqa64 zmm11, [rel q6k_mask_03_z]
    vmovdqa64 zmm12, [rel q6k_const_32_z]

.main_loop:
    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0
    vmovss      xmm1, [r12]
    vmulss      xmm14, xmm0, xmm1

    xor         r14d, r14d

    vmovdqu     ymm6, [r11 +   0]
    vmovdqu     ymm7, [r11 +  32]
    vmovdqu     ymm8, [r11 + 128]

    Q6K_QUARTER ymm6, lo, 0,   0, 0
    Q6K_QUARTER ymm7, lo, 2,  32, 2
    Q6K_QUARTER ymm6, hi, 4,  64, 4
    Q6K_QUARTER ymm7, hi, 6,  96, 6

    vmovdqu     ymm6, [r11 +  64]
    vmovdqu     ymm7, [r11 +  96]
    vmovdqu     ymm8, [r11 + 160]

    Q6K_QUARTER ymm6, lo, 0, 128,  8
    Q6K_QUARTER ymm7, lo, 2, 160, 10
    Q6K_QUARTER ymm6, hi, 4, 192, 12
    Q6K_QUARTER ymm7, hi, 6, 224, 14

    vcvtsi2ss   xmm0, xmm0, r14d
    vmulss      xmm0, xmm0, xmm14
    vaddss      xmm15, xmm15, xmm0

    add         r11, BS_Q6_K
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
