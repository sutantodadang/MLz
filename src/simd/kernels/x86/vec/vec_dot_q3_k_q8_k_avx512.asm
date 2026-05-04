;; =============================================================================
;; vec_dot_q3_k_q8_k_avx512.asm — Conservative AVX-512 port; per-block algorithm
;; identical to vec_dot_q3_k_q8_k_avx2.asm.  Constants live in zmm regs.
;; =============================================================================

section .data
    align 64
    q3k_mask_03_z:  times 64 db 0x03
    q3k_const_4_z:  times 64 db 4

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

%define BS_Q3_K     110
%define BS_Q8_K     292
%define HMASK_OFF   0
%define QS_OFF      32
%define SCALES_RAW  96
%define D_OFF       108
%define SCALES_DECODED_OFF 160

%macro Q3K_SUBDOT 3
    vpmovsxbw    ymm0, %1
    vpmovsxbw    ymm1, [r12 + 4 + (%2)]
    vpmaddwd     ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1
    vphaddd      xmm0, xmm0, xmm0
    vphaddd      xmm0, xmm0, xmm0
    vmovd        eax, xmm0
    movzx        ebx, byte [rsp + SCALES_DECODED_OFF + (%3)]
    sub          ebx, 32
    imul         eax, ebx
    add          r14d, eax
%endmacro

%macro Q3K_SUBBLOCK 4
    %if %1 == 0
        vmovdqa  ymm7, ymm6
    %else
        vpsrlw   ymm7, ymm6, %1
    %endif
    vpand        ymm7, ymm7, ymm10

    mov          eax, (1 << %2)
    vmovd        xmm12, eax
    vpbroadcastb ymm12, xmm12

    vpand        ymm0, ymm8, ymm12
    vpcmpeqb     ymm0, ymm0, ymm13
    vpand        ymm0, ymm0, ymm11
    vpsubb       ymm7, ymm7, ymm0

    Q3K_SUBDOT   xmm7, (%3),       (%4)
    vextracti128 xmm14, ymm7, 1
    Q3K_SUBDOT   xmm14, ((%3) + 16), ((%4) + 1)
%endmacro

global simd_vec_dot_q3_k_q8_k_avx512

simd_vec_dot_q3_k_q8_k_avx512:
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

    vmovdqa64 zmm10, [rel q3k_mask_03_z]
    vmovdqa64 zmm11, [rel q3k_const_4_z]
    vpxorq    zmm13, zmm13, zmm13

.main_loop:
    mov     eax, [r11 + SCALES_RAW + 0]
    mov     ebx, [r11 + SCALES_RAW + 4]
    mov     edx, [r11 + SCALES_RAW + 8]

    mov     ecx, edx
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, eax
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 0], esi

    mov     ecx, edx
    shr     ecx, 2
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, ebx
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 4], esi

    mov     ecx, edx
    shr     ecx, 4
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, eax
    shr     esi, 4
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 8], esi

    mov     ecx, edx
    shr     ecx, 6
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, ebx
    shr     esi, 4
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 12], esi

    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0
    vmovss      xmm1, [r12]
    vmulss      xmm9, xmm0, xmm1

    xor         r14d, r14d

    vmovdqu     ymm8, [r11 + HMASK_OFF]

    vmovdqu     ymm6, [r11 + QS_OFF + 0*32]
    Q3K_SUBBLOCK 0, 0,   0, 0
    Q3K_SUBBLOCK 2, 1,  32, 2
    Q3K_SUBBLOCK 4, 2,  64, 4
    Q3K_SUBBLOCK 6, 3,  96, 6

    vmovdqu     ymm6, [r11 + QS_OFF + 1*32]
    Q3K_SUBBLOCK 0, 4, 128,  8
    Q3K_SUBBLOCK 2, 5, 160, 10
    Q3K_SUBBLOCK 4, 6, 192, 12
    Q3K_SUBBLOCK 6, 7, 224, 14

    vcvtsi2ss   xmm0, xmm0, r14d
    vmulss      xmm0, xmm0, xmm9
    vaddss      xmm15, xmm15, xmm0

    add         r11, BS_Q3_K
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
