;; =============================================================================
;; vec_dot_q5_k_q8_k_avx512.asm — Handwritten AVX-512 implementation of
;; Q5_K x Q8_K dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q5_K_q8_K_generic`.
;;
;; STRATEGY
;; --------
;; Conservative AVX-512 port: the per-sub-block algorithm matches the AVX2
;; sibling (vec_dot_q5_k_q8_k_avx2.asm), but constants live in zmm regs and
;; broadcasts use EVEX encodings.  This guarantees correctness; a future
;; revision can fuse two sub-blocks into one zmm pass for throughput.
;;
;; Width: AVX-512 F + BW + DQ.  No VNNI required.
;; =============================================================================

section .data
    align 64
    ones_i16_z:    times 32 dw 1            ; 64 B (16 i16 per 256b lane)
    mask_lo4_z:    times 64 db 0x0F         ; 64 B
    const_16_z:    times 64 db 16           ; 64 B

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

%define BS_Q5_K 176
%define BS_Q8_K 292
%define SCALES_OFF 160

;; -----------------------------------------------------------------------------
;; SUB_DOT macro — same as AVX2 sibling.  ymm12/13/14 are the lower 256 bits of
;; the corresponding zmm constants loaded once in the prologue.
;; -----------------------------------------------------------------------------
%macro SUB_DOT 2
    %ifidn %1, lo
        vpand        ymm0, ymm6, ymm13
    %else
        vpsrlw       ymm0, ymm6, 4
        vpand        ymm0, ymm0, ymm13
    %endif

    mov              eax, (1 << %2)
    vmovd            xmm1, eax
    vpbroadcastb     ymm1, xmm1
    vpand            ymm2, ymm7, ymm1
    vpcmpeqb         ymm2, ymm2, ymm1
    vpand            ymm2, ymm2, ymm14
    vpaddb           ymm0, ymm0, ymm2

    vmovdqu          ymm1, [r12 + 4 + (%2)*32]

    vpmaddubsw       ymm0, ymm0, ymm1
    vpmaddwd         ymm0, ymm0, ymm12

    vextracti128     xmm1, ymm0, 1
    vpaddd           xmm0, xmm0, xmm1
    vphaddd          xmm0, xmm0, xmm0
    vphaddd          xmm0, xmm0, xmm0
    vmovd            eax, xmm0

    movzx            ebx, byte [rsp + SCALES_OFF + (%2)]
    imul             eax, ebx
    add              r14d, eax
%endmacro

global simd_vec_dot_q5_k_q8_k_avx512

simd_vec_dot_q5_k_q8_k_avx512:
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

    ;; EVEX 512-bit loads of the constants (lower halves used as ymm12/13/14).
    vmovdqa64 zmm12, [rel ones_i16_z]
    vmovdqa64 zmm13, [rel mask_lo4_z]
    vmovdqa64 zmm14, [rel const_16_z]

.main_loop:
    vmovd      xmm0, dword [r11]
    vcvtph2ps  xmm0, xmm0
    vmovss     xmm1, [r12]
    vshufps    xmm2, xmm0, xmm0, 0x00
    vshufps    xmm3, xmm0, xmm0, 0x55
    vmulss     xmm4, xmm2, xmm1
    vmulss     xmm5, xmm3, xmm1

    mov     eax, [r11 + 4 + 0]
    mov     ebx, [r11 + 4 + 4]
    mov     ecx, [r11 + 4 + 8]

    mov     edx, ecx
    shr     edx, 4
    and     edx, 0x0f0f0f0f
    mov     esi, ebx
    shr     esi, 6
    and     esi, 0x03030303
    shl     esi, 4
    or      edx, esi

    mov     edi, ebx
    and     edi, 0x3f3f3f3f

    mov     ebx, ecx
    and     ebx, 0x0f0f0f0f
    mov     esi, eax
    shr     esi, 6
    and     esi, 0x03030303
    shl     esi, 4
    or      ebx, esi

    mov     ecx, edi
    and     eax, 0x3f3f3f3f

    mov     [rsp + SCALES_OFF +  0], eax
    mov     [rsp + SCALES_OFF +  4], ebx
    mov     [rsp + SCALES_OFF +  8], ecx
    mov     [rsp + SCALES_OFF + 12], edx

    vmovdqu       ymm0, [r12 + 260]
    vmovq         xmm1, qword [rsp + SCALES_OFF + 8]
    vpunpcklbw    xmm1, xmm1, xmm1
    vpmovzxbw     ymm1, xmm1
    vpmaddwd      ymm0, ymm0, ymm1
    vextracti128  xmm1, ymm0, 1
    vpaddd        xmm0, xmm0, xmm1
    vphaddd       xmm0, xmm0, xmm0
    vphaddd       xmm0, xmm0, xmm0
    vmovd         r15d, xmm0

    vmovdqu       ymm7, [r11 + 16]

    xor           r14d, r14d

    vmovdqu       ymm6, [r11 + 48 + 0*32]
    SUB_DOT       lo, 0
    SUB_DOT       hi, 1

    vmovdqu       ymm6, [r11 + 48 + 1*32]
    SUB_DOT       lo, 2
    SUB_DOT       hi, 3

    vmovdqu       ymm6, [r11 + 48 + 2*32]
    SUB_DOT       lo, 4
    SUB_DOT       hi, 5

    vmovdqu       ymm6, [r11 + 48 + 3*32]
    SUB_DOT       lo, 6
    SUB_DOT       hi, 7

    vcvtsi2ss     xmm0, xmm0, r14d
    vmulss        xmm0, xmm0, xmm4
    vcvtsi2ss     xmm1, xmm1, r15d
    vmulss        xmm1, xmm1, xmm5
    vsubss        xmm0, xmm0, xmm1
    vaddss        xmm15, xmm15, xmm0

    add           r11, BS_Q5_K
    add           r12, BS_Q8_K
    dec           r10d
    jnz           .main_loop

.write_result:
    vmovss        [r13], xmm15

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
