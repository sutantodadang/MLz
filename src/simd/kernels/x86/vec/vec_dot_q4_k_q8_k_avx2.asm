;; =============================================================================
;; vec_dot_q4_k_q8_k_avx2.asm — Handwritten AVX2 implementation of
;; Q4_K x Q8_K dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q4_K_q8_K_generic` (deps/llama_cpp/ggml/src/ggml-cpu/quants.c).
;;
;; Q4_K is identical in structure to Q5_K *minus the high-bit byte array* — every
;; element is a plain 4-bit nibble in [0..15], no +16 injection.  The sibling
;; vec_dot_q5_k_q8_k_avx2.asm is the reference implementation; this kernel is a
;; direct fork with the qh/+16 logic removed.
;;
;; BLOCK LAYOUT
;; ------------
;;   block_q4_K  (144 B, QK_K = 256 elements):
;;       fp16   d           ; +0
;;       fp16   dmin        ; +2
;;       u8     scales[12]  ; +4   (same packing as Q5_K)
;;       u8     qs[128]     ; +16  low-nibble | high-nibble per byte
;;
;;   block_q8_K  (292 B): see vec_dot_q5_k_q8_k_avx2.asm
;;
;; CALLING CONVENTION
;; ------------------
;;   void simd_vec_dot_q4_k_q8_k_avx2(
;;       int          n,        ; ARG1  (must be a multiple of 256)
;;       float      * result,   ; ARG2
;;       const void * vx,       ; ARG3  pointer to N/256 block_q4_K
;;       const void * vy);      ; ARG4  pointer to N/256 block_q8_K
;; =============================================================================

section .data
    align 32
    q4k_ones_i16:    times 16 dw 1
    q4k_mask_lo4:    times 32 db 0x0F

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

%define BS_Q4_K 144
%define BS_Q8_K 292
%define SCALES_OFF 160

;; -----------------------------------------------------------------------------
;; SUB_DOT macro — one 32-element sub-block.  Inputs:
;;   ymm6  = current 32-byte qs chunk (broadcast not needed — same chunk used
;;           for both lo and hi nibble passes)
;;   ymm12 = ones_i16, ymm13 = mask_lo4
;;   r12   = vy super-block base, r14d = acc_total
;;   [rsp + SCALES_OFF + S] = scales[S]
;; Clobbers: ymm0, ymm1, eax, ebx
;; -----------------------------------------------------------------------------
%macro Q4K_SUB_DOT 2
    %ifidn %1, lo
        vpand        ymm0, ymm6, ymm13
    %else
        vpsrlw       ymm0, ymm6, 4
        vpand        ymm0, ymm0, ymm13
    %endif

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

global simd_vec_dot_q4_k_q8_k_avx2

simd_vec_dot_q4_k_q8_k_avx2:
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

    vmovdqa ymm12, [rel q4k_ones_i16]
    vmovdqa ymm13, [rel q4k_mask_lo4]

.main_loop:
    ;; ---- decode d, dmin (fp16) and y_d (f32) ----
    vmovd      xmm0, dword [r11]
    vcvtph2ps  xmm0, xmm0
    vmovss     xmm1, [r12]
    vshufps    xmm2, xmm0, xmm0, 0x00
    vshufps    xmm3, xmm0, xmm0, 0x55
    vmulss     xmm4, xmm2, xmm1                   ; d_x * y_d
    vmulss     xmm5, xmm3, xmm1                   ; dmin_x * y_d

    ;; ---- decode 12-byte scales/mins (identical to Q5_K) ----
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

    ;; ---- sumi_mins = sum_{j=0..15} bsums[j] * mins[j/2] ----
    vmovdqu       ymm0, [r12 + 260]
    vmovq         xmm1, qword [rsp + SCALES_OFF + 8]
    vpunpcklbw    xmm1, xmm1, xmm1
    vpmovzxbw     ymm1, xmm1
    vpmaddwd      ymm0, ymm0, ymm1
    vextracti128  xmm1, ymm0, 1
    vpaddd        xmm0, xmm0, xmm1
    vphaddd       xmm0, xmm0, xmm0
    vphaddd       xmm0, xmm0, xmm0
    vmovd         r15d, xmm0                      ; sumi_mins (i32)

    ;; ---- accumulate 8 sub-blocks ----
    xor           r14d, r14d

    vmovdqu       ymm6, [r11 + 16 + 0*32]
    Q4K_SUB_DOT   lo, 0
    Q4K_SUB_DOT   hi, 1

    vmovdqu       ymm6, [r11 + 16 + 1*32]
    Q4K_SUB_DOT   lo, 2
    Q4K_SUB_DOT   hi, 3

    vmovdqu       ymm6, [r11 + 16 + 2*32]
    Q4K_SUB_DOT   lo, 4
    Q4K_SUB_DOT   hi, 5

    vmovdqu       ymm6, [r11 + 16 + 3*32]
    Q4K_SUB_DOT   lo, 6
    Q4K_SUB_DOT   hi, 7

    ;; ---- sumf += d_x*y_d * acc_total - dmin_x*y_d * sumi_mins ----
    vcvtsi2ss     xmm0, xmm0, r14d
    vmulss        xmm0, xmm0, xmm4
    vcvtsi2ss     xmm1, xmm1, r15d
    vmulss        xmm1, xmm1, xmm5
    vsubss        xmm0, xmm0, xmm1
    vaddss        xmm15, xmm15, xmm0

    add           r11, BS_Q4_K
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
