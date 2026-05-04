;; =============================================================================
;; vec_dot_q4_0_q8_0_avx2.asm — Handwritten AVX2 implementation of Q4_0 x Q8_0
;; dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q4_0_q8_0_generic`.
;;
;;   void simd_vec_dot_q4_0_q8_0_avx2(
;;       int n,                   ; total elements (multiple of QK8_0=32)
;;       float * result,          ; out: scalar f32
;;       const block_q4_0 * vx,   ; weights (18 B per block)
;;       const block_q8_0 * vy);  ; activations (34 B per block)
;;
;; block_q4_0 (18 B):
;;   d (fp16) @ +0
;;   qs[16]   @ +2     (16 nibbles = 32 elements; low nibbles → x[0..15], high → x[16..31])
;;
;; block_q8_0 (34 B):
;;   d (fp16) @ +0
;;   qs[32]   @ +2     (signed i8)
;;
;; Algorithm per block:
;;   x_i = (qs[i&15] & 0x0F) - 8     for i in 0..15
;;   x_i = (qs[i&15] >>  4) - 8      for i in 16..31
;;   sumi = Σ x_i * y_qs[i]   (i32)
;;   sumf += d_x * d_y * (float)sumi
;; =============================================================================

section .data
    align 32
    q40_mask_lo4:   times 32 db 0x0F
    q40_const_8:    times 32 db 8

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

global simd_vec_dot_q4_0_q8_0_avx2

simd_vec_dot_q4_0_q8_0_avx2:
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
    shr     r10d, 5                          ; nb = n / 32
    mov     r13, ARG2
    vxorps  xmm15, xmm15, xmm15
    test    r10d, r10d
    jz      .write_result

    mov     r11, ARG3
    mov     r12, ARG4

    vmovdqa  ymm10, [rel q40_mask_lo4]
    vmovdqa  ymm11, [rel q40_const_8]

.main_loop:
    ;; -- decode 32 i8 weights into ymm6 ----------------------------------
    vmovdqu     xmm0, [r11 + 2]              ; 16 qs bytes
    vpand       xmm1, xmm0, xmm10            ; lo nibble (16 u8)
    vpsrlw      xmm2, xmm0, 4
    vpand       xmm2, xmm2, xmm10            ; hi nibble (16 u8)
    vinserti128 ymm6, ymm1, xmm2, 1          ; ymm6 = [lo16 | hi16]  (32 u8)
    vpsubb      ymm6, ymm6, ymm11            ; − 8  →  32 i8

    ;; -- load 32 i8 activations -----------------------------------------
    vmovdqu     ymm7, [r12 + 2]              ; 32 i8

    ;; -- 32 i8 × 32 i8 → i32 sumi ----------------------------------------
    vpmovsxbw    ymm0, xmm6                  ; 16 i16 (low half of x)
    vextracti128 xmm1, ymm6, 1
    vpmovsxbw    ymm1, xmm1                  ; 16 i16 (high half of x)

    vpmovsxbw    ymm2, xmm7                  ; 16 i16 (low half of y)
    vextracti128 xmm3, ymm7, 1
    vpmovsxbw    ymm3, xmm3                  ; 16 i16 (high half of y)

    vpmaddwd     ymm0, ymm0, ymm2            ; 8 i32 pair-sums
    vpmaddwd     ymm1, ymm1, ymm3            ; 8 i32 pair-sums
    vpaddd       ymm0, ymm0, ymm1            ; 8 i32

    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1            ; 4 i32
    vphaddd      xmm0, xmm0, xmm0            ; 2 i32
    vphaddd      xmm0, xmm0, xmm0            ; 1 i32 = sumi
    vmovd        eax, xmm0

    ;; -- factor = d_x * d_y ----------------------------------------------
    vmovd        xmm2, dword [r11]
    vcvtph2ps    xmm2, xmm2                  ; d_x  (only low f32 valid)
    vmovd        xmm3, dword [r12]
    vcvtph2ps    xmm3, xmm3                  ; d_y
    vmulss       xmm2, xmm2, xmm3            ; factor

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
