;; =============================================================================
;; vec_dot_q2_k_q8_k_avx2.asm — Handwritten AVX2 implementation of Q2_K x Q8_K
;; dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q2_K_q8_K_generic`.
;;
;;   void simd_vec_dot_q2_k_q8_k_avx2(
;;       int n,                   ; total elements (multiple of QK_K=256)
;;       float * result,          ; out: scalar f32
;;       const block_q2_K * vx,   ; weights (84 B per super-block)
;;       const block_q8_K * vy);  ; activations (292 B per super-block)
;;
;; block_q2_K (84 B):
;;   scales[16] @ +0     (low 4 bits = scale, high 4 bits = min)
;;   qs[64]     @ +16    (2-bit quants — 4 elements per byte across shifts 0/2/4/6)
;;   d (fp16)   @ +80
;;   dmin (fp16)@ +82
;;
;; Algorithm per super-block (matches generic reference):
;;   summs = Σ y_bsums[j] * (scales[j] >> 4)             for j ∈ [0,16)
;;   isum  = 0
;;   for k = 0..1 (two 128-element chunks):
;;       qs = ymm of 32 bytes from x.qs + k*32
;;       for shift in {0, 2, 4, 6}:
;;           a = (qs >> shift) & 0x03                    (32 u8 values 0..3)
;;           y = ymm of 32 i8 from y.qs + (k*128 + (shift/2)*32)
;;           scale_lo = scales[is++] & 0x0F
;;           isuml_lo = Σ a[l] * y[l]   for l ∈ [0,16)
;;           isum    += scale_lo * isuml_lo
;;           scale_hi = scales[is++] & 0x0F
;;           isuml_hi = Σ a[l] * y[l]   for l ∈ [16,32)
;;           isum    += scale_hi * isuml_hi
;;   sumf += dall * isum  -  dmin * summs
;; where dall = y_d * d_x  ;  dmin = y_d * dmin_x
;; =============================================================================

section .data
    align 32
    q2k_mask_03:    times 32 db 0x03
    q2k_mask_lo4:   times 32 db 0x0F
    q2k_ones_i16:   times 16 dw 1

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
%define BSUMS_OFF   260      ; in block_q8_K

;; Compute one 16-element (a × y) signed dot, multiply by scale, add to r14d.
;;   %1 = xmm holding 16 u8 a-values
;;   %2 = xmm holding 16 i8 y-values
;;   %3 = scratch r-byte holding scale (0..15)  → MUST be a 32-bit GPR alias (e.g. ebx)
;; Clobbers ymm0/ymm1, xmm1, eax.
%macro Q2K_SUBDOT 3
    vpmovzxbw    ymm0, %1                    ; 16 u16 a values
    vpmovsxbw    ymm1, %2                    ; 16 i16 y values
    vpmaddwd     ymm0, ymm0, ymm1            ; 8 × i32 pair-sums (signed since u8 ≤ 3)
    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1            ; 4 × i32
    vphaddd      xmm0, xmm0, xmm0            ; 2 × i32
    vphaddd      xmm0, xmm0, xmm0            ; 1 × i32  = isuml
    vmovd        eax, xmm0
    imul         eax, %3                     ; * scale
    add          r14d, eax
%endmacro

;; Run one shift level (low+high halves) on the current chunk.
;;   %1 = shift amount (0,2,4,6)
;;   %2 = byte offset into y.qs (relative to r12 + 4) for the 32 i8 lane
;;   %3 = scale base index (is)  → reads scales[%3] and scales[%3+1]
;; Inputs: ymm6 = qs ymm (32 bytes of x.qs for this 128-element chunk)
;;         ymm10 = mask 0x03 (broadcast)
;;         ymm11 = mask 0x0F (broadcast)
;;         r11   = block_q2_K base
;; Clobbers ymm7, ymm8 (decoded a), xmm9, xmm12, xmm13, eax, ebx, ecx
%macro Q2K_SHIFT 3
    %if %1 == 0
        vmovdqa  ymm7, ymm6
    %else
        vpsrlw   ymm7, ymm6, %1
    %endif
    vpand        ymm7, ymm7, ymm10           ; a = (qs >> shift) & 3   (32 u8)

    vmovdqu      ymm8, [r12 + 4 + (%2)]      ; 32 i8 y values

    movzx        ebx, byte [r11 + (%3)]      ; scale_lo nibble (low 4 bits)
    and          ebx, 0x0F
    movzx        ecx, byte [r11 + (%3) + 1]  ; scale_hi
    and          ecx, 0x0F

    ;; low 16:
    Q2K_SUBDOT   xmm7, xmm8, ebx

    ;; high 16:
    vextracti128 xmm12, ymm7, 1              ; a_hi (16 u8)
    vextracti128 xmm13, ymm8, 1              ; y_hi (16 i8)
    Q2K_SUBDOT   xmm12, xmm13, ecx
%endmacro

global simd_vec_dot_q2_k_q8_k_avx2

simd_vec_dot_q2_k_q8_k_avx2:
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
    shr     r10d, 8                          ; nb = n / 256
    mov     r13, ARG2
    vxorps  xmm15, xmm15, xmm15
    test    r10d, r10d
    jz      .write_result

    mov     r11, ARG3
    mov     r12, ARG4

    vmovdqa  ymm10, [rel q2k_mask_03]
    vmovdqa  ymm11, [rel q2k_mask_lo4]
    vmovdqa  ymm9,  [rel q2k_ones_i16]       ; (kept for potential future use)

.main_loop:
    ;; -- d, dmin, factor = y_d ---------------------------------------------
    vmovss      xmm14, [r12]                 ; y_d  (block_q8_K.d at +0, f32)

    ;; summs = Σ y_bsums[j] * (scales[j] >> 4)   for j=0..15
    vmovdqu     xmm0, [r11]                  ; 16 scales bytes
    vpsrlw      xmm0, xmm0, 4
    vpand       xmm0, xmm0, xmm11            ; (scales >> 4) & 0xF, 16 u8 mins
    vpmovzxbw   ymm0, xmm0                   ; 16 i16 mins
    vmovdqu     ymm1, [r12 + BSUMS_OFF]      ; 16 i16 bsums
    vpmaddwd    ymm0, ymm0, ymm1             ; 8 i32 pair-sums
    vextracti128 xmm1, ymm0, 1
    vpaddd      xmm0, xmm0, xmm1
    vphaddd     xmm0, xmm0, xmm0
    vphaddd     xmm0, xmm0, xmm0
    vmovd       r15d, xmm0                   ; summs (i32)

    xor         r14d, r14d                   ; isum = 0

    ;; -- chunk 0: q2 = qs[0..31], y_qs[0..127], scales[0..7] ------------
    vmovdqu     ymm6, [r11 + 16 + 0*32]
    Q2K_SHIFT   0,   0, 0
    Q2K_SHIFT   2,  32, 2
    Q2K_SHIFT   4,  64, 4
    Q2K_SHIFT   6,  96, 6

    ;; -- chunk 1: q2 = qs[32..63], y_qs[128..255], scales[8..15] --------
    vmovdqu     ymm6, [r11 + 16 + 1*32]
    Q2K_SHIFT   0, 128,  8
    Q2K_SHIFT   2, 160, 10
    Q2K_SHIFT   4, 192, 12
    Q2K_SHIFT   6, 224, 14

    ;; -- combine: sumf += dall * isum - dmin * summs ---------------------
    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0                   ; d_x
    vmulss      xmm0, xmm0, xmm14            ; dall = d_x * y_d
    vcvtsi2ss   xmm1, xmm1, r14d
    vmulss      xmm0, xmm0, xmm1             ; dall * isum

    vmovd       xmm2, dword [r11 + DMIN_OFF]
    vcvtph2ps   xmm2, xmm2                   ; dmin_x
    vmulss      xmm2, xmm2, xmm14            ; dmin = dmin_x * y_d
    vcvtsi2ss   xmm3, xmm3, r15d
    vmulss      xmm2, xmm2, xmm3             ; dmin * summs

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
