;; =============================================================================
;; vec_dot_q6_k_q8_k_avx2.asm — Handwritten AVX2 implementation of Q6_K x Q8_K
;; dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q6_K_q8_K_generic`.
;;
;;   void simd_vec_dot_q6_k_q8_k_avx2(
;;       int n,                   ; total elements (multiple of QK_K=256)
;;       float * result,          ; out: scalar f32
;;       const block_q6_K * vx,   ; weights (210 B per super-block)
;;       const block_q8_K * vy);  ; activations (292 B per super-block)
;;
;; block_q6_K (210 B):
;;   ql[128]    @ +0     (low 4 bits, 2 elements per byte)
;;   qh[64]     @ +128   (high 2 bits, 4 elements per byte)
;;   scales[16] @ +192   (signed i8 — one per 16-element sub-sub-block)
;;   d (fp16)   @ +208
;;
;; Algorithm per super-block (Q4 nibble + Q2 high bits − 32):
;;   for each of 2 chunks (128 elems each, 8 sub-sub-blocks of 16):
;;     decode 4 quarters of 32 elems each from (ql,qh) into signed i8 a-quarter
;;     for each 16-elem half of the quarter:
;;       subdot = Σ a[k] * y_qs[off+k]   (signed × signed via pmovsxbw+pmaddwd)
;;       acc_i32 += scales[s] * subdot
;;   sumf += d_x * y_d * (float)acc_i32
;; =============================================================================

section .data
    align 32
    q6k_mask_lo4:   times 32 db 0x0F
    q6k_mask_03:    times 32 db 0x03
    q6k_const_32:   times 32 db 32     ; signed bias

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

;; Decode a 32-element quarter from ql/qh into ymm dest.
;;   %1 = output ymm (decoded a-quarter, 32 signed i8)
;;   %2 = ql source ymm (ymm6 or ymm7)
;;   %3 = "lo" (use ql nibble &0x0F) or "hi" (use ql >> 4)
;;   %4 = qh right-shift (0, 2, 4, or 6) — selects 2 high bits per element
;; Clobbers ymm0.
%macro DECODE_Q6_QUARTER 4
    %ifidn %3, lo
        vpand    %1, %2, ymm10           ; a = ql & 0x0F
    %else
        vpsrlw   %1, %2, 4
        vpand    %1, %1, ymm10           ; a = (ql >> 4) & 0x0F
    %endif

    %if %4 == 0
        vmovdqa  ymm0, ymm8              ; qh
    %else
        vpsrlw   ymm0, ymm8, %4          ; qh >> shift
    %endif
    vpand    ymm0, ymm0, ymm11           ; & 0x03
    vpsllw   ymm0, ymm0, 4               ; << 4 (safe: source <= 3)
    vpor     %1, %1, ymm0                ; merge high 2 bits
    vpsubb   %1, %1, ymm12               ; − 32 (signed)
%endmacro

;; Compute one 16-element signed dot product and add scale*subdot to r14d.
;;   %1 = xmm holding 16 signed i8 of a
;;   %2 = byte offset into y_qs (relative to r12 + 4)
;;   %3 = scale index (0..15)
;; Clobbers ymm0/ymm1, xmm1, eax, ebx.
%macro Q6K_SUBDOT 3
    vpmovsxbw    ymm0, %1
    vpmovsxbw    ymm1, [r12 + 4 + (%2)]
    vpmaddwd     ymm0, ymm0, ymm1            ; 8 × i32 pair-sums
    vextracti128 xmm1, ymm0, 1
    vpaddd       xmm0, xmm0, xmm1            ; 4 × i32
    vphaddd      xmm0, xmm0, xmm0            ; 2 × i32
    vphaddd      xmm0, xmm0, xmm0            ; 1 × i32
    vmovd        eax, xmm0                   ; subdot
    movsx        ebx, byte [r11 + SCALES_OFF + (%3)]
    imul         eax, ebx
    add          r14d, eax
%endmacro

;; Helper: decode one quarter into ymm9 then run two sub-dots (low/high halves).
;;   %1=ql ymm, %2=lo|hi, %3=qh shift, %4=y_qs base offset, %5=scale base
%macro Q6K_QUARTER 5
    DECODE_Q6_QUARTER ymm9, %1, %2, %3
    Q6K_SUBDOT        xmm9, (%4),       (%5)
    vextracti128      xmm2, ymm9, 1
    Q6K_SUBDOT        xmm2, ((%4) + 16), ((%5) + 1)
%endmacro

global simd_vec_dot_q6_k_q8_k_avx2

simd_vec_dot_q6_k_q8_k_avx2:
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

    vmovdqa  ymm10, [rel q6k_mask_lo4]
    vmovdqa  ymm11, [rel q6k_mask_03]
    vmovdqa  ymm12, [rel q6k_const_32]

.main_loop:
    ;; -- load super-block scaling factor ----------------------------------
    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0                   ; d_x (f32)
    vmovss      xmm1, [r12]                  ; y_d
    vmulss      xmm14, xmm0, xmm1            ; factor = d_x * y_d

    xor         r14d, r14d                   ; acc_i32 = 0

    ;; -- chunk 0: ql[0..63], qh[0..31], y_qs[0..127], scales[0..7] -------
    vmovdqu     ymm6, [r11 +   0]            ; ql[0..31]
    vmovdqu     ymm7, [r11 +  32]            ; ql[32..63]
    vmovdqu     ymm8, [r11 + 128]            ; qh[0..31]

    Q6K_QUARTER ymm6, lo, 0,   0, 0          ; a[0..31]   y_qs[0..31]   scales 0,1
    Q6K_QUARTER ymm7, lo, 2,  32, 2          ; a[32..63]  y_qs[32..63]  scales 2,3
    Q6K_QUARTER ymm6, hi, 4,  64, 4          ; a[64..95]  y_qs[64..95]  scales 4,5
    Q6K_QUARTER ymm7, hi, 6,  96, 6          ; a[96..127] y_qs[96..127] scales 6,7

    ;; -- chunk 1: ql[64..127], qh[32..63], y_qs[128..255], scales[8..15]
    vmovdqu     ymm6, [r11 +  64]            ; ql[64..95]
    vmovdqu     ymm7, [r11 +  96]            ; ql[96..127]
    vmovdqu     ymm8, [r11 + 160]            ; qh[32..63]

    Q6K_QUARTER ymm6, lo, 0, 128,  8
    Q6K_QUARTER ymm7, lo, 2, 160, 10
    Q6K_QUARTER ymm6, hi, 4, 192, 12
    Q6K_QUARTER ymm7, hi, 6, 224, 14

    ;; -- combine: sumf += factor * acc_i32 -------------------------------
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
