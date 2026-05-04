;; =============================================================================
;; vec_dot_q3_k_q8_k_avx2.asm — Handwritten AVX2 implementation of Q3_K x Q8_K
;; dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q3_K_q8_K_generic`.
;;
;;   void simd_vec_dot_q3_k_q8_k_avx2(
;;       int n,                   ; total elements (multiple of QK_K=256)
;;       float * result,          ; out: scalar f32
;;       const block_q3_K * vx,   ; weights (110 B per super-block)
;;       const block_q8_K * vy);  ; activations (292 B per super-block)
;;
;; block_q3_K (110 B):
;;   hmask[32]  @ +0     (one bit per element across 8 sub-blocks of 32)
;;   qs[64]     @ +32    (two low bits per element, 4 elements packed per byte
;;                        across shifts {0,2,4,6})
;;   scales[12] @ +96    (16 packed 6-bit signed-biased scales)
;;   d (fp16)   @ +108
;;
;; Algorithm per super-block:
;;   decode 16 6-bit scales → 16 signed-biased u8 values (subtract 32 on use)
;;   for each of 8 sub-blocks (b = 0..7):
;;       chunk = b / 4    (0 or 1)              ; q3 bytes index 32*chunk
;;       shift = (b & 3) * 2                    ; 0,2,4,6
;;       q3   = ymm of 32 bytes from x.qs + 32*chunk
;;       hm_b = bit b of each hmask byte
;;       a    = ((q3 >> shift) & 3) − (4 if hm_b == 0 else 0)   ; 32 i8
;;       y    = ymm of 32 i8 from y.qs + b*32
;;       isum += (scales[2*b+0] − 32) * Σ a[ 0..15] * y[ 0..15]
;;       isum += (scales[2*b+1] − 32) * Σ a[16..31] * y[16..31]
;;   sumf += d_x * y_d * isum
;; =============================================================================

section .data
    align 32
    q3k_mask_03:    times 32 db 0x03
    q3k_const_4:    times 32 db 4

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

%define SCALES_DECODED_OFF 160      ; 16 bytes of decoded 6-bit scales

;; 16 i8 a × 16 i8 y → i32 sumi → isum += (scale[%3] − 32) * sumi
;;   %1 = xmm holding 16 i8 a-values
;;   %2 = byte offset into y.qs (relative to r12 + 4)
;;   %3 = scale index (0..15)
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

;; Decode one sub-block of 32 elements into ymm7 (signed i8) and run two sub-dots.
;;   %1 = shift (0,2,4,6)
;;   %2 = global sub-block index b (0..7)  → bit position within hmask
;;   %3 = y_qs byte offset (b*32)
;;   %4 = scale base index (b*2)
;; Inputs: ymm6=q3, ymm8=hmask, ymm10=mask03, ymm11=const4, ymm13=zero
%macro Q3K_SUBBLOCK 4
    %if %1 == 0
        vmovdqa  ymm7, ymm6
    %else
        vpsrlw   ymm7, ymm6, %1
    %endif
    vpand        ymm7, ymm7, ymm10           ; (q3 >> shift) & 3, 32 u8 in 0..3

    ;; build broadcast of (1 << b) into ymm12
    mov          eax, (1 << %2)
    vmovd        xmm12, eax
    vpbroadcastb ymm12, xmm12

    vpand        ymm0, ymm8, ymm12           ; 0 or (1<<b) per byte
    vpcmpeqb     ymm0, ymm0, ymm13           ; 0xFF where bit==0, 0x00 where bit==1
    vpand        ymm0, ymm0, ymm11           ; 4 where bit==0, 0 elsewhere
    vpsubb       ymm7, ymm7, ymm0            ; a in [-4, 3]

    ;; low 16 sub-dot
    Q3K_SUBDOT   xmm7, (%3),       (%4)

    ;; high 16 sub-dot
    vextracti128 xmm14, ymm7, 1
    Q3K_SUBDOT   xmm14, ((%3) + 16), ((%4) + 1)
%endmacro

global simd_vec_dot_q3_k_q8_k_avx2

simd_vec_dot_q3_k_q8_k_avx2:
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

    vmovdqa  ymm10, [rel q3k_mask_03]
    vmovdqa  ymm11, [rel q3k_const_4]
    vpxor    ymm13, ymm13, ymm13

.main_loop:
    ;; -- decode 16 6-bit scales into [rsp + SCALES_DECODED_OFF .. +16] ---
    ;;   auxs[0] = (auxs[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4)
    ;;   auxs[1] = (auxs[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4)
    ;;   auxs[2] = ((auxs[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4)
    ;;   auxs[3] = ((auxs[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4)
    mov     eax, [r11 + SCALES_RAW + 0]      ; auxs0_orig
    mov     ebx, [r11 + SCALES_RAW + 4]      ; auxs1_orig
    mov     edx, [r11 + SCALES_RAW + 8]      ; tmp (auxs2)

    ;; auxs[0]
    mov     ecx, edx
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, eax
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 0], esi

    ;; auxs[1]
    mov     ecx, edx
    shr     ecx, 2
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, ebx
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 4], esi

    ;; auxs[2]
    mov     ecx, edx
    shr     ecx, 4
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, eax
    shr     esi, 4
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 8], esi

    ;; auxs[3]
    mov     ecx, edx
    shr     ecx, 6
    and     ecx, 0x03030303
    shl     ecx, 4
    mov     esi, ebx
    shr     esi, 4
    and     esi, 0x0f0f0f0f
    or      esi, ecx
    mov     [rsp + SCALES_DECODED_OFF + 12], esi

    ;; -- factor = d_x * y_d ----------------------------------------------
    vmovd       xmm0, dword [r11 + D_OFF]
    vcvtph2ps   xmm0, xmm0                   ; d_x
    vmovss      xmm1, [r12]                  ; y_d (block_q8_K.d at +0, f32)
    vmulss      xmm9, xmm0, xmm1             ; factor (held in xmm9 for the loop)

    xor         r14d, r14d                   ; isum = 0

    ;; -- load hmask once -------------------------------------------------
    vmovdqu     ymm8, [r11 + HMASK_OFF]      ; 32 bytes of hmask

    ;; -- chunk 0: q3 = qs[0..31], y_qs[0..127], scales[0..7] ------------
    vmovdqu     ymm6, [r11 + QS_OFF + 0*32]
    Q3K_SUBBLOCK 0, 0,   0, 0
    Q3K_SUBBLOCK 2, 1,  32, 2
    Q3K_SUBBLOCK 4, 2,  64, 4
    Q3K_SUBBLOCK 6, 3,  96, 6

    ;; -- chunk 1: q3 = qs[32..63], y_qs[128..255], scales[8..15] --------
    vmovdqu     ymm6, [r11 + QS_OFF + 1*32]
    Q3K_SUBBLOCK 0, 4, 128,  8
    Q3K_SUBBLOCK 2, 5, 160, 10
    Q3K_SUBBLOCK 4, 6, 192, 12
    Q3K_SUBBLOCK 6, 7, 224, 14

    ;; -- combine: sumf += factor * isum ----------------------------------
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
