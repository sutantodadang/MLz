;; =============================================================================
;; vec_dot_f32_f32_avx512.asm — F32 × F32 dot product kernel (AVX-512F)
;; =============================================================================
;;
;; void simd_vec_dot_f32_f32_avx512(
;;     int n,                  ; number of elements
;;     float * r,              ; pointer to output scalar
;;     const void * vx,        ; const float * x
;;     const void * vy         ; const float * y
;; );
;;
;; Computes:  *r = sum_{i=0}^{n-1} x[i] * y[i]
;;
;; Win64:  rcx=n, rdx=r, r8=vx, r9=vy
;; SysV:   edi=n, rsi=r, rdx=vx, rcx=vy
;;
;; Algorithm:
;;   1. Parallel FMA: acc += x[i] * y[i]  for i in [0, n-16)  (zmm, 16-wide)
;;   2. Horizontal sum of zmm accumulator:
;;        vextractf32x8 ymm, zmm, 1  — extract hi 8 lanes
;;        vaddps ymm, ymm, ymm_low
;;        vextractf128 xmm, ymm, 1   — extract hi 4
;;        vaddps xmm, xmm, xmm_low
;;        vhaddps + vhaddps          — reduce to scalar
;;   3. Scalar tail for n % 16
;; =============================================================================

bits 64
default rel

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_R    rdx
    %define ARG_X    r8
    %define ARG_Y    r9
%else
    %define ARG_N    rdi
    %define ARG_R    rsi
    %define ARG_X    rdx
    %define ARG_Y    rcx
%endif

section .text
global simd_vec_dot_f32_f32_avx512

simd_vec_dot_f32_f32_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168
    vmovdqu [rsp+ 0],  xmm6
    vmovdqu [rsp+16],  xmm7
    vmovdqu [rsp+32],  xmm8
    vmovdqu [rsp+48],  xmm9
    vmovdqu [rsp+64],  xmm10
    vmovdqu [rsp+80],  xmm11
    vmovdqu [rsp+96],  xmm12
    vmovdqu [rsp+112], xmm13
    vmovdqu [rsp+128], xmm14
    vmovdqu [rsp+144], xmm15

    mov     r10, ARG_N
    mov     r11, ARG_X
    mov     r12, ARG_Y
    mov     r13, ARG_R

    ; Zero accumulator
    vpxorq  zmm0, zmm0, zmm0

    ; --- Main vector loop (16 floats / iter) ----------------------------------
    mov     r14d, r10d
    shr     r14d, 4               ; n / 16
    test    r14d, r14d
    jz      .tail_check

    mov     r8, r11
    mov     r9, r12
    align 64

.vec_loop:
    vmovups zmm1, [r8]            ; x[0..15]
    vmovups zmm2, [r9]            ; y[0..15]
    vfmadd231ps zmm0, zmm1, zmm2  ; acc += x * y

    add     r8, 64
    add     r9, 64
    dec     r14d
    jnz     .vec_loop

.tail_check:
    mov     ecx, r10d
    and     ecx, 15               ; n % 16
    jz      .reduce

.tail_loop:
    vmovss  xmm1, [r8]
    vmovss  xmm2, [r9]
    vfmadd231ss xmm0, xmm1, xmm2

    add     r8, 4
    add     r9, 4
    dec     ecx
    jnz     .tail_loop

    ; --- Horizontal sum of zmm0 -> scalar -------------------------------------
.reduce:
    ; zmm0 = [a0..a15]
    vextractf32x8 ymm1, zmm0, 1   ; ymm1 = hi 8: [a8..a15]
    vaddps  ymm0, ymm0, ymm1       ; ymm0 = [a0+a8 .. a7+a15]
    vextractf128 xmm1, ymm0, 1    ; xmm1 = hi 4: [a4+a12 .. a7+a15]
    vaddps  xmm0, xmm0, xmm1      ; xmm0 = [sum0 sum1 sum2 sum3]
    vhaddps xmm0, xmm0, xmm0      ; [sum0+sum1 sum0+sum1 sum2+sum3 sum2+sum3]
    vhaddps xmm0, xmm0, xmm0      ; [total total total total]

    vmovss  [r13], xmm0

    ; --- Epilogue -------------------------------------------------------------
    vmovdqu  xmm6,  [rsp+ 0]
    vmovdqu  xmm7,  [rsp+16]
    vmovdqu  xmm8,  [rsp+32]
    vmovdqu  xmm9,  [rsp+48]
    vmovdqu  xmm10, [rsp+64]
    vmovdqu  xmm11, [rsp+80]
    vmovdqu  xmm12, [rsp+96]
    vmovdqu  xmm13, [rsp+112]
    vmovdqu  xmm14, [rsp+128]
    vmovdqu  xmm15, [rsp+144]
    add     rsp, 168
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret
