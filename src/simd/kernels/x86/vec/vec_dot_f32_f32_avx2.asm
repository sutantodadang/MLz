;; =============================================================================
;; vec_dot_f32_f32_avx2.asm — F32 × F32 dot product kernel (AVX2 + FMA3)
;; =============================================================================
;;
;; void simd_vec_dot_f32_f32_avx2(
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
;;   1. Parallel FMA: acc += x[i] * y[i]  for i in [0, n-8)  (ymm, 8-wide)
;;   2. Horizontal sum of ymm accumulator
;;   3. Scalar tail for n % 8
;; =============================================================================

bits 64
default rel

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_R    rdx      ; float * r (pointer)
    %define ARG_X    r8       ; const void * vx
    %define ARG_Y    r9       ; const void * vy
%else
    %define ARG_N    rdi
    %define ARG_R    rsi      ; float * r (pointer)
    %define ARG_X    rdx
    %define ARG_Y    rcx
%endif

section .text
global simd_vec_dot_f32_f32_avx2

simd_vec_dot_f32_f32_avx2:
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
    mov     r13, ARG_R            ; output pointer

    ; Zero accumulator
    vxorps  ymm0, ymm0, ymm0

    ; --- Main vector loop (8 floats / iter) -----------------------------------
    mov     r14d, r10d
    shr     r14d, 3               ; n / 8
    test    r14d, r14d
    jz      .tail_check

    mov     r8, r11               ; x ptr
    mov     r9, r12               ; y ptr
    align 32

.vec_loop:
    vmovups ymm1, [r8]            ; x[0..7]
    vmovups ymm2, [r9]            ; y[0..7]
    vfmadd231ps ymm0, ymm1, ymm2   ; acc += x * y

    add     r8, 32
    add     r9, 32
    dec     r14d
    jnz     .vec_loop

.tail_check:
    mov     ecx, r10d
    and     ecx, 7                ; n % 8
    jz      .reduce

    ; Update x/y ptrs for tail
    ; (r8/r9 already point past the vector-processed region)
    ; Scalar tail
.tail_loop:
    vmovss  xmm1, [r8]
    vmovss  xmm2, [r9]
    vfmadd231ss xmm0, xmm1, xmm2

    add     r8, 4
    add     r9, 4
    dec     ecx
    jnz     .tail_loop

    ; --- Horizontal sum of ymm0 -> xmm0 (scalar) -----------------------------
.reduce:
    ; ymm0 = [a0 a1 a2 a3 a4 a5 a6 a7]
    vextractf128 xmm1, ymm0, 1     ; xmm1 = hi 128: [a4 a5 a6 a7]
    vaddps  xmm0, xmm0, xmm1       ; xmm0 = [a0+a4 a1+a5 a2+a6 a3+a7]
    vhaddps xmm0, xmm0, xmm0       ; xmm0 = [a0+a4+a1+a5 a0+a4+a1+a5 a2+a6+a3+a7 a2+a6+a3+a7]
    vhaddps xmm0, xmm0, xmm0       ; xmm0 = [sum sum sum sum]

    vmovss  [r13], xmm0            ; store to *r

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
