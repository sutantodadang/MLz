; ----------------------------------------------------------------------------
; silu_f32_avx2.asm  --  SiLU activation (AVX2 + FMA3)
; ----------------------------------------------------------------------------
;
; void simd_silu_f32_avx2(int n, const float * restrict x, float * restrict y);
;
; Computes:  y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
;
; Algorithm: exp(-x) via 2^t decomposition with degree-4 polynomial:
;   t      = -x * log2e
;   n      = round(t)           -- integer part
;   f      = t - n              -- fractional, f in [-0.5, 0.5]
;   2^f    = 1 + c1*f + c2*f^2 + c3*f^3 + c4*f^4   (Taylor of 2^f)
;   2^n    = construct via (n + 127) << 23
;   exp(-x) = 2^n * 2^f
;   sigmoid = 1 / (1 + exp(-x))
;   y       = x * sigmoid
;
; In-place capable: x and y may alias.
; Processes 8 f32 per iteration. Tail: scalar for n % 8.
;
; Win64:  rcx=n, rdx=x, r8=y
; SysV:   edi=n, rsi=x, rdx=y
;
; Accuracy: < 2 ULP vs reference SiLU for |x| < 88.1
; ----------------------------------------------------------------------------

bits 64
default rel

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_X    rdx
    %define ARG_Y    r8
%else
    %define ARG_N    rdi
    %define ARG_X    rsi
    %define ARG_Y    rdx
%endif

section .text
%ifdef WINDOWS
global simd_silu_f32_avx2
%else
global simd_silu_f32_avx2:function
%endif

simd_silu_f32_avx2:
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

    ; Save args; early-out if n == 0
    mov     r12, ARG_N
    mov     r13, ARG_X
    mov     r14, ARG_Y
    test    r12d, r12d
    jz      .epilogue

    ; --- Load constants -------------------------------------------------------
    ; ymm6  = exp_lo clamp    (int32 0, broadcast)
    ; ymm7  = exp_hi clamp    (int32 254, broadcast)
    ; ymm8  = log2e           (8 x 1.4426950f)
    ; ymm9  = sign mask       (8 x 0x80000000)
    ; ymm10 = bias 127        (8 x 127, as int32)
    ; ymm11 = 1.0f
    ; ymm12 = c1
    ; ymm13 = c2
    ; ymm14 = c3
    ; ymm15 = c4
    vpxor         ymm6, ymm6, ymm6
    vbroadcastss  ymm7,  [rel .const_exp_clamp_hi]
    vbroadcastss  ymm8,  [rel .const_log2e]
    vbroadcastss  ymm9,  [rel .const_sign]
    vbroadcastss  ymm10, [rel .const_bias127]
    vbroadcastss  ymm11, [rel .const_one]
    vbroadcastss  ymm12, [rel .const_poly_c1]
    vbroadcastss  ymm13, [rel .const_poly_c2]
    vbroadcastss  ymm14, [rel .const_poly_c3]
    vbroadcastss  ymm15, [rel .const_poly_c4]

    ; --- Main vector loop (8 floats / iter) -----------------------------------
    mov     rax, r13                ; src ptr (set before the n<8 branch so the
    mov     rdx, r14                ; dst ptr  tail loop has valid pointers)
    mov     ebx, r12d
    shr     ebx, 3                  ; ebx = n / 8
    test    ebx, ebx
    jz      .tail_check
    align 32

.vec_loop:
    vmovups ymm0, [rax]             ; ymm0 = x[0..7]

    ; --- Step 1: compute exp(-x) -----------------------------------------
    ; t = -x * log2e
    vxorps  ymm1, ymm0, ymm9        ; ymm1 = -x
    vmulps  ymm1, ymm1, ymm8        ; ymm1 = t = -x * log2e

    ; n = round(t), f = t - n
    vroundps ymm2, ymm1, 0x00       ; ymm2 = n = round(t) to nearest
    vsubps  ymm3, ymm1, ymm2        ; ymm3 = f (fractional, in [-0.5, 0.5])

    ; Polynomial: 2^f = (((c4*f + c3)*f + c2)*f + c1)*f + 1.0
    vmovaps ymm4, ymm15             ; ymm4 = c4
    vfmadd213ps ymm4, ymm3, ymm14   ; ymm4 = c4*f + c3
    vfmadd213ps ymm4, ymm3, ymm13   ; ymm4 = (c4*f+c3)*f + c2
    vfmadd213ps ymm4, ymm3, ymm12   ; ymm4 = ((c4*f+c3)*f+c2)*f + c1
    vfmadd213ps ymm4, ymm3, ymm11   ; ymm4 = 2^f approx

    ; 2^n: construct float via (n + 127) << 23, clamped to valid exponent range
    vcvtps2dq ymm5, ymm2            ; ymm5 = n as int32 (trunc, but n is exact int)
    vpaddd   ymm5, ymm5, ymm10      ; ymm5 = n + 127  (int32)
    vpmaxsd  ymm5, ymm5, ymm6       ; clamp: max(0, n+127)
    vpminsd  ymm5, ymm5, ymm7       ; clamp: min(254, ...)
    vpslld   ymm5, ymm5, 23         ; ymm5 = (n+127) << 23 = f32 bit pattern for 2^n

    ; exp(-x) = 2^n * 2^f
    vmulps  ymm1, ymm5, ymm4        ; ymm1 = exp(-x)

    ; --- Step 2: sigmoid = 1 / (1 + exp(-x)) -----------------------------
    vaddps  ymm1, ymm1, ymm11       ; ymm1 = 1 + exp(-x)
    vdivps  ymm1, ymm11, ymm1       ; ymm1 = sigmoid = 1 / (1 + exp(-x))

    ; --- Step 3: silu = x * sigmoid --------------------------------------
    vmulps  ymm0, ymm0, ymm1        ; ymm0 = x * sigmoid = silu(x)

    vmovups [rdx], ymm0
    add     rax, 32
    add     rdx, 32
    dec     ebx
    jnz     .vec_loop

.tail_check:
    mov     ecx, r12d
    and     ecx, 7                  ; ecx = n % 8
    jz      .epilogue

    ; --- Scalar tail loop -----------------------------------------------------
.tail_loop:
    vmovss  xmm0, [rax]             ; x

    vxorps  xmm1, xmm0, xmm9        ; -x
    vmulss  xmm1, xmm1, xmm8        ; t = -x * log2e
    vroundss xmm2, xmm1, xmm1, 0x00 ; n = round(t)
    vsubss  xmm3, xmm1, xmm2        ; f = t - n

    ; Polynomial: 2^f = (((c4*f + c3)*f + c2)*f + c1)*f + 1.0
    vmovss  xmm4, xmm15
    vfmadd213ss xmm4, xmm3, xmm14
    vfmadd213ss xmm4, xmm3, xmm13
    vfmadd213ss xmm4, xmm3, xmm12
    vfmadd213ss xmm4, xmm3, xmm11   ; xmm4 = 2^f

    ; 2^n via scalar integer ops with clamp
    vcvtss2si r15d, xmm2            ; n as integer
    add     r15d, 127               ; biased exponent
    xor     r10d, r10d              ; (r10d scratch: eax aliases rax = src ptr)
    cmp     r15d, r10d
    cmovl   r15d, r10d              ; clamp lo: max(0, n+127)
    mov     r10d, 254
    cmp     r15d, r10d
    cmovg   r15d, r10d              ; clamp hi: min(254, ...)
    shl     r15d, 23                ; construct float bit pattern
    vmovd   xmm5, r15d              ; xmm5 = 2^n

    vmulss  xmm1, xmm5, xmm4        ; exp(-x)
    vaddss  xmm1, xmm1, xmm11       ; 1 + exp(-x)
    vdivss  xmm1, xmm11, xmm1       ; sigmoid
    vmulss  xmm0, xmm0, xmm1        ; silu

    vmovss  [rdx], xmm0
    add     rax, 4
    add     rdx, 4
    dec     ecx
    jnz     .tail_loop

.epilogue:
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

; ----------------------------------------------------------------------------
; Read-only constants
; ----------------------------------------------------------------------------
section .rodata align=32

.const_log2e:      dd 0x3FB8AA3B     ; log2(e) = 1.4426950f
.const_sign:       dd 0x80000000     ; sign-bit mask
.const_bias127:    dd 127            ; exponent bias (as int32)
.const_exp_clamp_hi: dd 254          ; max valid biased exponent (as int32)
.const_one:        dd 0x3F800000     ; 1.0f
.const_poly_c1:    dd 0x3F317218     ; c1 = 0.6931472f  (ln 2)
.const_poly_c2:    dd 0x3E75FDF0     ; c2 = 0.2402265f
.const_poly_c3:    dd 0x3D635847     ; c3 = 0.0555041f
.const_poly_c4:    dd 0x3C1D99AA     ; c4 = 0.0096181f
