; ----------------------------------------------------------------------------
; silu_f32_avx512.asm  --  SiLU activation (AVX-512F)
; ----------------------------------------------------------------------------
;
; void simd_silu_f32_avx512(int n, const float * restrict x, float * restrict y);
;
; Computes:  y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
;
; Algorithm identical to AVX2 variant but uses zmm (16-wide) registers:
;   t      = -x * log2e
;   n      = vrndscaleps(t, 0)   -- round to nearest
;   f      = t - n
;   2^f    = polynomial via vfmadd213ps
;   2^n    = construct via integer ops on exponent field
;   exp(-x) = 2^n * 2^f
;   sigmoid = 1 / (1 + exp(-x))
;   y       = x * sigmoid
;
; Processes 16 f32 per iteration. Tail: scalar for n % 16.
;
; Win64:  rcx=n, rdx=x, r8=y
; SysV:   edi=n, rsi=x, rdx=y
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
global simd_silu_f32_avx512
%else
global simd_silu_f32_avx512:function
%endif

simd_silu_f32_avx512:
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

    mov     r12d, ARG_N
    mov     r13, ARG_X
    mov     r14, ARG_Y
    test    r12d, r12d
    jz      .epilogue

    ; --- Load constants into zmm20-zmm28 (no ABI save needed) -----------------
    vpxorq        zmm20, zmm20, zmm20        ; zmm20 = 0 (clamp lo)
    vbroadcastss  zmm21, [rel .const_exp_clamp_hi]
    vbroadcastss  zmm22, [rel .const_log2e]
    vbroadcastss  zmm23, [rel .const_sign]
    vbroadcastss  zmm24, [rel .const_bias127]
    vbroadcastss  zmm25, [rel .const_one]
    vbroadcastss  zmm26, [rel .const_poly_c1]
    vbroadcastss  zmm27, [rel .const_poly_c2]
    vbroadcastss  zmm28, [rel .const_poly_c3]
    vbroadcastss  zmm29, [rel .const_poly_c4]

    ; --- Main vector loop (16 floats / iter) ----------------------------------
    mov     ebx, r12d
    shr     ebx, 4                  ; ebx = n / 16
    test    ebx, ebx
    jz      .tail_check

    mov     rax, r13
    mov     rdx, r14
    align 64

.vec_loop:
    vmovups zmm0, [rax]             ; zmm0 = x[0..15]

    ; Step 1: exp(-x)
    vxorps  zmm1, zmm0, zmm23       ; zmm1 = -x
    vmulps  zmm1, zmm1, zmm22       ; zmm1 = t = -x * log2e
    vrndscaleps zmm2, zmm1, 0x00    ; zmm2 = n = round(t)
    vsubps  zmm3, zmm1, zmm2        ; zmm3 = f

    ; Polynomial: 2^f
    vmovaps zmm4, zmm29             ; zmm4 = c4
    vfmadd213ps zmm4, zmm3, zmm28   ; c4*f + c3
    vfmadd213ps zmm4, zmm3, zmm27   ; (...) * f + c2
    vfmadd213ps zmm4, zmm3, zmm26   ; (...) * f + c1
    vfmadd213ps zmm4, zmm3, zmm25   ; (...) * f + 1.0 = 2^f

    ; 2^n via (n + 127) << 23, clamped to valid exponent range
    vcvttps2dq zmm5, zmm2           ; zmm5 = n as int32
    vpaddd    zmm5, zmm5, zmm24     ; zmm5 = n + 127
    vpmaxsd   zmm5, zmm5, zmm20     ; clamp lo: max(0, ...)
    vpminsd   zmm5, zmm5, zmm21     ; clamp hi: min(254, ...)
    vpslld    zmm5, zmm5, 23        ; zmm5 = float bit pattern for 2^n

    vmulps  zmm1, zmm5, zmm4        ; zmm1 = exp(-x)

    ; Step 2: sigmoid = 1 / (1 + exp(-x))
    vaddps  zmm1, zmm1, zmm25       ; 1 + exp(-x)
    vdivps  zmm1, zmm25, zmm1       ; sigmoid

    ; Step 3: silu = x * sigmoid
    vmulps  zmm0, zmm0, zmm1

    vmovups [rdx], zmm0
    add     rax, 64
    add     rdx, 64
    dec     ebx
    jnz     .vec_loop

.tail_check:
    mov     ecx, r12d
    and     ecx, 15                 ; ecx = n % 16
    jz      .epilogue

    ; --- Scalar tail ----------------------------------------------------------
.tail_loop:
    vmovss  xmm0, [rax]

    vxorps  xmm1, xmm0, xmm23       ; -x
    vmulss  xmm1, xmm1, xmm22       ; t
    vrndscaless xmm2, xmm1, xmm1, 0x00  ; n
    vsubss  xmm3, xmm1, xmm2        ; f

    vmovss  xmm4, xmm29
    vfmadd213ss xmm4, xmm3, xmm28
    vfmadd213ss xmm4, xmm3, xmm27
    vfmadd213ss xmm4, xmm3, xmm26
    vfmadd213ss xmm4, xmm3, xmm25   ; 2^f

    vcvtss2si r15d, xmm2
    add     r15d, 127
    xor     eax, eax
    cmp     r15d, eax
    cmovl   r15d, eax               ; clamp lo
    cmp     r15d, 254
    cmovg   r15d, 254               ; clamp hi
    shl     r15d, 23
    vmovd   xmm5, r15d

    vmulss  xmm1, xmm5, xmm4        ; exp(-x)
    vaddss  xmm1, xmm1, xmm25       ; 1 + exp(-x)
    vdivss  xmm1, xmm25, xmm1       ; sigmoid
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
section .rodata align=64

.const_log2e:      dd 0x3FB8AA3B
.const_sign:       dd 0x80000000
.const_bias127:    dd 127
.const_exp_clamp_hi: dd 254
.const_one:        dd 0x3F800000
.const_poly_c1:    dd 0x3F317218
.const_poly_c2:    dd 0x3E75FDF0
.const_poly_c3:    dd 0x3D635847
.const_poly_c4:    dd 0x3C1D99AA
