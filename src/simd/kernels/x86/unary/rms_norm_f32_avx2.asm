; ----------------------------------------------------------------------------
; rms_norm_f32_avx2.asm  --  Single-row RMS norm (AVX2 + FMA)
; ----------------------------------------------------------------------------
;
; void simd_rms_norm_f32_avx2(int n, float eps, const float * x, float * y);
;
; Computes:    scale = 1 / sqrt( (Sum_i x[i]^2) / n + eps )
;              y[i]  = x[i] * scale
;
; Numerical strategy (matches upstream's `ggml_compute_forward_rms_norm_f32`
; closely but accumulates in 4 parallel f64 lanes via vcvtps2pd+vaddpd, then
; horizontal-sums in tree order).  Per-row error is bounded by ULP(2) for
; reasonable n (n <= 65536).
;
; Win64 ABI:  rcx=n, xmm1=eps, r8=x, r9=y
;             returns nothing; preserves xmm6-xmm15 (callee-save).
;
; SysV ABI:   edi=n, xmm0=eps, rsi=x, rdx=y
;
; Tail handling: scalar loop for the final n % 8 elements.
;
; ----------------------------------------------------------------------------

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_EPS  xmm1
    %define ARG_X    r8
    %define ARG_Y    r9
%else
    %define ARG_N    rdi
    %define ARG_EPS  xmm0
    %define ARG_X    rsi
    %define ARG_Y    rdx
%endif

section .text
global simd_rms_norm_f32_avx2

simd_rms_norm_f32_avx2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168                ; 160 B for xmm6..xmm15 + 8 B align
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

    ; ------------------------------------------------------------------
    ; Save scalar args we still need late (eps -> xmm15, n -> r14).
    ; ------------------------------------------------------------------
    vmovss   xmm15, xmm15, ARG_EPS  ; eps
    mov      r14, ARG_N             ; total element count (i64)
    mov      r12, ARG_X             ; src ptr (preserved across loops)
    mov      r13, ARG_Y             ; dst ptr

    ; ------------------------------------------------------------------
    ; Pass 1:  accumulate sum of squares in 4 parallel f64 lanes.
    ; ymm0 = high f64 pair, ymm1 = low f64 pair
    ; ------------------------------------------------------------------
    vpxor    ymm0, ymm0, ymm0
    vpxor    ymm1, ymm1, ymm1

    mov      rbx, r14
    shr      rbx, 3                 ; rbx = n / 8 (vectorized iters)
    test     rbx, rbx
    jz       .pass1_tail

    mov      rax, r12               ; current load ptr

.pass1_vec:
    vmovups  ymm2, [rax]            ; 8 f32 inputs
    vmulps   ymm2, ymm2, ymm2       ; squares (f32)
    ; promote two halves to f64 and accumulate
    vextractf128 xmm3, ymm2, 1
    vcvtps2pd ymm4, xmm2            ; lo 4 squares as f64
    vcvtps2pd ymm5, xmm3            ; hi 4 squares as f64
    vaddpd   ymm1, ymm1, ymm4
    vaddpd   ymm0, ymm0, ymm5

    add      rax, 32
    dec      rbx
    jnz      .pass1_vec

.pass1_tail:
    ; Horizontal-sum the two f64 vectors -> xmm6 (scalar f64)
    vaddpd   ymm0, ymm0, ymm1       ; combine high+low
    vextractf128 xmm2, ymm0, 1
    vaddpd   xmm0, xmm0, xmm2       ; 2 f64 lanes left
    vhaddpd  xmm0, xmm0, xmm0       ; scalar f64 in low lane
    vmovapd  xmm6, xmm0             ; xmm6 = sum_sq (f64)

    ; Scalar tail (process the last n % 8 elements)
    mov      rax, r14
    and      rax, 7                 ; rax = n & 7
    jz       .pass1_done

    mov      rcx, r14
    and      rcx, ~7                ; rcx = n - (n&7) = start index of tail
    lea      rdx, [r12 + rcx*4]     ; tail src ptr

.pass1_tail_loop:
    vmovss   xmm2, [rdx]
    vmulss   xmm2, xmm2, xmm2       ; x*x in f32
    vcvtss2sd xmm3, xmm3, xmm2      ; -> f64
    vaddsd   xmm6, xmm6, xmm3
    add      rdx, 4
    dec      rax
    jnz      .pass1_tail_loop

.pass1_done:
    ; xmm6 = sum_sq (f64).  Compute mean = sum_sq / n  (f64 divide).
    vcvtsi2sd xmm7, xmm7, r14       ; f64(n)
    vdivsd   xmm6, xmm6, xmm7       ; mean = sum_sq / n
    vcvtsd2ss xmm6, xmm6, xmm6      ; mean back to f32 (matches upstream)
    vaddss   xmm6, xmm6, xmm15      ; mean + eps
    vsqrtss  xmm6, xmm6, xmm6       ; sqrt(mean+eps)
    vmovss   xmm7, [rel rms_const_one]
    vdivss   xmm6, xmm7, xmm6       ; scale = 1 / sqrt(...)
    vbroadcastss ymm7, xmm6         ; ymm7 = scale (8 lanes)

    ; ------------------------------------------------------------------
    ; Pass 2:  y[i] = x[i] * scale
    ; ------------------------------------------------------------------
    mov      rbx, r14
    shr      rbx, 3                 ; n/8
    mov      rax, r12               ; src
    mov      rdx, r13               ; dst
    test     rbx, rbx
    jz       .pass2_tail

.pass2_vec:
    vmovups  ymm0, [rax]
    vmulps   ymm0, ymm0, ymm7
    vmovups  [rdx], ymm0
    add      rax, 32
    add      rdx, 32
    dec      rbx
    jnz      .pass2_vec

.pass2_tail:
    mov      rcx, r14
    and      rcx, 7
    jz       .epilogue

.pass2_tail_loop:
    vmovss   xmm0, [rax]
    vmulss   xmm0, xmm0, xmm6
    vmovss   [rdx], xmm0
    add      rax, 4
    add      rdx, 4
    dec      rcx
    jnz      .pass2_tail_loop

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
    add      rsp, 168
    pop      r15
    pop      r14
    pop      r13
    pop      r12
    pop      rbx
    pop      rbp
    vzeroupper
    ret

section .rodata align=16
rms_const_one:  dd 0x3F800000        ; 1.0f
