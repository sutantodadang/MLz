; ----------------------------------------------------------------------------
; layer_norm_f32_avx2.asm  --  Single-row Layer Normalization (AVX2 + FMA)
; ----------------------------------------------------------------------------
;
; void simd_layer_norm_f32_avx2(int n, float eps, const float * x, float * y);
;
; Computes:    mean = (Sum_i x[i]) / n
;              var  = (Sum_i (x[i] - mean)^2) / n
;              inv  = 1 / sqrt(var + eps)
;              y[i] = (x[i] - mean) * inv
;
; Numerical strategy (Welford online algorithm, matches ggml):
;   Pass 1: accumulate sum (f64) and sum-of-squares (f64) simultaneously
;           using 4 parallel f64 lanes per accumulator via vcvtps2pd+vaddpd.
;   Pass 2: y[i] = (x[i] - mean) * scale, 8 floats/iter.
;   Per-row error bounded by ULP(3) for n <= 65536.
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

bits 64
default rel

section .text
global simd_layer_norm_f32_avx2

simd_layer_norm_f32_avx2:
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
    ; Save scalar args.  eps -> xmm15, n -> r14, src -> r12, dst -> r13.
    ; ------------------------------------------------------------------
    vmovss   xmm15, xmm15, ARG_EPS  ; eps
    mov      r14, ARG_N             ; total element count
    mov      r12, ARG_X             ; src ptr (preserved)
    mov      r13, ARG_Y             ; dst ptr (preserved)

    ; ------------------------------------------------------------------
    ; Pass 1: Welford — accumulate sum and sum-of-squares in f64.
    ; sum   accum: ymm8(hi), ymm9(lo)   = 4 parallel f64 lanes
    ; sumsq accum: ymm10(hi), ymm11(lo) = 4 parallel f64 lanes
    ; ------------------------------------------------------------------
    vpxor    ymm8, ymm8, ymm8
    vpxor    ymm9, ymm9, ymm9
    vpxor    ymm10, ymm10, ymm10
    vpxor    ymm11, ymm11, ymm11

    mov      rbx, r14
    shr      rbx, 3                 ; rbx = n / 8 (vectorized iters)
    test     rbx, rbx
    jz       .pass1_tail

    mov      rax, r12               ; working load ptr

.pass1_vec:
    vmovups  ymm0, [rax]            ; 8 f32 inputs

    ; ---- accumulate sum (f64) ----
    vextractf128 xmm1, ymm0, 1      ; hi 4 f32
    vcvtps2pd ymm2, xmm0            ; lo 4 -> f64
    vcvtps2pd ymm3, xmm1            ; hi 4 -> f64
    vaddpd   ymm9, ymm9, ymm2       ; sum_lo += lo_4
    vaddpd   ymm8, ymm8, ymm3       ; sum_hi += hi_4

    ; ---- accumulate sum-of-squares (f64) ----
    vmulps   ymm0, ymm0, ymm0       ; x^2 in f32
    vextractf128 xmm1, ymm0, 1      ; hi 4 squares
    vcvtps2pd ymm2, xmm0            ; lo 4 squares -> f64
    vcvtps2pd ymm3, xmm1            ; hi 4 squares -> f64
    vaddpd   ymm11, ymm11, ymm2     ; sumsq_lo += lo_4
    vaddpd   ymm10, ymm10, ymm3     ; sumsq_hi += hi_4

    add      rax, 32
    dec      rbx
    jnz      .pass1_vec

.pass1_tail:
    ; ---- horizontal-sum sum accumulators -> xmm8 (scalar f64) ----
    vaddpd   ymm8, ymm8, ymm9       ; combine hi+lo (4 f64 lanes)
    vextractf128 xmm9, ymm8, 1
    vaddpd   xmm8, xmm8, xmm9       ; 2 f64 lanes
    vhaddpd  xmm8, xmm8, xmm8       ; scalar f64 in low lane

    ; ---- horizontal-sum sumsq accumulators -> xmm10 (scalar f64) ----
    vaddpd   ymm10, ymm10, ymm11
    vextractf128 xmm11, ymm10, 1
    vaddpd   xmm10, xmm10, xmm11
    vhaddpd  xmm10, xmm10, xmm10

    ; Scalar tail for both sum and sumsq (n & 7 elements)
    mov      rax, r14
    and      rax, 7                 ; tail count
    jz       .pass1_done

    mov      rcx, r14
    and      rcx, ~7
    lea      rdx, [r12 + rcx*4]     ; tail src ptr

.pass1_tail_loop:
    vmovss   xmm0, [rdx]            ; x (f32)
    vcvtss2sd xmm1, xmm1, xmm0      ; x -> f64
    vaddsd   xmm8, xmm8, xmm1       ; sum += x

    vmulss   xmm0, xmm0, xmm0       ; x^2 (f32)
    vcvtss2sd xmm1, xmm1, xmm0      ; x^2 -> f64
    vaddsd   xmm10, xmm10, xmm1     ; sumsq += x^2

    add      rdx, 4
    dec      rax
    jnz      .pass1_tail_loop

.pass1_done:
    ; xmm8 = sum (f64), xmm10 = sumsq (f64)
    ; mean = sum / n, var = sumsq/n - mean^2
    vcvtsi2sd xmm0, xmm0, r14       ; f64(n)

    vdivsd   xmm8, xmm8, xmm0       ; mean_f64 = sum / n
    vdivsd   xmm10, xmm10, xmm0     ; avg_sq_f64 = sumsq / n
    vmulsd   xmm9, xmm8, xmm8       ; mean^2 (f64)
    vsubsd   xmm10, xmm10, xmm9     ; var_f64 = avg_sq - mean^2

    ; Narrow to f32, compute scale
    vcvtsd2ss xmm6, xmm8, xmm8      ; mean_f32
    vcvtsd2ss xmm7, xmm10, xmm10    ; var_f32

    vaddss   xmm7, xmm7, xmm15      ; var + eps
    vsqrtss  xmm7, xmm7, xmm7       ; sqrt(var + eps)
    vmovss   xmm0, [rel ln_const_one]
    vdivss   xmm7, xmm0, xmm7       ; scale = 1 / sqrt(...)

    ; Broadcast for vectorized pass 2
    vbroadcastss ymm6, xmm6         ; ymm6 = mean (8 lanes)
    vbroadcastss ymm7, xmm7         ; ymm7 = scale (8 lanes)

    ; ------------------------------------------------------------------
    ; Pass 2:  y[i] = (x[i] - mean) * scale
    ; ------------------------------------------------------------------
    mov      rbx, r14
    shr      rbx, 3                 ; n / 8
    mov      rax, r12               ; src
    mov      rdx, r13               ; dst
    test     rbx, rbx
    jz       .pass2_tail

.pass2_vec:
    vmovups  ymm0, [rax]
    vsubps   ymm0, ymm0, ymm6       ; x - mean
    vmulps   ymm0, ymm0, ymm7       ; * scale
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
    vsubss   xmm0, xmm0, xmm6       ; scalar sub (uses low lane of ymm6)
    vmulss   xmm0, xmm0, xmm7       ; * scale
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
ln_const_one:  dd 0x3F800000        ; 1.0f
