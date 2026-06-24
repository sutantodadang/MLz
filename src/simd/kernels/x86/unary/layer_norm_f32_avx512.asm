; ----------------------------------------------------------------------------
; layer_norm_f32_avx512.asm  --  Single-row Layer Normalization (AVX-512F)
; ----------------------------------------------------------------------------
;
; Same contract as the AVX2 variant; processes 16 f32 / iter.
; Numerical strategy: parallel f64 accumulators (8 lanes each for lo/hi)
; using vcvtps2pd zmm, ymm halves; matches AVX2 ULP envelope (<= 3 ULP).
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
global simd_layer_norm_f32_avx512

simd_layer_norm_f32_avx512:
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

    vmovss   xmm15, xmm15, ARG_EPS  ; eps
    mov      r14, ARG_N             ; n
    mov      r12, ARG_X             ; src
    mov      r13, ARG_Y             ; dst

    ; --- Pass 1: Welford in f64 ---
    ; zmm0(lo sum), zmm1(hi sum)       — 16 f64 lanes for sum
    ; zmm2(lo sumsq), zmm3(hi sumsq)   — 16 f64 lanes for sum-of-squares
    vpxorq   zmm0, zmm0, zmm0
    vpxorq   zmm1, zmm1, zmm1
    vpxorq   zmm2, zmm2, zmm2
    vpxorq   zmm3, zmm3, zmm3

    mov      rbx, r14
    shr      rbx, 4                 ; n / 16
    test     rbx, rbx
    jz       .pass1_tail

    mov      rax, r12

.pass1_vec:
    vmovups  zmm4, [rax]            ; 16 f32

    ; ---- accumulate sum (f64) ----
    vextractf32x8 ymm5, zmm4, 1     ; hi 8 f32
    vcvtps2pd zmm6, ymm4            ; lo 8 -> f64 (8 lanes)
    vcvtps2pd zmm7, ymm5            ; hi 8 -> f64 (8 lanes)
    vaddpd   zmm0, zmm0, zmm6       ; sum_lo
    vaddpd   zmm1, zmm1, zmm7       ; sum_hi

    ; ---- accumulate sum-of-squares (f64) ----
    vmulps   zmm4, zmm4, zmm4       ; x^2 in f32
    vextractf32x8 ymm5, zmm4, 1     ; hi 8 squares
    vcvtps2pd zmm6, ymm4            ; lo 8 -> f64
    vcvtps2pd zmm7, ymm5            ; hi 8 -> f64
    vaddpd   zmm2, zmm2, zmm6       ; sumsq_lo
    vaddpd   zmm3, zmm3, zmm7       ; sumsq_hi

    add      rax, 64
    dec      rbx
    jnz      .pass1_vec

.pass1_tail:
    ; ---- horizontal-sum sum -> xmm6 (scalar f64) ----
    vaddpd   zmm0, zmm0, zmm1       ; combine lo+hi (8 f64 lanes)
    vextractf64x4 ymm1, zmm0, 1
    vaddpd   ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddpd   xmm0, xmm0, xmm1
    vhaddpd  xmm0, xmm0, xmm0
    vmovapd  xmm6, xmm0             ; xmm6 = sum (f64)

    ; ---- horizontal-sum sumsq -> xmm7 (scalar f64) ----
    vaddpd   zmm2, zmm2, zmm3
    vextractf64x4 ymm3, zmm2, 1
    vaddpd   ymm2, ymm2, ymm3
    vextractf128 xmm3, ymm2, 1
    vaddpd   xmm2, xmm2, xmm3
    vhaddpd  xmm2, xmm2, xmm2
    vmovapd  xmm7, xmm2             ; xmm7 = sumsq (f64)

    ; Scalar tail (n & 15)
    mov      rax, r14
    and      rax, 15
    jz       .pass1_done

    mov      rcx, r14
    and      rcx, ~15
    lea      rdx, [r12 + rcx*4]

.pass1_tail_loop:
    vmovss   xmm0, [rdx]            ; x (f32)
    vcvtss2sd xmm1, xmm1, xmm0      ; x -> f64
    vaddsd   xmm6, xmm6, xmm1       ; sum += x

    vmulss   xmm0, xmm0, xmm0       ; x^2
    vcvtss2sd xmm1, xmm1, xmm0      ; x^2 -> f64
    vaddsd   xmm7, xmm7, xmm1       ; sumsq += x^2

    add      rdx, 4
    dec      rax
    jnz      .pass1_tail_loop

.pass1_done:
    ; xmm6 = sum (f64), xmm7 = sumsq (f64)
    vcvtsi2sd xmm0, xmm0, r14       ; f64(n)

    vdivsd   xmm6, xmm6, xmm0       ; mean_f64 = sum / n
    vdivsd   xmm7, xmm7, xmm0       ; avg_sq_f64 = sumsq / n
    vmulsd   xmm8, xmm6, xmm6       ; mean^2
    vsubsd   xmm7, xmm7, xmm8       ; var_f64 = avg_sq - mean^2

    ; Narrow to f32
    vcvtsd2ss xmm6, xmm6, xmm6      ; mean_f32
    vcvtsd2ss xmm7, xmm7, xmm7      ; var_f32

    vaddss   xmm7, xmm7, xmm15      ; var + eps
    vsqrtss  xmm7, xmm7, xmm7       ; sqrt(var + eps)
    vmovss   xmm0, [rel ln_const_one_z]
    vdivss   xmm7, xmm0, xmm7       ; scale = 1 / sqrt(...)

    ; Broadcast for 16-wide pass 2
    vbroadcastss zmm6, xmm6         ; zmm6 = mean (16 lanes)
    vbroadcastss zmm7, xmm7         ; zmm7 = scale (16 lanes)

    ; --- Pass 2: y = (x - mean) * scale (16-wide) ---
    mov      rbx, r14
    shr      rbx, 4                 ; n / 16
    mov      rax, r12               ; src
    mov      rdx, r13               ; dst
    test     rbx, rbx
    jz       .pass2_tail

.pass2_vec:
    vmovups  zmm0, [rax]
    vsubps   zmm0, zmm0, zmm6       ; x - mean
    vmulps   zmm0, zmm0, zmm7       ; * scale
    vmovups  [rdx], zmm0
    add      rax, 64
    add      rdx, 64
    dec      rbx
    jnz      .pass2_vec

.pass2_tail:
    mov      rcx, r14
    and      rcx, 15
    jz       .epilogue

.pass2_tail_loop:
    vmovss   xmm0, [rax]
    vsubss   xmm0, xmm0, xmm6
    vmulss   xmm0, xmm0, xmm7
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
ln_const_one_z:  dd 0x3F800000
