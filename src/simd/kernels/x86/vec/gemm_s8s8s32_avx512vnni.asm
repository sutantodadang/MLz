; ----------------------------------------------------------------------------
; gemm_s8s8s32_avx512vnni.asm — INT8 GEMM microkernel (AVX512-VNNI, 256-bit)
; ----------------------------------------------------------------------------
;
; void simd_gemm_s8s8s32_avx512vnni(int M, int N, int K,
;                                   const int8_t * A,   // M x K, row-major
;                                   const int8_t * B,   // N x K, row-major
;                                   int32_t * C);        // M x N, row-major
;
; Same contract/layout as the AVX2 variant. Uses vpdpbusd (VNNI) for the inner
; product: one instruction folds 32 u8*s8 products into 8 int32 accumulators,
; replacing the vpmaddubsw + vpmaddwd + vpaddd chain.
;
; Requires AVX512-VNNI + AVX512VL (vpdpbusd on ymm). vpdpbusd is u8*s8, so the
; same sign trick supplies an unsigned |a| and a sign-folded b.
;
; ponytail: still a naive triple loop (no tiling). VNNI only shortens the inner
; chain; blocking is the next lever if throughput matters.
;
; Win64:  rcx=M, rdx=N, r8=K, r9=A, [rbp+48]=B, [rbp+56]=C
; SysV:   edi=M, esi=N, edx=K, rcx=A, r8=B, r9=C
; ----------------------------------------------------------------------------

bits 64
default rel

section .text
%ifdef WINDOWS
    global simd_gemm_s8s8s32_avx512vnni
%else
    global simd_gemm_s8s8s32_avx512vnni:function hidden
%endif

simd_gemm_s8s8s32_avx512vnni:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rsi
    push    rdi
    sub     rsp, 168
    vmovdqu [rsp+  0], xmm6
    vmovdqu [rsp+ 16], xmm7
    vmovdqu [rsp+ 32], xmm8
    vmovdqu [rsp+ 48], xmm9
    vmovdqu [rsp+ 64], xmm10
    vmovdqu [rsp+ 80], xmm11
    vmovdqu [rsp+ 96], xmm12
    vmovdqu [rsp+112], xmm13
    vmovdqu [rsp+128], xmm14
    vmovdqu [rsp+144], xmm15

%ifdef WINDOWS
    mov     rbx, rcx
    mov     r12, rdx
    mov     r13, r8
    mov     r14, r9
    mov     r15, [rbp+48]
    mov     r10, [rbp+56]
%else
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    mov     r14, rcx
    mov     r15, r8
    mov     r10, r9
%endif
    movsxd  rbx, ebx
    movsxd  r12, r12d
    movsxd  r13, r13d

    test    rbx, rbx
    jle     .done
    test    r12, r12
    jle     .done

.m_loop:
    mov     r11, r15                  ; B_col = B base
    mov     r9, r12                   ; n counter = N
.n_loop:
    vpxor   ymm0, ymm0, ymm0          ; int32 accumulator

    mov     rdx, r14                  ; pa = A_row
    mov     rax, r11                  ; pb = B_col
    mov     r8, r13
    shr     r8, 5                     ; k / 32
    test    r8, r8
    jz      .k_reduce

.k_loop:
    vmovdqu ymm1, [rdx]
    vmovdqu ymm2, [rax]
    add     rdx, 32
    add     rax, 32
    vpsignb ymm3, ymm1, ymm1          ; |a| (unsigned)
    vpsignb ymm2, ymm2, ymm1          ; b * sign(a)
    vpdpbusd ymm0, ymm3, ymm2         ; acc += sum(u8 * s8)  (VNNI)
    dec     r8
    jnz     .k_loop

.k_reduce:
    vextracti128 xmm1, ymm0, 1
    vpaddd  xmm0, xmm0, xmm1
    vphaddd xmm0, xmm0, xmm0
    vphaddd xmm0, xmm0, xmm0
    vmovd   esi, xmm0

    mov     rcx, r13
    and     rcx, 31
    jz      .store
.tail_loop:
    movsx   edi, byte [rdx]
    movsx   r8d, byte [rax]
    imul    edi, r8d
    add     esi, edi
    inc     rdx
    inc     rax
    dec     rcx
    jnz     .tail_loop

.store:
    mov     [r10], esi
    add     r10, 4
    add     r11, r13
    dec     r9
    jnz     .n_loop

    add     r14, r13
    dec     rbx
    jnz     .m_loop

.done:
    vmovdqu xmm6,  [rsp+  0]
    vmovdqu xmm7,  [rsp+ 16]
    vmovdqu xmm8,  [rsp+ 32]
    vmovdqu xmm9,  [rsp+ 48]
    vmovdqu xmm10, [rsp+ 64]
    vmovdqu xmm11, [rsp+ 80]
    vmovdqu xmm12, [rsp+ 96]
    vmovdqu xmm13, [rsp+112]
    vmovdqu xmm14, [rsp+128]
    vmovdqu xmm15, [rsp+144]
    add     rsp, 168
    pop     rdi
    pop     rsi
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret
