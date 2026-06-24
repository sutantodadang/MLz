; ----------------------------------------------------------------------------
; gemm_s8s8s32_avx2.asm — INT8 GEMM microkernel (AVX2)
; ----------------------------------------------------------------------------
;
; void simd_gemm_s8s8s32_avx2(int M, int N, int K,
;                             const int8_t * A,   // M x K, row-major
;                             const int8_t * B,   // N x K, row-major (i.e. B^T)
;                             int32_t * C);        // M x N, row-major
;
; Computes  C[m*N + n] = sum_{k} A[m*K + k] * B[n*K + k]   (signed int8 products,
; accumulated in int32). Each output is the dot product of an A row and a B row.
;
; x86 has no direct s8*s8 multiply; use ggml's sign trick:
;   |a| = vpsignb(a, a)          (abs, in [0,127])
;   b'  = vpsignb(b, a)          (b folded with sign of a)
;   vpmaddubsw(|a|, b') -> int16 pair sums   (no overflow: <= 2*127*127 < 32768)
;   vpmaddwd(., ones16) -> int32             (sum adjacent pairs)
;
; ponytail: naive triple loop, no register/cache blocking. O(M*N*K). Upgrade to
; an MR x NR tiled microkernel if GEMM throughput ever matters. K must be a
; multiple of 32 for the vector path; a scalar tail covers K % 32.
;
; Win64:  rcx=M, rdx=N, r8=K, r9=A, [rbp+48]=B, [rbp+56]=C
; SysV:   edi=M, esi=N, edx=K, rcx=A, r8=B, r9=C
; ----------------------------------------------------------------------------

bits 64
default rel

section .text
%ifdef WINDOWS
    global simd_gemm_s8s8s32_avx2
%else
    global simd_gemm_s8s8s32_avx2:function hidden
%endif

simd_gemm_s8s8s32_avx2:
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

    ; --- gather args into callee-saved regs (uniform across ABIs) ---
%ifdef WINDOWS
    mov     rbx, rcx                  ; M
    mov     r12, rdx                  ; N
    mov     r13, r8                   ; K
    mov     r14, r9                   ; A base
    mov     r15, [rbp+48]             ; B base   (Win64 5th arg, past shadow)
    mov     r10, [rbp+56]             ; C base   (Win64 6th arg)
%else
    mov     rbx, rdi                  ; M
    mov     r12, rsi                  ; N
    mov     r13, rdx                  ; K
    mov     r14, rcx                  ; A base
    mov     r15, r8                   ; B base
    mov     r10, r9                   ; C base
%endif
    movsxd  rbx, ebx                  ; sign-extend the int args to 64-bit
    movsxd  r12, r12d
    movsxd  r13, r13d

    ; ones16 = {1,1,...} int16 x16  (for vpmaddwd reduction)
    vpcmpeqw ymm15, ymm15, ymm15      ; all 0xFFFF (-1 per word)
    vpsrlw   ymm15, ymm15, 15         ; -> 0x0001 per word

    test    rbx, rbx
    jle     .done
    test    r12, r12
    jle     .done

.m_loop:
    mov     r11, r15                  ; B_col = B base (reset each A row)
    mov     r9, r12                   ; n counter = N
.n_loop:
    vpxor   ymm0, ymm0, ymm0          ; int32 accumulator (8 lanes)

    mov     rdx, r14                  ; pa = A_row
    mov     rax, r11                  ; pb = B_col
    mov     r8, r13                   ; k remaining
    shr     r8, 5                     ; k / 32
    test    r8, r8
    jz      .k_reduce

.k_loop:
    vmovdqu ymm1, [rdx]               ; 32 s8 from A
    vmovdqu ymm2, [rax]               ; 32 s8 from B
    add     rdx, 32
    add     rax, 32
    vpsignb ymm3, ymm1, ymm1          ; |a|
    vpsignb ymm2, ymm2, ymm1          ; b * sign(a)
    vpmaddubsw ymm3, ymm3, ymm2       ; 16 int16 pair-sums
    vpmaddwd  ymm3, ymm3, ymm15       ; 8 int32
    vpaddd  ymm0, ymm0, ymm3
    dec     r8
    jnz     .k_loop

.k_reduce:
    ; --- horizontal reduce ymm0 (8 int32) -> esi ---
    vextracti128 xmm1, ymm0, 1
    vpaddd  xmm0, xmm0, xmm1          ; 4 int32
    vphaddd xmm0, xmm0, xmm0          ; 2 int32
    vphaddd xmm0, xmm0, xmm0          ; 1 int32
    vmovd   esi, xmm0                 ; partial dot in esi

    ; --- scalar tail for K % 32  (rdx=pa, rax=pb already past vector region) ---
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
    mov     [r10], esi                ; C[m,n]
    add     r10, 4                    ; next C element (row-major)
    add     r11, r13                  ; B_col += K  (next B row)
    dec     r9
    jnz     .n_loop

    add     r14, r13                  ; A_row += K  (next A row)
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
