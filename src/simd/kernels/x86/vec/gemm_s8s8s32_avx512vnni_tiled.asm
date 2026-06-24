; ----------------------------------------------------------------------------
; gemm_s8s8s32_avx512vnni_tiled.asm — tiled INT8 GEMM (AVX512-VNNI, 256-bit)
; ----------------------------------------------------------------------------
;
; void simd_gemm_s8s8s32_avx512vnni_t(int M, int N, int K,
;                                     const int8_t * A,   // M x K row-major
;                                     const int8_t * B,   // N x K row-major
;                                     int32_t * C);        // M x N row-major
;
; MR x NR = 4 x 2 register-blocked microkernel. Each A row chunk feeds NR=2
; outputs and each B row chunk feeds MR=4 outputs per K-step, so an A/B load is
; reused across the tile instead of being re-read once per (m,n). That cuts the
; memory traffic that makes the naive kernel bandwidth-bound (where VNNI buys
; nothing) and lets vpdpbusd actually pull ahead.
;
; vpdpbusd is u8*s8. To avoid a per-row sign fold (which would defeat B reuse)
; use the unsigned-offset trick:
;   A_u8 = A XOR 0x80            (== A + 128, reinterpreted unsigned in [1,255])
;   acc_raw[m][n] = sum_k A_u8[m][k] * B[n][k]                       (vpdpbusd)
;   csum[n]       = sum_k B[n][k]            (vpdpbusd with a u8=1 vector)
;   C[m][n]       = acc_raw[m][n] - 128 * csum[n]
; vpdpbusd accumulates in int32 directly, so u8 in [0,255] never overflows
; (the AVX2 vpmaddubsw path WOULD overflow int16 here — that's why this trick is
; VNNI-only; the AVX2 kernel keeps the |a| sign trick instead).
;
; ponytail: requires M%4==0, N%2==0, K%32==0. The C dispatcher routes only the
; aligned bulk here and falls back to the naive kernel for edges. MR/NR could
; grow (more reuse) at the cost of register pressure; 4x2 already fits in 15 ymm.
;
; Win64:  rcx=M, rdx=N, r8=K, r9=A, [rbp+48]=B, [rbp+56]=C
; SysV:   edi=M, esi=N, edx=K, rcx=A, r8=B, r9=C
; ----------------------------------------------------------------------------

bits 64
default rel

section .text
%ifdef WINDOWS
    global simd_gemm_s8s8s32_avx512vnni_t
%else
    global simd_gemm_s8s8s32_avx512vnni_t:function hidden
%endif

simd_gemm_s8s8s32_avx512vnni_t:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rsi
    push    rdi
    sub     rsp, 176                  ; xmm6-15 (160) + 16 scratch for csum
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
    ; [rsp+160], [rsp+164] = 128*csum0, 128*csum1 scratch

%ifdef WINDOWS
    mov     eax, ecx                  ; M
    mov     r12d, edx                 ; N
    mov     r13d, r8d                 ; K
    mov     r14, r9                   ; A base
    mov     r15, [rbp+48]             ; B base
    mov     rbx, [rbp+56]             ; C base
%else
    mov     eax, edi                  ; M
    mov     r12d, esi                 ; N
    mov     r13d, edx                 ; K
    mov     r14, rcx                  ; A base
    mov     r15, r8                   ; B base
    mov     rbx, r9                   ; C base
%endif
    movsxd  rax, eax                  ; M (64-bit)
    movsxd  r12, r12d                 ; N
    movsxd  r13, r13d                 ; K
    mov     [rsp+168], rax            ; stash M (need rax as scratch)

    ; constants: ymm14 = 0x80 bytes (XOR mask), ymm13 = 0x01 bytes (u8 ones)
    mov     eax, 0x80808080
    vmovd   xmm14, eax
    vpbroadcastd ymm14, xmm14
    mov     eax, 0x01010101
    vmovd   xmm13, eax
    vpbroadcastd ymm13, xmm13

    ; m_tile loop: M/4 iterations. r14 = A row-block base (advances 4*K).
    mov     rax, [rsp+168]
    shr     rax, 2                    ; M/4
    test    rax, rax
    jz      .done
    mov     [rsp+168], rax            ; remaining m_tiles

.m_tile:
    ; n_tile loop: N/2 iterations. B column-block base advances 2*K.
    mov     r10, r15                  ; B_block = B base
    mov     rdx, r12
    shr     rdx, 1                    ; rdx = N/2  (n_tile counter)
.n_tile:
    ; zero accumulators + csums
    vpxor   ymm0, ymm0, ymm0
    vpxor   ymm1, ymm1, ymm1
    vpxor   ymm2, ymm2, ymm2
    vpxor   ymm3, ymm3, ymm3
    vpxor   ymm4, ymm4, ymm4
    vpxor   ymm5, ymm5, ymm5
    vpxor   ymm6, ymm6, ymm6
    vpxor   ymm7, ymm7, ymm7
    vpxor   ymm8, ymm8, ymm8          ; csum0
    vpxor   ymm9, ymm9, ymm9          ; csum1

    ; A row pointers a0..a3 = r14 + {0,K,2K,3K}
    mov     rsi, r14                  ; a0
    lea     rdi, [r14 + r13]          ; a1
    lea     r8,  [r14 + r13*2]        ; a2
    lea     r9,  [r8  + r13]          ; a3
    ; B col pointers b0,b1 = r10 + {0,K}
    mov     r11, r10                  ; b0
    ; b1 = r10 + K -> keep in a reg; reuse rcx
    lea     rcx, [r10 + r13]          ; b1 (overwritten as counter? need both) -> use stack
    mov     [rsp+160], rcx            ; b1 ptr saved
    mov     rcx, r13
    shr     rcx, 5                    ; K/32 (k-step count)

.k_loop:
    vmovdqu ymm10, [r11]              ; B0
    mov     rax, [rsp+160]
    vmovdqu ymm11, [rax]             ; B1
    vpdpbusd ymm8, ymm13, ymm10       ; csum0 += 1 * B0
    vpdpbusd ymm9, ymm13, ymm11       ; csum1 += 1 * B1

    vpxor   ymm12, ymm14, [rsi]       ; a0_u8
    vpdpbusd ymm0, ymm12, ymm10       ; acc00
    vpdpbusd ymm1, ymm12, ymm11       ; acc01
    vpxor   ymm12, ymm14, [rdi]       ; a1_u8
    vpdpbusd ymm2, ymm12, ymm10       ; acc10
    vpdpbusd ymm3, ymm12, ymm11       ; acc11
    vpxor   ymm12, ymm14, [r8]        ; a2_u8
    vpdpbusd ymm4, ymm12, ymm10       ; acc20
    vpdpbusd ymm5, ymm12, ymm11       ; acc21
    vpxor   ymm12, ymm14, [r9]        ; a3_u8
    vpdpbusd ymm6, ymm12, ymm10       ; acc30
    vpdpbusd ymm7, ymm12, ymm11       ; acc31

    add     rsi, 32
    add     rdi, 32
    add     r8, 32
    add     r9, 32
    add     r11, 32
    add     qword [rsp+160], 32       ; advance b1
    dec     rcx
    jnz     .k_loop

    ; --- reduce csum0,csum1 -> 128*csum into scratch dwords ---
    vextracti128 xmm10, ymm8, 1
    vpaddd  xmm10, xmm8, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    shl     eax, 7                    ; *128
    mov     [rsp+160], eax            ; 128*csum0
    vextracti128 xmm11, ymm9, 1
    vpaddd  xmm11, xmm9, xmm11
    vphaddd xmm11, xmm11, xmm11
    vphaddd xmm11, xmm11, xmm11
    vmovd   eax, xmm11
    shl     eax, 7
    mov     [rsp+164], eax            ; 128*csum1

    ; --- C tile base: C + (m_row*N + n_col)*4 ; m_row=(M/4 done)*4, n_col=(N/2 done)*2
    ; We track via rbx (C base for current tile). Compute element offsets inline.
    ; Row stride in bytes = N*4. Output 8 values acc[m][n].
    ; C_tile pointer is maintained in rbx across n_tiles/m_tiles (advances 2 cols
    ; per n_tile, and to next 4-row block per m_tile). Here rbx points at C[m0][n0].
    mov     r9, r12                   ; N
    shl     r9, 2                     ; row stride bytes = N*4

    ; helper: reduce ymmX (still live in ymm0..7) -> eax, subtract 128*csum[n]
    ; m=0
    vextracti128 xmm10, ymm0, 1
    vpaddd  xmm10, xmm0, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+160]
    mov     [rbx], eax                ; C[0][0]
    vextracti128 xmm10, ymm1, 1
    vpaddd  xmm10, xmm1, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+164]
    mov     [rbx+4], eax              ; C[0][1]
    ; m=1
    lea     r8, [rbx + r9]
    vextracti128 xmm10, ymm2, 1
    vpaddd  xmm10, xmm2, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+160]
    mov     [r8], eax
    vextracti128 xmm10, ymm3, 1
    vpaddd  xmm10, xmm3, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+164]
    mov     [r8+4], eax
    ; m=2
    lea     r8, [rbx + r9*2]
    vextracti128 xmm10, ymm4, 1
    vpaddd  xmm10, xmm4, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+160]
    mov     [r8], eax
    vextracti128 xmm10, ymm5, 1
    vpaddd  xmm10, xmm5, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+164]
    mov     [r8+4], eax
    ; m=3
    lea     r8, [rbx + r9*2]
    add     r8, r9
    vextracti128 xmm10, ymm6, 1
    vpaddd  xmm10, xmm6, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+160]
    mov     [r8], eax
    vextracti128 xmm10, ymm7, 1
    vpaddd  xmm10, xmm7, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   eax, xmm10
    sub     eax, [rsp+164]
    mov     [r8+4], eax

    ; advance to next n_tile: B_block += 2*K, C tile += 2 cols (8 bytes)
    lea     r10, [r10 + r13*2]
    add     rbx, 8
    dec     rdx
    jnz     .n_tile

    ; advance to next m_tile: A block += 4*K, C += 4 rows - (N already advanced
    ; back? rbx advanced by N/2 * 8 = N*4 = one row. Need 4 rows total minus the
    ; one already added => add 3 row strides.)
    lea     r14, [r14 + r13*4]        ; A += 4*K
    mov     r9, r12
    shl     r9, 2                     ; N*4
    lea     rbx, [rbx + r9*2]
    add     rbx, r9                   ; rbx += 3*N*4  (already +1 row from n_tiles)
    mov     rax, [rsp+168]
    dec     rax
    mov     [rsp+168], rax
    jnz     .m_tile

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
    add     rsp, 176
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
