; ----------------------------------------------------------------------------
; gemm_s8s8s32_avx2_tiled.asm — tiled INT8 GEMM (AVX2 + FMA), 4x2 register block
; ----------------------------------------------------------------------------
;
; void simd_gemm_s8s8s32_avx2_t(int M, int N, int K,
;                               const int8_t * A,   // M x K row-major
;                               const int8_t * B,   // N x K row-major
;                               int32_t * C);        // M x N row-major
;
; Same MR x NR = 4 x 2 register blocking as the VNNI tile, but AVX2 has no
; vpdpbusd and the unsigned-offset trick would overflow int16 in vpmaddubsw
; (u8 in [0,255] * s8 -> up to 2*255*127 > 32767). So this keeps ggml's |a|
; sign trick, which bounds the unsigned operand to [0,127]:
;   |a| = vpsignb(a,a),  b' = vpsignb(b,a),  vpmaddubsw(|a|,b') -> int16 pairs,
;   vpmaddwd(.,ones16) -> int32.
; The sign fold is per (m,n) (b folded with the sign of that A row), but the A
; and B *loads* are still shared across the tile — that is the reuse that beats
; the naive kernel's M-fold re-read of B.
;
; ponytail: requires M%4==0, N%2==0, K%32==0; the C dispatcher routes aligned
; shapes here (when VNNI is absent) and any other shape to the naive kernel.
;
; Win64:  rcx=M, rdx=N, r8=K, r9=A, [rbp+48]=B, [rbp+56]=C
; SysV:   edi=M, esi=N, edx=K, rcx=A, r8=B, r9=C
; ----------------------------------------------------------------------------

bits 64
default rel

section .text
%ifdef WINDOWS
    global simd_gemm_s8s8s32_avx2_t
%else
    global simd_gemm_s8s8s32_avx2_t:function hidden
%endif

simd_gemm_s8s8s32_avx2_t:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rsi
    push    rdi
    sub     rsp, 176
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
    mov     eax, ecx
    mov     r12d, edx
    mov     r13d, r8d
    mov     r14, r9
    mov     r15, [rbp+48]
    mov     rbx, [rbp+56]
%else
    mov     eax, edi
    mov     r12d, esi
    mov     r13d, edx
    mov     r14, rcx
    mov     r15, r8
    mov     rbx, r9
%endif
    movsxd  rax, eax
    movsxd  r12, r12d
    movsxd  r13, r13d
    mov     [rsp+168], rax            ; M

    ; ones16 = {1}x16 int16 for vpmaddwd reduction
    vpcmpeqw ymm15, ymm15, ymm15
    vpsrlw   ymm15, ymm15, 15

    mov     rax, [rsp+168]
    shr     rax, 2                    ; M/4
    test    rax, rax
    jz      .done
    mov     [rsp+168], rax

.m_tile:
    mov     r10, r15                  ; B_block = B base
    mov     rdx, r12
    shr     rdx, 1                    ; n_tile counter = N/2
.n_tile:
    vpxor   ymm0, ymm0, ymm0
    vpxor   ymm1, ymm1, ymm1
    vpxor   ymm2, ymm2, ymm2
    vpxor   ymm3, ymm3, ymm3
    vpxor   ymm4, ymm4, ymm4
    vpxor   ymm5, ymm5, ymm5
    vpxor   ymm6, ymm6, ymm6
    vpxor   ymm7, ymm7, ymm7

    mov     rsi, r14                  ; a0
    lea     rdi, [r14 + r13]          ; a1
    lea     r8,  [r14 + r13*2]        ; a2
    lea     r9,  [r8  + r13]          ; a3
    mov     r11, r10                  ; b0
    lea     rcx, [r10 + r13]
    mov     [rsp+160], rcx            ; b1 ptr
    mov     rcx, r13
    shr     rcx, 5                    ; K/32

.k_loop:
    vmovdqu ymm8, [r11]               ; B0
    mov     rax, [rsp+160]
    vmovdqu ymm9, [rax]               ; B1

    ; m=0
    vmovdqu ymm10, [rsi]              ; a0
    vpsignb ymm11, ymm10, ymm10       ; |a0|
    vpsignb ymm12, ymm8, ymm10        ; b0 folded w/ sign(a0)
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm0, ymm0, ymm12
    vpsignb ymm12, ymm9, ymm10        ; b1 folded
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm1, ymm1, ymm12
    ; m=1
    vmovdqu ymm10, [rdi]
    vpsignb ymm11, ymm10, ymm10
    vpsignb ymm12, ymm8, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm2, ymm2, ymm12
    vpsignb ymm12, ymm9, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm3, ymm3, ymm12
    ; m=2
    vmovdqu ymm10, [r8]
    vpsignb ymm11, ymm10, ymm10
    vpsignb ymm12, ymm8, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm4, ymm4, ymm12
    vpsignb ymm12, ymm9, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm5, ymm5, ymm12
    ; m=3
    vmovdqu ymm10, [r9]
    vpsignb ymm11, ymm10, ymm10
    vpsignb ymm12, ymm8, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm6, ymm6, ymm12
    vpsignb ymm12, ymm9, ymm10
    vpmaddubsw ymm12, ymm11, ymm12
    vpmaddwd  ymm12, ymm12, ymm15
    vpaddd  ymm7, ymm7, ymm12

    add     rsi, 32
    add     rdi, 32
    add     r8, 32
    add     r9, 32
    add     r11, 32
    add     qword [rsp+160], 32
    dec     rcx
    jnz     .k_loop

    ; --- C tile base in rbx; row stride bytes = N*4 in r9 ---
    mov     r9, r12
    shl     r9, 2

    ; m=0
    vextracti128 xmm10, ymm0, 1
    vpaddd  xmm10, xmm0, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [rbx], xmm10
    vextracti128 xmm10, ymm1, 1
    vpaddd  xmm10, xmm1, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [rbx+4], xmm10
    ; m=1
    lea     r8, [rbx + r9]
    vextracti128 xmm10, ymm2, 1
    vpaddd  xmm10, xmm2, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8], xmm10
    vextracti128 xmm10, ymm3, 1
    vpaddd  xmm10, xmm3, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8+4], xmm10
    ; m=2
    lea     r8, [rbx + r9*2]
    vextracti128 xmm10, ymm4, 1
    vpaddd  xmm10, xmm4, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8], xmm10
    vextracti128 xmm10, ymm5, 1
    vpaddd  xmm10, xmm5, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8+4], xmm10
    ; m=3
    lea     r8, [rbx + r9*2]
    add     r8, r9
    vextracti128 xmm10, ymm6, 1
    vpaddd  xmm10, xmm6, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8], xmm10
    vextracti128 xmm10, ymm7, 1
    vpaddd  xmm10, xmm7, xmm10
    vphaddd xmm10, xmm10, xmm10
    vphaddd xmm10, xmm10, xmm10
    vmovd   [r8+4], xmm10

    lea     r10, [r10 + r13*2]        ; B_block += 2*K
    add     rbx, 8                    ; C tile += 2 cols
    dec     rdx
    jnz     .n_tile

    lea     r14, [r14 + r13*4]        ; A += 4*K
    mov     r9, r12
    shl     r9, 2
    lea     rbx, [rbx + r9*2]
    add     rbx, r9                   ; rbx += 3*N*4 (one row already added by n_tiles)
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
