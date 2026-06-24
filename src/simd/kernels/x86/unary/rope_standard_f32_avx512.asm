; ===========================================================================
; rope_standard_f32_avx512.asm  --  Standard (interleaved) RoPE, f32, 8-pair.
; ---------------------------------------------------------------------------
;   void simd_rope_standard_f32_avx512(int64_t n_pairs,
;                                      const float * cache,
;                                      const float * src,
;                                      float       * dst);
;
;   for ic in [0, n_pairs):
;     cos = cache[2*ic + 0]; sin = cache[2*ic + 1]
;     x0  = src[2*ic + 0];   x1   = src[2*ic + 1]
;     dst[2*ic + 0] = x0*cos - x1*sin
;     dst[2*ic + 1] = x0*sin + x1*cos
;
; Processes 8 pairs (16 floats) per iteration.
; Cache and src deinterleaving via vpermps with single-source index regs.
; Result interleaving via vpermi2ps (two-source).
;
; Win64 ABI: rcx=n_pairs, rdx=cache, r8=src, r9=dst.
; Saves xmm6-xmm15 + rbx,rsi,rdi,r12-r15. vzeroupper before ret.
; ===========================================================================

bits 64
default rel

%ifdef WINDOWS
global simd_rope_standard_f32_avx512
%else
global simd_rope_standard_f32_avx512:function
%endif

section .rodata align=64

; vpermps indices to deinterleave [a0,b0,a1,b1,...,a7,b7] -> evens[8]/odds[8].
; Single-source: indices 0..15 map to lanes 0..15 of the source zmm.
align 64
rstd_cos_idx:
        dd      0,  2,  4,  6,  8, 10, 12, 14,  0,  2,  4,  6,  8, 10, 12, 14
align 64
rstd_sin_idx:
        dd      1,  3,  5,  7,  9, 11, 13, 15,  1,  3,  5,  7,  9, 11, 13, 15

; vpermi2ps interleave indices: combines dst_even (source 0, lanes 0-7)
; and dst_odd (source 1, lanes 16-23) back into interleaved order.
; Result: [E0,O0,E1,O1,E2,O2,E3,O3,E4,O4,E5,O5,E6,O6,E7,O7]
align 64
rstd_interleave_idx:
        dd      0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23

section .text

simd_rope_standard_f32_avx512:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 176

        vmovaps [rsp + 0x00], xmm6
        vmovaps [rsp + 0x10], xmm7
        vmovaps [rsp + 0x20], xmm8
        vmovaps [rsp + 0x30], xmm9
        vmovaps [rsp + 0x40], xmm10
        vmovaps [rsp + 0x50], xmm11
        vmovaps [rsp + 0x60], xmm12
        vmovaps [rsp + 0x70], xmm13
        vmovaps [rsp + 0x80], xmm14
        vmovaps [rsp + 0x90], xmm15

        mov     r10, rcx                ; n_pairs
        test    r10, r10
        jle     .done

        xor     rax, rax                ; ic = 0

        ; vector loop: 8 pairs per iteration
        mov     r15, r10
        and     r15, ~7                 ; round down to 8
        cmp     r15, 0
        je      .scalar_tail

        ; Pre-load index constants into zmm registers
        vmovaps zmm14, [rel rstd_cos_idx]
        vmovaps zmm15, [rel rstd_sin_idx]
        vmovaps zmm13, [rel rstd_interleave_idx]

.vec_loop:
        ; --- Load and deinterleave cache (8 cos/sin pairs = 16 floats) ---
        lea     rcx, [rdx + rax*8]      ; cache + ic*8
        vmovups zmm0, [rcx]             ; [c0,s0,c1,s1,...,c7,s7]

        vpermps zmm1, zmm14, zmm0       ; cos = [c0,c1,...,c7 | dup8]
        vpermps zmm2, zmm15, zmm0       ; sin = [s0,s1,...,s7 | dup8]

        ; --- Load and deinterleave src (8 x0/x1 pairs = 16 floats) ---
        lea     rcx, [r8 + rax*8]       ; src + ic*8
        vmovups zmm3, [rcx]             ; [x0_0,x1_0,...,x0_7,x1_7]

        vpermps zmm4, zmm14, zmm3       ; x0 = [x0_0,...,x0_7 | dup8]
        vpermps zmm5, zmm15, zmm3       ; x1 = [x1_0,...,x1_7 | dup8]

        ; --- Compute ---
        ; dst_even = x0*cos - x1*sin
        vmulps       zmm6, zmm4, zmm1
        vfnmadd231ps zmm6, zmm5, zmm2

        ; dst_odd = x0*sin + x1*cos
        vmulps       zmm7, zmm4, zmm2
        vfmadd231ps  zmm7, zmm5, zmm1

        ; --- Interleave results and store ---
        ; vpermi2ps uses indices in first operand (modified) to select from
        ; zmm6 (even, source 0) and zmm7 (odd, source 1).
        vmovaps   zmm8, zmm13           ; copy interleave indices
        vpermi2ps zmm8, zmm6, zmm7      ; zmm8 = interleaved [E0,O0,...,E7,O7]

        lea     rcx, [r9 + rax*8]
        vmovups [rcx], zmm8

        add     rax, 8                  ; ic += 8 pairs
        cmp     rax, r15
        jb      .vec_loop

.scalar_tail:
        cmp     rax, r10
        jge     .done

.scalar_loop:
        lea     rcx, [rdx + rax*8]
        vmovss  xmm0, [rcx]             ; cos
        vmovss  xmm1, [rcx + 4]         ; sin

        lea     rcx, [r8 + rax*8]
        vmovss  xmm2, [rcx]             ; x0
        vmovss  xmm3, [rcx + 4]         ; x1

        vmulss       xmm4, xmm2, xmm0
        vfnmadd231ss xmm4, xmm3, xmm1
        lea     rcx, [r9 + rax*8]
        vmovss  [rcx], xmm4

        vmulss       xmm4, xmm2, xmm1
        vfmadd231ss  xmm4, xmm3, xmm0
        vmovss  [rcx + 4], xmm4

        inc     rax
        cmp     rax, r10
        jb      .scalar_loop

.done:
        vmovaps xmm6,  [rsp + 0x00]
        vmovaps xmm7,  [rsp + 0x10]
        vmovaps xmm8,  [rsp + 0x20]
        vmovaps xmm9,  [rsp + 0x30]
        vmovaps xmm10, [rsp + 0x40]
        vmovaps xmm11, [rsp + 0x50]
        vmovaps xmm12, [rsp + 0x60]
        vmovaps xmm13, [rsp + 0x70]
        vmovaps xmm14, [rsp + 0x80]
        vmovaps xmm15, [rsp + 0x90]
        add     rsp, 176
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        vzeroupper
        ret
