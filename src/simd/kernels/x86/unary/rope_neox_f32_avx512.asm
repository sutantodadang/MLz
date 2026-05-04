; ===========================================================================
; rope_neox_f32_avx512.asm  --  RoPE NEOX rotation for f32, 16-pair stride.
; ---------------------------------------------------------------------------
;   void simd_rope_neox_f32_avx512(int64_t n_pairs,
;                                  const float * cache,
;                                  const float * src,
;                                  float       * dst);
;
;   for ic in [0, n_pairs):
;     cos = cache[2*ic + 0]; sin = cache[2*ic + 1]
;     x0  = src[ic]; x1 = src[ic + n_pairs]
;     dst[ic]            = x0*cos - x1*sin
;     dst[ic + n_pairs]  = x0*sin + x1*cos
;
; Win64 ABI: rcx=n_pairs, rdx=cache, r8=src, r9=dst.
; Saves xmm6-xmm15 + rbx,rsi,rdi,r12-r15. vzeroupper before ret.
;
; Cache deinterleave uses vpermi2ps with index regs preserved across the
; loop (zmm14 = cos indices, zmm15 = sin indices).  Per iteration we copy
; the index reg into a scratch reg before the vpermi2ps (which destroys
; its first operand).
; ===========================================================================

bits 64
default rel

%ifdef WINDOWS
global simd_rope_neox_f32_avx512
%else
global simd_rope_neox_f32_avx512:function
%endif

section .rodata align=64
align 64
rope_avx512_cos_idx:
        dd      0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
align 64
rope_avx512_sin_idx:
        dd      1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31

section .text

simd_rope_neox_f32_avx512:
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

        mov     r10, rcx
        test    r10, r10
        jle     .done

        mov     r11, r10
        shl     r11, 2
        lea     r13, [r8 + r11]
        lea     r14, [r9 + r11]

        xor     rax, rax

        mov     r15, r10
        and     r15, ~15
        cmp     r15, 0
        je      .scalar_tail

        vmovaps zmm14, [rel rope_avx512_cos_idx]
        vmovaps zmm15, [rel rope_avx512_sin_idx]

.vec_loop:
        lea     rcx, [rdx + rax*8]
        vmovups zmm0, [rcx]
        vmovups zmm1, [rcx + 64]

        vmovaps   zmm2, zmm14
        vpermi2ps zmm2, zmm0, zmm1
        vmovaps   zmm3, zmm15
        vpermi2ps zmm3, zmm0, zmm1

        vmovups zmm4, [r8  + rax*4]
        vmovups zmm5, [r13 + rax*4]

        vmulps       zmm6, zmm4, zmm2
        vfnmadd231ps zmm6, zmm5, zmm3
        vmovups      [r9 + rax*4], zmm6

        vmulps       zmm7, zmm4, zmm3
        vfmadd231ps  zmm7, zmm5, zmm2
        vmovups      [r14 + rax*4], zmm7

        add     rax, 16
        cmp     rax, r15
        jb      .vec_loop

.scalar_tail:
        cmp     rax, r10
        jge     .done
.scalar_loop:
        lea     rcx, [rdx + rax*8]
        vmovss  xmm0, [rcx]
        vmovss  xmm1, [rcx + 4]
        vmovss  xmm2, [r8 + rax*4]
        vmovss  xmm3, [r13 + rax*4]

        vmulss       xmm4, xmm2, xmm0
        vfnmadd231ss xmm4, xmm3, xmm1
        vmovss       [r9 + rax*4], xmm4

        vmulss       xmm4, xmm2, xmm1
        vfmadd231ss  xmm4, xmm3, xmm0
        vmovss       [r14 + rax*4], xmm4

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
