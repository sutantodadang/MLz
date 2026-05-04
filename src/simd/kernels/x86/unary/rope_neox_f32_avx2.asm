; ===========================================================================
; rope_neox_f32_avx2.asm  --  RoPE NEOX rotation (and MROPE/IMROPE) for f32.
; ---------------------------------------------------------------------------
; Replaces the scalar `rotate_pairs<float>` upstream uses for any rope mode
; that splits the head into [low | high] halves of width n_dims/2.
;
; Kernel contract:
;   void simd_rope_neox_f32_avx2(int64_t n_pairs,
;                                const float * cache,   // 2*n_pairs floats
;                                const float * src,     // 2*n_pairs floats
;                                float       * dst);    // 2*n_pairs floats
;
; Layout (NEOX / MROPE / IMROPE):
;   for ic in [0, n_pairs):
;     cos = cache[2*ic + 0]
;     sin = cache[2*ic + 1]
;     x0  = src[ic]
;     x1  = src[ic + n_pairs]
;     dst[ic]            = x0*cos - x1*sin
;     dst[ic + n_pairs]  = x0*sin + x1*cos
;
; Win64 ABI:  rcx = n_pairs, rdx = cache, r8 = src, r9 = dst
; Saves xmm6-xmm15 + rbx, rsi, rdi, r12-r15. vzeroupper before ret.
; ===========================================================================

bits 64
default rel

%ifdef WINDOWS
global simd_rope_neox_f32_avx2
%else
global simd_rope_neox_f32_avx2:function
%endif

section .rodata align=32
; vpermd indices to deinterleave a 16-element [c0,s0,c1,s1,...,c7,s7] block
; loaded as two ymm registers into cos = [c0..c7], sin = [s0..s7].
; vshufps + vpermpd path used instead (no rodata required for that).

section .text

simd_rope_neox_f32_avx2:
        ; ---- prologue ------------------------------------------------------
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 176                ; 160 B for xmm6-xmm15 + 16 B align

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

        ; rcx = n_pairs (int64)
        ; rdx = cache (2*n_pairs floats)
        ; r8  = src (2*n_pairs floats)
        ; r9  = dst (2*n_pairs floats)
        mov     r10, rcx                ; n_pairs
        test    r10, r10
        jle     .done

        ; r11 = src_high = src + n_pairs*4
        ; r12 = dst_high = dst + n_pairs*4
        mov     r11, r10
        shl     r11, 2
        lea     r13, [r8 + r11]         ; src_high
        lea     r14, [r9 + r11]         ; dst_high

        xor     rax, rax                ; ic = 0

        ; vector loop: 8 pairs per iteration
        mov     r15, r10
        and     r15, ~7                 ; r15 = n_pairs rounded down to 8
        cmp     r15, 0
        je      .scalar_tail

.vec_loop:
        ; Load 16 cache floats: ymm0 = [c0,s0,c1,s1,c2,s2,c3,s3]
        ;                       ymm1 = [c4,s4,c5,s5,c6,s6,c7,s7]
        lea     rcx, [rdx + rax*8]      ; cache + 2*ic*4 = cache + ic*8
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + 32]

        ; Deinterleave per 128-bit lane:
        ;   vshufps imm=0x88 picks [a0,a2,b0,b2] from inputs [a0,a1,a2,a3],[b0,b1,b2,b3]
        ;   vshufps imm=0xDD picks [a1,a3,b1,b3]
        ; ymm2 (cos pre) = [c0,c1,c4,c5, c2,c3,c6,c7]
        ; ymm3 (sin pre) = [s0,s1,s4,s5, s2,s3,s6,s7]
        vshufps ymm2, ymm0, ymm1, 0x88
        vshufps ymm3, ymm0, ymm1, 0xDD

        ; vpermpd imm=0xD8 reorders 64-bit lanes [0,1,2,3] -> [0,2,1,3]
        ;   yields cos = [c0,c1,c2,c3,c4,c5,c6,c7]
        vpermpd ymm4, ymm2, 0xD8        ; cos
        vpermpd ymm5, ymm3, 0xD8        ; sin

        ; Load src_low and src_high
        vmovups ymm6, [r8  + rax*4]     ; x0
        vmovups ymm7, [r13 + rax*4]     ; x1

        ; dst_low = x0*cos - x1*sin
        vmulps  ymm8, ymm6, ymm4        ; x0*cos
        vfnmadd231ps ymm8, ymm7, ymm5   ; - x1*sin
        vmovups [r9 + rax*4], ymm8

        ; dst_high = x0*sin + x1*cos
        vmulps  ymm9, ymm6, ymm5        ; x0*sin
        vfmadd231ps  ymm9, ymm7, ymm4   ; + x1*cos
        vmovups [r14 + rax*4], ymm9

        add     rax, 8
        cmp     rax, r15
        jb      .vec_loop

.scalar_tail:
        cmp     rax, r10
        jge     .done
.scalar_loop:
        ; cos = cache[2*ic], sin = cache[2*ic+1]
        lea     rcx, [rdx + rax*8]
        vmovss  xmm0, [rcx]             ; cos
        vmovss  xmm1, [rcx + 4]         ; sin
        vmovss  xmm2, [r8 + rax*4]      ; x0
        vmovss  xmm3, [r13 + rax*4]     ; x1

        vmulss  xmm4, xmm2, xmm0
        vfnmadd231ss xmm4, xmm3, xmm1
        vmovss  [r9 + rax*4], xmm4

        vmulss  xmm4, xmm2, xmm1
        vfmadd231ss  xmm4, xmm3, xmm0
        vmovss  [r14 + rax*4], xmm4

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
