; ===========================================================================
; rope_standard_f32_avx2.asm  --  Standard (interleaved) RoPE rotation, f32.
; ---------------------------------------------------------------------------
; Replaces the scalar rotation for the STANDARD rope layout where src pairs
; are interleaved: [x0_0, x1_0, x0_1, x1_1, ...] rather than half-split.
;
; Kernel contract:
;   void simd_rope_standard_f32_avx2(int64_t n_pairs,
;                                    const float * cache,   // 2*n_pairs floats
;                                    const float * src,     // 2*n_pairs floats
;                                    float       * dst);    // 2*n_pairs floats
;
; Layout (STANDARD / interleaved pairs):
;   for ic in [0, n_pairs):
;     cos = cache[2*ic + 0]
;     sin = cache[2*ic + 1]
;     x0  = src[2*ic + 0]       // EVEN index (adjacent pair)
;     x1  = src[2*ic + 1]       // ODD index (adjacent pair)
;     dst[2*ic + 0] = x0*cos - x1*sin
;     dst[2*ic + 1] = x0*sin + x1*cos
;
; Cache layout is interleaved [cos,sin,cos,sin,...] — same as NEOX.
; Src layout is ALSO interleaved — DIFFERENT from NEOX (which is half-split).
;
; Processes 4 pairs (8 floats) per iteration using vshufps+vpermpd
; for deinterleaving.  vunpcklps for interleaving results.
;
; Win64 ABI:  rcx = n_pairs, rdx = cache, r8 = src, r9 = dst
; Saves xmm6-xmm15 + rbx, rsi, rdi, r12-r15. vzeroupper before ret.
; ===========================================================================

bits 64
default rel

%ifdef WINDOWS
global simd_rope_standard_f32_avx2
%else
global simd_rope_standard_f32_avx2:function
%endif

section .text

simd_rope_standard_f32_avx2:
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
        ; rdx = cache (2*n_pairs floats, interleaved [c0,s0,c1,s1,...])
        ; r8  = src (2*n_pairs floats, interleaved [x0_0,x1_0,x0_1,x1_1,...])
        ; r9  = dst (2*n_pairs floats, interleaved)
        mov     r10, rcx                ; n_pairs
        test    r10, r10
        jle     .done

        xor     rax, rax                ; ic = 0

        ; vector loop: 4 pairs per iteration (8 floats)
        mov     r15, r10
        and     r15, ~3                 ; r15 = n_pairs rounded down to 4
        cmp     r15, 0
        je      .scalar_tail

.vec_loop:
        ; --- Load and deinterleave cache (4 cos/sin pairs) ---
        ; cache[8*ic..8*ic+7] = [c0,s0,c1,s1,c2,s2,c3,s3]
        lea     rcx, [rdx + rax*8]      ; cache + 2*ic*4 = cache + ic*8
        vmovups ymm0, [rcx]             ; [c0,s0,c1,s1,c2,s2,c3,s3]

        ; vshufps picks evens/odds per 128-bit lane
        vshufps ymm1, ymm0, ymm0, 0x88  ; [c0,c1,c0,c1 | c2,c3,c2,c3]
        vshufps ymm2, ymm0, ymm0, 0xDD  ; [s0,s1,s0,s1 | s2,s3,s2,s3]

        ; vpermpd 0xD8: reorder 64b lanes [0,2,1,3] to compact
        vpermpd ymm1, ymm1, 0xD8        ; cos = [c0,c1,c2,c3 | c0,c1,c2,c3]
        vpermpd ymm2, ymm2, 0xD8        ; sin = [s0,s1,s2,s3 | s0,s1,s2,s3]

        ; --- Load and deinterleave src (4 x0/x1 pairs) ---
        ; src[8*ic..8*ic+7] = [x0_0,x1_0,x0_1,x1_1,x0_2,x1_2,x0_3,x1_3]
        lea     rcx, [r8 + rax*8]
        vmovups ymm3, [rcx]             ; [x0_0,x1_0,x0_1,x1_1 | ...]

        vshufps ymm4, ymm3, ymm3, 0x88  ; [x0_0,x0_1,... | x0_2,x0_3,...]
        vshufps ymm5, ymm3, ymm3, 0xDD  ; [x1_0,x1_1,... | x1_2,x1_3,...]

        vpermpd ymm4, ymm4, 0xD8        ; x0 = [x0_0,x0_1,x0_2,x0_3 | ...]
        vpermpd ymm5, ymm5, 0xD8        ; x1 = [x1_0,x1_1,x1_2,x1_3 | ...]

        ; --- Compute ---
        ; dst_even = x0*cos - x1*sin
        vmulps       ymm6, ymm4, ymm1        ; x0*cos
        vfnmadd231ps ymm6, ymm5, ymm2        ; - x1*sin

        ; dst_odd = x0*sin + x1*cos
        vmulps       ymm7, ymm4, ymm2        ; x0*sin
        vfmadd231ps  ymm7, ymm5, ymm1        ; + x1*cos

        ; --- Interleave results and store ---
        ; ymm6 = [de0,de1,de2,de3 | dup], ymm7 = [do0,do1,do2,do3 | dup]
        ; (vpermpd duplicated the low lane into the high lane, so all 4 results
        ; live in lane 0.) Interleave: low pair via unpcklps, high pair via
        ; unpckhps, then assemble the two 128-bit halves.
        vunpcklps   ymm8, ymm6, ymm7        ; lane0 = [de0,do0,de1,do1]
        vunpckhps   ymm9, ymm6, ymm7        ; lane0 = [de2,do2,de3,do3]
        vinsertf128 ymm8, ymm8, xmm9, 1     ; [de0,do0,de1,do1 | de2,do2,de3,do3]
        lea     rcx, [r9 + rax*8]
        vmovups [rcx], ymm8

        add     rax, 4                 ; ic += 4 pairs
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

        ; x0 = src[2*ic], x1 = src[2*ic+1]
        lea     rcx, [r8 + rax*8]
        vmovss  xmm2, [rcx]             ; x0
        vmovss  xmm3, [rcx + 4]         ; x1

        ; dst[2*ic] = x0*cos - x1*sin
        vmulss       xmm4, xmm2, xmm0
        vfnmadd231ss xmm4, xmm3, xmm1
        lea     rcx, [r9 + rax*8]
        vmovss  [rcx], xmm4

        ; dst[2*ic+1] = x0*sin + x1*cos
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
