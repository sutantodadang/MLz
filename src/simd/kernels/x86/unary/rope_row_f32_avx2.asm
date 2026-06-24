; ----------------------------------------------------------------------------
; rope_row_f32_avx2.asm — rotate one row with on-the-fly vectorised sin/cos
; ----------------------------------------------------------------------------
;
; void simd_rope_row_avx2(int half, int pos, const float * freq,
;                         const float * x, float * out);
;
; Rotates `half` adjacent f32 pairs: theta_i = pos * freq[i],
;   out[2i]   = x[2i]*cos - x[2i+1]*sin
;   out[2i+1] = x[2i]*sin + x[2i+1]*cos
; sin/cos computed 8 pairs at a time with a Cephes single-precision polynomial
; (Pommier sincos_ps, AVX2 port) — the asm counterpart of fra_rope_row_vec for
; the asm-vs-intrinsics comparison.
;
; Requires half % 8 == 0 (the comparison uses D=128 -> half=64). vpermps expands
; each pair's cos/sin to its two slots; rotation uses out = x*cos + sign*swap(x)*sin.
;
; Win64: rcx=half, rdx=pos, r8=freq, r9=x, [rbp+48]=out
; SysV:  edi=half, esi=pos, rdx=freq, rcx=x, r8=out
; ----------------------------------------------------------------------------

bits 64
default rel

section .rodata align=32
fopi:   dd 1.27323954473516
dp1:    dd -0.78515625
dp2:    dd -2.4187564849853515625e-4
dp3:    dd -3.77489497744594108e-8
sc0:    dd -1.9515295891e-4
sc1:    dd  8.3321608736e-3
sc2:    dd -1.6666654611e-1
cc0:    dd  2.443315711809948e-5
cc1:    dd -1.388731625493765e-3
cc2:    dd  4.166664568298827e-2
chalf:  dd 0.5
cone:   dd 1.0
c_signmask: dd 0x80000000
c_absmask:  dd 0x7fffffff
c_int1: dd 1
c_intn1: dd 0xFFFFFFFE
c_int2: dd 2
c_int4: dd 4
align 32
idx_lo: dd 0,0,1,1,2,2,3,3
idx_hi: dd 4,4,5,5,6,6,7,7
align 32
sign_pat: dd 0xBF800000,0x3F800000,0xBF800000,0x3F800000,0xBF800000,0x3F800000,0xBF800000,0x3F800000

section .text
%ifdef WINDOWS
    global simd_rope_row_avx2
%else
    global simd_rope_row_avx2:function hidden
%endif

simd_rope_row_avx2:
    push    rbp
    mov     rbp, rsp
    push    rbx
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
    mov     r10d, ecx                 ; half
    mov     eax, edx                  ; pos
    ; freq already in r8, x already in r9
    mov     r11, [rbp+48]             ; out
%else
    mov     r10d, edi                 ; half
    mov     eax, esi                  ; pos
    mov     r11, r8                   ; out
    mov     r9, rcx                   ; x
    mov     r8, rdx                   ; freq
%endif

    ; pos -> float broadcast in ymm15
    vcvtsi2ss xmm0, xmm0, eax
    vbroadcastss ymm15, xmm0

    ; idx + sign constants
    vmovdqa ymm13, [rel idx_lo]
    vmovdqa ymm14, [rel idx_hi]

    xor     ebx, ebx                  ; i = 0
    shr     r10d, 3                   ; groups = half/8
    test    r10d, r10d
    jz      .done

.loop:
    ; theta = freq[i..] * pos
    vmovups ymm0, [r8 + rbx*4]
    vmulps  ymm0, ymm0, ymm15

    ; ===== sincos_ps(ymm0) -> sin=ymm10, cos=ymm11 =====
    vbroadcastss ymm1, [rel c_signmask]
    vandps  ymm2, ymm0, ymm1              ; sign_bit_sin = x & signmask
    vbroadcastss ymm1, [rel c_absmask]
    vandps  ymm0, ymm0, ymm1              ; x = abs(x)

    vbroadcastss ymm1, [rel fopi]
    vmulps  ymm3, ymm0, ymm1              ; y = x*FOPI
    vcvttps2dq ymm4, ymm3                 ; emm2 = (int)y
    vbroadcastss ymm1, [rel c_int1]
    vpaddd  ymm4, ymm4, ymm1              ; +1
    vbroadcastss ymm1, [rel c_intn1]
    vpand   ymm4, ymm4, ymm1              ; & ~1
    vcvtdq2ps ymm3, ymm4                  ; y = (float)emm2
    vmovdqa ymm5, ymm4                    ; emm4 = emm2

    vbroadcastss ymm1, [rel c_int4]
    vpand   ymm6, ymm4, ymm1              ; emm0 = emm2 & 4
    vpslld  ymm6, ymm6, 29                ; swap_sign_sin = ymm6
    vbroadcastss ymm1, [rel c_int2]
    vpand   ymm4, ymm4, ymm1              ; emm2 & 2
    vpxor   ymm1, ymm1, ymm1
    vpcmpeqd ymm7, ymm4, ymm1             ; poly_mask = (emm2&2)==0  -> ymm7

    ; Cody-Waite: x += y*DP1 + y*DP2 + y*DP3
    vbroadcastss ymm1, [rel dp1]
    vfmadd231ps ymm0, ymm3, ymm1
    vbroadcastss ymm1, [rel dp2]
    vfmadd231ps ymm0, ymm3, ymm1
    vbroadcastss ymm1, [rel dp3]
    vfmadd231ps ymm0, ymm3, ymm1          ; x reduced -> ymm0

    ; sign_bit_cos: emm4 = ~(emm4-2) & 4 ; <<29
    vbroadcastss ymm1, [rel c_int2]
    vpsubd  ymm5, ymm5, ymm1
    vbroadcastss ymm1, [rel c_int4]
    vpandn  ymm5, ymm5, ymm1              ; ~emm5 & 4
    vpslld  ymm5, ymm5, 29                ; sign_bit_cos = ymm5
    vxorps  ymm2, ymm2, ymm6             ; sign_bit_sin ^= swap_sign_sin

    vmulps  ymm8, ymm0, ymm0              ; z = x*x

    ; cos poly -> ymm11
    vbroadcastss ymm11, [rel cc0]
    vbroadcastss ymm1, [rel cc1]
    vfmadd213ps ymm11, ymm8, ymm1         ; cc0*z+cc1
    vbroadcastss ymm1, [rel cc2]
    vfmadd213ps ymm11, ymm8, ymm1         ; *z+cc2
    vmulps  ymm11, ymm11, ymm8
    vmulps  ymm11, ymm11, ymm8            ; *z*z
    vbroadcastss ymm1, [rel chalf]
    vfnmadd231ps ymm11, ymm8, ymm1        ; -0.5*z
    vbroadcastss ymm1, [rel cone]
    vaddps  ymm11, ymm11, ymm1            ; +1   -> cos poly

    ; sin poly -> ymm10
    vbroadcastss ymm10, [rel sc0]
    vbroadcastss ymm1, [rel sc1]
    vfmadd213ps ymm10, ymm8, ymm1
    vbroadcastss ymm1, [rel sc2]
    vfmadd213ps ymm10, ymm8, ymm1
    vmulps  ymm10, ymm10, ymm8
    vmulps  ymm10, ymm10, ymm0            ; *z*x
    vaddps  ymm10, ymm10, ymm0            ; +x   -> sin poly

    ; select: sin = mask? sinpoly : cospoly ; cos = mask? cospoly : sinpoly
    ; vblendvps dst,a,b,mask = mask?b:a
    vblendvps ymm9, ymm11, ymm10, ymm7    ; sin = mask? sinpoly(ymm10):cospoly(ymm11)
    vblendvps ymm11, ymm10, ymm11, ymm7   ; cos = mask? cospoly:sinpoly
    vxorps  ymm10, ymm9, ymm2             ; sin ^= sign_bit_sin
    vxorps  ymm11, ymm11, ymm5            ; cos ^= sign_bit_cos
    ; ===== end sincos: sin=ymm10 cos=ymm11 =====

    ; expand cos/sin to pair slots
    vpermps ymm0, ymm13, ymm11            ; c_lo
    vpermps ymm1, ymm14, ymm11            ; c_hi
    vpermps ymm2, ymm13, ymm10            ; s_lo
    vpermps ymm3, ymm14, ymm10            ; s_hi
    vmovaps ymm12, [rel sign_pat]

    ; rotate lo 4 pairs (x[2i..2i+7])
    lea     rax, [rbx*2]                  ; 2i (element index)
    vmovups ymm4, [r9 + rax*4]            ; x_lo
    vpermilps ymm5, ymm4, 0xB1            ; swap(x_lo)
    vmulps  ymm5, ymm5, ymm2
    vmulps  ymm5, ymm5, ymm12             ; sign*swap*s_lo
    vfmadd231ps ymm5, ymm4, ymm0          ; + x*c_lo  (ymm5 = x*c + ...)? need x*c+t
    ; NOTE: vfmadd231ps ymm5, ymm4, ymm0 = ymm5 + ymm4*ymm0 -> correct (t + x*c)
    vmovups [r11 + rax*4], ymm5

    ; rotate hi 4 pairs (x[2i+8..2i+15])
    vmovups ymm4, [r9 + rax*4 + 32]
    vpermilps ymm6, ymm4, 0xB1
    vmulps  ymm6, ymm6, ymm3
    vmulps  ymm6, ymm6, ymm12
    vfmadd231ps ymm6, ymm4, ymm1
    vmovups [r11 + rax*4 + 32], ymm6

    add     ebx, 8                        ; i += 8 pairs
    dec     r10d
    jnz     .loop

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
    pop     rbx
    pop     rbp
    vzeroupper
    ret
