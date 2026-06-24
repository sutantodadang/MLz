; ----------------------------------------------------------------------------
; quantize_q8_0_f32_avx2.asm  --  Q8_0 quantization of f32 floats (AVX2)
; ----------------------------------------------------------------------------
;
; void simd_quantize_q8_0_f32_avx2(int n, const float * x, void * y);
;
; Quantizes groups of 32 floats into block_q8_0 structs (34 bytes each):
;
;   typedef struct {
;       ggml_half d;       // f16 scale — amax / 127.0f  (2 bytes)
;       int8_t    qs[32];  // 32 quantised int8 values
;   } block_q8_0;
;
; Algorithm — bit-exact with ggml's quantize_row_q8_0 (arch/x86/quants.c):
;   For each block of 32 floats:
;     1. Load 4 × ymm (32 floats)
;     2. Compute amax = max(|x[i]|) via ANDNOT with -0.0f mask
;     3. Horizontal max reduction (ymm → scalar)
;     4. d = amax / 127.0f  → store as f16 (vcvtps2ph)
;     5. id = (amax != 0) ? 127.0f / amax : 0.0f
;     6. Multiply all 32 floats by id
;     7. Round to nearest integer (vroundps, imm8=0x00)
;     8. Convert f32 → i32 (vcvtps2dq)
;     9. Pack i32 → i16 (vpackssdw), i16 → i8 (vpacksswb)
;    10. Fix byte order with vpermd — AVX2 pack operates on 128-bit lanes
;        independently, perm = {0,4,1,5,2,6,3,7}
;    11. Store 32 int8 values to y[i].qs
;
; Win64 ABI:  rcx = n, rdx = x, r8 = y
; SysV ABI:   rdi = n, rsi = x, rdx = y
; Saves xmm6-xmm15 + rbp, rbx, r12-r15.  vzeroupper before ret.
; ----------------------------------------------------------------------------

bits 64
default rel

%ifdef WINDOWS
    %define ARG_N  rcx
    %define ARG_X  rdx
    %define ARG_Y  r8
    global simd_quantize_q8_0_f32_avx2
%else
    %define ARG_N  rdi
    %define ARG_X  rsi
    %define ARG_Y  rdx
    global simd_quantize_q8_0_f32_avx2:function
%endif

section .rodata align=32
align 32
q8_sign_mask:
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
q8_const_127:
    dd 0x42FE0000                     ; 127.0f
q8_perm_mask:
    dd 0, 4, 1, 5, 2, 6, 3, 7        ; vpermd dword shuffle

section .text

simd_quantize_q8_0_f32_avx2:
    ; ------------------------------------------------------------------
    ; Prologue — save callee-saved GPRs and xmm6-xmm15
    ; ------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168                ; 10 × 16 B + 8 B align
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

    ; ------------------------------------------------------------------
    ; Save args to callee-saved regs & load constants
    ; ------------------------------------------------------------------
    mov     r12, ARG_N              ; r12 = n
    mov     r13, ARG_X              ; r13 = x ptr (src)
    mov     r14, ARG_Y              ; r14 = y ptr (dst)

    vmovaps ymm10, [rel q8_sign_mask]   ; ymm10 = {-0.0f, ...} × 8
    vmovss  xmm11, [rel q8_const_127]   ; xmm11 = 127.0f
    vmovaps ymm12, [rel q8_perm_mask]   ; ymm12 = {0,4,1,5,2,6,3,7}

    ; ------------------------------------------------------------------
    ; nb = n / 32   (number of blocks)
    ; ------------------------------------------------------------------
    mov     rbx, r12
    shr     rbx, 5
    test    rbx, rbx
    jz      .epilogue

    ; ------------------------------------------------------------------
    ; Main loop — one block (32 floats → 34 bytes) per iteration
    ; ------------------------------------------------------------------
.block_loop:

    ; --- 1. Load 32 floats (4 × ymm) --------------------------------
    vmovups ymm0, [r13]
    vmovups ymm1, [r13 + 32]
    vmovups ymm2, [r13 + 64]
    vmovups ymm3, [r13 + 96]
    add     r13, 128                ; advance src past 32 floats

    ; --- 2. Compute amax = max(|x[i]|) -------------------------------
    ; ANDNOT with -0.0f clears the sign bit → absolute value
    vandnps ymm4, ymm10, ymm0       ; abs(v0)
    vandnps ymm5, ymm10, ymm1       ; abs(v1)
    vmaxps  ymm4, ymm4, ymm5
    vandnps ymm5, ymm10, ymm2       ; abs(v2)
    vmaxps  ymm4, ymm4, ymm5
    vandnps ymm5, ymm10, ymm3       ; abs(v3)
    vmaxps  ymm4, ymm4, ymm5

    ; --- 3. Horizontal max reduction (ymm → scalar xmm) ------------
    ; Matches _mm256_extractf128_ps + _mm_max_ps + _mm_movehl_ps
    ;          + _mm_movehdup_ps + _mm_max_ss pattern from ggml.
    vextractf128 xmm5, ymm4, 1      ; xmm5 = high 4 lanes
    vmaxps  xmm4, xmm4, xmm5        ; xmm4 = max(lo4, hi4) — 4 values

    vmovhlps xmm5, xmm4, xmm4       ; xmm5 = [xmm4[2], xmm4[3], ...]
    vmaxps   xmm4, xmm4, xmm5       ; xmm4[0] = max(xmm4[0], xmm4[2])

    vmovshdup xmm5, xmm4            ; xmm5[0] = xmm4[1]
    vmaxss    xmm4, xmm4, xmm5      ; xmm4[0] = amax (scalar)

    ; --- 4. Compute d = amax / 127.0f  → store as f16 -------------
    vdivss  xmm6, xmm4, xmm11       ; xmm6 = d = amax / 127.0f
    vcvtps2ph xmm6, xmm6, 0         ; convert f32 → f16 in xmm6 low 64 bits
    vmovd    eax, xmm6              ; eax = low 32 bits (2 f16 values)
    mov      word [r14], ax         ; store f16 at block_q8_0.d (offset 0, 2 bytes)

    ; --- 5. Compute id = 127.0f / amax  (or 0 if amax == 0) --------
    vxorps  xmm7, xmm7, xmm7
    vucomiss xmm4, xmm7
    jz      .zero_id

    vdivss  xmm5, xmm11, xmm4       ; xmm5 = 127.0f / amax
    jmp     .apply_scale

.zero_id:
    vxorps  xmm5, xmm5, xmm5        ; id = 0.0f

.apply_scale:
    ; --- 6. Multiply all 32 floats by id -----------------------------
    vbroadcastss ymm5, xmm5         ; ymm5 = {id, id, ...} × 8

    vmulps  ymm0, ymm0, ymm5
    vmulps  ymm1, ymm1, ymm5
    vmulps  ymm2, ymm2, ymm5
    vmulps  ymm3, ymm3, ymm5

    ; --- 7. Round to nearest integer ---------------------------------
    ; imm8 = 0x00  →  _MM_FROUND_TO_NEAREST_INT (no SAE)
    ; Matches ggml's _MM_ROUND_NEAREST exactly.
    vroundps ymm0, ymm0, 0x00
    vroundps ymm1, ymm1, 0x00
    vroundps ymm2, ymm2, 0x00
    vroundps ymm3, ymm3, 0x00

    ; --- 8. Convert f32 → i32 ---------------------------------------
    vcvtps2dq ymm0, ymm0
    vcvtps2dq ymm1, ymm1
    vcvtps2dq ymm2, ymm2
    vcvtps2dq ymm3, ymm3

    ; --- 9. Pack i32 → i16 → i8 --------------------------------------
    ; packssdw: hi-128 of dst = src1 (ymm1/ymm3), lo-128 = src0 (ymm0/ymm2)
    vpackssdw ymm0, ymm0, ymm1
    ; ymm0 = [i0_0..i0_7 (lo), i1_0..i1_7 (hi)] — 16 i16 values
    vpackssdw ymm2, ymm2, ymm3
    ; ymm2 = [i2_0..i2_7 (lo), i3_0..i3_7 (hi)]

    ; packsswb: hi-128 = src1 (ymm2), lo-128 = src0 (ymm0)
    vpacksswb ymm0, ymm0, ymm2
    ; ymm0 = 32 i8 values, but lanes are interleaved across 128b halves

    ; --- 10. Fix byte order -------------------------------------------
    ; AVX2 pack treats each 128-bit lane independently.
    ; vpermd with {0,4,1,5,2,6,3,7} restores natural order.
    vpermd  ymm0, ymm12, ymm0

    ; --- 11. Store 32 int8 values -------------------------------------
    vmovdqu [r14 + 2], ymm0         ; store at block_q8_0.qs (offset 2)

    add     r14, 34                 ; advance dst by 34 bytes (one block)

    dec     rbx
    jnz     .block_loop

    ; ------------------------------------------------------------------
    ; Epilogue
    ; ------------------------------------------------------------------
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
    add     rsp, 168
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret
