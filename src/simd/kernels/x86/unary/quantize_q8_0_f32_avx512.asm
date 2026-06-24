; ----------------------------------------------------------------------------
; quantize_q8_0_f32_avx512.asm  --  Q8_0 quantization of f32 floats (AVX-512)
; ----------------------------------------------------------------------------
;
; void simd_quantize_q8_0_f32_avx512(int n, const float * x, void * y);
;
; Same algorithm as the AVX2 variant but exploits AVX-512 features:
;   - 2 × zmm loads (32 floats) instead of 4 × ymm
;   - vrcp14ps for fast approximate reciprocal (14-bit)
;   - vrndscaleps for rounding
;   - vpmovsdb for direct int32 → int8 saturating pack (no pack chain)
;   - vinserti128 to combine two 16-byte results into 32-byte store
;
; Algorithm — bit-exact with ggml's quantize_row_q8_0 (arch/x86/quants.c):
;   For each block of 32 floats:
;     1. Load 2 × zmm (32 floats)
;     2. Compute amax = max(|x[i]|) via ANDNOT with -0.0f mask
;     3. Horizontal max reduction (zmm → scalar)
;     4. d = amax / 127.0f  → store as f16 (vcvtps2ph + mov word)
;     5. id = (amax != 0) ? 127.0f / amax : 0.0f  (vrcp14ps for reciprocal)
;     6. Multiply all 32 floats by id
;     7. Round to nearest (vrndscaleps, imm8=0x00)
;     8. Convert f32 → i32 (vcvtps2dq)
;     9. Pack i32 → i8 (vpmovsdb × 2, vinserti128 to combine)
;    10. Store 32 int8 values to y[i].qs
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
    global simd_quantize_q8_0_f32_avx512
%else
    %define ARG_N  rdi
    %define ARG_X  rsi
    %define ARG_Y  rdx
    global simd_quantize_q8_0_f32_avx512:function
%endif

section .rodata align=64
align 64
q8_sign_mask_k:
    dd 0x80000000                     ; -0.0f — single copy, broadcast to zmm
q8_const_127_k:
    dd 0x42FE0000                     ; 127.0f — single copy, broadcast to zmm

section .text

simd_quantize_q8_0_f32_avx512:
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
    ; Save args & load constants into zmm regs
    ; ------------------------------------------------------------------
    mov     r12, ARG_N              ; r12 = n
    mov     r13, ARG_X              ; r13 = x ptr (src)
    mov     r14, ARG_Y              ; r14 = y ptr (dst)

    ; Sign-bit mask: broadcast single -0.0f to zmm14 (16 lanes)
    vmovss      xmm0, [rel q8_sign_mask_k]
    vpbroadcastd zmm14, xmm0

    ; 127.0f: broadcast to zmm15 (16 lanes)
    vmovss      xmm0, [rel q8_const_127_k]
    vbroadcastss zmm15, xmm0

    ; Also keep scalar 127.0f in xmm11 for the d computation
    vmovaps xmm11, xmm0

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

    ; --- 1. Load 32 floats (2 × zmm = 2 × 16) -----------------------
    vmovups zmm0, [r13]
    vmovups zmm1, [r13 + 64]
    add     r13, 128                ; advance src past 32 floats

    ; --- 2. Compute amax = max(|x[i]|) -------------------------------
    ; ANDNOT with -0.0f clears sign → abs, then max across both regs.
    vandnps zmm2, zmm14, zmm0       ; zmm2 = abs(v0) — 16 floats
    vandnps zmm3, zmm14, zmm1       ; zmm3 = abs(v1) — 16 floats
    vmaxps  zmm2, zmm2, zmm3        ; zmm2 = max(abs(v0), abs(v1)) — 16 floats

    ; --- 3. Horizontal max reduction (zmm → scalar) ------------------
    ; Tree reduction matching the x86 SSE pattern:
    ;   extract high 8 → max → extract high 4 → max →
    ;   movehl → max → movshdup → max
    vextractf32x8 ymm3, zmm2, 1     ; ymm3 = high 8 lanes
    vmaxps  ymm2, ymm2, ymm3        ; ymm2 = 8 values (max per pair)

    vextractf128 xmm3, ymm2, 1      ; xmm3 = high 4 lanes
    vmaxps  xmm2, xmm2, xmm3        ; xmm2 = 4 values (max per pair)

    vmovhlps xmm3, xmm2, xmm2       ; xmm3 = [xmm2[2], xmm2[3], ...]
    vmaxps   xmm2, xmm2, xmm3       ; xmm2[0] = max(xmm2[0], xmm2[2])

    vmovshdup xmm3, xmm2            ; xmm3[0] = xmm2[1]
    vmaxss    xmm4, xmm2, xmm3      ; xmm4 = amax (scalar)

    ; --- 4. Compute d = amax / 127.0f  → store as f16 ----------------
    vdivss  xmm5, xmm4, xmm11       ; xmm5 = d = amax / 127.0f

    ; Convert f32 → f16 and store 2 bytes at y[i].d (offset 0)
    vcvtps2ph xmm5, xmm5, 0         ; convert to f16 in xmm5 low 64 bits
    vmovd    eax, xmm5              ; eax = low 32 bits (2 f16 values)
    mov      word [r14], ax         ; store low 16 bits = f16(d)

    ; --- 5. Compute id = 127.0f / amax  (or 0 if amax == 0) ----------
    vxorps  xmm7, xmm7, xmm7
    vucomiss xmm4, xmm7
    jz      .zero_id

    ; id = 127/amax. Use EXACT scalar division (vrcp14ps is only 14-bit and
    ; produces off-by-one quantization vs the reference), then broadcast.
    vdivss   xmm5, xmm11, xmm4      ; xmm5 = 127.0 / amax (exact)
    vbroadcastss zmm5, xmm5         ; zmm5 = {id, ...} × 16
    jmp      .apply_scale

.zero_id:
    vpxorq  zmm5, zmm5, zmm5        ; id = 0.0f (all lanes)

.apply_scale:
    ; --- 6. Multiply all 32 floats by id -----------------------------
    vmulps  zmm0, zmm0, zmm5
    vmulps  zmm1, zmm1, zmm5

    ; --- 7. Round to nearest integer ---------------------------------
    ; imm8 = 0x00 → RC=nearest-even, SAE=0 (matches _MM_ROUND_NEAREST)
    vrndscaleps zmm0, zmm0, 0x00
    vrndscaleps zmm1, zmm1, 0x00

    ; --- 8. Convert f32 → i32 ----------------------------------------
    vcvtps2dq zmm0, zmm0
    vcvtps2dq zmm1, zmm1

    ; --- 9. Pack i32 → i8 (signed saturating) ------------------------
    ; vpmovsdb: 16 i32 in zmm → 16 i8 in xmm (signed, saturating)
    vpmovsdb xmm2, zmm0             ; xmm2 = qs[ 0..15]
    vpmovsdb xmm3, zmm1             ; xmm3 = qs[16..31]

    ; Combine into ymm2: low 128 = xmm2, high 128 = xmm3
    ; After vpmovsdb, ymm2's upper 128 is undefined; vinserti128
    ; replaces it with xmm3 while keeping lower 128 from ymm2 (= xmm2).
    vinserti128 ymm2, ymm2, xmm3, 1

    ; --- 10. Store 32 int8 values -------------------------------------
    vmovdqu [r14 + 2], ymm2         ; store at block_q8_0.qs (offset 2)

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
