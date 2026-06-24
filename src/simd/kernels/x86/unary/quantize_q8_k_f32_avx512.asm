; ----------------------------------------------------------------------------
; quantize_q8_k_f32_avx512.asm — Q8_K quantization of f32 floats (AVX-512F)
; ----------------------------------------------------------------------------
;
; void simd_quantize_q8_k_f32_avx512(int n, const float * x, void * y);
;
; Quantizes groups of 256 floats into block_q8_K structs (292 bytes each).
; Bit-exact with ggml quantize_row_q8_K_ref (ggml-quants.c:2692).
;
; Uses AVX-512 zmm registers (16 f32 per register), k-masking for conditional
; blend, vpmovsdb for direct int32->int8 truncation.
;
; struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
;
; Win64 ABI:  rcx=n, rdx=x, r8=y
; SysV ABI:   rdi=n, rsi=x, rdx=y
; ----------------------------------------------------------------------------

bits 64
default rel

%ifdef WINDOWS
    %define ARG_N    rcx
    %define ARG_X    rdx
    %define ARG_Y    r8
%else
    %define ARG_N    rdi
    %define ARG_X    rsi
    %define ARG_Y    rdx
%endif

section .text
%ifdef WINDOWS
    global simd_quantize_q8_k_f32_avx512
%else
    global simd_quantize_q8_k_f32_avx512:function hidden
%endif

simd_quantize_q8_k_f32_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 232
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

    ; save args; block count = n / 256
    mov     r12, ARG_N
    mov     r13, ARG_X
    mov     r14, ARG_Y
    shr     r12, 8
    jz      .epilogue

    ; pre-load constants
    vbroadcastss zmm12, [rel q8k512_sign_mask]  ; 0x80000000 x16
    vpbroadcastd zmm13, [rel q8k512_127_int]     ; 127 x16 int32
    vmovss  xmm11, [rel q8k512_neg_127f]          ; -127.0f
    vmovss  xmm10, [rel q8k512_onef]              ;  1.0f

.block_loop:
    mov     r15, r13                        ; save block-start source ptr

    ; ======================================================================
    ; PASS 1: Per-lane max-abs search (16 lanes, 16 zmm loads)
    ; zmm15 = per-lane amax (16 f32)
    ; zmm14 = per-lane smax (16 f32)
    ; ======================================================================
    vpxord  zmm15, zmm15, zmm15
    vpxord  zmm14, zmm14, zmm14
    mov     ecx, 16                         ; 256 / 16 = 16 iterations

.p1_loop:
    vmovups zmm0, [r13]
    add     r13, 64
    vandnps zmm1, zmm12, zmm0              ; abs
    vcmpps  k1, zmm1, zmm15, 0x1E          ; GT (quiet) mask -> k1
    vmaxps  zmm15, zmm15, zmm1             ; update per-lane amax
    vblendmps zmm14{k1}, zmm14, zmm0        ; update per-lane smax where GT
    dec     ecx
    jnz     .p1_loop

    ; --- horizontal-reduce zmm15 (16 f32) -> scalar amax ---
    vextractf32x8 ymm9, zmm15, 1            ; upper 8 f32
    vmaxps  ymm8, ymm15, ymm9               ; max(low8, hi8) = 8 f32
    vextractf128 xmm9, ymm8, 1              ; upper 4
    vmaxps  xmm8, xmm8, xmm9                ; max(low4, hi4) = 4 f32
    vmovhlps xmm9, xmm8, xmm8               ; [2, 3, 2, 3]
    vmaxps  xmm8, xmm8, xmm9                ; max of two pairs
    vpermilps xmm9, xmm8, 0x01              ; [1, 0, 1, 0]
    vmaxss  xmm8, xmm8, xmm9                ; xmm8[0] = global scalar amax

    ; --- zero-block check ---
    vxorps  xmm0, xmm0, xmm0
    vucomiss xmm0, xmm8
    je      .zero_block

    ; --- find lane index (0..15) holding global amax ---
    vbroadcastss zmm13_t, xmm8
    vcmpps  k1, zmm15, zmm13_t, 0           ; EQ mask (16 bits)
    kmovw   eax, k1
    tzcnt   eax, eax                         ; first matching lane

    ; --- extract smax at that lane from zmm14 ---
    vmovd   xmm0, eax
    vpbroadcastd zmm0, xmm0
    vpermps zmm0, zmm0, zmm14                ; all lanes = zmm14[lane_index]
    vmovss  xmm7, xmm0                        ; xmm7[0] = smax

    ; ======================================================================
    ; Compute iscale = -127.0f / smax,  d = 1.0f / iscale
    ; ======================================================================
    vdivss  xmm6, xmm11, xmm7                ; iscale
    vdivss  xmm5, xmm10, xmm6                ; d (store later)
    vbroadcastss zmm6, xmm6                   ; broadcast iscale for pass 2

    ; ======================================================================
    ; PASS 2: Quantize 256 floats -> int8 via vpmovsdb
    ; 16 iterations of 16 f32 -> 16 int8 = 256 int8 total
    ; ======================================================================
    mov     r13, r15                         ; restore source ptr
    lea     rbx, [r14 + 4]                   ; qs dest
    mov     r11d, 16
.q_loop:
    vmovups zmm0, [r13]
    add     r13, 64
    vmulps  zmm0, zmm0, zmm6                 ; * iscale
    vcvtps2dq zmm0, zmm0                      ; f32 -> int32 (round to nearest)
    vpminsd zmm0, zmm0, zmm13                 ; clamp to 127
    vpmovsdb [rbx], zmm0                      ; truncate 16 int32 -> 16 int8, store
    add     rbx, 16
    dec     r11d
    jnz     .q_loop

    ; ======================================================================
    ; PASS 3: bsums — 16 groups, each sum of 16 consecutive int8
    ; ======================================================================
    lea     rbx, [r14 + 4]
    lea     rdx, [r14 + 260]
    mov     r11d, 16
.bsum_loop:
    vmovdqu xmm0, [rbx]                      ; load 16 int8
    add     rbx, 16
    vpmovsxbw ymm0, xmm0                     ; sign-extend 16 int8 -> 16 int16 in ymm
    vextracti128 xmm1, ymm0, 1               ; high 8 int16
    vpaddw  xmm0, xmm0, xmm1                 ; 8 pairwise sums
    vphaddw xmm0, xmm0, xmm0                 ; 4 sums
    vphaddw xmm0, xmm0, xmm0                 ; 2 sums
    vphaddw xmm0, xmm0, xmm0                 ; 1 sum in lane 0
    vpextrw [rdx], xmm0, 0                   ; store int16
    add     rdx, 2
    dec     r11d
    jnz     .bsum_loop

    ; ======================================================================
    ; Store d = 1.0f / iscale
    ; ======================================================================
    vmovss  [r14], xmm5
    jmp     .block_advance

.zero_block:
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r14], xmm0                       ; d = 0
    vpxord  zmm0, zmm0, zmm0
    lea     rbx, [r14 + 4]
    mov     ecx, 4                            ; 4 × 64 byte stores = 256 bytes
.zero_qs_loop:
    vmovdqu32 [rbx], zmm0
    add     rbx, 64
    dec     ecx
    jnz     .zero_qs_loop

.block_advance:
    add     r15, 1024                         ; x += 256 floats
    mov     r13, r15
    add     r14, 292                          ; y += sizeof(block_q8_K)
    dec     r12
    jnz     .block_loop

.epilogue:
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
    add     rsp, 232
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret

; ----------------------------------------------------------------------------
; Read-only constants
; ----------------------------------------------------------------------------
section .rodata align=64
q8k512_sign_mask:
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
q8k512_127_int:
    dd 127, 127, 127, 127, 127, 127, 127, 127
    dd 127, 127, 127, 127, 127, 127, 127, 127
q8k512_neg_127f: dd 0xC2FE0000   ; -127.0f
q8k512_onef:     dd 0x3F800000   ;  1.0f
