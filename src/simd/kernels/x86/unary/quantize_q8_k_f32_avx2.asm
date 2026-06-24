; ----------------------------------------------------------------------------
; quantize_q8_k_f32_avx2.asm — Q8_K quantization of f32 floats (AVX2 + FMA)
; ----------------------------------------------------------------------------
;
; void simd_quantize_q8_k_f32_avx2(int n, const float * x, void * y);
;
; Quantizes groups of 256 floats into block_q8_K structs (292 bytes each).
; Bit-exact with ggml quantize_row_q8_K_ref (ggml-quants.c:2692).
;
; struct block_q8_K {
;     float   d;                // offset   0,  4 bytes
;     int8_t  qs[QK_K];         // offset   4, 256 bytes
;     int16_t bsums[QK_K/16];   // offset 260,  32 bytes (16 int16)
; }; // sizeof = 292
;
; Algorithm per block of 256 floats:
;   1. Find amax = max(abs(x)) AND smax = the signed value with max abs
;   2. If amax == 0: d=0, memset qs=0, skip bsums
;   3. iscale = -127.0f / smax   (uses SIGNED max, not amax)
;   4. qs[j] = MIN(127, nearest_int(iscale * x[j]))
;   5. bsums[g] = sum of qs[g*16 .. g*16+15] for g in [0,16)
;   6. d = 1.0f / iscale
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
    global simd_quantize_q8_k_f32_avx2
%else
    global simd_quantize_q8_k_f32_avx2:function hidden
%endif

simd_quantize_q8_k_f32_avx2:
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

    ; save args, pre-load constants
    mov     r12, ARG_N
    mov     r13, ARG_X
    mov     r14, ARG_Y
    shr     r12, 8                     ; block count = n / 256
    jz      .epilogue

    vbroadcastss ymm12, [rel q8k_sign_mask]   ; 0x80000000 x8  (abs mask invert)
    vpbroadcastd ymm13, [rel q8k_127_int]      ; 127 x8 int32   (clamp constant)
    vmovss  xmm11, [rel q8k_neg_127f]           ; -127.0f         (iscale numerator)
    vmovss  xmm10, [rel q8k_onef]               ;  1.0f           (d numerator)

.block_loop:
    mov     r15, r13                        ; save block-start source ptr

    ; ======================================================================
    ; PASS 1: Per-lane max-abs search over 256 floats (32 x 8-float iters)
    ;   ymm15 = per-lane amax  (8 f32, max abs per index-mod-8)
    ;   ymm14 = per-lane smax  (8 f32, signed value giving per-lane amax)
    ; Strategy: for each 8-float load, compute abs, compare with running
    ; per-lane amax, and for lanes where abs > amax, update both amax
    ; and smax simultaneously.  Matches reference strict > semantics.
    ; ======================================================================
    vpxor   ymm15, ymm15, ymm15             ; amax = 0
    vpxor   ymm14, ymm14, ymm14             ; smax = 0
    mov     ecx, 32                         ; 256 / 8
.p1_loop:
    vmovups ymm0, [r13]                     ; load 8 f32
    add     r13, 32
    vandnps ymm1, ymm12, ymm0               ; ymm1 = abs(ymm0)
    vcmpps  ymm2, ymm1, ymm15, 0x1E        ; GT (quiet): abs > current amax?
    vmaxps  ymm15, ymm15, ymm1              ; update per-lane amax
    vblendvps ymm14, ymm14, ymm0, ymm2      ; update per-lane smax where GT
    dec     ecx
    jnz     .p1_loop

    ; --- horizontal-reduce ymm15 (8 per-lane amax) -> scalar amax ---
    vextractf128 xmm9, ymm15, 1             ; amax[4..7]
    vmaxps  xmm8, xmm15, xmm9               ; max(low4, hi4) per lane -> 4 vals
    vmovhlps xmm9, xmm8, xmm8               ; [xmm8[2], xmm8[3], dup]
    vmaxps  xmm8, xmm8, xmm9                ; max of two pairs -> 2 distinct vals
    vpermilps xmm9, xmm8, 0x01              ; [xmm8[1], xmm8[0], ...]
    vmaxss  xmm8, xmm8, xmm9                ; xmm8[0] = global scalar amax

    ; --- zero-block check ---
    vxorps  xmm0, xmm0, xmm0
    vucomiss xmm0, xmm8
    je      .zero_block

    ; --- find which lane (0..7) holds the global amax ---
    vbroadcastss ymm13, xmm8
    vcmpps  ymm15, ymm15, ymm13, 0        ; EQ: which lanes match global amax?
    vmovmskps eax, ymm15
    tzcnt   eax, eax                         ; first matching lane (0-7)

    ; --- extract smax from ymm14 at that lane via vpermps ---
    vmovd   xmm0, eax
    vpbroadcastd ymm0, xmm0
    vpermps ymm0, ymm0, ymm14                ; all lanes = ymm14[lane_index]
    vmovss  xmm7, xmm0                        ; xmm7[0] = smax (signed max)

    ; ======================================================================
    ; Compute iscale = -127.0f / smax,  d = 1.0f / iscale
    ; ======================================================================
    vdivss  xmm6, xmm11, xmm7                ; iscale = -127.0f / smax
    vdivss  xmm5, xmm10, xmm6                ; d = 1.0f / iscale  (store later)
    vbroadcastss ymm6, xmm6                   ; broadcast iscale for pass 2

    ; ======================================================================
    ; PASS 2: Quantize 256 floats to int8  (8 batches of 32 -> 256 int8)
    ; Each batch: 4 ymm f32 -> 4 ymm int32 -> pack -> 1 ymm int8 (32 bytes)
    ; ======================================================================
    mov     r13, r15                         ; restore source ptr
    lea     rbx, [r14 + 4]                   ; qs dest (y + 4)
    mov     r11d, 8                          ; 8 batches
.q_loop:
    vmovups ymm0, [r13]
    vmovups ymm1, [r13 + 32]
    vmovups ymm2, [r13 + 64]
    vmovups ymm3, [r13 + 96]
    add     r13, 128

    vmulps  ymm0, ymm0, ymm6
    vmulps  ymm1, ymm1, ymm6
    vmulps  ymm2, ymm2, ymm6
    vmulps  ymm3, ymm3, ymm6

    vcvtps2dq ymm0, ymm0                     ; f32 -> int32 (round ties to even)
    vcvtps2dq ymm1, ymm1
    vcvtps2dq ymm2, ymm2
    vcvtps2dq ymm3, ymm3

    vpminsd ymm0, ymm0, ymm13                ; clamp upper: MIN(v, 127)
    vpminsd ymm1, ymm1, ymm13
    vpminsd ymm2, ymm2, ymm13
    vpminsd ymm3, ymm3, ymm13

    ; --- pack 4 ymm int32 -> 1 ymm int8  (SSE pack on xmm, then combine) ---
    ; ymm0: extract high 128, packdw with low 128 -> xmm with 8 int16
    vextracti128 xmm8, ymm0, 1
    vpackssdw xmm8, xmm0, xmm8               ; 8 int16 from ymm0

    vextracti128 xmm9, ymm1, 1
    vpackssdw xmm9, xmm1, xmm9               ; 8 int16 from ymm1

    vextracti128 xmm10, ymm2, 1
    vpackssdw xmm10, xmm2, xmm10             ; 8 int16 from ymm2

    vextracti128 xmm11, ymm3, 1
    vpackssdw xmm11, xmm3, xmm11             ; 8 int16 from ymm3

    ; int16 -> int8: combine pairs into 16-byte xmm registers
    vpacksswb xmm12, xmm8, xmm9            ; 16 int8 from ymm0+ymm1
    vpacksswb xmm13, xmm10, xmm11          ; 16 int8 from ymm2+ymm3

    ; combine into 32 int8 in ymm
    vinserti128 ymm14, ymm12, xmm13, 1  ; [xmm12 lo, xmm13 hi]

    vmovdqu [rbx], ymm14
    add     rbx, 32

    dec     r11d
    jnz     .q_loop

    ; ======================================================================
    ; PASS 3: bsums — sum of every 16 consecutive qs values (16 groups)
    ; bsums at y + 4 + 256 = y + 260
    ; ======================================================================
    lea     rbx, [r14 + 4]                   ; qs base
    lea     rdx, [r14 + 260]                 ; bsums base
    mov     r11d, 16
.bsum_loop:
    vmovdqu xmm0, [rbx]                      ; load 16 int8
    add     rbx, 16
    vpmovsxbw ymm0, xmm0                     ; sign-extend 16 bytes -> 16 int16 in ymm

    ; sum all 16 int16: extract high 8, add to low 8, then h-reduce
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
    ; Store d = 1.0f / iscale  at block offset 0
    ; ======================================================================
    vmovss  [r14], xmm5
    jmp     .block_advance

.zero_block:
    ; amax == 0: d = 0, memset qs = 0, skip bsums (leave untouched)
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r14], xmm0
    vpxor   ymm0, ymm0, ymm0
    lea     rbx, [r14 + 4]
    mov     ecx, 8
.zero_qs_loop:
    vmovdqu [rbx], ymm0
    add     rbx, 32
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
; Read-only constants (32-byte aligned)
; ----------------------------------------------------------------------------
section .rodata align=32
q8k_sign_mask:
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
    dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
q8k_127_int:
    dd 127, 127, 127, 127, 127, 127, 127, 127
q8k_neg_127f:   dd 0xC2FE0000    ; -127.0f
q8k_onef:       dd 0x3F800000    ;  1.0f
