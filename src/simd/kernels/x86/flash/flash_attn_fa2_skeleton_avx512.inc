; ============================================================================
; Flash Attention 2 - AVX-512 Skeleton (Shared Include)
; ============================================================================
;
; This file implements the full FA2 online-softmax algorithm for AVX-512.
; Each quantized kernel file defines type-specific macros and then includes
; this skeleton via:
;   %include "flash_attn_fa2_skeleton_avx512.inc"
;
; REQUIRED DEFINITIONS (must be set before including):
;   KERNEL_NAME          - global symbol name (e.g., simd_flash_attn_q4_0_avx512)
;   K_BLOCK_BYTES        - bytes per K quant block (e.g., 18 for Q4_0)
;   V_BLOCK_BYTES        - bytes per V quant block (e.g., 18 for Q4_0)
;   K_BLOCK_VALUES       - values per K block (e.g., 32 for Q4_0)
;   V_BLOCK_VALUES       - values per V block (e.g., 32 for Q4_0)
;
; REQUIRED MACROS (must be defined before including):
;   DECLARE_QUANT_CONSTANTS  - emit .rodata constants for dequant
;   INIT_QUANT_REGS          - load constant registers (e.g., nibble_mask)
;                              May use: zmm11, zmm12, zmm13.
;                              Must NOT touch: zmm30, zmm31.
;
;   DEQUANT_K_DOT_BLOCK  k_block_ptr, q_ptr, acc_zmm
;     Compute dot(Q_f32[0:block_values], dequant(K_block)) and ADD to acc_zmm.
;     Q is always F32. k_block_ptr = register pointing to quant block start.
;     q_ptr = register pointing to corresponding Q floats.
;     acc_zmm = zmm register to accumulate into.
;     Must preserve: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx.
;     May clobber: zmm0-zmm5, rax.
;
;   DEQUANT_V_ACCUM_BLOCK  v_block_ptr, prob_zmm, o_base_ptr, block_idx
;     Dequantize one V block to F32, multiply by prob_zmm (broadcast prob),
;     and accumulate into O[block_idx * V_BLOCK_VALUES * 4 + o_base_ptr].
;     Must preserve: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx.
;     May clobber: zmm0-zmm5, rax.
;
; REGISTER ALLOCATION:
;   zmm11-zmm13: quant constants (set by INIT_QUANT_REGS)
;   zmm30: -inf broadcast (persistent)
;   zmm31: scale broadcast (persistent)
;   r12: config pointer
;   r13: tile_start (during tile loop)
;   r14: j loop variable
;   r15: work_idx
;   k2: V head_dim remainder opmask
;
; DESIGN NOTES:
;   - FA_TILE_KV = 64 (doubled from AVX2's 32)
;   - 64-byte stack alignment
;   - row_max and row_sum stored on stack (not in registers, unlike AVX2)
;   - No push/pop inside loops — all state on stack, rsp never changes
;   - Fused softmax+PV: compute p=exp(score-max), immediately accumulate O
;   - Uses vscalefss-based FAST_EXP (from F32 AVX-512 kernel)
;   - k2 opmask for V head_dim remainder in final store
;
; ============================================================================

bits 64
default rel

; --- ABI detection ---
%ifidn __OUTPUT_FORMAT__, win64
  %define WIN64
%endif

; --- Config struct offsets (must match flash_attention.h) ---
%define CFG_HEAD_DIM_K       0
%define CFG_HEAD_DIM_V       8
%define CFG_N_QUERIES       16
%define CFG_N_KV            24
%define CFG_N_HEAD_Q        32
%define CFG_N_HEAD_KV       40
%define CFG_BATCH_SIZE      48
%define CFG_SCALE           56
%define CFG_LOGIT_SOFTCAP   60
%define CFG_MAX_BIAS        64
%define CFG_MODE            68
%define CFG_WINDOW_SIZE     72
%define CFG_Q_PTR           80
%define CFG_K_PTR           88
%define CFG_V_PTR           96
%define CFG_MASK_PTR       104
%define CFG_DST_PTR        112
%define CFG_Q_NB0          120
%define CFG_Q_NB1          128
%define CFG_Q_NB2          136
%define CFG_Q_NB3          144
%define CFG_K_NB0          152
%define CFG_K_NB1          160
%define CFG_K_NB2          168
%define CFG_K_NB3          176
%define CFG_V_NB0          184
%define CFG_V_NB1          192
%define CFG_V_NB2          200
%define CFG_V_NB3          208
%define CFG_MASK_NB0       216
%define CFG_MASK_NB1       224
%define CFG_MASK_NB2       232
%define CFG_MASK_NB3       240
%define CFG_DST_NB0        248
%define CFG_DST_NB1        256
%define CFG_DST_NB2        264
%define CFG_DST_NB3        272
%define CFG_K_TYPE         280
%define CFG_V_TYPE         284
%define CFG_ITH            328
%define CFG_NTH            332

; --- FA2 parameters ---
%define FA_TILE_KV          64
%define FA_MAX_HEAD_DIM    256

; --- Stack frame layout (64-byte aligned) ---
;
; O accumulator: 256 floats max = 1024 bytes
; Scores: 64 floats = 256 bytes
; Locals: dimensions, pointers, strides, work decomposition
;
%define STK_O                0      ; float[256] = 1024 bytes
%define STK_SCORES        1024      ; float[64]  = 256 bytes
; --- Saved locals (1280+) ---
%define STK_CONFIG        1280
%define STK_Q_ROW         1288
%define STK_K_BASE        1296
%define STK_V_BASE        1304
%define STK_DST_ROW       1312
%define STK_MASK_PTR_L    1320
%define STK_HEAD_DIM_K    1328
%define STK_HEAD_DIM_V    1336
%define STK_N_KV          1344
%define STK_N_QUERIES     1352
%define STK_N_HEAD_Q      1360
%define STK_N_HEAD_KV     1368
%define STK_BATCH_SIZE    1376
; --- Strides ---
%define STK_Q_NB1         1384
%define STK_K_NB1         1392
%define STK_V_NB1         1400
%define STK_DST_NB1       1408
%define STK_MASK_NB1      1416
%define STK_Q_NB2         1424
%define STK_K_NB2         1432
%define STK_V_NB2         1440
%define STK_DST_NB2       1448
%define STK_Q_NB3         1456
%define STK_K_NB3         1464
%define STK_V_NB3         1472
%define STK_DST_NB3       1480
; --- Work decomposition ---
%define STK_WORK_START    1488
%define STK_WORK_END      1496
; --- Quant block counts ---
%define STK_K_NBLOCKS     1504
%define STK_V_NBLOCKS     1512
; --- V head_dim chunking ---
%define STK_HDV_FULL16    1520
%define STK_HDV_REM       1528
; --- Online softmax state ---
%define STK_ROW_MAX       1536      ; float (4 bytes)
%define STK_ROW_SUM       1540      ; float (4 bytes)
; --- Misc ---
%define STK_IQ_SAVE       1544
%define STK_TILE_END      1552
%define STK_NQ_X_NHQ      1560
%define STK_MASK_NB0      1568
; --- Windows XMM save area ---
%define STK_XMM_SAVE      1600      ; 10 * 16 = 160 bytes
; --- Total frame (aligned to 64) ---
%define FRAME_SIZE         1856

; ============================================================================
; Data Section
; ============================================================================
section .rodata
align 64

; Shared FA2 constants (16-wide for zmm)
fa2_neg_inf:      times 16 dd 0xFF800000     ; -infinity
fa2_ones_f32:     times 16 dd 0x3F800000     ; 1.0

; Exp approximation constants (Cody-Waite + degree-4 Horner + vscalef)
fa2_exp_log2e:    times 16 dd 0x3FB8AA3B     ; log2(e) = 1.44269504
fa2_exp_ln2:      times 16 dd 0x3F317218     ; ln(2)   = 0.69314718
fa2_exp_c1:       times 16 dd 0x3F800000     ; 1.0
fa2_exp_c2:       times 16 dd 0x3F000000     ; 0.5
fa2_exp_c3:       times 16 dd 0x3E2AAAAB     ; 1/6
fa2_exp_c4:       times 16 dd 0x3D2AAAAB     ; 1/24
fa2_exp_clamp_lo: times 16 dd 0xC2AEAC50     ; -87.3365
fa2_exp_clamp_hi: times 16 dd 0x42B17218     ; 88.7228

; Quant-specific constants (emitted by the kernel file)
DECLARE_QUANT_CONSTANTS

; ============================================================================
; Code Section
; ============================================================================
section .text

; --- Inline scalar exp macro (vscalefss-based) ---
; Input/Output: %1 = xmm register
; Clobbers: xmm16, xmm17, xmm18, xmm19
; Preserves: all other xmm/zmm, all GPRs
%macro FAST_EXP_SS 1
    ; Clamp input to valid range
    vmovss  xmm16, [rel fa2_exp_clamp_lo]
    vmaxss  %1, %1, xmm16
    vmovss  xmm16, [rel fa2_exp_clamp_hi]
    vminss  %1, %1, xmm16

    ; n = round(x * log2(e))
    vmovss  xmm16, [rel fa2_exp_log2e]
    vmulss  xmm16, %1, xmm16                 ; x * log2(e)
    vrndscaless xmm17, xmm16, xmm16, 0x08   ; n = round nearest

    ; r = x - n * ln(2)  (Cody-Waite reduction)
    vmovss  xmm18, [rel fa2_exp_ln2]
    vmulss  xmm18, xmm17, xmm18              ; n * ln(2)
    vsubss  xmm18, %1, xmm18                 ; r = x - n*ln(2)

    ; Horner polynomial: p = (((c4*r + c3)*r + c2)*r + c1)*r + 1
    vmovss  xmm19, [rel fa2_exp_c4]
    vfmadd213ss xmm19, xmm18, [rel fa2_exp_c3]
    vfmadd213ss xmm19, xmm18, [rel fa2_exp_c2]
    vfmadd213ss xmm19, xmm18, [rel fa2_exp_c1]
    vfmadd213ss xmm19, xmm18, [rel fa2_ones_f32]

    ; result = p * 2^n (using vscalefss)
    vscalefss %1, xmm19, xmm17
%endmacro

; ============================================================================
; Main kernel function
; ============================================================================
global KERNEL_NAME
KERNEL_NAME:
    ; === Prologue ===
    push      rbp
    mov       rbp, rsp
    push      rbx
    push      r12
    push      r13
    push      r14
    push      r15

%ifdef WIN64
    push      rdi
    push      rsi
    ; Allocate frame and align to 64 bytes
    sub       rsp, FRAME_SIZE + 64
    and       rsp, ~63

    ; Save Windows callee-saved XMM6-XMM15
    vmovups   [rsp+STK_XMM_SAVE],       xmm6
    vmovups   [rsp+STK_XMM_SAVE+16],    xmm7
    vmovups   [rsp+STK_XMM_SAVE+32],    xmm8
    vmovups   [rsp+STK_XMM_SAVE+48],    xmm9
    vmovups   [rsp+STK_XMM_SAVE+64],    xmm10
    vmovups   [rsp+STK_XMM_SAVE+80],    xmm11
    vmovups   [rsp+STK_XMM_SAVE+96],    xmm12
    vmovups   [rsp+STK_XMM_SAVE+112],   xmm13
    vmovups   [rsp+STK_XMM_SAVE+128],   xmm14
    vmovups   [rsp+STK_XMM_SAVE+144],   xmm15

    ; Remap Windows ABI: rcx -> r12
    mov       r12, rcx
%else
    sub       rsp, FRAME_SIZE + 64
    and       rsp, ~63
    mov       r12, rdi              ; SysV: first arg in rdi
%endif

    ; Store config pointer on stack
    mov       [rsp+STK_CONFIG], r12

    ; === Early exit checks ===
    mov       rax, [r12+CFG_HEAD_DIM_K]
    test      rax, rax
    jz        .epilogue
    mov       rax, [r12+CFG_HEAD_DIM_V]
    test      rax, rax
    jz        .epilogue
    mov       rax, [r12+CFG_N_QUERIES]
    test      rax, rax
    jz        .epilogue
    mov       rax, [r12+CFG_N_KV]
    test      rax, rax
    jz        .epilogue
    mov       rax, [r12+CFG_N_HEAD_Q]
    test      rax, rax
    jz        .epilogue
    mov       rax, [r12+CFG_BATCH_SIZE]
    test      rax, rax
    jz        .epilogue

    ; === Load config into stack locals ===
    mov       rax, [r12+CFG_HEAD_DIM_K]
    mov       [rsp+STK_HEAD_DIM_K], rax
    mov       rax, [r12+CFG_HEAD_DIM_V]
    mov       [rsp+STK_HEAD_DIM_V], rax
    mov       rax, [r12+CFG_N_QUERIES]
    mov       [rsp+STK_N_QUERIES], rax
    mov       rax, [r12+CFG_N_KV]
    mov       [rsp+STK_N_KV], rax
    mov       rax, [r12+CFG_N_HEAD_Q]
    mov       [rsp+STK_N_HEAD_Q], rax
    mov       rax, [r12+CFG_N_HEAD_KV]
    mov       [rsp+STK_N_HEAD_KV], rax
    mov       rax, [r12+CFG_BATCH_SIZE]
    mov       [rsp+STK_BATCH_SIZE], rax

    ; Precompute nq * nhq
    mov       rax, [rsp+STK_N_QUERIES]
    imul      rax, [rsp+STK_N_HEAD_Q]
    mov       [rsp+STK_NQ_X_NHQ], rax

    ; Load pointers
    mov       rax, [r12+CFG_MASK_PTR]
    mov       [rsp+STK_MASK_PTR_L], rax

    ; Load strides nb1
    mov       rax, [r12+CFG_Q_NB1]
    mov       [rsp+STK_Q_NB1], rax
    mov       rax, [r12+CFG_K_NB1]
    mov       [rsp+STK_K_NB1], rax
    mov       rax, [r12+CFG_V_NB1]
    mov       [rsp+STK_V_NB1], rax
    mov       rax, [r12+CFG_DST_NB1]
    mov       [rsp+STK_DST_NB1], rax
    mov       rax, [r12+CFG_MASK_NB1]
    mov       [rsp+STK_MASK_NB1], rax
    mov       rax, [r12+CFG_MASK_NB0]
    mov       [rsp+STK_MASK_NB0], rax
    ; nb2
    mov       rax, [r12+CFG_Q_NB2]
    mov       [rsp+STK_Q_NB2], rax
    mov       rax, [r12+CFG_K_NB2]
    mov       [rsp+STK_K_NB2], rax
    mov       rax, [r12+CFG_V_NB2]
    mov       [rsp+STK_V_NB2], rax
    mov       rax, [r12+CFG_DST_NB2]
    mov       [rsp+STK_DST_NB2], rax
    ; nb3
    mov       rax, [r12+CFG_Q_NB3]
    mov       [rsp+STK_Q_NB3], rax
    mov       rax, [r12+CFG_K_NB3]
    mov       [rsp+STK_K_NB3], rax
    mov       rax, [r12+CFG_V_NB3]
    mov       [rsp+STK_V_NB3], rax
    mov       rax, [r12+CFG_DST_NB3]
    mov       [rsp+STK_DST_NB3], rax

    ; Compute number of quant blocks per head_dim
    mov       rax, [rsp+STK_HEAD_DIM_K]
    xor       rdx, rdx
    mov       rcx, K_BLOCK_VALUES
    div       rcx
    mov       [rsp+STK_K_NBLOCKS], rax

    mov       rax, [rsp+STK_HEAD_DIM_V]
    xor       rdx, rdx
    mov       rcx, V_BLOCK_VALUES
    div       rcx
    mov       [rsp+STK_V_NBLOCKS], rax

    ; Compute V head_dim chunking for 16-wide store
    mov       rax, [rsp+STK_HEAD_DIM_V]
    mov       rcx, rax
    shr       rax, 4                          ; head_dim_v / 16
    mov       [rsp+STK_HDV_FULL16], rax
    and       rcx, 15                         ; head_dim_v % 16
    mov       [rsp+STK_HDV_REM], rcx

    ; Set k2 opmask for V remainder
    test      rcx, rcx
    jz        .no_v_rem_mask
    mov       eax, 1
    shl       eax, cl
    dec       eax                             ; (1 << rem) - 1
    kmovw     k2, eax
    jmp       .v_rem_mask_done
.no_v_rem_mask:
    mov       eax, 0xFFFF
    kmovw     k2, eax
.v_rem_mask_done:

    ; === Load persistent ZMM constants ===
    vmovaps   zmm30, [rel fa2_neg_inf]        ; zmm30 = -inf
    vbroadcastss zmm31, dword [r12+CFG_SCALE] ; zmm31 = scale

    ; === Initialize quant-specific constant registers ===
    INIT_QUANT_REGS

    ; === Thread partitioning ===
    ; total_work = n_queries * n_head_q * batch_size
    mov       rax, [rsp+STK_NQ_X_NHQ]
    imul      rax, [rsp+STK_BATCH_SIZE]
    mov       rcx, rax                        ; rcx = total_work

    movsxd    r8, dword [r12+CFG_NTH]
    movsxd    r9, dword [r12+CFG_ITH]

    ; work_start = (total_work * ith) / nth
    mov       rbx, rax                        ; rbx = total_work
    imul      rax, r9                         ; total_work * ith
    cqo
    idiv      r8                              ; rax = (total_work * ith) / nth
    mov       [rsp+STK_WORK_START], rax

    ; work_end = (total_work * (ith + 1)) / nth
    mov       rax, r9
    inc       rax                             ; ith + 1
    imul      rax, rbx                        ; total_work * (ith + 1)
    cqo
    idiv      r8
    cmp       rax, rcx
    cmovg     rax, rcx
    mov       [rsp+STK_WORK_END], rax

    ; Check if this thread has work
    mov       r15, [rsp+STK_WORK_START]
    cmp       r15, [rsp+STK_WORK_END]
    jge       .epilogue

    ; === Work loop ===
    align     64
.work_loop:
    cmp       r15, [rsp+STK_WORK_END]
    jge       .work_done

    ; --- Decompose work_idx -> (iq, ihq, ib) ---
    ; work_idx = ib * (nq * nhq) + ihq * nq + iq
    mov       rax, r15
    xor       edx, edx
    mov       rcx, [rsp+STK_NQ_X_NHQ]
    div       rcx
    ; rax = ib, rdx = remainder
    mov       r8, rax                         ; r8 = ib

    mov       rax, rdx
    xor       edx, edx
    mov       rcx, [rsp+STK_N_QUERIES]
    div       rcx
    ; rax = ihq, rdx = iq
    mov       r9, rax                         ; r9 = ihq
    mov       [rsp+STK_IQ_SAVE], rdx          ; save iq
    mov       r14, rdx                        ; r14 = iq (temp)

    ; --- GQA: map ihq -> ihkv ---
    mov       rax, r9
    imul      rax, [rsp+STK_N_HEAD_KV]
    xor       edx, edx
    div       qword [rsp+STK_N_HEAD_Q]
    mov       rbx, rax                        ; rbx = ihkv

    ; --- Compute Q_row pointer ---
    ; Q_row = q_ptr + iq*q_nb1 + ihq*q_nb2 + ib*q_nb3
    mov       rdi, [r12+CFG_Q_PTR]
    mov       rcx, r14
    imul      rcx, [rsp+STK_Q_NB1]
    add       rdi, rcx
    mov       rcx, r9
    imul      rcx, [rsp+STK_Q_NB2]
    add       rdi, rcx
    mov       rcx, r8
    imul      rcx, [rsp+STK_Q_NB3]
    add       rdi, rcx
    mov       [rsp+STK_Q_ROW], rdi

    ; --- Compute K_base pointer ---
    ; K_base = k_ptr + ihkv*k_nb2 + ib*k_nb3
    mov       rsi, [r12+CFG_K_PTR]
    mov       rcx, rbx
    imul      rcx, [rsp+STK_K_NB2]
    add       rsi, rcx
    mov       rcx, r8
    imul      rcx, [rsp+STK_K_NB3]
    add       rsi, rcx
    mov       [rsp+STK_K_BASE], rsi

    ; --- Compute V_base pointer ---
    ; V_base = v_ptr + ihkv*v_nb2 + ib*v_nb3
    mov       rax, [r12+CFG_V_PTR]
    mov       rcx, rbx
    imul      rcx, [rsp+STK_V_NB2]
    add       rax, rcx
    mov       rcx, r8
    imul      rcx, [rsp+STK_V_NB3]
    add       rax, rcx
    mov       [rsp+STK_V_BASE], rax

    ; --- Compute dst_row pointer ---
    ; dst_row = dst_ptr + iq*dst_nb1 + ihq*dst_nb2 + ib*dst_nb3
    mov       rax, [r12+CFG_DST_PTR]
    mov       rcx, r14
    imul      rcx, [rsp+STK_DST_NB1]
    add       rax, rcx
    mov       rcx, r9
    imul      rcx, [rsp+STK_DST_NB2]
    add       rax, rcx
    mov       rcx, r8
    imul      rcx, [rsp+STK_DST_NB3]
    add       rax, rcx
    mov       [rsp+STK_DST_ROW], rax

    ; --- Zero the O accumulator ---
    vpxord    zmm0, zmm0, zmm0
    lea       rdi, [rsp+STK_O]
    mov       rcx, [rsp+STK_HDV_FULL16]
    test      rcx, rcx
    jz        .zero_o_rem
.zero_o_full:
    vmovaps   [rdi], zmm0
    add       rdi, 64
    dec       rcx
    jnz       .zero_o_full
.zero_o_rem:
    mov       rcx, [rsp+STK_HDV_REM]
    test      rcx, rcx
    jz        .zero_o_done
    vmovaps   [rdi], zmm0                     ; safe: buffer has 1024 bytes
.zero_o_done:

    ; --- Initialize online softmax state on stack ---
    mov       dword [rsp+STK_ROW_MAX], 0xFF800000   ; row_max = -inf
    mov       dword [rsp+STK_ROW_SUM], 0x00000000   ; row_sum = 0.0

    ; --- KV Tile Loop ---
    xor       r13, r13                        ; r13 = tile_start = 0

    align     32
.tile_loop:
    cmp       r13, [rsp+STK_N_KV]
    jge       .tile_done

    ; tile_end = min(tile_start + FA_TILE_KV, n_kv)
    lea       rbx, [r13 + FA_TILE_KV]
    cmp       rbx, [rsp+STK_N_KV]
    cmovg     rbx, [rsp+STK_N_KV]
    mov       [rsp+STK_TILE_END], rbx
    ; rbx = tile_end, tile_count = rbx - r13

    ; ============================================================
    ; Step 1: QK dot products -> STK_SCORES
    ; ============================================================
    mov       r14, r13                        ; j = tile_start

    align     16
.qk_dot_loop:
    cmp       r14, [rsp+STK_TILE_END]
    jge       .qk_dot_done

    ; K_row = K_base + j * k_nb1
    mov       rax, r14
    imul      rax, [rsp+STK_K_NB1]
    mov       rsi, [rsp+STK_K_BASE]
    lea       r8, [rsi + rax]                 ; r8 = K_row (quantized)

    ; Q row pointer
    mov       rdi, [rsp+STK_Q_ROW]

    ; Clear dot accumulator
    vpxord    zmm6, zmm6, zmm6

    ; Iterate over quant blocks
    xor       r9, r9                          ; block_idx = 0
    mov       r10, [rsp+STK_K_NBLOCKS]

.qk_block_loop:
    cmp       r9, r10
    jge       .qk_block_done

    ; K block pointer: K_row + block_idx * K_BLOCK_BYTES
    mov       rax, r9
    imul      rax, K_BLOCK_BYTES
    lea       r11, [r8 + rax]                 ; r11 = K block ptr

    ; Q pointer for this block: Q_row + block_idx * BLOCK_VALUES * 4
    mov       rax, r9
    imul      rax, K_BLOCK_VALUES * 4
    lea       rdx, [rdi + rax]                ; rdx = Q ptr

    ; Quant-specific dot product: dot(Q[rdx], K[r11]) -> add to zmm6
    DEQUANT_K_DOT_BLOCK r11, rdx, zmm6

    inc       r9
    jmp       .qk_block_loop
.qk_block_done:

    ; Horizontal sum zmm6 -> scalar xmm6
    vextractf32x8 ymm0, zmm6, 1
    vaddps    ymm6, ymm6, ymm0
    vextractf32x4 xmm0, ymm6, 1
    vaddps    xmm6, xmm6, xmm0
    vpermilps xmm0, xmm6, 0x4E
    vaddps    xmm6, xmm6, xmm0
    vpermilps xmm0, xmm6, 0xB1
    vaddss    xmm6, xmm6, xmm0

    ; score = dot * scale
    vmulss    xmm6, xmm6, xmm31

    ; Store to tile scores buffer
    mov       rax, r14
    sub       rax, r13                        ; j - tile_start
    vmovss    [rsp+STK_SCORES+rax*4], xmm6

    inc       r14
    jmp       .qk_dot_loop
.qk_dot_done:

    ; ============================================================
    ; Step 2: Apply mask (if present)
    ; ============================================================
    mov       rax, [rsp+STK_MASK_PTR_L]
    test      rax, rax
    jz        .mask_done

    ; mask_row = mask_ptr + iq * mask_nb1
    mov       rcx, [rsp+STK_IQ_SAVE]
    imul      rcx, [rsp+STK_MASK_NB1]
    add       rax, rcx                        ; rax = mask row base

    ; Add mask values to scores
    lea       rdi, [rsp+STK_SCORES]
    mov       r14, r13                        ; j = tile_start
    xor       rdx, rdx                        ; j_local = 0
.mask_loop:
    cmp       r14, [rsp+STK_TILE_END]
    jge       .mask_done
    ; mask value at mask_row + j * mask_nb0
    mov       rcx, r14
    imul      rcx, [rsp+STK_MASK_NB0]
    vmovss    xmm0, [rdi+rdx*4]              ; score
    vaddss    xmm0, xmm0, [rax+rcx]          ; + mask value
    vmovss    [rdi+rdx*4], xmm0
    inc       rdx
    inc       r14
    jmp       .mask_loop
.mask_done:

    ; ============================================================
    ; Step 3: Find tile max
    ; ============================================================
    mov       rcx, [rsp+STK_TILE_END]
    sub       rcx, r13                        ; tile_count
    lea       rsi, [rsp+STK_SCORES]

    ; Start with -inf
    vmovss    xmm20, [rel fa2_neg_inf]

    ; Process full 16-wide chunks
    mov       rax, rcx
    shr       rax, 4                          ; tile_count / 16
    test      rax, rax
    jz        .tile_max_scalar_setup

    vmovaps   zmm20, [rel fa2_neg_inf]
    mov       rdi, rsi
.tile_max_vec:
    vmaxps    zmm20, zmm20, [rdi]
    add       rdi, 64
    dec       rax
    jnz       .tile_max_vec

    ; Horizontal max: zmm20 -> xmm20[0]
    vextractf32x8 ymm21, zmm20, 1
    vmaxps    ymm20, ymm20, ymm21
    vextractf32x4 xmm21, ymm20, 1
    vmaxps    xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0x4E
    vmaxps    xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0xB1
    vmaxps    xmm20, xmm20, xmm21
    jmp       .tile_max_scalar_check

.tile_max_scalar_setup:
    mov       rdi, rsi

.tile_max_scalar_check:
    ; Process remaining (tile_count % 16) scalars
    mov       rax, rcx
    and       rax, 15
    test      rax, rax
    jz        .tile_max_done
.tile_max_scalar:
    vmaxss    xmm20, xmm20, [rdi]
    add       rdi, 4
    dec       rax
    jnz       .tile_max_scalar
.tile_max_done:
    ; xmm20[0] = tile_max

    ; ============================================================
    ; Step 4: Online softmax update
    ; ============================================================
    ; new_max = max(row_max, tile_max)
    vmovss    xmm21, [rsp+STK_ROW_MAX]       ; old row_max
    vmaxss    xmm22, xmm21, xmm20            ; new_max

    ; correction = exp(old_max - new_max)
    vsubss    xmm23, xmm21, xmm22
    FAST_EXP_SS xmm23                        ; xmm23 = correction

    ; row_sum *= correction
    vmovss    xmm24, [rsp+STK_ROW_SUM]
    vmulss    xmm24, xmm24, xmm23
    vmovss    [rsp+STK_ROW_SUM], xmm24

    ; row_max = new_max
    vmovss    [rsp+STK_ROW_MAX], xmm22

    ; O *= correction (vectorised with zmm)
    vbroadcastss zmm25, xmm23

    lea       rdi, [rsp+STK_O]
    mov       rcx, [rsp+STK_HDV_FULL16]
    test      rcx, rcx
    jz        .rescale_o_rem
.rescale_o_full:
    vmulps    zmm26, zmm25, [rdi]
    vmovaps   [rdi], zmm26
    add       rdi, 64
    dec       rcx
    jnz       .rescale_o_full
.rescale_o_rem:
    mov       rcx, [rsp+STK_HDV_REM]
    test      rcx, rcx
    jz        .rescale_o_done
    vmovaps   zmm26, [rdi]
    vmulps    zmm26, zmm26, zmm25
    vmovaps   [rdi], zmm26
.rescale_o_done:

    ; ============================================================
    ; Step 5: Fused softmax+PV accumulation
    ;   For each j in tile: p = exp(score[j]-max), sum += p, O += p*V[j]
    ; ============================================================
    mov       r14, r13                        ; j = tile_start
    mov       r8, [rsp+STK_V_BASE]

    align     16
.pv_loop:
    cmp       r14, [rsp+STK_TILE_END]
    jge       .pv_done

    ; p = exp(score[j - tile_start] - new_max)
    mov       rax, r14
    sub       rax, r13                        ; j_local
    vmovss    xmm23, [rsp+STK_SCORES+rax*4]
    vsubss    xmm23, xmm23, xmm22            ; score - new_max
    FAST_EXP_SS xmm23                        ; xmm23 = p

    ; row_sum += p
    vaddss    xmm24, xmm23, [rsp+STK_ROW_SUM]
    vmovss    [rsp+STK_ROW_SUM], xmm24

    ; Broadcast p for PV multiply
    vbroadcastss zmm6, xmm23

    ; V_row = V_base + j * v_nb1
    mov       rax, r14
    imul      rax, [rsp+STK_V_NB1]
    lea       r11, [r8 + rax]                 ; r11 = V_row (quantized)

    ; Iterate over V blocks: O += p * dequant(V_block)
    xor       r9, r9                          ; block_idx = 0
    mov       r10, [rsp+STK_V_NBLOCKS]

.pv_block_loop:
    cmp       r9, r10
    jge       .pv_block_done

    ; V block pointer
    mov       rax, r9
    imul      rax, V_BLOCK_BYTES
    lea       rax, [r11 + rax]                ; V block ptr

    ; O base pointer
    lea       rcx, [rsp+STK_O]

    ; Quant-specific PV accumulation
    DEQUANT_V_ACCUM_BLOCK rax, zmm6, rcx, r9

    inc       r9
    jmp       .pv_block_loop
.pv_block_done:

    inc       r14
    jmp       .pv_loop
.pv_done:

    ; Advance tile
    add       r13, FA_TILE_KV
    jmp       .tile_loop

.tile_done:

    ; ============================================================
    ; Step 6: Normalize O /= row_sum
    ; ============================================================
    vmovss    xmm24, [rsp+STK_ROW_SUM]

    ; Guard against zero sum
    vpxord    xmm25, xmm25, xmm25
    vucomiss  xmm24, xmm25
    jz        .store_output
    jp        .store_output                   ; NaN check

    ; Reciprocal 1/sum via vrcp14ss + Newton-Raphson
    vrcp14ss  xmm25, xmm25, xmm24            ; ~1/sum (14-bit accuracy)
    ; NR iteration: x' = x * (2 - sum*x)
    vmulss    xmm26, xmm24, xmm25            ; sum * x
    vmovss    xmm27, [rel fa2_ones_f32]
    vaddss    xmm27, xmm27, xmm27            ; 2.0
    vsubss    xmm27, xmm27, xmm26            ; 2 - sum*x
    vmulss    xmm25, xmm25, xmm27            ; x * (2 - sum*x)

    ; Broadcast and normalize O
    vbroadcastss zmm25, xmm25

    lea       rdi, [rsp+STK_O]
    mov       rcx, [rsp+STK_HDV_FULL16]
    test      rcx, rcx
    jz        .norm_rem
.norm_full:
    vmulps    zmm26, zmm25, [rdi]
    vmovaps   [rdi], zmm26
    add       rdi, 64
    dec       rcx
    jnz       .norm_full
.norm_rem:
    mov       rcx, [rsp+STK_HDV_REM]
    test      rcx, rcx
    jz        .store_output
    vmovaps   zmm26, [rdi]
    vmulps    zmm26, zmm26, zmm25
    vmovaps   [rdi], zmm26

    ; ============================================================
    ; Step 7: Store O -> dst_row
    ; ============================================================
.store_output:
    lea       rsi, [rsp+STK_O]
    mov       rdi, [rsp+STK_DST_ROW]

    mov       rcx, [rsp+STK_HDV_FULL16]
    test      rcx, rcx
    jz        .store_rem
.store_full:
    vmovaps   zmm26, [rsi]
    vmovups   [rdi], zmm26                    ; dst may not be 64-byte aligned
    add       rsi, 64
    add       rdi, 64
    dec       rcx
    jnz       .store_full
.store_rem:
    mov       rcx, [rsp+STK_HDV_REM]
    test      rcx, rcx
    jz        .store_done
    vmovaps   zmm26, [rsi]
    vmovups   [rdi]{k2}, zmm26               ; masked store for partial tail
.store_done:

    ; Next work item
    inc       r15
    jmp       .work_loop

.work_done:
    ; === Epilogue ===
.epilogue:
    vzeroupper

%ifdef WIN64
    ; Restore XMM6-15
    vmovups   xmm6,  [rsp+STK_XMM_SAVE]
    vmovups   xmm7,  [rsp+STK_XMM_SAVE+16]
    vmovups   xmm8,  [rsp+STK_XMM_SAVE+32]
    vmovups   xmm9,  [rsp+STK_XMM_SAVE+48]
    vmovups   xmm10, [rsp+STK_XMM_SAVE+64]
    vmovups   xmm11, [rsp+STK_XMM_SAVE+80]
    vmovups   xmm12, [rsp+STK_XMM_SAVE+96]
    vmovups   xmm13, [rsp+STK_XMM_SAVE+112]
    vmovups   xmm14, [rsp+STK_XMM_SAVE+128]
    vmovups   xmm15, [rsp+STK_XMM_SAVE+144]

    ; Restore rsp from rbp
    lea       rsp, [rbp-56]                   ; 7 pushes below rbp
    pop       rsi
    pop       rdi
%else
    lea       rsp, [rbp-40]                   ; 5 pushes below rbp
%endif

    pop       r15
    pop       r14
    pop       r13
    pop       r12
    pop       rbx
    pop       rbp
    ret
