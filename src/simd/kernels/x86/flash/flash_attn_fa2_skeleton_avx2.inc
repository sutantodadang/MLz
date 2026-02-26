; ============================================================================
; Flash Attention 2 - AVX2 Skeleton (Shared Include)
; ============================================================================
;
; This file implements the full FA2 online-softmax algorithm for AVX2.
; Each quantized kernel file defines type-specific macros and then includes
; this skeleton via:
;   %include "flash_attn_fa2_skeleton_avx2.inc"
;
; REQUIRED DEFINITIONS (must be set before including):
;   KERNEL_NAME          - global symbol name (e.g., simd_flash_attn_q4_0_avx2)
;   K_BLOCK_BYTES        - bytes per K quant block (e.g., 18 for Q4_0)
;   V_BLOCK_BYTES        - bytes per V quant block (e.g., 18 for Q4_0)
;   K_BLOCK_VALUES       - values per K block (e.g., 32 for Q4_0)
;   V_BLOCK_VALUES       - values per V block (e.g., 32 for Q4_0)
;
; REQUIRED MACROS (must be defined before including):
;   DECLARE_QUANT_CONSTANTS  - emit .rodata constants for dequant
;   INIT_QUANT_REGS          - load constant registers (e.g., nibble_mask)
;                              May use: ymm11, ymm12, ymm13.
;                              Must NOT touch: ymm14, ymm15.
;
;   DEQUANT_K_DOT_BLOCK  k_block_ptr, q_ptr, acc_ymm
;     Compute dot(Q_f32[0:31], dequant(K_block)) and ADD to acc_ymm.
;     Q is always F32. k_block_ptr = register pointing to quant block start.
;     q_ptr = register pointing to corresponding Q floats.
;     acc_ymm = ymm register to accumulate into.
;     Must preserve: ymm11-ymm15, rdi, rsi, r12-r15, rbx.
;     May clobber: ymm0-ymm5, rax.
;
;   DEQUANT_V_ACCUM_BLOCK  v_block_ptr, prob_ymm, o_base_ptr, block_idx
;     Dequantize one V block to F32, multiply by prob_ymm (broadcast prob),
;     and accumulate into O[block_idx * V_BLOCK_VALUES * 4 + o_base_ptr].
;     Must preserve: ymm11-ymm15, rdi, rsi, r12-r15, rbx.
;     May clobber: ymm0-ymm5, rax.
;
; REGISTER ALLOCATION:
;   ymm11-ymm13: quant constants (set by INIT_QUANT_REGS)
;   ymm14: row_sum (scalar in xmm14[0])
;   ymm15: row_max (scalar in xmm15[0])
;   r12: config pointer
;   r13: work_idx
;   r14: tile_start (during tile loop)
;   r15: n_kv (during tile loop)
;
; ============================================================================

bits 64

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
%define FA_TILE_KV          32
%define FA_MAX_HEAD_DIM    256

; --- Stack frame layout (32-byte aligned) ---
;
; O accumulator: 256 floats max = 1024 bytes
; Scores/probs: 32 floats each = 128 bytes each
; Locals: dimensions, pointers, strides, work decomposition
;
%define STK_O                0      ; float[256] = 1024 bytes
%define STK_SCORES        1024      ; float[32]  = 128 bytes
%define STK_PROBS         1152      ; float[32]  = 128 bytes
; --- Saved locals (1280+) ---
%define STK_CONFIG        1280
%define STK_Q_ROW         1288      ; Q row pointer for current work item
%define STK_K_BASE        1296      ; K base pointer for current head/batch
%define STK_V_BASE        1304      ; V base pointer for current head/batch
%define STK_DST_ROW       1312      ; dst row pointer for current work item
%define STK_MASK_PTR_L    1320      ; mask pointer (local copy)
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
; --- Scale vector (32 bytes, ymm-aligned) ---
%define STK_SCALE_VEC     1504
; --- Quant block counts ---
%define STK_K_NBLOCKS     1536
%define STK_V_NBLOCKS     1544
; --- Saved iq for mask ---
%define STK_IQ_SAVE       1552
; --- RBP save ---
%define STK_RBP_SAVE      1568
%define FRAME_SIZE         1600

; ============================================================================
; Data Section
; ============================================================================
section .rodata
align 32

; Shared FA2 constants
fa2_neg_inf:      times 8 dd 0xFF800000     ; -infinity
fa2_exp_log2e:    times 8 dd 0x3FB8AA3B     ; log2(e) = 1.44269504
fa2_exp_half:     times 8 dd 0x3F000000     ; 0.5
fa2_exp_ln2_hi:   times 8 dd 0x3F317218     ; ln(2) hi
fa2_exp_ln2_lo:   times 8 dd 0xB102E308     ; ln(2) lo (correction)
fa2_exp_c1:       times 8 dd 0x3E2AAAAB     ; 1/6
fa2_exp_c2:       times 8 dd 0x3D2AAAA5     ; 1/24
fa2_exp_c3:       times 8 dd 0x3C091461     ; 1/120
fa2_exp_c4:       times 8 dd 0x3AB74468     ; 1/720
fa2_exp_c5:       times 8 dd 0x39509FC9     ; 1/5040
fa2_exp_clamp_hi: times 8 dd 0x42B17218     ; 88.7228
fa2_exp_clamp_lo: times 8 dd 0xC2CE8ED0     ; -103.278
fa2_exp_magic:    times 8 dd 0x4B00007F     ; 2^23 + 127
fa2_ones_f32:     times 8 dd 0x3F800000     ; 1.0
fa2_zeros_f32:    times 8 dd 0x00000000     ; 0.0

; Quant-specific constants (emitted by the kernel file)
DECLARE_QUANT_CONSTANTS

; ============================================================================
; Code Section
; ============================================================================
section .text

; --- Inline exp macro (avoids call/rsp issues) ---
; Input: ymm0 = x (8 floats)
; Output: ymm0 = exp(x)
; Clobbers: ymm1-ymm5
; Preserves: ymm6-ymm15
%macro FAST_EXP_AVX2 0
    vminps    ymm0, ymm0, [rel fa2_exp_clamp_hi]
    vmaxps    ymm0, ymm0, [rel fa2_exp_clamp_lo]
    vmulps    ymm1, ymm0, [rel fa2_exp_log2e]
    vaddps    ymm2, ymm1, [rel fa2_exp_magic]
    vsubps    ymm1, ymm2, [rel fa2_exp_magic]
    vmulps    ymm3, ymm1, [rel fa2_exp_ln2_hi]
    vsubps    ymm0, ymm0, ymm3
    vmulps    ymm3, ymm1, [rel fa2_exp_ln2_lo]
    vsubps    ymm0, ymm0, ymm3
    vmovaps   ymm3, [rel fa2_exp_c5]
    vfmadd213ps ymm3, ymm0, [rel fa2_exp_c4]
    vfmadd213ps ymm3, ymm0, [rel fa2_exp_c3]
    vfmadd213ps ymm3, ymm0, [rel fa2_exp_c2]
    vfmadd213ps ymm3, ymm0, [rel fa2_exp_c1]
    vfmadd213ps ymm3, ymm0, [rel fa2_exp_half]
    vfmadd213ps ymm3, ymm0, [rel fa2_ones_f32]
    vfmadd213ps ymm3, ymm0, [rel fa2_ones_f32]
    vpslld    ymm2, ymm2, 23
    vpaddd    ymm0, ymm3, ymm2
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
    sub       rsp, 160
    vmovaps   [rsp],      xmm6
    vmovaps   [rsp+16],   xmm7
    vmovaps   [rsp+32],   xmm8
    vmovaps   [rsp+48],   xmm9
    vmovaps   [rsp+64],   xmm10
    vmovaps   [rsp+80],   xmm11
    vmovaps   [rsp+96],   xmm12
    vmovaps   [rsp+112],  xmm13
    vmovaps   [rsp+128],  xmm14
    vmovaps   [rsp+144],  xmm15
    mov       rdi, rcx          ; Win64: first arg rcx -> rdi
%endif

    ; Save original rbp, setup aligned frame
    mov       r12, rdi              ; r12 = config pointer (persistent)
    and       rsp, ~31              ; 32-byte align stack
    sub       rsp, FRAME_SIZE
    mov       [rsp+STK_RBP_SAVE], rbp
    mov       [rsp+STK_CONFIG], r12

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

    ; Broadcast scale
    vbroadcastss ymm0, [r12+CFG_SCALE]
    vmovaps   [rsp+STK_SCALE_VEC], ymm0

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

    ; === Initialize quant-specific constant registers ===
    INIT_QUANT_REGS

    ; === Thread partitioning ===
    ; total_work = n_queries * n_head_q * batch_size
    mov       rax, [rsp+STK_N_QUERIES]
    imul      rax, [rsp+STK_N_HEAD_Q]
    imul      rax, [rsp+STK_BATCH_SIZE]
    mov       rcx, rax              ; rcx = total_work

    movsxd    r8, dword [r12+CFG_NTH]
    movsxd    r9, dword [r12+CFG_ITH]

    ; work_per_thread = total / nth
    xor       rdx, rdx
    div       r8                    ; rax = base_work, rdx = remainder
    mov       rbx, rax              ; rbx = base_work

    ; start = ith * base_work + min(ith, remainder)
    mov       rax, r9
    imul      rax, rbx
    cmp       r9, rdx
    jge       .start_no_extra
    add       rax, r9
    jmp       .start_done
.start_no_extra:
    add       rax, rdx
.start_done:
    mov       [rsp+STK_WORK_START], rax

    ; end = start + base_work + (ith < remainder ? 1 : 0)
    add       rax, rbx
    cmp       r9, rdx
    jge       .end_no_extra
    inc       rax
.end_no_extra:
    cmp       rax, rcx
    cmovg     rax, rcx
    mov       [rsp+STK_WORK_END], rax

    ; === Work loop ===
    mov       r13, [rsp+STK_WORK_START]

.work_loop:
    cmp       r13, [rsp+STK_WORK_END]
    jge       .work_done

    ; --- Decompose work_idx -> (iq, ihq, ib) ---
    ; work_idx = ib * (nq * nhq) + ihq * nq + iq
    mov       rax, r13
    xor       rdx, rdx
    mov       rcx, [rsp+STK_N_QUERIES]
    div       rcx                   ; rax = work_idx / nq, rdx = iq
    mov       [rsp+STK_IQ_SAVE], rdx  ; save iq for mask later
    mov       r14, rdx              ; r14 = iq (temp)

    xor       rdx, rdx
    mov       rcx, [rsp+STK_N_HEAD_Q]
    div       rcx                   ; rax = ib, rdx = ihq
    mov       r15, rdx              ; r15 = ihq (temp)
    mov       r8, rax               ; r8 = ib (temp)

    ; --- GQA: map ihq -> ihkv ---
    mov       rcx, r15
    imul      rcx, [rsp+STK_N_HEAD_KV]
    mov       rax, rcx
    xor       rdx, rdx
    div       qword [rsp+STK_N_HEAD_Q]
    mov       rbx, rax              ; rbx = ihkv

    ; --- Compute Q_row pointer ---
    ; Q_row = q_ptr + iq*q_nb1 + ihq*q_nb2 + ib*q_nb3
    mov       rdi, [r12+CFG_Q_PTR]
    mov       rcx, r14
    imul      rcx, [rsp+STK_Q_NB1]
    add       rdi, rcx
    mov       rcx, r15
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
    mov       rcx, r15
    imul      rcx, [rsp+STK_DST_NB2]
    add       rax, rcx
    mov       rcx, r8
    imul      rcx, [rsp+STK_DST_NB3]
    add       rax, rcx
    mov       [rsp+STK_DST_ROW], rax

    ; --- Zero the O accumulator ---
    vxorps    ymm0, ymm0, ymm0
    mov       rdx, [rsp+STK_HEAD_DIM_V]
    xor       rax, rax
.zero_o_loop:
    cmp       rax, rdx
    jge       .zero_o_done
    vmovaps   [rsp+STK_O+rax*4], ymm0
    add       rax, 8
    jmp       .zero_o_loop
.zero_o_done:

    ; --- Initialize online softmax state ---
    vmovss    xmm15, [rel fa2_neg_inf]  ; row_max = -inf (scalar)
    vxorps    xmm14, xmm14, xmm14      ; row_sum = 0.0 (scalar)

    ; --- KV Tile Loop ---
    xor       r14, r14                  ; r14 = tile_start = 0
    mov       r15, [rsp+STK_N_KV]      ; r15 = n_kv

.tile_loop:
    cmp       r14, r15
    jge       .tile_done

    ; tile_end = min(tile_start + FA_TILE_KV, n_kv)
    lea       rbx, [r14 + FA_TILE_KV]
    cmp       rbx, r15
    cmovg     rbx, r15
    ; rbx = tile_end, tile_count = rbx - r14

    ; ============================================================
    ; Step 1: QK dot products -> STK_SCORES
    ; ============================================================
    mov       rcx, r14              ; j = tile_start
    mov       rsi, [rsp+STK_K_BASE]
    mov       rdi, [rsp+STK_Q_ROW]

.qk_dot_loop:
    cmp       rcx, rbx
    jge       .qk_dot_done

    ; Save loop counter
    push      rcx

    ; K_row = K_base + j * k_nb1
    mov       rax, rcx
    imul      rax, [rsp+8+STK_K_NB1]     ; +8 for push
    lea       r8, [rsi + rax]             ; r8 = K_row (quantized)

    ; Clear dot accumulator
    vxorps    ymm6, ymm6, ymm6

    ; Iterate over quant blocks
    xor       r9, r9                      ; block_idx = 0
    mov       r10, [rsp+8+STK_K_NBLOCKS] ; +8 for push

.qk_block_loop:
    cmp       r9, r10
    jge       .qk_block_done

    ; K block pointer: K_row + block_idx * K_BLOCK_BYTES
    mov       rax, r9
    imul      rax, K_BLOCK_BYTES
    lea       r11, [r8 + rax]             ; r11 = K block ptr

    ; Q pointer for this block: Q_row + block_idx * BLOCK_VALUES * 4
    mov       rax, r9
    imul      rax, K_BLOCK_VALUES * 4
    lea       rdx, [rdi + rax]            ; rdx = Q ptr

    ; Quant-specific dot product: dot(Q[rdx], K[r11]) -> add to ymm6
    DEQUANT_K_DOT_BLOCK r11, rdx, ymm6

    inc       r9
    jmp       .qk_block_loop
.qk_block_done:

    ; Horizontal sum ymm6 -> scalar xmm6
    vextractf128 xmm0, ymm6, 1
    vaddps    xmm6, xmm6, xmm0
    vpermilps xmm0, xmm6, 0x4E
    vaddps    xmm6, xmm6, xmm0
    vpermilps xmm0, xmm6, 0xB1
    vaddss    xmm6, xmm6, xmm0

    ; Restore loop counter, store score
    pop       rcx
    mov       rax, rcx
    sub       rax, r14                    ; j - tile_start
    vmovss    [rsp+STK_SCORES+rax*4], xmm6

    inc       rcx
    jmp       .qk_dot_loop
.qk_dot_done:

    ; ============================================================
    ; Step 2: Scale scores
    ; ============================================================
    mov       rcx, rbx
    sub       rcx, r14                    ; tile_count
    lea       rax, [rsp+STK_SCORES]
    xor       rdx, rdx
.scale_loop:
    cmp       rdx, rcx
    jge       .scale_done
    vmovss    xmm0, [rax+rdx*4]
    vmulss    xmm0, xmm0, [rsp+STK_SCALE_VEC]
    vmovss    [rax+rdx*4], xmm0
    inc       rdx
    jmp       .scale_loop
.scale_done:

    ; ============================================================
    ; Step 3: Apply mask (if present)
    ; ============================================================
    mov       rax, [rsp+STK_MASK_PTR_L]
    test      rax, rax
    jz        .mask_done

    ; mask_val = mask_ptr + iq * mask_nb1
    mov       rcx, [rsp+STK_IQ_SAVE]
    imul      rcx, [rsp+STK_MASK_NB1]
    add       rax, rcx                    ; rax = mask row base

    ; Add mask values to scores
    lea       rdi, [rsp+STK_SCORES]
    mov       rcx, r14                    ; j = tile_start
    xor       rdx, rdx
.mask_loop:
    mov       rsi, rbx
    sub       rsi, r14
    cmp       rdx, rsi
    jge       .mask_done
    ; mask[iq, tile_start + j_local] - assume mask is contiguous F32 per row
    mov       rsi, r14
    add       rsi, rdx                    ; absolute kv index
    vmovss    xmm0, [rdi+rdx*4]          ; score
    vaddss    xmm0, xmm0, [rax+rsi*4]    ; + mask value
    vmovss    [rdi+rdx*4], xmm0
    inc       rdx
    jmp       .mask_loop
.mask_done:

    ; ============================================================
    ; Step 4: Online softmax
    ; ============================================================
    ; Find tile_max
    vmovss    xmm8, [rel fa2_neg_inf]
    lea       rax, [rsp+STK_SCORES]
    mov       rcx, rbx
    sub       rcx, r14                    ; tile_count
    xor       rdx, rdx
.tile_max_loop:
    cmp       rdx, rcx
    jge       .tile_max_done
    vmaxss    xmm8, xmm8, [rax+rdx*4]
    inc       rdx
    jmp       .tile_max_loop
.tile_max_done:

    ; new_max = max(row_max, tile_max)
    vmaxss    xmm9, xmm15, xmm8

    ; correction = exp(old_max - new_max)
    vsubss    xmm0, xmm15, xmm9
    vbroadcastss ymm0, xmm0
    FAST_EXP_AVX2
    ; ymm0[0] = correction
    vmovss    xmm10, xmm0                ; xmm10 = correction (scalar)

    ; Rescale O *= correction
    vbroadcastss ymm10, xmm10
    mov       rdx, [rsp+STK_HEAD_DIM_V]
    xor       rcx, rcx
.rescale_o_loop:
    cmp       rcx, rdx
    jge       .rescale_o_done
    vmovaps   ymm0, [rsp+STK_O+rcx*4]
    vmulps    ymm0, ymm0, ymm10
    vmovaps   [rsp+STK_O+rcx*4], ymm0
    add       rcx, 8
    jmp       .rescale_o_loop
.rescale_o_done:

    ; row_sum *= correction (scalar)
    vmulss    xmm14, xmm14, xmm10

    ; Update row_max = new_max
    vmovss    xmm15, xmm9

    ; Compute p[j] = exp(score[j] - new_max)
    lea       rax, [rsp+STK_SCORES]
    lea       r9, [rsp+STK_PROBS]
    mov       rcx, rbx
    sub       rcx, r14                    ; tile_count
    xor       rdx, rdx
.prob_loop:
    cmp       rdx, rcx
    jge       .prob_done
    vmovss    xmm0, [rax+rdx*4]
    vsubss    xmm0, xmm0, xmm9
    vbroadcastss ymm0, xmm0
    FAST_EXP_AVX2
    vmovss    [r9+rdx*4], xmm0
    vaddss    xmm14, xmm14, xmm0         ; row_sum += p[j]
    inc       rdx
    jmp       .prob_loop
.prob_done:

    ; ============================================================
    ; Step 5: PV accumulation
    ; ============================================================
    ; For each j in tile: O += p[j] * dequant(V_row[j])
    mov       r8, [rsp+STK_V_BASE]
    mov       rcx, rbx
    sub       rcx, r14                    ; tile_count
    xor       rdx, rdx                    ; j_local = 0

.pv_loop:
    cmp       rdx, rcx
    jge       .pv_done

    ; Save j_local and tile_count
    push      rdx
    push      rcx

    ; Broadcast p[j_local]
    vbroadcastss ymm6, [rsp+16+STK_PROBS+rdx*4]  ; +16 for two pushes

    ; V_row = V_base + (tile_start + j_local) * v_nb1
    mov       rax, r14
    add       rax, rdx                    ; absolute kv index
    imul      rax, [rsp+16+STK_V_NB1]    ; +16 for pushes
    lea       r11, [r8 + rax]             ; r11 = V_row (quantized)

    ; Iterate over V blocks
    xor       r9, r9                      ; block_idx = 0
    mov       r10, [rsp+16+STK_V_NBLOCKS]

.pv_block_loop:
    cmp       r9, r10
    jge       .pv_block_done

    ; V block pointer
    mov       rax, r9
    imul      rax, V_BLOCK_BYTES
    lea       rax, [r11 + rax]            ; V block ptr

    ; O base pointer
    lea       rcx, [rsp+16+STK_O]         ; +16 for pushes

    ; Quant-specific PV accumulation
    DEQUANT_V_ACCUM_BLOCK rax, ymm6, rcx, r9

    inc       r9
    jmp       .pv_block_loop
.pv_block_done:

    ; Restore j_local and tile_count
    pop       rcx
    pop       rdx
    inc       rdx
    jmp       .pv_loop
.pv_done:

    ; Advance tile
    add       r14, FA_TILE_KV
    jmp       .tile_loop

.tile_done:

    ; ============================================================
    ; Step 6: Normalize O /= row_sum
    ; ============================================================
    vmovss    xmm0, [rel fa2_ones_f32]
    vdivss    xmm0, xmm0, xmm14
    vbroadcastss ymm0, xmm0

    mov       rdx, [rsp+STK_HEAD_DIM_V]
    xor       rcx, rcx
.normalize_loop:
    cmp       rcx, rdx
    jge       .normalize_done
    vmovaps   ymm1, [rsp+STK_O+rcx*4]
    vmulps    ymm1, ymm1, ymm0
    vmovaps   [rsp+STK_O+rcx*4], ymm1
    add       rcx, 8
    jmp       .normalize_loop
.normalize_done:

    ; ============================================================
    ; Step 7: Store O -> dst_row
    ; ============================================================
    mov       r9, [rsp+STK_DST_ROW]
    mov       rdx, [rsp+STK_HEAD_DIM_V]
    xor       rcx, rcx
.store_loop:
    cmp       rcx, rdx
    jge       .store_done
    vmovaps   ymm0, [rsp+STK_O+rcx*4]
    vmovups   [r9+rcx*4], ymm0
    add       rcx, 8
    jmp       .store_loop
.store_done:

    ; Next work item
    inc       r13
    jmp       .work_loop

.work_done:
    ; === Epilogue ===
    mov       rbp, [rsp+STK_RBP_SAVE]

%ifdef WIN64
    lea       rax, [rbp-216]
    vmovaps   xmm6,  [rax]
    vmovaps   xmm7,  [rax+16]
    vmovaps   xmm8,  [rax+32]
    vmovaps   xmm9,  [rax+48]
    vmovaps   xmm10, [rax+64]
    vmovaps   xmm11, [rax+80]
    vmovaps   xmm12, [rax+96]
    vmovaps   xmm13, [rax+112]
    vmovaps   xmm14, [rax+128]
    vmovaps   xmm15, [rax+144]
    lea       rsp, [rbp-56]
    pop       rsi
    pop       rdi
%else
    lea       rsp, [rbp-40]
%endif

    pop       r15
    pop       r14
    pop       r13
    pop       r12
    pop       rbx
    pop       rbp
    vzeroupper
    ret
