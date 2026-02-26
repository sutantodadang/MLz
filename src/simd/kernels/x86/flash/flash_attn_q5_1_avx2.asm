; ============================================================================
; Flash Attention 2 - Q5_1 AVX2 Kernel
; ============================================================================
; Quantized K/V with Q5_1 format (24 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q5_1 block layout (24 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: fp16 min   'm' (2 bytes)
;   offset 4: uint32 qh      (4 bytes) — 5th bit of each quant
;   offset 8: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;
; Dequantization: value = d * (nibble | (qh_bit << 4)) + m
;   5-bit unsigned [0,31], no bias subtraction — uses min offset m.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q5_1_avx2
%define K_BLOCK_BYTES       24
%define V_BLOCK_BYTES       24
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q5_1_nibble_mask:   times 32 db 0x0F
fa2_q5_1_shift_counts:  dd 0, 1, 2, 3, 4, 5, 6, 7
fa2_q5_1_dword_ones:    times 8 dd 1
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F for xmm-level nibble extraction)
; ymm12 = shift_counts [0..7] for variable-shift qh bit extraction
; ymm13 = dword_ones   (8x 1) for masking after shift
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q5_1_nibble_mask]
    vmovdqu   ymm12, [rel fa2_q5_1_shift_counts]
    vmovdqu   ymm13, [rel fa2_q5_1_dword_ones]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q5_1_block))
;
; %1 = k_block_ptr (register, points to Q5_1 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product:
;   1. Load fp16 scale d, convert to f32, broadcast
;   2. Load fp16 min m, convert to f32, broadcast
;   3. Load qh (4 bytes) for 5th bit extraction
;   4. Extract low/high nibbles from packed bytes
;   5. Interleave to get 32 unsigned bytes in sequential order
;   6. For each group of 8: zero-extend, inject 5th bit, convert,
;      compute d*val+m, dot with Q, accumulate
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale d -> broadcast ymm5
    movzx     eax, word [%1]
    vmovd     xmm5, eax
    vcvtph2ps xmm5, xmm5
    vbroadcastss ymm5, xmm5            ; ymm5 = broadcast d

    ; 2. Load fp16 min m -> broadcast ymm0
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm0, xmm0            ; ymm0 = broadcast m

    ; 3. Load 16 packed bytes (32 nibbles) from qs
    vmovdqu   xmm1, [%1+8]

    ; 4. Extract nibbles
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles >> 4 & 0x0F

    ; 5. Interleave: low[i], high[i] -> 32 bytes in ymm4
    vpunpcklbw xmm4, xmm2, xmm3        ; interleave low halves
    vpunpckhbw xmm1, xmm2, xmm3        ; interleave high halves
    vinserti128 ymm4, ymm4, xmm1, 1    ; ymm4 = 32 unsigned nibble bytes

    ; --- Group 0: values 0-7 ---
    vpmovzxbd ymm1, xmm4               ; zero-extend 8 bytes -> 8 uint32
    ; Extract qh byte 0, broadcast, variable-shift, mask, position at bit 4
    movzx     eax, byte [%1+4]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12        ; shift by [0,1,2,3,4,5,6,7]
    vpand     ymm3, ymm3, ymm13        ; isolate bit 0 -> 0 or 1
    vpslld    ymm3, ymm3, 4            ; -> 0 or 16
    vpor      ymm1, ymm1, ymm3         ; combine: 5-bit unsigned value
    vcvtdq2ps ymm1, ymm1               ; -> f32
    vfmadd213ps ymm1, ymm5, ymm0       ; ymm1 = d * val + m
    vmovups   ymm2, [%2]               ; Q[0:7]
    vfmadd231ps %3, ymm2, ymm1         ; acc += Q * dequant_val

    ; --- Group 1: values 8-15 ---
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend -> 8 uint32
    movzx     eax, byte [%1+5]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    vmovups   ymm2, [%2+32]            ; Q[8:15]
    vfmadd231ps %3, ymm2, ymm1         ; acc += Q * dequant_val

    ; --- Group 2: values 16-23 ---
    vextracti128 xmm1, ymm4, 1         ; high 16 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend first 8 -> uint32
    movzx     eax, byte [%1+6]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3         ; combine: 5-bit unsigned value
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    vmovups   ymm2, [%2+64]            ; Q[16:23]
    vfmadd231ps %3, ymm2, ymm1         ; acc += Q * dequant_val

    ; --- Group 3: values 24-31 ---
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend -> 8 uint32
    movzx     eax, byte [%1+7]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    vmovups   ymm2, [%2+96]            ; Q[24:31]
    vfmadd231ps %3, ymm2, ymm1         ; acc += Q * dequant_val
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q5_1_block)
;
; %1 = v_block_ptr (register, points to Q5_1 block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
; O offset = block_idx * 32 * 4 = block_idx << 7 (128 bytes per block)
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale d -> broadcast ymm5
    movzx     eax, word [%1]
    vmovd     xmm5, eax
    vcvtph2ps xmm5, xmm5
    vbroadcastss ymm5, xmm5            ; ymm5 = broadcast d

    ; 2. Load fp16 min m -> broadcast ymm0
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm0, xmm0            ; ymm0 = broadcast m

    ; 3. Load + extract nibbles, interleave
    vmovdqu   xmm1, [%1+8]
    vpand     xmm2, xmm1, xmm11
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11
    vpunpcklbw xmm4, xmm2, xmm3
    vpunpckhbw xmm1, xmm2, xmm3
    vinserti128 ymm4, ymm4, xmm1, 1    ; ymm4 = 32 unsigned nibble bytes

    ; 4. Compute O offset = block_idx * 128
    mov       rax, %4
    shl       rax, 7

    ; --- Group 0: values 0-7 ---
    vpmovzxbd ymm1, xmm4
    movzx     eax, byte [%1+4]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m

    ; Recompute O offset (rax was clobbered by movzx eax)
    mov       rax, %4
    shl       rax, 7

    vmovaps   ymm2, [%3+rax]           ; O[0:7]
    vfmadd231ps ymm2, %2, ymm1         ; O += prob * dequant_val
    vmovaps   [%3+rax], ymm2

    ; --- Group 1: values 8-15 ---
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    movzx     eax, byte [%1+5]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    mov       rax, %4
    shl       rax, 7
    vmovaps   ymm2, [%3+rax+32]        ; O[8:15]
    vfmadd231ps ymm2, %2, ymm1
    vmovaps   [%3+rax+32], ymm2

    ; --- Group 2: values 16-23 ---
    vextracti128 xmm1, ymm4, 1
    vpmovzxbd ymm1, xmm1
    movzx     eax, byte [%1+6]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    mov       rax, %4
    shl       rax, 7
    vmovaps   ymm2, [%3+rax+64]        ; O[16:23]
    vfmadd231ps ymm2, %2, ymm1
    vmovaps   [%3+rax+64], ymm2

    ; --- Group 3: values 24-31 ---
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    movzx     eax, byte [%1+7]
    vmovd     xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpsrlvd   ymm3, ymm3, ymm12
    vpand     ymm3, ymm3, ymm13
    vpslld    ymm3, ymm3, 4
    vpor      ymm1, ymm1, ymm3
    vcvtdq2ps ymm1, ymm1
    vfmadd213ps ymm1, ymm5, ymm0       ; d * val + m
    mov       rax, %4
    shl       rax, 7
    vmovaps   ymm2, [%3+rax+96]        ; O[24:31]
    vfmadd231ps ymm2, %2, ymm1
    vmovaps   [%3+rax+96], ymm2
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
