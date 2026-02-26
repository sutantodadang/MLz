; ============================================================================
; Flash Attention 2 - Q4_0 AVX2 Kernel
; ============================================================================
; Quantized K/V with Q4_0 format (18 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q4_0 block layout (18 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;   Low nibble = q[2i], high nibble = q[2i+1], subtract 8 for signed
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_0_avx2
%define K_BLOCK_BYTES       18
%define V_BLOCK_BYTES       18
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q4_nibble_mask: times 32 db 0x0F
fa2_q4_bias:        times 32 db 8
fa2_q4_ones_16:     times 16 dw 1
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; ymm12 = ones_16    (16x 1w)
; ymm13 = q4_bias    (32x 8)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q4_nibble_mask]
    vmovdqu   ymm12, [rel fa2_q4_ones_16]
    vmovdqu   ymm13, [rel fa2_q4_bias]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q4_0_block))
;
; %1 = k_block_ptr (register, points to Q4_0 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product:
;   1. Load fp16 scale, convert to f32
;   2. Extract low/high nibbles from packed bytes
;   3. Interleave to get 32 unsigned bytes, subtract 8 for signed
;   4. Dot product with Q using FMA approach:
;      - Convert Q4 signed bytes to f32 in 4 groups of 8
;      - FMA with corresponding Q f32 values
;      - Scale by block scale
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale -> xmm0
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0           ; xmm0[0] = scale (f32)

    ; 2. Load 16 packed bytes (32 nibbles)
    vmovdqu   xmm1, [%1+2]

    ; 3. Extract nibbles
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles >> 4 & 0x0F

    ; 4. Interleave: low[i], high[i] -> 32 bytes in ymm
    vpunpcklbw xmm4, xmm2, xmm3        ; interleave low halves
    vpunpckhbw xmm5, xmm2, xmm3        ; interleave high halves
    vinserti128 ymm4, ymm4, xmm5, 1    ; ymm4 = 32 unsigned bytes

    ; 5. Subtract bias (8) to get signed values
    vpsubb    ymm4, ymm4, ymm13        ; ymm4 = 32 signed int8

    ; 6. Convert i8 -> f32 and dot with Q, accumulating into %3
    ; Group 0: bytes 0-7
    vpmovsxbd ymm1, xmm4               ; sign-extend 8 bytes -> 8 int32
    vcvtdq2ps ymm1, ymm1               ; -> 8 f32 dequantized values
    vmovups   ymm2, [%2]               ; Q[0:7]
    vmulps    ymm1, ymm1, ymm2         ; Q * dequant
    vbroadcastss ymm5, xmm0            ; broadcast scale
    vfmadd231ps %3, ymm1, ymm5         ; acc += (Q * dequant) * scale

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0         ; xmm1 = low 16 bytes
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1               ; sign-extend -> 8 int32
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+32]            ; Q[8:15]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5         ; acc += (Q * dequant) * scale

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1         ; xmm1 = high 16 bytes
    vpmovsxbd ymm1, xmm1               ; sign-extend first 8
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+64]            ; Q[16:23]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5         ; acc += (Q * dequant) * scale

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1         ; xmm1 = high 16 bytes
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1               ; sign-extend -> 8 int32
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+96]            ; Q[24:31]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5         ; acc += (Q * dequant) * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_0_block)
;
; %1 = v_block_ptr (register, points to Q4_0 block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale -> xmm0
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm5, xmm0            ; ymm5 = broadcast scale

    ; 2. Load + extract nibbles (same as K)
    vmovdqu   xmm1, [%1+2]
    vpand     xmm2, xmm1, xmm11
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11
    vpunpcklbw xmm4, xmm2, xmm3
    vpunpckhbw xmm1, xmm2, xmm3        ; reuse xmm1
    vinserti128 ymm4, ymm4, xmm1, 1
    vpsubb    ymm4, ymm4, ymm13

    ; 3. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                    ; * 128

    ; 4. Dequant + prob * V + accumulate into O
    ; Group 0: bytes 0-7
    vpmovsxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5          ; dequant * scale
    vmovaps   ymm0, [%3+rax]            ; O[0:7]
    vfmadd231ps ymm0, %2, ymm1          ; O += prob * V_dequant
    vmovaps   [%3+rax], ymm0

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax+32]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+32], ymm0

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax+64]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+64], ymm0

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax+96]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+96], ymm0
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
