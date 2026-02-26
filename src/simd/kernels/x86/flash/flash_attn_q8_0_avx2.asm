; ============================================================================
; Flash Attention 2 - Q8_0 AVX2 Kernel
; ============================================================================
; Quantized K/V with Q8_0 format (34 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q8_0 block layout (34 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: int8[32] quantized values (32 bytes)
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q8_0_avx2
%define K_BLOCK_BYTES       34
%define V_BLOCK_BYTES       34
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
; Q8_0 needs no special constants beyond what skeleton provides
align 32
fa2_q8_ones_16:     times 16 dw 1
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm12 = ones_16 (for vpmaddwd identity)
; ymm11, ymm13 unused but must be preserved
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm12, [rel fa2_q8_ones_16]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q8_0_block))
;
; %1 = k_block_ptr (register)
; %2 = q_ptr (register, 32 F32 values)
; %3 = acc_ymm (accumulator)
;
; Q8_0 dequant: simply load 32 int8, convert to f32, scale by d
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm5, xmm0            ; ymm5 = broadcast scale

    ; 2. Load 32 int8 values directly
    vmovdqu   ymm4, [%1+2]             ; 32 signed int8

    ; 3. Convert i8 -> f32 in 4 groups of 8 and dot with Q
    ; Group 0: bytes 0-7
    vpmovsxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+32]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+64]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+96]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q8_0_block)
;
; %1 = v_block_ptr (register)
; %2 = prob_ymm (broadcast probability)
; %3 = o_base_ptr (register)
; %4 = block_idx (register)
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm5, xmm0

    ; 2. Load 32 int8
    vmovdqu   ymm4, [%1+2]

    ; 3. O offset = block_idx * 32 * 4
    mov       rax, %4
    shl       rax, 7

    ; 4. Dequant + accumulate
    ; Group 0
    vpmovsxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax], ymm0

    ; Group 1
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax+32]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+32], ymm0

    ; Group 2
    vextracti128 xmm1, ymm4, 1
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5
    vmovaps   ymm0, [%3+rax+64]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+64], ymm0

    ; Group 3
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
