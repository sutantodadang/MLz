; ============================================================================
; Flash Attention 2 - F16 AVX2 Kernel
; ============================================================================
; K/V stored as F16 (IEEE 754 half-precision, 2 bytes per value)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; F16 layout: contiguous fp16 values, no block structure.
; We treat it as "blocks" of 8 values (16 bytes) to match the skeleton's
; block-iteration pattern, since vcvtph2ps operates on 8 fp16 -> 8 fp32.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_f16_avx2
%define K_BLOCK_BYTES       16       ; 8 fp16 values = 16 bytes
%define V_BLOCK_BYTES       16
%define K_BLOCK_VALUES      8        ; 8 values per "block"
%define V_BLOCK_VALUES      8

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
; F16 needs no special data constants
%endmacro

; ============================================================================
; Initialize quant constant registers
; F16 doesn't need any constant registers
; ============================================================================
%macro INIT_QUANT_REGS 0
    ; Nothing to initialize for F16
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:7], cvt_f16_to_f32(K_f16[0:7]))
;
; %1 = k_block_ptr (register, 8 fp16 values = 16 bytes)
; %2 = q_ptr (register, 8 F32 values)
; %3 = acc_ymm (accumulator)
;
; F16 dequant: just vcvtph2ps, then FMA with Q
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Load 8 fp16 values and convert to f32
    vmovdqu   xmm0, [%1]               ; 8 fp16 = 16 bytes
    vcvtph2ps ymm0, xmm0               ; -> 8 f32

    ; Load Q and FMA
    vmovups   ymm1, [%2]               ; Q[0:7] f32
    vfmadd231ps %3, ymm0, ymm1         ; acc += K_f32 * Q_f32
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * cvt_f16_to_f32(V_f16[0:7])
;
; %1 = v_block_ptr (register, 8 fp16 values)
; %2 = prob_ymm (broadcast probability)
; %3 = o_base_ptr (register)
; %4 = block_idx (register)
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; Convert 8 fp16 -> 8 f32
    vmovdqu   xmm0, [%1]
    vcvtph2ps ymm0, xmm0

    ; O offset = block_idx * 8 * 4 = block_idx * 32
    mov       rax, %4
    shl       rax, 5                    ; * 32

    ; Accumulate: O += prob * V_f32
    vmovaps   ymm1, [%3+rax]
    vfmadd231ps ymm1, %2, ymm0
    vmovaps   [%3+rax], ymm1
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
