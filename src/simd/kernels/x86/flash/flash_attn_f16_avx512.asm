; ============================================================================
; Flash Attention 2 - F16 AVX-512 Kernel
; ============================================================================
; K/V stored as F16 (IEEE 754 half-precision, 2 bytes per value)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; F16 layout: contiguous fp16 values, no block structure.
; We treat it as "blocks" of 16 values (32 bytes) to match the skeleton's
; block-iteration pattern, since vcvtph2ps zmm, ymm operates on 16 fp16.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_f16_avx512
%define K_BLOCK_BYTES       32       ; 16 fp16 values = 32 bytes
%define V_BLOCK_BYTES       32
%define K_BLOCK_VALUES      16       ; 16 values per "block"
%define V_BLOCK_VALUES      16

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
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:15], cvt_f16_to_f32(K_f16[0:15]))
;
; %1 = k_block_ptr (register, 16 fp16 values = 32 bytes)
; %2 = q_ptr (register, 16 F32 values)
; %3 = acc_zmm (accumulator)
;
; F16 dequant: single vcvtph2ps zmm (16 fp16 -> 16 f32), then FMA with Q
;
; Clobbers: zmm0-zmm1, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Load 16 fp16 values and convert to f32
    vmovdqu   ymm0, [%1]               ; 16 fp16 = 32 bytes
    vcvtph2ps zmm0, ymm0               ; -> 16 f32

    ; Load Q and FMA
    vmovups   zmm1, [%2]               ; Q[0:15] f32
    vfmadd231ps %3, zmm0, zmm1         ; acc += K_f32 * Q_f32
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * cvt_f16_to_f32(V_f16[0:15])
;
; %1 = v_block_ptr (register, 16 fp16 values)
; %2 = prob_zmm (broadcast probability)
; %3 = o_base_ptr (register)
; %4 = block_idx (register)
;
; O offset = block_idx * 16 * 4 = block_idx << 6 (64 bytes per block)
;
; Clobbers: zmm0-zmm1, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; Convert 16 fp16 -> 16 f32
    vmovdqu   ymm0, [%1]
    vcvtph2ps zmm0, ymm0

    ; O offset = block_idx * 16 * 4 = block_idx * 64
    mov       rax, %4
    shl       rax, 6                    ; * 64

    ; Accumulate: O += prob * V_f32
    vmovaps   zmm1, [%3+rax]
    vfmadd231ps zmm1, %2, zmm0
    vmovaps   [%3+rax], zmm1
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
