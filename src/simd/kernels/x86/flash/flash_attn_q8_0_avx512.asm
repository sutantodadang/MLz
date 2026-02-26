; ============================================================================
; Flash Attention 2 - Q8_0 AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q8_0 format (34 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q8_0 block layout (34 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: int8[32] quantized values (32 bytes)
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8.
; vpmovsxbd zmm, xmm sign-extends 16 int8 -> 16 int32 in one instruction.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q8_0_avx512
%define K_BLOCK_BYTES       34
%define V_BLOCK_BYTES       34
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
; Q8_0 needs no special constants
%endmacro

; ============================================================================
; Initialize quant constant registers
; Q8_0 doesn't need any constant registers (zmm11-13 unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    ; Nothing to initialize for Q8_0
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q8_0_block))
;
; %1 = k_block_ptr (register)
; %2 = q_ptr (register, 32 F32 values)
; %3 = acc_zmm (accumulator)
;
; Q8_0 dequant: load 32 int8, convert to f32, scale by d.
; Two groups of 16 using vpmovsxbd zmm, [ptr].
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm5, xmm0            ; zmm5 = broadcast scale

    ; 2. Group 0: bytes 0-15 -> 16 int32 -> 16 f32
    vpmovsxbd zmm1, [%1+2]             ; sign-extend 16 int8 -> 16 int32
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vmovups   zmm2, [%2]               ; Q[0:15]
    vmulps    zmm1, zmm1, zmm2         ; Q * dequant
    vfmadd231ps %3, zmm1, zmm5         ; acc += (Q * dequant) * scale

    ; 3. Group 1: bytes 16-31 -> 16 int32 -> 16 f32
    vpmovsxbd zmm1, [%1+18]            ; sign-extend next 16 int8
    vcvtdq2ps zmm1, zmm1
    vmovups   zmm2, [%2+64]            ; Q[16:31]
    vmulps    zmm1, zmm1, zmm2
    vfmadd231ps %3, zmm1, zmm5
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q8_0_block)
;
; %1 = v_block_ptr (register)
; %2 = prob_zmm (broadcast probability)
; %3 = o_base_ptr (register)
; %4 = block_idx (register)
;
; O offset = block_idx * 32 * 4 = block_idx << 7 (128 bytes per block)
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm5, xmm0

    ; 2. O offset = block_idx * 128
    mov       rax, %4
    shl       rax, 7

    ; 3. Group 0: bytes 0-15
    vpmovsxbd zmm1, [%1+2]
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5          ; dequant * scale
    vmovaps   zmm0, [%3+rax]            ; O[0:15]
    vfmadd231ps zmm0, %2, zmm1          ; O += prob * V_dequant
    vmovaps   [%3+rax], zmm0

    ; 4. Group 1: bytes 16-31
    vpmovsxbd zmm1, [%1+18]
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5
    vmovaps   zmm0, [%3+rax+64]         ; O[16:31]
    vfmadd231ps zmm0, %2, zmm1
    vmovaps   [%3+rax+64], zmm0
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
