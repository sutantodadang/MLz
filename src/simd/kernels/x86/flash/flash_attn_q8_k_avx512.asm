; ============================================================================
; Flash Attention 2 - Q8_K AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q8_K format (292 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q8_K block layout (292 bytes):
;   offset 0:   d      (float32, 4 bytes) — scale factor (NOTE: f32, not f16!)
;   offset 4:   qs     (int8_t[256], 256 bytes) — signed 8-bit quantized values
;   offset 260: bsums  (int16_t[16], 32 bytes) — block sums (unused in FA)
;
; Dequantization: value[i] = d * qs[i]
;
; AVX-512 processes 16 groups of 16 instead of AVX2's 32 groups of 8.
; vpmovsxbd zmm, xmmword [ptr] sign-extends 16 int8 -> 16 int32.
;
; NOTE: K_BLOCK_VALUES=256 means these kernels only work when head_dim >= 256
;       and head_dim is a multiple of 256.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q8_k_avx512
%define K_BLOCK_BYTES       292
%define V_BLOCK_BYTES       292
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
; Q8_K needs no special constants — direct signed int8 with f32 scale
%endmacro

; ============================================================================
; Initialize quant constant registers
; Q8_K doesn't need any constant registers (zmm11-13 unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    ; Nothing to initialize for Q8_K
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q8_K_block))
;
; %1 = k_block_ptr (register, points to Q8_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; Q8_K dequant: load 256 int8 values, convert to f32 in groups of 16,
; dot product with Q, then scale by f32 'd' at the end.
;
; Strategy: accumulate raw dot products into local zmm5, then
;   acc += local_dot * scale
;
; Processes 256 values as 16 groups of 16 values.
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load f32 scale -> zmm0 (broadcast)
    vmovss    xmm0, [%1]                ; xmm0[0] = d (f32 scale)
    vbroadcastss zmm0, xmm0             ; zmm0 = broadcast scale

    ; 2. Zero local accumulator
    vxorps    zmm5, zmm5, zmm5          ; zmm5 = raw dot product accumulator

    ; 3. Process 256 int8 values: 16 groups of 16
    %assign grp 0
    %rep 16
        vpmovsxbd zmm1, [%1 + 4 + grp*16]
        vcvtdq2ps zmm1, zmm1
        vfmadd231ps zmm5, zmm1, [%2 + grp*16*4]
        %assign grp grp+1
    %endrep

    ; 4. Scale accumulated dot product and add to output accumulator
    vfmadd231ps %3, zmm5, zmm0          ; acc += local_dot * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q8_K_block)
;
; %1 = v_block_ptr (register, points to Q8_K block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (immediate/register, block index for offset calculation)
;
; Dequant each value (d * qs[i]), multiply by prob, accumulate into O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 (compile-time constant
; when block_idx is immediate from skeleton's %rep loop).
;
; Processes 256 values as 16 groups of 16 values.
;
; Clobbers: zmm0-zmm5, rax, rdx
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load f32 scale -> zmm0 (broadcast)
    vmovss    xmm0, [%1]
    vbroadcastss zmm0, xmm0             ; zmm0 = broadcast scale
    ; 2. Pre-compute O base pointer: rdx = %3 + %4 * 1024
    ;    (block_idx * 256 values * 4 bytes = block_idx * 1024)
    mov       rdx, %4
    shl       rdx, 10                    ; rdx = block_idx * 1024
    add       rdx, %3                    ; rdx = O base for this block

    ; 3. Process 256 values: dequant, weight by prob, accumulate into O
    %assign grp 0
    %rep 16
        vpmovsxbd zmm1, [%1 + 4 + grp*16]
        vcvtdq2ps zmm1, zmm1
        vmulps    zmm1, zmm1, zmm0       ; dequant = scale * int8
        vmulps    zmm1, zmm1, %2         ; weighted = dequant * prob
        vaddps    zmm1, zmm1, [rdx + grp*64]
        vmovaps   [rdx + grp*64], zmm1
        %assign grp grp+1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
