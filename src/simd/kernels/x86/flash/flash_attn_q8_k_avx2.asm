; ============================================================================
; Flash Attention 2 - Q8_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q8_K format (292 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q8_K block layout (292 bytes):
;   offset 0:   d      (float32, 4 bytes) — scale factor (NOTE: f32, not f16!)
;   offset 4:   qs     (int8_t[256], 256 bytes) — signed 8-bit quantized values
;   offset 260: bsums  (int16_t[16], 32 bytes) — block sums (unused in FA)
;
; Dequantization: value[i] = d * qs[i]
;
; NOTE: K_BLOCK_VALUES=256 means these kernels only work when head_dim >= 256
;       and head_dim is a multiple of 256.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q8_k_avx2
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
; Q8_K doesn't need any constant registers (ymm11-13 unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    ; Nothing to initialize for Q8_K
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q8_K_block))
;
; %1 = k_block_ptr (register, points to Q8_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Q8_K dequant: load 256 int8 values, convert to f32 in groups of 8,
; dot product with Q, then scale by f32 'd' at the end.
;
; Strategy: accumulate raw dot products into local ymm5, then
;   acc += local_dot * scale
;
; Processes 256 values as 8 sub-blocks of 32 values, each with 4 groups of 8.
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load f32 scale -> ymm0 (broadcast)
    vmovss    xmm0, [%1]                ; xmm0[0] = d (f32 scale)
    vbroadcastss ymm0, xmm0             ; ymm0 = broadcast scale

    ; 2. Zero local accumulator
    vxorps    ymm5, ymm5, ymm5          ; ymm5 = raw dot product accumulator

    ; 3. Process 256 int8 values: 8 sub-blocks × 4 groups × 8 values
    %assign _sb 0
    %rep 8
        %assign grp 0
        %rep 4
            vpmovsxbd ymm1, qword [%1 + 4 + _sb*32 + grp*8]
            vcvtdq2ps ymm1, ymm1
            vfmadd231ps ymm5, ymm1, [%2 + (_sb*32 + grp*8)*4]
            %assign grp grp+1
        %endrep
        %assign _sb _sb+1
    %endrep

    ; 4. Scale accumulated dot product and add to output accumulator
    vfmadd231ps %3, ymm5, ymm0          ; acc += local_dot * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q8_K_block)
;
; %1 = v_block_ptr (register, points to Q8_K block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (immediate/register, block index for offset calculation)
;
; Dequant each value (d * qs[i]), multiply by prob, accumulate into O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 (compile-time constant
; when block_idx is immediate from skeleton's %rep loop).
;
; Processes 256 values as 8 sub-blocks of 32, each with 4 groups of 8.
;
; Clobbers: ymm0-ymm5, rax, rdx
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load f32 scale -> ymm0 (broadcast)
    vmovss    xmm0, [%1]
    vbroadcastss ymm0, xmm0             ; ymm0 = broadcast scale
    ; 2. Pre-compute O base pointer: rdx = %3 + %4 * 1024
    ;    (block_idx * 256 values * 4 bytes = block_idx * 1024)
    mov       rdx, %4
    shl       rdx, 10                    ; rdx = block_idx * 1024
    add       rdx, %3                    ; rdx = O base for this block

    ; 3. Process 256 values: dequant, weight by prob, accumulate into O
    %assign _sb 0
    %rep 8
        %assign grp 0
        %rep 4
            vpmovsxbd ymm1, qword [%1 + 4 + _sb*32 + grp*8]
            vcvtdq2ps ymm1, ymm1
            vmulps    ymm1, ymm1, ymm0   ; dequant = scale * int8
            vmulps    ymm1, ymm1, %2     ; weighted = dequant * prob
            vaddps    ymm1, ymm1, [rdx + (_sb*32 + grp*8)*4]
            vmovaps   [rdx + (_sb*32 + grp*8)*4], ymm1
            %assign grp grp+1
        %endrep
        %assign _sb _sb+1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
