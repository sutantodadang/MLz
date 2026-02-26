; ============================================================================
; Flash Attention 2 - Q2_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q2_K format (84 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q2_K block layout (84 bytes):
;   offset  0: scales  (uint8_t[16], 16 bytes) — packed 4-bit scales and mins
;              For sub-block j: scale = scales[j] & 0x0F, min = scales[j] >> 4
;   offset 16: qs      (uint8_t[64], 64 bytes) — 2-bit quantized values
;              4 values packed per byte, 256 values total
;   offset 80: d       (ggml_half, 2 bytes) — super-block scale
;   offset 82: dmin    (ggml_half, 2 bytes) — super-block min
;
; 16 sub-blocks of 16 values each.
;
; Sub-block j mapping to qs bytes:
;   qs_offset = 16 + (j/8)*32 + (j & 1)*16
;   bit_shift = ((j >> 1) & 3) * 2
;   Values extracted: (qs[byte] >> bit_shift) & 0x03 for 16 consecutive bytes
;
; Dequantization: value = d * scale_bits * q_val - dmin * min_bits
;   Reformulated:  value = eff_scale * q_val + eff_min
;   where: eff_scale = d * (scales[j] & 0x0F)
;          eff_min   = -dmin * (scales[j] >> 4)
;
; NOTE: K_BLOCK_VALUES=256 means these kernels only work when head_dim >= 256
;       and head_dim is a multiple of 256.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q2_k_avx2
%define K_BLOCK_BYTES       84
%define V_BLOCK_BYTES       84
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q2k_2bit_mask: times 32 db 0x03
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = 2-bit mask (32x 0x03) — used for masking extracted 2-bit values
; ymm12 = (unused)
; ymm13 = (unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q2k_2bit_mask]
%endmacro

; ============================================================================
; Helper: Dequant + dot one Q2_K sub-block of 16 values (AVX2)
;
; Extracts 16 two-bit values from packed qs bytes, converts to f32,
; applies: result = eff_scale * q_val + eff_min
; Then dot-products with Q and accumulates.
;
; %1 = block_ptr register
; %2 = qs_offset (immediate) — byte offset within block to 16 qs bytes
; %3 = bit_shift (immediate) — right shift amount (0, 2, 4, or 6)
; %4 = ymm register with broadcast eff_scale
; %5 = ymm register with broadcast eff_min
; %6 = q_ptr_with_offset (memory operand for Q f32 values)
; %7 = acc_ymm (ymm to accumulate dot product)
;
; Clobbers: ymm0-ymm2
; ============================================================================
%macro DEQUANT_Q2K_SUBBLOCK_DOT 7
    ; Load 16 packed bytes, shift and mask to get 16 two-bit values
    vmovdqu   xmm2, [%1 + %2]
%if %3 > 0
    vpsrlw    xmm2, xmm2, %3
%endif
    vpand     xmm2, xmm2, xmm11             ; 16 bytes, each 0-3

    ; Group 0: first 8 values -> f32, dequant, dot with Q
    vpmovzxbd ymm1, xmm2                    ; zero-extend 8 bytes -> 8 int32
    vcvtdq2ps ymm1, ymm1                    ; -> 8 f32
    vfmadd132ps ymm1, %5, %4                ; dequant = eff_scale * q + eff_min
    vfmadd231ps %7, ymm1, [%6]              ; acc += dequant * Q[0:7]

    ; Group 1: next 8 values
    vpsrldq   xmm2, xmm2, 8                 ; shift right 8 bytes
    vpmovzxbd ymm1, xmm2
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %5, %4
    vfmadd231ps %7, ymm1, [%6 + 32]         ; acc += dequant * Q[8:15]
%endmacro

; ============================================================================
; Helper: Dequant + accumulate one Q2_K sub-block for V path (AVX2)
;
; %1 = block_ptr register
; %2 = qs_offset (immediate)
; %3 = bit_shift (immediate)
; %4 = ymm register with broadcast eff_scale
; %5 = ymm register with broadcast eff_min
; %6 = prob_ymm (broadcast probability)
; %7 = o_ptr (register, points to O accumulator for this sub-block's 16 values)
;
; Clobbers: ymm0-ymm2
; ============================================================================
%macro DEQUANT_Q2K_SUBBLOCK_V_ACCUM 7
    ; Load 16 packed bytes, shift and mask
    vmovdqu   xmm2, [%1 + %2]
%if %3 > 0
    vpsrlw    xmm2, xmm2, %3
%endif
    vpand     xmm2, xmm2, xmm11

    ; Group 0: first 8 values
    vpmovzxbd ymm1, xmm2
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %5, %4                ; dequant = eff_scale * q + eff_min
    vmovaps   ymm0, [%7]                    ; O[0:7]
    vfmadd231ps ymm0, %6, ymm1              ; O += prob * dequant
    vmovaps   [%7], ymm0

    ; Group 1: next 8 values
    vpsrldq   xmm2, xmm2, 8
    vpmovzxbd ymm1, xmm2
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %5, %4
    vmovaps   ymm0, [%7 + 32]
    vfmadd231ps ymm0, %6, ymm1
    vmovaps   [%7 + 32], ymm0
%endmacro

; ============================================================================
; Helper: Extract scale and min for sub-block j, compute eff_scale/eff_min
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
;
; Outputs:
;   ymm3 = broadcast eff_scale = d * float(scales[j] & 0x0F)
;   ymm4 = broadcast eff_min   = -dmin * float(scales[j] >> 4)
;
; Clobbers: ymm0-ymm4, eax, edx
; ============================================================================
%macro EXTRACT_Q2K_SCALE_MIN 2
    ; Load scale byte
    movzx     eax, byte [%1 + %2]            ; scales[j]
    mov       edx, eax
    and       eax, 0x0F                       ; scale_bits = low nibble
    shr       edx, 4                          ; min_bits = high nibble

    ; eff_scale = d * float(scale_bits)
    vmovd     xmm3, eax
    vcvtdq2ps xmm3, xmm3                    ; float(scale_bits)
    movzx     eax, word [%1 + 80]            ; d (fp16)
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; d as f32
    vmulss    xmm3, xmm3, xmm0              ; d * scale_bits
    vbroadcastss ymm3, xmm3                  ; broadcast eff_scale

    ; eff_min = -dmin * float(min_bits)
    vmovd     xmm4, edx
    vcvtdq2ps xmm4, xmm4                    ; float(min_bits)
    movzx     eax, word [%1 + 82]            ; dmin (fp16)
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; dmin as f32
    vxorps    xmm1, xmm1, xmm1
    vsubss    xmm0, xmm1, xmm0              ; -dmin
    vmulss    xmm4, xmm4, xmm0              ; -dmin * min_bits
    vbroadcastss ymm4, xmm4                  ; broadcast eff_min
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q2_k_block))
;
; %1 = k_block_ptr (register, points to Q2_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Processes 16 sub-blocks of 16 values each. For each sub-block:
;   1. Extract 4-bit scale and min from scales[j]
;   2. Compute eff_scale = d * scale, eff_min = -dmin * min
;   3. Extract 16 two-bit values from qs
;   4. Dequant, dot with Q, accumulate
;
; Clobbers: ymm0-ymm5, eax, edx
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Zero local accumulator
    vxorps    ymm5, ymm5, ymm5

    ; Process 16 sub-blocks
    %assign _j 0
    %rep 16
        ; Compute qs location for sub-block _j
        ; qs_offset = 16 + (_j/8)*32 + (_j & 1)*16
        ; bit_shift = ((_j >> 1) & 3) * 2
        %assign _qs_off 16 + (_j / 8) * 32 + (_j & 1) * 16
        %assign _shift ((_j >> 1) & 3) * 2

        ; Extract scale/min and compute eff_scale(ymm3), eff_min(ymm4)
        EXTRACT_Q2K_SCALE_MIN %1, _j

        ; Dequant 16 values and dot with Q
        DEQUANT_Q2K_SUBBLOCK_DOT %1, _qs_off, _shift, ymm3, ymm4, %2 + _j * 64, ymm5

        %assign _j _j + 1
    %endrep

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, ymm5
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q2_k_block)
;
; %1 = v_block_ptr (register, points to Q2_K block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Dequant each value, multiply by prob, accumulate into O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 bytes.
;
; Processes 16 sub-blocks of 16 values each.
;
; Clobbers: ymm0-ymm5, eax, edx, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; Compute O base offset = block_idx * 1024
    mov       rax, %4
    shl       rax, 10                        ; * 1024
    lea       rax, [%3 + rax]                ; rax = O base for this block

    ; Process 16 sub-blocks
    %assign _j 0
    %rep 16
        %assign _qs_off 16 + (_j / 8) * 32 + (_j & 1) * 16
        %assign _shift ((_j >> 1) & 3) * 2

        ; Extract scale/min and compute eff_scale(ymm3), eff_min(ymm4)
        EXTRACT_Q2K_SCALE_MIN %1, _j

        ; Dequant 16 values, weight by prob, accumulate to O
        DEQUANT_Q2K_SUBBLOCK_V_ACCUM %1, _qs_off, _shift, ymm3, ymm4, %2, rax + _j * 64

        %assign _j _j + 1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
