; ============================================================================
; Flash Attention 2 - Q2_K AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q2_K format (84 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
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
; AVX-512 advantage: vpmovzxbd zmm, xmm processes all 16 bytes in one
; instruction — no group split needed (vs AVX2's 2×8 approach).
;
; NOTE: K_BLOCK_VALUES=256 means these kernels only work when head_dim >= 256
;       and head_dim is a multiple of 256.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q2_k_avx512
%define K_BLOCK_BYTES       84
%define V_BLOCK_BYTES       84
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q2k_2bit_mask: times 64 db 0x03
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = 2-bit mask (64x 0x03) — used for masking extracted 2-bit values
; zmm12 = (unused)
; zmm13 = (unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqa64 zmm11, [rel fa2_q2k_2bit_mask]
%endmacro

; ============================================================================
; Helper: Extract scale and min for sub-block j, compute eff_scale/eff_min
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
;
; Outputs:
;   zmm3 = broadcast eff_scale = d * float(scales[j] & 0x0F)
;   zmm4 = broadcast eff_min   = -dmin * float(scales[j] >> 4)
;
; Clobbers: zmm0, zmm3, zmm4, xmm1, eax, ecx
; ============================================================================
%macro EXTRACT_Q2K_SCALE_MIN 2
    ; Load scale byte
    movzx     eax, byte [%1 + %2]            ; scales[j]
    mov       ecx, eax
    and       eax, 0x0F                       ; scale_bits = low nibble
    shr       ecx, 4                          ; min_bits = high nibble

    ; eff_scale = d * float(scale_bits)
    vmovd     xmm3, eax
    vcvtdq2ps xmm3, xmm3                    ; float(scale_bits)
    movzx     eax, word [%1 + 80]            ; d (fp16)
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; d as f32
    vmulss    xmm3, xmm3, xmm0              ; d * scale_bits
    vbroadcastss zmm3, xmm3                  ; broadcast eff_scale

    ; eff_min = -dmin * float(min_bits)
    vmovd     xmm4, ecx
    vcvtdq2ps xmm4, xmm4                    ; float(min_bits)
    movzx     eax, word [%1 + 82]            ; dmin (fp16)
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; dmin as f32
    vxorps    xmm1, xmm1, xmm1
    vsubss    xmm0, xmm1, xmm0              ; -dmin
    vmulss    xmm4, xmm4, xmm0              ; -dmin * min_bits
    vbroadcastss zmm4, xmm4                  ; broadcast eff_min
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q2_k_block))
;
; %1 = k_block_ptr (register, points to Q2_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; Processes 16 sub-blocks of 16 values each. For each sub-block:
;   1. Extract 4-bit scale and min from scales[j]
;   2. Compute eff_scale = d * scale, eff_min = -dmin * min
;   3. Extract 16 two-bit values from qs
;   4. Dequant via FMA, dot with Q, accumulate
;
; AVX-512: each sub-block is one vpmovzxbd zmm (16 values at once).
;
; Clobbers: zmm0-zmm5, eax, ecx
; Preserves: zmm11-zmm13, zmm30-zmm31, rdx, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Zero local accumulator
    vxorps    zmm5, zmm5, zmm5

    ; Process 16 sub-blocks of 16 values
    %assign _j 0
    %rep 16
        ; Compute qs location for sub-block _j
        ; qs_offset = 16 + (_j/8)*32 + (_j & 1)*16
        ; bit_shift = ((_j >> 1) & 3) * 2
        %assign _qs_off 16 + (_j / 8) * 32 + (_j & 1) * 16
        %assign _shift ((_j >> 1) & 3) * 2

        ; Extract scale/min and compute eff_scale(zmm3), eff_min(zmm4)
        EXTRACT_Q2K_SCALE_MIN %1, _j

        ; Load 16 packed bytes, shift and mask to get 16 two-bit values
        vmovdqu   xmm2, [%1 + _qs_off]
%if _shift > 0
        vpsrlw    xmm2, xmm2, _shift
%endif
        vpand     xmm2, xmm2, xmm11         ; 16 bytes, each 0-3

        ; Zero-extend 16 bytes -> 16 int32, convert to f32
        vpmovzxbd zmm1, xmm2                ; all 16 values at once
        vcvtdq2ps zmm1, zmm1                ; -> 16 f32

        ; Dequant: value = eff_scale * q + eff_min
        vfmadd132ps zmm1, zmm4, zmm3

        ; Dot with Q and accumulate
        vfmadd231ps zmm5, zmm1, [%2 + _j * 64]

        %assign _j _j + 1
    %endrep

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, zmm5
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q2_k_block)
;
; %1 = v_block_ptr (register, points to Q2_K block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (immediate, block index for offset — from skeleton's %rep)
;
; Dequant each value, multiply by prob, accumulate into O.
; O offset = (block_idx * 256 + sub_block * 16) * 4 bytes.
;
; Processes 16 sub-blocks of 16 values each.
;
; Clobbers: zmm0-zmm5, eax, ecx, rdx
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; Compute O base for this block: rdx = o_base + block_idx * 1024
    mov       rdx, %4
    shl       rdx, 10                        ; * 1024 (= 256 values * 4 bytes)
    add       rdx, %3                        ; rdx = O base for this block
    ; Process 16 sub-blocks
    %assign _j 0
    %rep 16
        %assign _qs_off 16 + (_j / 8) * 32 + (_j & 1) * 16
        %assign _shift ((_j >> 1) & 3) * 2
        EXTRACT_Q2K_SCALE_MIN %1, _j
        vmovdqu   xmm2, [%1 + _qs_off]
%if _shift > 0
        vpsrlw    xmm2, xmm2, _shift
%endif
        vpand     xmm2, xmm2, xmm11
        vpmovzxbd zmm1, xmm2
        vcvtdq2ps zmm1, zmm1
        vfmadd132ps zmm1, zmm4, zmm3
        vmulps    zmm1, zmm1, %2
        vaddps    zmm1, zmm1, [rdx + _j * 64]
        vmovaps   [rdx + _j * 64], zmm1
        %assign _j _j + 1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
