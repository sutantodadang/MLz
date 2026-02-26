; ============================================================================
; Flash Attention 2 - IQ4_NL AVX2 Kernel
; ============================================================================
; Quantized K/V with IQ4_NL format (18 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; IQ4_NL block layout (18 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: uint8[16] packed nibbles (2 nibbles per byte = 32 indices)
;   Low nibble = index[2i], high nibble = index[2i+1]
;   Each 4-bit index selects from a 16-entry non-linear lookup table.
;
; Dequantization: value = d * kvalues_iq4nl[nibble]
; Key trick: vpshufb performs parallel byte lookup using low 4 bits as index.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_iq4_nl_avx2
%define K_BLOCK_BYTES       18
%define V_BLOCK_BYTES       18
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_iq4nl_nibble_mask: times 32 db 0x0F
fa2_iq4nl_lut:         db -127, -104, -83, -65, -49, -35, -22, -10
                       db    1,   13,  25,  38,  53,  69,  89, 113
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; xmm12 = IQ4_NL lookup table (16 signed int8 values)
; ymm13 = unused
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_iq4nl_nibble_mask]
    vmovdqu   xmm12, [rel fa2_iq4nl_lut]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_iq4_nl_block))
;
; %1 = k_block_ptr (register, points to IQ4_NL block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product:
;   1. Load fp16 scale, convert to f32
;   2. Extract low/high nibbles from packed bytes
;   3. vpshufb lookup: nibble index → signed int8 via kvalues_iq4nl table
;   4. Interleave to get 32 signed int8 values in sequential order
;   5. Sign-extend to int32, convert to f32, scale by d, dot with Q
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm15, rdi, rsi, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale -> xmm0
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0           ; xmm0[0] = scale (f32)

    ; 2. Load 16 packed bytes (32 nibbles)
    vmovdqu   xmm1, [%1+2]

    ; 3. Extract nibbles
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F (indices for even positions)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles >> 4 & 0x0F (indices for odd positions)

    ; 4. Lookup table via vpshufb: index → signed int8 value
    vpshufb   xmm2, xmm12, xmm2        ; low nibble indices → looked-up int8 values
    vpshufb   xmm3, xmm12, xmm3        ; high nibble indices → looked-up int8 values

    ; 5. Interleave: low[i], high[i] -> 32 bytes in ymm (sequential value order)
    vpunpcklbw xmm4, xmm2, xmm3        ; interleave low halves: values 0-15
    vpunpckhbw xmm5, xmm2, xmm3        ; interleave high halves: values 16-31
    vinserti128 ymm4, ymm4, xmm5, 1    ; ymm4 = 32 signed int8 values

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
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_iq4_nl_block)
;
; %1 = v_block_ptr (register, points to IQ4_NL block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm15, rdi, rsi, r12-r15, rbx
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

    ; 3. Lookup table via vpshufb
    vpshufb   xmm2, xmm12, xmm2        ; low nibble indices → int8 values
    vpshufb   xmm3, xmm12, xmm3        ; high nibble indices → int8 values

    ; 4. Interleave to get 32 signed int8 values
    vpunpcklbw xmm4, xmm2, xmm3
    vpunpckhbw xmm1, xmm2, xmm3        ; reuse xmm1
    vinserti128 ymm4, ymm4, xmm1, 1

    ; 5. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                    ; * 128

    ; 6. Dequant + prob * V + accumulate into O
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
