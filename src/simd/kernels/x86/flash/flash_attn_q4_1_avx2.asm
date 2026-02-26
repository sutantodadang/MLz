; ============================================================================
; Flash Attention 2 - Q4_1 AVX2 Kernel
; ============================================================================
; Quantized K/V with Q4_1 format (20 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q4_1 block layout (20 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: fp16 min   'm' (2 bytes)
;   offset 4: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;   Low nibble = q[2i], high nibble = q[2i+1], unsigned [0,15]
;   Dequantization: value = d * nibble + m
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_1_avx2
%define K_BLOCK_BYTES       20
%define V_BLOCK_BYTES       20
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q4_nibble_mask: times 32 db 0x0F
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; ymm12 = unused (reserved)
; ymm13 = unused (reserved — no bias needed for Q4_1)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q4_nibble_mask]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q4_1_block))
;
; %1 = k_block_ptr (register, points to Q4_1 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product:
;   1. Load fp16 scale 'd', convert to f32, broadcast
;   2. Load fp16 min 'm', convert to f32, broadcast
;   3. Extract low/high nibbles from packed bytes at offset 4
;   4. Interleave to get 32 unsigned bytes
;   5. Zero-extend to f32 in 4 groups of 8 (vpmovzxbd)
;   6. Compute: dequant = d * nibble + m
;   7. Dot product with Q values, accumulate into acc
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm13, ymm14-ymm15
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale 'd' -> broadcast ymm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0               ; xmm0[0] = d (f32)
    vbroadcastss ymm5, xmm0            ; ymm5 = broadcast d

    ; 2. Load fp16 min 'm' -> broadcast ymm3
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0               ; xmm0[0] = m (f32)
    vbroadcastss ymm3, xmm0            ; ymm3 = broadcast m

    ; 3. Load 16 packed bytes (32 nibbles) at offset 4
    vmovdqu   xmm1, [%1+4]

    ; 4. Extract nibbles
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F
    vpsrlw    xmm0, xmm1, 4
    vpand     xmm0, xmm0, xmm11        ; high nibbles >> 4 & 0x0F

    ; 5. Interleave: low[i], high[i] -> 32 bytes in ymm
    vpunpcklbw xmm4, xmm2, xmm0        ; interleave low halves
    vpunpckhbw xmm1, xmm2, xmm0        ; interleave high halves
    vinserti128 ymm4, ymm4, xmm1, 1    ; ymm4 = 32 unsigned bytes [0,15]

    ; 6. Convert u8 -> f32 and dot with Q, accumulating into %3
    ; Formula per element: acc += Q[i] * (d * nibble + m)
    ;   = Q[i] * d * nibble + Q[i] * m
    ; We compute: dequant = d * nibble_f32 + m, then FMA with Q

    ; Group 0: bytes 0-7
    vpmovzxbd ymm1, xmm4               ; zero-extend 8 bytes -> 8 uint32
    vcvtdq2ps ymm1, ymm1               ; -> 8 f32 nibble values
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovups   ymm2, [%2]               ; Q[0:7]
    vfmadd231ps %3, ymm1, ymm2         ; acc += dequant * Q

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0         ; xmm1 = low 16 bytes
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend -> 8 uint32
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovups   ymm2, [%2+32]            ; Q[8:15]
    vfmadd231ps %3, ymm1, ymm2         ; acc += dequant * Q

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1         ; xmm1 = high 16 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend first 8
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovups   ymm2, [%2+64]            ; Q[16:23]
    vfmadd231ps %3, ymm1, ymm2         ; acc += dequant * Q

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1         ; xmm1 = high 16 bytes
    vpsrldq   xmm1, xmm1, 8            ; shift right 8 bytes
    vpmovzxbd ymm1, xmm1               ; zero-extend -> 8 uint32
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovups   ymm2, [%2+96]            ; Q[24:31]
    vfmadd231ps %3, ymm1, ymm2         ; acc += dequant * Q
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_1_block)
;
; %1 = v_block_ptr (register, points to Q4_1 block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K (value = d * nibble + m), then multiply by prob
; and add to O.
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm13, ymm14-ymm15
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale 'd' -> broadcast ymm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm5, xmm0            ; ymm5 = broadcast d

    ; 2. Load fp16 min 'm' -> broadcast ymm3
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm3, xmm0            ; ymm3 = broadcast m

    ; 3. Load + extract nibbles (same as K)
    vmovdqu   xmm1, [%1+4]
    vpand     xmm2, xmm1, xmm11        ; low nibbles
    vpsrlw    xmm0, xmm1, 4
    vpand     xmm0, xmm0, xmm11        ; high nibbles
    vpunpcklbw xmm4, xmm2, xmm0
    vpunpckhbw xmm1, xmm2, xmm0        ; reuse xmm1
    vinserti128 ymm4, ymm4, xmm1, 1    ; ymm4 = 32 unsigned bytes

    ; 4. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                    ; * 128

    ; 5. Dequant + prob * V + accumulate into O
    ; Group 0: bytes 0-7
    vpmovzxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovaps   ymm0, [%3+rax]            ; O[0:7]
    vfmadd231ps ymm0, %2, ymm1          ; O += prob * V_dequant
    vmovaps   [%3+rax], ymm0

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovaps   ymm0, [%3+rax+32]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+32], ymm0

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovaps   ymm0, [%3+rax+64]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+64], ymm0

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, ymm3, ymm5       ; ymm1 = d * nibble + m
    vmovaps   ymm0, [%3+rax+96]
    vfmadd231ps ymm0, %2, ymm1
    vmovaps   [%3+rax+96], ymm0
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
