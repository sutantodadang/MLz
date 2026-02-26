; ============================================================================
; Flash Attention 2 - Q5_0 AVX2 Kernel
; ============================================================================
; Quantized K/V with Q5_0 format (22 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q5_0 block layout (22 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: uint8[4] qh   (4 bytes) — 5th bit of each quant (32 bits)
;   offset 6: uint8[16] qs  (16 bytes) — lower 4 bits packed as nibbles
;
; Dequantization:
;   For byte j in qs (j=0..15):
;     value[j]    = d * (((qs[j] & 0x0F) | ((qh >> j)     & 1) << 4) - 16)
;     value[j+16] = d * (((qs[j] >> 4)   | ((qh >> (j+16)) & 1) << 4) - 16)
;   Each 5-bit unsigned value [0,31] is biased by -16 → signed [-16,15]
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q5_0_avx2
%define K_BLOCK_BYTES       22
%define V_BLOCK_BYTES       22
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q5_nibble_mask: times 32 db 0x0F
fa2_q5_bias:        times 32 db 16
; Shuffle pattern: broadcast qh byte 0 to positions 0-7, byte 1 to 8-15
fa2_q5_qh_shuf_lo:  db 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
; Shuffle pattern: broadcast qh byte 2 to positions 0-7, byte 3 to 8-15
fa2_q5_qh_shuf_hi:  db 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
; Bit mask to isolate individual bits within each byte
fa2_q5_bit_mask:     db 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
; The 5th bit value to OR into nibbles (0x10 = 16)
fa2_q5_high_bit:    times 16 db 0x10
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; ymm12 = unused (available)
; ymm13 = q5_bias    (32x 16)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q5_nibble_mask]
    vmovdqu   ymm13, [rel fa2_q5_bias]
%endmacro

; ============================================================================
; Helper: Extract 5th bits from qh for 16 values
;
; Given xmm_qh containing the 4 qh bytes, extract 5th bits for one group
; of 16 values and produce an xmm with 0x10 where bit is set, 0x00 otherwise.
;
; %1 = destination xmm (output: 16 bytes of 0x00 or 0x10)
; %2 = xmm_qh (source: 4 bytes of qh, will be read not modified)
; %3 = shuffle constant label (fa2_q5_qh_shuf_lo or fa2_q5_qh_shuf_hi)
; %4 = scratch xmm
;
; Clobbers: %1, %4
; ============================================================================
%macro EXTRACT_QH_BITS 4
    vpshufb   %1, %2, [rel %3]               ; broadcast qh bytes to positions
    vpand     %1, %1, [rel fa2_q5_bit_mask]   ; isolate individual bits
    vpxor     %4, %4, %4                      ; zero
    vpcmpeqb  %4, %1, %4                      ; 0xFF where bit=0, 0x00 where bit=1
    vpandn    %1, %4, [rel fa2_q5_high_bit]   ; 0x10 where bit=1, 0x00 where bit=0
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q5_0_block))
;
; %1 = k_block_ptr (register, points to Q5_0 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product:
;   1. Load fp16 scale, convert to f32
;   2. Load qh (4 bytes) and extract 5th bits for both nibble groups
;   3. Extract low/high nibbles, OR in 5th bits
;   4. Interleave to get 32 unsigned 5-bit bytes, subtract 16 for signed
;   5. Dot product with Q using FMA approach (4 groups of 8)
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm15, rdi, rsi, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale -> broadcast to ymm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                 ; xmm0[0] = scale (f32)
    vbroadcastss ymm5, xmm0              ; ymm5 = broadcast scale

    ; 2. Load qh (4 bytes) into xmm0
    vmovd     xmm0, dword [%1+2]         ; xmm0 = [qh0, qh1, qh2, qh3, 0...]

    ; 3. Extract 5th bits for low nibble group (values 0-15)
    ;    xmm2 = 0x10 where qh bit set for values 0-15, else 0x00
    EXTRACT_QH_BITS xmm2, xmm0, fa2_q5_qh_shuf_lo, xmm3

    ; 4. Extract 5th bits for high nibble group (values 16-31)
    ;    xmm3 = 0x10 where qh bit set for values 16-31, else 0x00
    EXTRACT_QH_BITS xmm3, xmm0, fa2_q5_qh_shuf_hi, xmm4

    ; 5. Load 16 packed nibble bytes
    vmovdqu   xmm1, [%1+6]

    ; 6. Extract nibbles and OR in 5th bits
    vpand     xmm0, xmm1, xmm11          ; low nibbles & 0x0F (values 0-15)
    vpor      xmm0, xmm0, xmm2           ; OR in 5th bits for values 0-15

    vpsrlw    xmm1, xmm1, 4
    vpand     xmm1, xmm1, xmm11          ; high nibbles >> 4 & 0x0F (values 16-31)
    vpor      xmm1, xmm1, xmm3           ; OR in 5th bits for values 16-31

    ; 7. Interleave to 32 bytes: [v0,v16, v1,v17, ..., v15,v31]
    vpunpcklbw xmm4, xmm0, xmm1          ; interleave low halves
    vpunpckhbw xmm2, xmm0, xmm1          ; interleave high halves
    vinserti128 ymm4, ymm4, xmm2, 1      ; ymm4 = 32 unsigned 5-bit bytes

    ; 8. Subtract bias (16) to get signed values [-16, 15]
    vpsubb    ymm4, ymm4, ymm13

    ; 9. Convert i8 -> f32 and dot with Q, accumulating into %3
    ; Group 0: bytes 0-7
    vpmovsxbd ymm1, xmm4                 ; sign-extend 8 bytes -> 8 int32
    vcvtdq2ps ymm1, ymm1                 ; -> 8 f32 dequantized values
    vmovups   ymm2, [%2]                 ; Q[0:7]
    vmulps    ymm1, ymm1, ymm2           ; Q * dequant
    vfmadd231ps %3, ymm1, ymm5           ; acc += (Q * dequant) * scale

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm4, 0           ; xmm1 = low 16 bytes
    vpsrldq   xmm1, xmm1, 8              ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1                 ; sign-extend -> 8 int32
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+32]              ; Q[8:15]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5           ; acc += (Q * dequant) * scale

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm4, 1           ; xmm1 = high 16 bytes
    vpmovsxbd ymm1, xmm1                 ; sign-extend first 8
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+64]              ; Q[16:23]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5           ; acc += (Q * dequant) * scale

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm4, 1           ; xmm1 = high 16 bytes
    vpsrldq   xmm1, xmm1, 8              ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1                 ; sign-extend -> 8 int32
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2+96]              ; Q[24:31]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm5           ; acc += (Q * dequant) * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q5_0_block)
;
; %1 = v_block_ptr (register, points to Q5_0 block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
;
; Clobbers: ymm0-ymm5, rax
; Preserves: ymm11-ymm15, rdi, rsi, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale -> broadcast to ymm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss ymm5, xmm0              ; ymm5 = broadcast scale

    ; 2. Load qh and extract 5th bits (same as K)
    vmovd     xmm0, dword [%1+2]

    EXTRACT_QH_BITS xmm2, xmm0, fa2_q5_qh_shuf_lo, xmm3
    EXTRACT_QH_BITS xmm3, xmm0, fa2_q5_qh_shuf_hi, xmm4

    ; 3. Load nibbles, extract, OR in 5th bits
    vmovdqu   xmm1, [%1+6]
    vpand     xmm0, xmm1, xmm11
    vpor      xmm0, xmm0, xmm2

    vpsrlw    xmm1, xmm1, 4
    vpand     xmm1, xmm1, xmm11
    vpor      xmm1, xmm1, xmm3

    ; 4. Interleave and subtract bias
    vpunpcklbw xmm4, xmm0, xmm1
    vpunpckhbw xmm2, xmm0, xmm1          ; reuse xmm2
    vinserti128 ymm4, ymm4, xmm2, 1
    vpsubb    ymm4, ymm4, ymm13

    ; 5. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                      ; * 128

    ; 6. Dequant + prob * V + accumulate into O
    ; Group 0: bytes 0-7
    vpmovsxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm5            ; dequant * scale
    vmovaps   ymm0, [%3+rax]              ; O[0:7]
    vfmadd231ps ymm0, %2, ymm1            ; O += prob * V_dequant
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
