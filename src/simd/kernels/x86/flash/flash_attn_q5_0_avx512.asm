; ============================================================================
; Flash Attention 2 - Q5_0 AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q5_0 format (22 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
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
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q5_0_avx512
%define K_BLOCK_BYTES       22
%define V_BLOCK_BYTES       22
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q5_nibble_mask: times 16 db 0x0F    ; 16 bytes for xmm ops
                    times 48 db 0       ; pad to 64 bytes
fa2_q5_bias:        times 16 db 16      ; 16 bytes for xmm ops
                    times 48 db 0       ; pad to 64 bytes
; Shuffle pattern: broadcast qh byte 0 to positions 0-7, byte 1 to 8-15
fa2_q5_qh_shuf_lo:  db 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
                    times 48 db 0       ; pad to 64 bytes
; Shuffle pattern: broadcast qh byte 2 to positions 0-7, byte 3 to 8-15
fa2_q5_qh_shuf_hi:  db 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
                    times 48 db 0       ; pad to 64 bytes
; Bit mask to isolate individual bits within each byte
fa2_q5_bit_mask:     db 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
                    times 48 db 0       ; pad to 64 bytes
; The 5th bit value to OR into nibbles (0x10 = 16)
fa2_q5_high_bit:    times 16 db 0x10
                    times 48 db 0       ; pad to 64 bytes
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = nibble_mask (only xmm11 used — low 16 bytes)
; zmm13 = q5_bias     (only xmm13 used — low 16 bytes)
; zmm12 = unused
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q5_nibble_mask]
    vmovdqu   xmm13, [rel fa2_q5_bias]
%endmacro

; ============================================================================
; Helper: Extract 5th bits from qh for 16 values (xmm-level)
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
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; AVX-512 dequantization + dot product:
;   1. Load fp16 scale, convert to f32, broadcast to zmm
;   2. Extract 5th bits from qh for both nibble groups
;   3. Extract low/high nibbles, OR in 5th bits
;   4. Sign via subtract 16
;   5. vpmovsxbd zmm for 16 values at a time (two groups)
;   6. Dot with Q and accumulate
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale -> xmm0, broadcast to zmm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                 ; xmm0[0] = scale (f32)
    vbroadcastss zmm5, xmm0              ; zmm5 = broadcast scale

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

    ; 6. Extract low nibbles (values 0-15), OR in 5th bits, subtract bias
    vpand     xmm0, xmm1, xmm11          ; low nibbles & 0x0F
    vpor      xmm0, xmm0, xmm2           ; OR in 5th bits
    vpsubb    xmm2, xmm0, xmm13          ; subtract bias 16 for signed

    ; 7. Extract high nibbles (values 16-31), OR in 5th bits, subtract bias
    vpsrlw    xmm0, xmm1, 4
    vpand     xmm0, xmm0, xmm11          ; high nibbles >> 4 & 0x0F
    vpor      xmm0, xmm0, xmm3           ; OR in 5th bits
    vpsubb    xmm3, xmm0, xmm13          ; subtract bias 16 for signed

    ; 8. Group 0: low nibbles (16 values 0-15)
    vpmovsxbd zmm1, xmm2                 ; sign-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1                 ; -> 16 f32
    vmovups   zmm2, [%2]                 ; Q[0:15]
    vmulps    zmm1, zmm1, zmm2           ; Q * dequant
    vfmadd231ps %3, zmm1, zmm5           ; acc += (Q * dequant) * scale

    ; 9. Group 1: high nibbles (16 values 16-31)
    vpmovsxbd zmm1, xmm3                 ; sign-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1                 ; -> 16 f32
    vmovups   zmm2, [%2+64]              ; Q[16:31]
    vmulps    zmm1, zmm1, zmm2           ; Q * dequant
    vfmadd231ps %3, zmm1, zmm5           ; acc += (Q * dequant) * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q5_0_block)
;
; %1 = v_block_ptr (register, points to Q5_0 block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
; O offset = block_idx * 32 * 4 = block_idx << 7 (128 bytes per block)
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale -> xmm0, broadcast to zmm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm5, xmm0              ; zmm5 = broadcast scale

    ; 2. Load qh and extract 5th bits (same as K)
    vmovd     xmm0, dword [%1+2]

    EXTRACT_QH_BITS xmm2, xmm0, fa2_q5_qh_shuf_lo, xmm3
    EXTRACT_QH_BITS xmm3, xmm0, fa2_q5_qh_shuf_hi, xmm4

    ; 3. Load nibbles, extract, OR in 5th bits, subtract bias
    vmovdqu   xmm1, [%1+6]

    vpand     xmm0, xmm1, xmm11          ; low nibbles
    vpor      xmm0, xmm0, xmm2           ; OR in 5th bits
    vpsubb    xmm2, xmm0, xmm13          ; subtract bias

    vpsrlw    xmm0, xmm1, 4
    vpand     xmm0, xmm0, xmm11          ; high nibbles
    vpor      xmm0, xmm0, xmm3           ; OR in 5th bits
    vpsubb    xmm3, xmm0, xmm13          ; subtract bias

    ; 4. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                      ; * 128

    ; 5. Group 0: low nibbles (values 0-15) -> O[0:15]
    vpmovsxbd zmm1, xmm2
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5            ; dequant * scale
    vmovaps   zmm0, [%3+rax]              ; O[0:15]
    vfmadd231ps zmm0, %2, zmm1            ; O += prob * V_dequant
    vmovaps   [%3+rax], zmm0

    ; 6. Group 1: high nibbles (values 16-31) -> O[16:31]
    vpmovsxbd zmm1, xmm3
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5            ; dequant * scale
    vmovaps   zmm0, [%3+rax+64]           ; O[16:31]
    vfmadd231ps zmm0, %2, zmm1            ; O += prob * V_dequant
    vmovaps   [%3+rax+64], zmm0
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
