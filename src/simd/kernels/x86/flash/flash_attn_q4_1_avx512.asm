; ============================================================================
; Flash Attention 2 - Q4_1 AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q4_1 format (20 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q4_1 block layout (20 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: fp16 min   'm' (2 bytes)
;   offset 4: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;   Low nibble = q[2i], high nibble = q[2i+1], unsigned [0,15]
;   Dequantization: value = d * nibble + m
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_1_avx512
%define K_BLOCK_BYTES       20
%define V_BLOCK_BYTES       20
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q4_nibble_mask: times 16 db 0x0F    ; 16 bytes for xmm ops
                    times 48 db 0       ; pad to 64 bytes
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = nibble_mask (only xmm11 used — low 16 bytes)
; zmm12 = unused
; zmm13 = unused (no bias needed for Q4_1)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q4_nibble_mask]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q4_1_block))
;
; %1 = k_block_ptr (register, points to Q4_1 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; AVX-512 dequantization + dot product:
;   1. Load fp16 scale 'd', convert to f32, broadcast to zmm
;   2. Load fp16 min 'm', convert to f32, broadcast to zmm
;   3. Extract low/high nibbles from packed bytes at offset 4
;   4. Zero-extend 16 bytes -> 16 uint32 via vpmovzxbd zmm, xmm
;   5. Two groups of 16: dequant = d * nibble + m, dot with Q
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale 'd' -> broadcast zmm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0               ; xmm0[0] = d (f32)
    vbroadcastss zmm5, xmm0            ; zmm5 = broadcast d

    ; 2. Load fp16 min 'm' -> broadcast zmm3
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0               ; xmm0[0] = m (f32)
    vbroadcastss zmm3, xmm0            ; zmm3 = broadcast m

    ; 3. Load 16 packed bytes (32 nibbles) at offset 4
    vmovdqu   xmm1, [%1+4]

    ; 4. Extract low nibbles (values 0-15) — unsigned [0,15]
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F

    ; 5. Extract high nibbles (values 16-31) — unsigned [0,15]
    vpsrlw    xmm4, xmm1, 4
    vpand     xmm4, xmm4, xmm11        ; high nibbles >> 4 & 0x0F

    ; 6. Group 0: low nibbles (16 values 0-15)
    vpmovzxbd zmm1, xmm2               ; zero-extend 16 bytes -> 16 uint32
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vfmadd132ps zmm1, zmm3, zmm5       ; zmm1 = d * nibble + m
    vmovups   zmm2, [%2]               ; Q[0:15]
    vfmadd231ps %3, zmm1, zmm2         ; acc += dequant * Q

    ; 7. Group 1: high nibbles (16 values 16-31)
    vpmovzxbd zmm1, xmm4               ; zero-extend 16 bytes -> 16 uint32
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vfmadd132ps zmm1, zmm3, zmm5       ; zmm1 = d * nibble + m
    vmovups   zmm2, [%2+64]            ; Q[16:31]
    vfmadd231ps %3, zmm1, zmm2         ; acc += dequant * Q
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_1_block)
;
; %1 = v_block_ptr (register, points to Q4_1 block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K (value = d * nibble + m), then multiply by prob
; and add to O.
; O offset = block_idx * 32 * 4 = block_idx << 7 (128 bytes per block)
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 scale 'd' -> broadcast zmm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm5, xmm0            ; zmm5 = broadcast d

    ; 2. Load fp16 min 'm' -> broadcast zmm3
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm3, xmm0            ; zmm3 = broadcast m

    ; 3. Load + extract nibbles (same as K)
    vmovdqu   xmm1, [%1+4]
    vpand     xmm2, xmm1, xmm11        ; low nibbles
    vpsrlw    xmm4, xmm1, 4
    vpand     xmm4, xmm4, xmm11        ; high nibbles

    ; 4. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                    ; * 128

    ; 5. Group 0: low nibbles (values 0-15) -> O[0:15]
    vpmovzxbd zmm1, xmm2
    vcvtdq2ps zmm1, zmm1
    vfmadd132ps zmm1, zmm3, zmm5       ; zmm1 = d * nibble + m
    vmovaps   zmm0, [%3+rax]            ; O[0:15]
    vfmadd231ps zmm0, %2, zmm1          ; O += prob * V_dequant
    vmovaps   [%3+rax], zmm0

    ; 6. Group 1: high nibbles (values 16-31) -> O[16:31]
    vpmovzxbd zmm1, xmm4
    vcvtdq2ps zmm1, zmm1
    vfmadd132ps zmm1, zmm3, zmm5       ; zmm1 = d * nibble + m
    vmovaps   zmm0, [%3+rax+64]         ; O[16:31]
    vfmadd231ps zmm0, %2, zmm1          ; O += prob * V_dequant
    vmovaps   [%3+rax+64], zmm0
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
