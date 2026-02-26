; ============================================================================
; Flash Attention 2 - Q4_0 AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q4_0 format (18 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q4_0 block layout (18 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;   Low nibble = q[2i], high nibble = q[2i+1], subtract 8 for signed
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_0_avx512
%define K_BLOCK_BYTES       18
%define V_BLOCK_BYTES       18
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q4_nibble_mask: times 16 db 0x0F    ; 16 bytes for xmm ops
                    times 48 db 0       ; pad to 64 bytes
fa2_q4_bias:        times 16 db 8       ; 16 bytes for xmm ops
                    times 48 db 0       ; pad to 64 bytes
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = nibble_mask (only xmm11 used — low 16 bytes)
; zmm13 = q4_bias     (only xmm13 used — low 16 bytes)
; zmm12 = unused
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q4_nibble_mask]
    vmovdqu   xmm13, [rel fa2_q4_bias]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q4_0_block))
;
; %1 = k_block_ptr (register, points to Q4_0 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; AVX-512 dequantization + dot product:
;   1. Load fp16 scale, convert to f32, broadcast to zmm
;   2. Extract low/high nibbles from packed bytes
;   3. Sign-extend 16 bytes -> 16 int32 via vpmovsxbd zmm, xmm
;   4. Two groups of 16: dot with Q and accumulate
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale -> xmm0, broadcast to zmm5
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0               ; xmm0[0] = scale (f32)
    vbroadcastss zmm5, xmm0            ; zmm5 = broadcast scale

    ; 2. Load 16 packed bytes (32 nibbles)
    vmovdqu   xmm1, [%1+2]

    ; 3. Extract low nibbles (values 0-15)
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F
    vpsubb    xmm2, xmm2, xmm13        ; subtract bias 8 for signed

    ; 4. Extract high nibbles (values 16-31)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles >> 4 & 0x0F
    vpsubb    xmm3, xmm3, xmm13        ; subtract bias 8 for signed

    ; 5. Group 0: low nibbles (16 values 0-15)
    vpmovsxbd zmm1, xmm2               ; sign-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vmovups   zmm2, [%2]               ; Q[0:15]
    vmulps    zmm1, zmm1, zmm2         ; Q * dequant
    vfmadd231ps %3, zmm1, zmm5         ; acc += (Q * dequant) * scale

    ; 6. Group 1: high nibbles (16 values 16-31)
    vpmovsxbd zmm1, xmm3               ; sign-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vmovups   zmm2, [%2+64]            ; Q[16:31]
    vmulps    zmm1, zmm1, zmm2         ; Q * dequant
    vfmadd231ps %3, zmm1, zmm5         ; acc += (Q * dequant) * scale
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_0_block)
;
; %1 = v_block_ptr (register, points to Q4_0 block)
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
    vbroadcastss zmm5, xmm0            ; zmm5 = broadcast scale

    ; 2. Load + extract nibbles (same as K)
    vmovdqu   xmm1, [%1+2]
    vpand     xmm2, xmm1, xmm11        ; low nibbles
    vpsubb    xmm2, xmm2, xmm13        ; subtract bias
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles
    vpsubb    xmm3, xmm3, xmm13        ; subtract bias

    ; 3. Compute O offset = block_idx * 32 * 4 = block_idx * 128
    mov       rax, %4
    shl       rax, 7                    ; * 128

    ; 4. Group 0: low nibbles (values 0-15) -> O[0:15]
    vpmovsxbd zmm1, xmm2
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5          ; dequant * scale
    vmovaps   zmm0, [%3+rax]            ; O[0:15]
    vfmadd231ps zmm0, %2, zmm1          ; O += prob * V_dequant
    vmovaps   [%3+rax], zmm0

    ; 5. Group 1: high nibbles (values 16-31) -> O[16:31]
    vpmovsxbd zmm1, xmm3
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, zmm5          ; dequant * scale
    vmovaps   zmm0, [%3+rax+64]         ; O[16:31]
    vfmadd231ps zmm0, %2, zmm1          ; O += prob * V_dequant
    vmovaps   [%3+rax+64], zmm0
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
