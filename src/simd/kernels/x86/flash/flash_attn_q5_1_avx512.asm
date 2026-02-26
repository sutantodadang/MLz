; ============================================================================
; Flash Attention 2 - Q5_1 AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q5_1 format (24 bytes/block, 32 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q5_1 block layout (24 bytes):
;   offset 0: fp16 scale 'd' (2 bytes)
;   offset 2: fp16 min   'm' (2 bytes)
;   offset 4: uint32 qh      (4 bytes) — 5th bit of each quant
;   offset 8: uint8[16] packed nibbles (2 nibbles per byte = 32 values)
;
; Dequantization: value = d * (nibble | (qh_bit << 4)) + m
;   5-bit unsigned [0,31], no bias subtraction — uses min offset m.
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q5_1_avx512
%define K_BLOCK_BYTES       24
%define V_BLOCK_BYTES       24
%define K_BLOCK_VALUES      32
%define V_BLOCK_VALUES      32

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q5_1_nibble_mask:   times 16 db 0x0F       ; 16 bytes for xmm ops
                        times 48 db 0           ; pad to 64 bytes
fa2_q5_1_shift_counts:  dd 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
fa2_q5_1_dword_ones:    times 16 dd 1
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = nibble_mask (only xmm11 used — low 16 bytes for nibble extraction)
; zmm12 = shift_counts [0..15] for variable-shift qh bit extraction (16 dwords)
; zmm13 = dword_ones   (16x 1) for masking after shift
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q5_1_nibble_mask]
    vmovdqu32 zmm12, [rel fa2_q5_1_shift_counts]
    vmovdqu32 zmm13, [rel fa2_q5_1_dword_ones]
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:31], dequant(K_q5_1_block))
;
; %1 = k_block_ptr (register, points to Q5_1 block start)
; %2 = q_ptr (register, points to 32 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; AVX-512 dequantization + dot product:
;   1. Load fp16 scale d, convert to f32, broadcast to zmm
;   2. Load fp16 min m, convert to f32, broadcast to zmm
;   3. Extract low/high nibbles from packed bytes
;   4. Extract 5th bits from qh, combine with nibbles
;   5. Zero-extend 16 bytes -> 16 int32 via vpmovzxbd zmm, xmm
;   6. Two groups of 16: dequant, dot with Q, accumulate
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load fp16 scale d -> broadcast zmm5
    movzx     eax, word [%1]
    vmovd     xmm5, eax
    vcvtph2ps xmm5, xmm5
    vbroadcastss zmm5, xmm5            ; zmm5 = broadcast d

    ; 2. Load fp16 min m -> broadcast zmm0
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm0, xmm0            ; zmm0 = broadcast m

    ; 3. Load 16 packed bytes (32 nibbles) from qs
    vmovdqu   xmm1, [%1+8]

    ; 4. Extract low nibbles (16 bytes -> values for group 0)
    vpand     xmm2, xmm1, xmm11        ; low nibbles & 0x0F

    ; 5. Extract high nibbles (16 bytes -> values for group 1)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles >> 4 & 0x0F

    ; 6. Load qh bits for 5th bit extraction
    ; (loaded per-group as word from qh field)
    ; --- Group 0: low nibbles (16 values) with lower 16 qh bits ---
    vpmovzxbd zmm1, xmm2               ; zero-extend 16 bytes -> 16 uint32
    ; Extract 5th bits: broadcast lower 16 bits of qh, variable-shift, mask
    movzx     eax, word [%1+4]          ; lower 16 bits of qh
    vmovd     xmm4, eax
    vpbroadcastd zmm4, xmm4
    vpsrlvd   zmm4, zmm4, zmm12        ; shift by [0..15]
    vpandd    zmm4, zmm4, zmm13        ; isolate bit 0 -> 0 or 1
    vpslld    zmm4, zmm4, 4            ; -> 0 or 16
    vpord     zmm1, zmm1, zmm4         ; combine: 5-bit unsigned value
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vfmadd213ps zmm1, zmm5, zmm0       ; zmm1 = d * val + m
    vmovups   zmm2, [%2]               ; Q[0:15]
    vfmadd231ps %3, zmm2, zmm1         ; acc += Q * dequant_val

    ; --- Group 1: high nibbles (16 values) with upper 16 qh bits ---
    vpmovzxbd zmm1, xmm3               ; zero-extend 16 bytes -> 16 uint32
    ; Extract 5th bits: broadcast upper 16 bits of qh
    movzx     eax, word [%1+6]          ; upper 16 bits of qh (bytes 6-7)
    vmovd     xmm4, eax
    vpbroadcastd zmm4, xmm4
    vpsrlvd   zmm4, zmm4, zmm12        ; shift by [0..15]
    vpandd    zmm4, zmm4, zmm13        ; isolate bit 0
    vpslld    zmm4, zmm4, 4            ; -> 0 or 16
    vpord     zmm1, zmm1, zmm4         ; combine: 5-bit unsigned value
    vcvtdq2ps zmm1, zmm1               ; -> 16 f32
    vfmadd213ps zmm1, zmm5, zmm0       ; zmm1 = d * val + m
    vmovups   zmm2, [%2+64]            ; Q[16:31]
    vfmadd231ps %3, zmm2, zmm1         ; acc += Q * dequant_val
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q5_1_block)
;
; %1 = v_block_ptr (register, points to Q5_1 block)
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
    ; 1. Load fp16 scale d -> broadcast zmm5
    movzx     eax, word [%1]
    vmovd     xmm5, eax
    vcvtph2ps xmm5, xmm5
    vbroadcastss zmm5, xmm5            ; zmm5 = broadcast d

    ; 2. Load fp16 min m -> broadcast zmm0
    movzx     eax, word [%1+2]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0
    vbroadcastss zmm0, xmm0            ; zmm0 = broadcast m

    ; 3. Load + extract nibbles (same as K)
    vmovdqu   xmm1, [%1+8]
    vpand     xmm2, xmm1, xmm11        ; low nibbles
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11        ; high nibbles

    ; 4. Compute O offset = block_idx * 128
    mov       rax, %4
    shl       rax, 7

    ; --- Group 0: low nibbles (values 0-15) with lower 16 qh bits ---
    vpmovzxbd zmm1, xmm2               ; zero-extend -> 16 uint32
    movzx     eax, word [%1+4]          ; lower 16 bits of qh
    vmovd     xmm4, eax
    vpbroadcastd zmm4, xmm4
    vpsrlvd   zmm4, zmm4, zmm12
    vpandd    zmm4, zmm4, zmm13
    vpslld    zmm4, zmm4, 4
    vpord     zmm1, zmm1, zmm4
    vcvtdq2ps zmm1, zmm1
    vfmadd213ps zmm1, zmm5, zmm0       ; d * val + m

    ; Recompute O offset (rax clobbered by movzx eax)
    mov       rax, %4
    shl       rax, 7

    vmovaps   zmm2, [%3+rax]           ; O[0:15]
    vfmadd231ps zmm2, %2, zmm1         ; O += prob * dequant_val
    vmovaps   [%3+rax], zmm2

    ; --- Group 1: high nibbles (values 16-31) with upper 16 qh bits ---
    vpmovzxbd zmm1, xmm3               ; zero-extend -> 16 uint32
    movzx     eax, word [%1+6]          ; upper 16 bits of qh
    vmovd     xmm4, eax
    vpbroadcastd zmm4, xmm4
    vpsrlvd   zmm4, zmm4, zmm12
    vpandd    zmm4, zmm4, zmm13
    vpslld    zmm4, zmm4, 4
    vpord     zmm1, zmm1, zmm4
    vcvtdq2ps zmm1, zmm1
    vfmadd213ps zmm1, zmm5, zmm0       ; d * val + m
    mov       rax, %4
    shl       rax, 7
    vmovaps   zmm2, [%3+rax+64]        ; O[16:31]
    vfmadd231ps zmm2, %2, zmm1         ; O += prob * dequant_val
    vmovaps   [%3+rax+64], zmm2
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
