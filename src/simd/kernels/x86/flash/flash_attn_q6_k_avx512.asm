; ============================================================================
; Flash Attention 2 - Q6_K AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q6_K format (210 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
;
; Q6_K block layout (210 bytes, 256 values):
;   offset 0:   ql[128]    — lower 4 bits of 6-bit values (nibble-packed)
;   offset 128: qh[64]     — upper 2 bits of 6-bit values (2-bit packed)
;   offset 192: scales[16] — per-sub-block scales (SIGNED int8)
;   offset 208: d          — super-block scale (fp16, 2 bytes)
;
; 6-bit reconstruction:
;   val_6bit = ql_nibble | (qh_2bits << 4)       ; range [0, 63]
;   val_signed = val_6bit - 32                    ; range [-32, 31]
;   dequantized = d * scales[sub_block] * val_signed
;
; 16 sub-blocks of 16 values each = 256 total.
; Processed as 8 chunks of 32 values (2 sub-blocks per chunk).
;
; AVX-512 processes 2 groups of 16 per chunk instead of AVX2's 4 groups of 8.
; Each group of 16 aligns with exactly one sub-block and one scale.
;
; Memory layout for 32-value chunk j (0..7):
;   ql_offset = j * 16      (16 bytes = 32 nibbles)
;   qh_offset = 128 + j * 8 (8 bytes = 32 x 2-bit)
;   scale_idx = j * 2       (2 signed int8 scales)
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q6_k_avx512
%define K_BLOCK_BYTES       210
%define V_BLOCK_BYTES       210
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q6k_nibble_mask: times 16 db 0x0F       ; 16 bytes for xmm ops
                     times 48 db 0           ; pad to 64 bytes

fa2_q6k_bias_32:     times 16 db 32          ; 16 bytes for xmm ops
                     times 48 db 0           ; pad to 64 bytes

; Shuffle mask to replicate each of 4 bytes 4 times: byte[i] = i/4
align 16
fa2_q6k_qh_shuf:   db 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3

; Masks for extracting 2-bit fields from replicated qh bytes.
; After vpshufb replication, each group of 4 bytes holds the same qh byte.
; Position 0,4,8,12 -> bits [1:0] (shift 0)
; Position 1,5,9,13 -> bits [3:2] (shift 2)
; Position 2,6,10,14 -> bits [5:4] (shift 4)
; Position 3,7,11,15 -> bits [7:6] (shift 6)
;
; Mask the target bits AT their original position, then word-shift right.
; The word-shift is safe: masking first ensures no cross-byte contamination.
align 16
fa2_q6k_qh_mask_s0: db 0x03,0x00,0x00,0x00, 0x03,0x00,0x00,0x00
                     db 0x03,0x00,0x00,0x00, 0x03,0x00,0x00,0x00
fa2_q6k_qh_mask_s2: db 0x00,0x0C,0x00,0x00, 0x00,0x0C,0x00,0x00
                     db 0x00,0x0C,0x00,0x00, 0x00,0x0C,0x00,0x00
fa2_q6k_qh_mask_s4: db 0x00,0x00,0x30,0x00, 0x00,0x00,0x30,0x00
                     db 0x00,0x00,0x30,0x00, 0x00,0x00,0x30,0x00
fa2_q6k_qh_mask_s6: db 0x00,0x00,0x00,0xC0, 0x00,0x00,0x00,0xC0
                     db 0x00,0x00,0x00,0xC0, 0x00,0x00,0x00,0xC0
%endmacro

; ============================================================================
; Initialize quant constant registers
; xmm11 = nibble_mask (16x 0x0F, low 128 bits of zmm11)
; xmm12 = bias_32     (16x 32, low 128 bits of zmm12)
; zmm13 = unused
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q6k_nibble_mask]
    vmovdqu   xmm12, [rel fa2_q6k_bias_32]
%endmacro

; ============================================================================
; QH_EXTRACT_16: Extract 16 x 2-bit values from 4 packed qh bytes
;
; Inputs:
;   %1 = source xmm (low 4 bytes contain packed qh data, loaded via vmovd)
;   %2 = destination xmm (16 bytes, each with a 2-bit value 0..3)
;
; Scratch: xmm1, xmm2, xmm3 (caller must not need these)
;
; Algorithm:
;   1. Replicate each byte 4 times via vpshufb
;   2. Mask bits at each 2-bit field position
;   3. Word-shift right to align each field to bits [1:0]
;   4. OR all results together
; ============================================================================
%macro QH_EXTRACT_16 2
    vpshufb %2, %1, [rel fa2_q6k_qh_shuf]

    ; Shift-0: bits [1:0] at positions 0,4,8,12
    vpand   xmm3, %2, [rel fa2_q6k_qh_mask_s0]

    ; Shift-2: bits [3:2] at positions 1,5,9,13
    vpand   xmm1, %2, [rel fa2_q6k_qh_mask_s2]
    vpsrlw  xmm1, xmm1, 2
    vpor    xmm3, xmm3, xmm1

    ; Shift-4: bits [5:4] at positions 2,6,10,14
    vpand   xmm1, %2, [rel fa2_q6k_qh_mask_s4]
    vpsrlw  xmm1, xmm1, 4
    vpor    xmm3, xmm3, xmm1

    ; Shift-6: bits [7:6] at positions 3,7,11,15
    vpand   xmm1, %2, [rel fa2_q6k_qh_mask_s6]
    vpsrlw  xmm1, xmm1, 6
    vpor    xmm3, xmm3, xmm1

    vmovdqa %2, xmm3
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q6_k_block))
;
; %1 = k_block_ptr (register, points to Q6_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; Processes 8 chunks of 32 values each. Each chunk has 2 sub-blocks
; of 16 values, each processed as one zmm group (16 f32 values).
;
; AVX-512 advantage: vpmovsxbd zmm, xmm sign-extends 16 bytes -> 16 dwords
; at once, so each sub-block is one group instead of two.
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Load fp16 super-block scale d -> xmm0[0]
    movzx     eax, word [%1 + 208]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                        ; xmm0[0] = d (f32)

    ; --- Process 8 chunks of 32 values ---
    %assign %%chunk 0
    %rep 8

    ; =================================================================
    ; STEP 1: Extract 32 low-nibble values from ql
    ; =================================================================
    vmovdqu   xmm2, [%1 + %%chunk * 16]         ; 16 packed bytes
    vpand     xmm4, xmm2, xmm11                 ; low nibbles (even values)
    vpsrlw    xmm3, xmm2, 4
    vpand     xmm3, xmm3, xmm11                 ; high nibbles (odd values)

    ; Interleave low and high nibbles into value order
    ; Low 8 bytes -> 16 values (sub-block 0)
    vpunpcklbw xmm2, xmm4, xmm3                 ; values 0-15
    ; High 8 bytes -> 16 values (sub-block 1)
    vpunpckhbw xmm4, xmm4, xmm3                 ; values 16-31
    ; xmm2 = sub-block 0 ql (16 bytes), xmm4 = sub-block 1 ql (16 bytes)

    ; =================================================================
    ; STEP 2: Extract 32 x 2-bit values from qh
    ; =================================================================
    ; Sub-block 0: 4 qh bytes -> 16 values
    vmovd     xmm3, [%1 + 128 + %%chunk * 8]
    QH_EXTRACT_16 xmm3, xmm5
    ; xmm5 = sub-block 0 qh (16 bytes)

    ; Sub-block 1: 4 qh bytes -> 16 values
    vmovd     xmm3, [%1 + 128 + %%chunk * 8 + 4]
    QH_EXTRACT_16 xmm3, xmm3
    ; xmm3 = sub-block 1 qh (16 bytes)

    ; =================================================================
    ; STEP 3: Combine ql and qh, subtract bias — per sub-block
    ; =================================================================

    ; --- Sub-block 0 (values 0-15) ---
    vpsllw    xmm5, xmm5, 4                     ; qh << 4 (safe: max 3 -> 0x30)
    vpor      xmm2, xmm2, xmm5                  ; val_6bit = ql | (qh << 4)
    vpsubb    xmm2, xmm2, xmm12                 ; val_signed = val_6bit - 32

    ; --- Sub-block 1 (values 16-31) ---
    vpsllw    xmm3, xmm3, 4
    vpor      xmm4, xmm4, xmm3
    vpsubb    xmm4, xmm4, xmm12

    ; =================================================================
    ; STEP 4: Load scales, dequant to f32, dot with Q
    ; =================================================================

    ; Load 2 scales at once via vpmovsxbd (no GPR clobber)
    vpmovsxbd xmm5, dword [%1 + 192 + %%chunk * 2]
    ; xmm5 = [scale0_i32, scale1_i32, ...]
    vcvtdq2ps xmm5, xmm5                        ; [scale0_f32, scale1_f32, ...]

    ; --- Group 0: sub-block 0, 16 values -> zmm ---
    ; d * scale0 broadcast
    vbroadcastss zmm3, xmm0                      ; zmm3 = d broadcast
    vbroadcastss zmm1, xmm5                      ; zmm1 = scale0 broadcast
    vmulps    zmm3, zmm3, zmm1                   ; zmm3 = d * scale0

    vpmovsxbd zmm1, xmm2                         ; sign-extend 16 bytes -> 16 i32
    vcvtdq2ps zmm1, zmm1                         ; -> 16 f32
    vmovups   zmm2, [%2 + %%chunk * 128]          ; Q[0:15]
    vmulps    zmm1, zmm1, zmm2                   ; dequant * Q
    vfmadd231ps %3, zmm1, zmm3                   ; acc += (dequant*Q) * (d*scale0)

    ; --- Group 1: sub-block 1, 16 values -> zmm ---
    ; d * scale1 broadcast
    vpshufd   xmm5, xmm5, 0x55                  ; broadcast scale1 element
    vbroadcastss zmm3, xmm0                      ; zmm3 = d broadcast
    vbroadcastss zmm1, xmm5                      ; zmm1 = scale1 broadcast
    vmulps    zmm3, zmm3, zmm1                   ; zmm3 = d * scale1

    vpmovsxbd zmm1, xmm4                         ; sign-extend 16 bytes -> 16 i32
    vcvtdq2ps zmm1, zmm1                         ; -> 16 f32
    vmovups   zmm2, [%2 + %%chunk * 128 + 64]    ; Q[16:31]
    vmulps    zmm1, zmm1, zmm2                   ; dequant * Q
    vfmadd231ps %3, zmm1, zmm3                   ; acc += (dequant*Q) * (d*scale1)

    %assign %%chunk %%chunk + 1
    %endrep
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q6_k_block)
;
; %1 = v_block_ptr (register, points to Q6_K block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then: O[i] += prob * dequant(V[i])
; O offset = block_idx * 256 * 4 = block_idx << 10 (1024 bytes per block)
;
; Clobbers: zmm0-zmm5, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load fp16 super-block scale d
    movzx     eax, word [%1 + 208]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                        ; xmm0[0] = d (f32)

    ; 2. Compute O base offset = block_idx * 256 * 4 = block_idx * 1024
    mov       rax, %4
    shl       rax, 10                            ; rax = block_idx * 1024

    ; --- Process 8 chunks of 32 values ---
    %assign %%chunk 0
    %rep 8

    ; =================================================================
    ; STEP 1: Extract 32 low-nibble values from ql
    ; =================================================================
    vmovdqu   xmm3, [%1 + %%chunk * 16]         ; 16 packed bytes
    vpand     xmm4, xmm3, xmm11                 ; low nibbles
    vpsrlw    xmm5, xmm3, 4
    vpand     xmm5, xmm5, xmm11                 ; high nibbles
    vpunpcklbw xmm3, xmm4, xmm5                 ; values 0-15
    vpunpckhbw xmm4, xmm4, xmm5                 ; values 16-31
    ; xmm3 = sub-block 0 ql, xmm4 = sub-block 1 ql

    ; =================================================================
    ; STEP 2: Extract 32 x 2-bit values from qh
    ; =================================================================
    vmovd     xmm5, [%1 + 128 + %%chunk * 8]
    QH_EXTRACT_16 xmm5, xmm5
    ; xmm5 = sub-block 0 qh

    ; Save sub-block 0 combined before processing sub-block 1 qh
    vpsllw    xmm5, xmm5, 4
    vpor      xmm3, xmm3, xmm5                  ; sub-block 0: ql | (qh << 4)
    vpsubb    xmm3, xmm3, xmm12                 ; sub-block 0: val_signed

    vmovd     xmm5, [%1 + 128 + %%chunk * 8 + 4]
    QH_EXTRACT_16 xmm5, xmm5
    ; xmm5 = sub-block 1 qh

    vpsllw    xmm5, xmm5, 4
    vpor      xmm4, xmm4, xmm5                  ; sub-block 1: ql | (qh << 4)
    vpsubb    xmm4, xmm4, xmm12                 ; sub-block 1: val_signed

    ; =================================================================
    ; STEP 3: Load scales, dequant to f32, accumulate into O
    ; =================================================================

    ; Load 2 scales at once (sign-extended i8 -> i32 -> f32)
    vpmovsxbd xmm5, dword [%1 + 192 + %%chunk * 2]
    vcvtdq2ps xmm5, xmm5                        ; [scale0_f32, scale1_f32, ...]

    ; --- Group 0: sub-block 0 (values 0-15) -> O[0:15] ---
    ; Save scale1 by shuffling into a temp before broadcasting scale0
    vpshufd   xmm1, xmm5, 0x55                  ; xmm1 = scale1 broadcast in xmm

    ; Now safe to clobber xmm5
    vbroadcastss zmm5, xmm5                      ; zmm5 = scale0 f32 broadcast
    vbroadcastss zmm2, xmm0                      ; zmm2 = d broadcast
    vmulps    zmm5, zmm5, zmm2                   ; zmm5 = d * scale0

    vpmovsxbd zmm2, xmm3                         ; sign-extend sub-block 0 -> 16 i32
    vcvtdq2ps zmm2, zmm2                         ; -> 16 f32
    vmulps    zmm2, zmm2, zmm5                   ; dequant_f32 = val * d*scale0
    vmovaps   zmm3, [%3 + rax + %%chunk * 128]   ; O[0:15]
    vfmadd231ps zmm3, %2, zmm2                   ; O += prob * dequant
    vmovaps   [%3 + rax + %%chunk * 128], zmm3

    ; --- Group 1: sub-block 1 (values 16-31) -> O[16:31] ---
    vbroadcastss zmm5, xmm1                      ; zmm5 = scale1 f32 broadcast
    vbroadcastss zmm2, xmm0                      ; zmm2 = d broadcast
    vmulps    zmm5, zmm5, zmm2                   ; zmm5 = d * scale1

    vpmovsxbd zmm2, xmm4                         ; sign-extend sub-block 1 -> 16 i32
    vcvtdq2ps zmm2, zmm2                         ; -> 16 f32
    vmulps    zmm2, zmm2, zmm5                   ; dequant_f32 = val * d*scale1
    vmovaps   zmm3, [%3 + rax + %%chunk * 128 + 64]
    vfmadd231ps zmm3, %2, zmm2                   ; O += prob * dequant
    vmovaps   [%3 + rax + %%chunk * 128 + 64], zmm3

    %assign %%chunk %%chunk + 1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
