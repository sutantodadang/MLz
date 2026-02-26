; ============================================================================
; Flash Attention 2 - Q6_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q6_K format (210 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
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
; Memory layout for 32-value chunk j (0..7):
;   ql_offset = j * 16      (16 bytes = 32 nibbles)
;   qh_offset = 128 + j * 8 (8 bytes = 32 x 2-bit)
;   scale_idx = j * 2       (2 signed int8 scales)
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q6_k_avx2
%define K_BLOCK_BYTES       210
%define V_BLOCK_BYTES       210
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q6k_nibble_mask: times 32 db 0x0F

fa2_q6k_bias_32:     times 32 db 32

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
; Approach: mask the target bits AT their original position, then word-shift
; right. The word-shift is safe because masking first ensures no cross-byte
; contamination (max masked value fits in a byte after shift).
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
; ymm11 = nibble_mask (32x 0x0F)
; ymm12 = bias_32     (32x 32)
; ymm13 = (unused, preserved)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q6k_nibble_mask]
    vmovdqu   ymm12, [rel fa2_q6k_bias_32]
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
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Processes 8 chunks of 32 values each. Each chunk spans 2 sub-blocks
; of 16 values with independent signed int8 scales.
;
; Dequantization + dot product per chunk:
;   1. Extract 32 low-nibble values from ql (4 bits each)
;   2. Extract 32 high-bit values from qh (2 bits each)
;   3. Combine: val_6bit = ql_nibble | (qh_2bits << 4)
;   4. Subtract 32 to center -> signed bytes
;   5. For each group of 8 values:
;      - sign-extend i8 -> i32 -> f32
;      - multiply with Q vector values
;      - FMA with (d * scale) into accumulator
;
; Clobbers: ymm0-ymm5, rax, edx
; Preserves: ymm11-ymm13, ymm14-ymm15
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; Load fp16 super-block scale d -> broadcast into ymm0
    movzx     eax, word [%1 + 208]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                ; xmm0[0] = d (f32)
    ; ymm0 will hold d broadcast; we set it per-scale below

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
    vpunpcklbw xmm2, xmm4, xmm3                 ; interleave bytes 0-7 -> 16 values
    vpunpckhbw xmm4, xmm4, xmm3                 ; interleave bytes 8-15 -> 16 values
    vinserti128 ymm4, ymm2, xmm4, 1             ; ymm4 = 32 ql values in order

    ; =================================================================
    ; STEP 2: Extract 32 x 2-bit values from qh
    ; =================================================================
    ; Low half: 4 qh bytes -> 16 values (for chunk values 0-15)
    vmovd     xmm2, [%1 + 128 + %%chunk * 8]
    QH_EXTRACT_16 xmm2, xmm5
    ; xmm5 = low 16 qh values

    ; High half: 4 qh bytes -> 16 values (for chunk values 16-31)
    vmovd     xmm2, [%1 + 128 + %%chunk * 8 + 4]
    QH_EXTRACT_16 xmm2, xmm2
    ; xmm2 = high 16 qh values

    ; Combine into ymm5
    vinserti128 ymm5, ymm5, xmm2, 1             ; ymm5 = 32 qh values

    ; =================================================================
    ; STEP 3: Combine ql and qh, subtract bias
    ; =================================================================
    vpsllw    ymm5, ymm5, 4                      ; qh << 4 (safe: max val 3 -> 0x30)
    vpor      ymm4, ymm4, ymm5                   ; val_6bit = ql | (qh << 4)
    vpsubb    ymm4, ymm4, ymm12                  ; val_signed = val_6bit - 32

    ; =================================================================
    ; STEP 4: Load scales and compute d * scale for 2 sub-blocks
    ; =================================================================

    ; --- Sub-block 0 (values 0-15): scale at [192 + chunk*2] ---
    ; Use vpmovsxbd to load scale byte without clobbering rax
    vpmovsxbd xmm5, dword [%1 + 192 + %%chunk * 2]
    ; xmm5 = [scale0_i32, scale1_i32, ...]  (4 sign-extended dwords)
    vcvtdq2ps xmm5, xmm5                        ; [scale0_f32, scale1_f32, ...]

    ; Compute d * scale0 broadcast
    vbroadcastss ymm2, xmm0                      ; ymm2 = d broadcast
    vbroadcastss ymm3, xmm5                      ; ymm3 = scale0 broadcast
    vmulps    ymm3, ymm3, ymm2                   ; ymm3 = d * scale0

    ; --- Group 0: bytes 0-7 -> 8 f32 ---
    vpmovsxbd ymm1, xmm4                         ; sign-extend 8 bytes -> 8 i32
    vcvtdq2ps ymm1, ymm1                         ; -> 8 f32
    vmovups   ymm2, [%2 + %%chunk * 128]          ; Q[0:7]
    vmulps    ymm1, ymm1, ymm2                   ; dequant * Q
    vfmadd231ps %3, ymm1, ymm3                   ; acc += (dequant*Q) * (d*scale0)

    ; --- Group 1: bytes 8-15 -> 8 f32 ---
    vextracti128 xmm1, ymm4, 0                   ; low 16 bytes
    vpsrldq   xmm1, xmm1, 8                      ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2 + %%chunk * 128 + 32]    ; Q[8:15]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm3

    ; --- Sub-block 1 (values 16-31): scale at [192 + chunk*2 + 1] ---
    ; Extract scale1 from xmm5 (element 1)
    vpshufd   xmm5, xmm5, 0x55                   ; broadcast element 1
    vbroadcastss ymm3, xmm5                      ; ymm3 = scale1 broadcast
    vbroadcastss ymm2, xmm0                      ; ymm2 = d broadcast
    vmulps    ymm3, ymm3, ymm2                   ; ymm3 = d * scale1

    ; --- Group 2: bytes 16-23 -> 8 f32 ---
    vextracti128 xmm1, ymm4, 1                   ; high 16 bytes
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2 + %%chunk * 128 + 64]    ; Q[16:23]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm3

    ; --- Group 3: bytes 24-31 -> 8 f32 ---
    vextracti128 xmm1, ymm4, 1                   ; high 16 bytes
    vpsrldq   xmm1, xmm1, 8                      ; shift right 8 bytes
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmovups   ymm2, [%2 + %%chunk * 128 + 96]    ; Q[24:31]
    vmulps    ymm1, ymm1, ymm2
    vfmadd231ps %3, ymm1, ymm3

    %assign %%chunk %%chunk + 1
    %endrep
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q6_k_block)
;
; %1 = v_block_ptr (register, points to Q6_K block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then: O[i] += prob * dequant(V[i])
; O offset = block_idx * 256 * 4 = block_idx << 10 (1024 bytes per block)
;
; Clobbers: ymm0-ymm5, rax, edx
; Preserves: ymm11-ymm13, ymm14-ymm15
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
    vpunpcklbw xmm3, xmm4, xmm5                 ; interleave low -> 16 values
    vpunpckhbw xmm4, xmm4, xmm5                 ; interleave high -> 16 values
    vinserti128 ymm4, ymm3, xmm4, 1             ; ymm4 = 32 ql values

    ; =================================================================
    ; STEP 2: Extract 32 x 2-bit values from qh
    ; =================================================================
    ; Low half
    vmovd     xmm3, [%1 + 128 + %%chunk * 8]
    QH_EXTRACT_16 xmm3, xmm5
    ; xmm5 = low 16 qh values

    ; High half
    vmovd     xmm3, [%1 + 128 + %%chunk * 8 + 4]
    QH_EXTRACT_16 xmm3, xmm3
    ; xmm3 = high 16 qh values

    vinserti128 ymm5, ymm5, xmm3, 1             ; ymm5 = 32 qh values

    ; =================================================================
    ; STEP 3: Combine and subtract bias
    ; =================================================================
    vpsllw    ymm5, ymm5, 4                      ; qh << 4
    vpor      ymm4, ymm4, ymm5                   ; val_6bit
    vpsubb    ymm4, ymm4, ymm12                  ; val_signed

    ; =================================================================
    ; STEP 4: Load scales, dequant to f32, accumulate into O
    ; =================================================================

    ; Load 2 scales at once (sign-extended i8 -> i32 -> f32)
    vpmovsxbd xmm5, dword [%1 + 192 + %%chunk * 2]
    vcvtdq2ps xmm5, xmm5                        ; [scale0_f32, scale1_f32, ...]

    ; d * scale0 broadcast
    vbroadcastss ymm3, xmm0                      ; d broadcast
    vbroadcastss ymm1, xmm5                      ; scale0 broadcast
    vmulps    ymm3, ymm3, ymm1                   ; ymm3 = d * scale0

    ; --- Group 0: bytes 0-7 -> O[0:7] ---
    vpmovsxbd ymm1, xmm4
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm3                   ; dequant_f32 = val * d * scale0
    vmovaps   ymm5, [%3 + rax + %%chunk * 128]
    vfmadd231ps ymm5, %2, ymm1                   ; O += prob * dequant
    vmovaps   [%3 + rax + %%chunk * 128], ymm5

    ; --- Group 1: bytes 8-15 -> O[8:15] ---
    vextracti128 xmm1, ymm4, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm3
    vmovaps   ymm5, [%3 + rax + %%chunk * 128 + 32]
    vfmadd231ps ymm5, %2, ymm1
    vmovaps   [%3 + rax + %%chunk * 128 + 32], ymm5

    ; Reload scales for sub-block 1 (need xmm5 again)
    vpmovsxbd xmm5, dword [%1 + 192 + %%chunk * 2]
    vcvtdq2ps xmm5, xmm5
    vpshufd   xmm5, xmm5, 0x55                   ; broadcast scale1
    vbroadcastss ymm3, xmm0                      ; d broadcast
    vbroadcastss ymm5, xmm5
    vmulps    ymm3, ymm3, ymm5                   ; ymm3 = d * scale1

    ; --- Group 2: bytes 16-23 -> O[16:23] ---
    vextracti128 xmm1, ymm4, 1
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm3
    vmovaps   ymm5, [%3 + rax + %%chunk * 128 + 64]
    vfmadd231ps ymm5, %2, ymm1
    vmovaps   [%3 + rax + %%chunk * 128 + 64], ymm5

    ; --- Group 3: bytes 24-31 -> O[24:31] ---
    vextracti128 xmm1, ymm4, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovsxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, ymm3
    vmovaps   ymm5, [%3 + rax + %%chunk * 128 + 96]
    vfmadd231ps ymm5, %2, ymm1
    vmovaps   [%3 + rax + %%chunk * 128 + 96], ymm5

    %assign %%chunk %%chunk + 1
    %endrep
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
