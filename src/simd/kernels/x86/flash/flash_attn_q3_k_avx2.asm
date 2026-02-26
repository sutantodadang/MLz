; ============================================================================
; Flash Attention 2 - Q3_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q3_K format (110 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q3_K block layout (110 bytes):
;   offset  0: hmask  (uint8_t[32], 32 bytes) — high bit (bit 2) per value
;   offset 32: qs     (uint8_t[64], 64 bytes) — low 2 bits per value (4 per byte)
;   offset 96: scales (uint8_t[12], 12 bytes) — packed 6-bit scales for 16 sub-blocks
;   offset 108: d     (ggml_half, 2 bytes)    — super-block scale
;
; 16 sub-blocks of 16 values each.
;
; 3-bit value reconstruction per value i in sub-block j:
;   qs_2bits  = (qs[i/4] >> (2*(i%4))) & 3
;   hmask_bit = (hmask[i/8] >> (i%8)) & 1
;   raw       = qs_2bits | (hmask_bit << 2)          ; range 0-7
;   signed    = raw - 4                               ; range -4 to +3
;   dequant   = d * scale * signed
;
; Scale extraction from scales[12] for sub-block j (0-15):
;   if j < 8: low4 = (scales[j/2] >> (4*(j%2))) & 0xF
;   else:     low4 = (scales[4 + (j-8)/2] >> (4*(j%2))) & 0xF
;   hi2 = (scales[8 + j/4] >> (2*(j%4))) & 3
;   scale_6bit = low4 | (hi2 << 4)                   ; range 0-63
;   scale_signed = scale_6bit - 32                    ; range -32..+31
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q3_k_avx2
%define K_BLOCK_BYTES       110
%define V_BLOCK_BYTES       110
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
; Shift amounts for extracting 2-bit values from a dword (8 values per ymm)
; Group 0 (values 0-7): bits [0:15] of 4-byte qs chunk
fa2_q3k_shifts_2bit_lo: dd 0, 2, 4, 6, 8, 10, 12, 14
; Group 1 (values 8-15): bits [16:31] of 4-byte qs chunk
fa2_q3k_shifts_2bit_hi: dd 16, 18, 20, 22, 24, 26, 28, 30

; Shift amounts for extracting 1-bit hmask values from a word (8 values per ymm)
fa2_q3k_shifts_1bit_lo: dd 0, 1, 2, 3, 4, 5, 6, 7
fa2_q3k_shifts_1bit_hi: dd 8, 9, 10, 11, 12, 13, 14, 15

; Masks
fa2_q3k_mask_2bit: times 8 dd 0x03
fa2_q3k_mask_1bit: times 8 dd 0x01

; Subtraction constant for centering: raw_3bit - 4 → signed
fa2_q3k_sub4: times 8 dd 4

; Bias for 6-bit scales: scale_6bit - 32 → signed
fa2_q3k_scale_bias: times 8 dd 32
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = mask_2bit (times 8 dd 0x03)
; ymm12 = mask_1bit (times 8 dd 0x01)
; ymm13 = sub4 (times 8 dd 4)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q3k_mask_2bit]
    vmovdqu   ymm12, [rel fa2_q3k_mask_1bit]
    vmovdqu   ymm13, [rel fa2_q3k_sub4]
%endmacro

; ============================================================================
; Helper: Extract Q3_K 6-bit signed scale for sub-block j
;
; Inputs:
;   %1 = block_ptr register (points to start of Q3_K block)
;   %2 = sub-block index j (immediate 0-15)
;
; Outputs:
;   eax = signed scale (scale_6bit - 32, range -32..+31)
;
; Clobbers: eax, ecx, edx
; ============================================================================
%macro EXTRACT_Q3K_SCALE 2
    ; --- Low 4 bits ---
%if %2 < 8
    ; j < 8: byte_idx = 96 + j/2
    movzx     eax, byte [%1 + 96 + %2 / 2]
%else
    ; j >= 8: byte_idx = 96 + 4 + (j-8)/2
    movzx     eax, byte [%1 + 96 + 4 + (%2 - 8) / 2]
%endif
%if (%2 % 2) == 1
    shr       eax, 4                          ; odd j: high nibble
%endif
    and       eax, 0x0F                        ; low4

    ; --- High 2 bits ---
    movzx     edx, byte [%1 + 96 + 8 + %2 / 4]
%if (%2 % 4) != 0
    shr       edx, 2 * (%2 % 4)
%endif
    and       edx, 0x03                        ; hi2

    ; --- Combine ---
    shl       edx, 4
    or        eax, edx                         ; scale_6bit = low4 | (hi2 << 4)
    sub       eax, 32                          ; signed scale = scale_6bit - 32
%endmacro

; ============================================================================
; Helper: Extract and reconstruct 8 Q3_K values to int32 in ymm
;
; Extracts 2-bit values from qs, 1-bit values from hmask, combines to 3-bit,
; subtracts 4 to get signed values, and zero/sign-extends to int32 in ymm.
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = group index (0 = values 0-7, 1 = values 8-15 within sub-block)
; %4 = output ymm register (will contain 8 x int32 signed values)
;
; Uses ymm11 (mask_2bit), ymm12 (mask_1bit), ymm13 (sub4)
; Clobbers: ymm0, ymm1, the output register
; ============================================================================
%macro EXTRACT_Q3K_8VALUES 4
    ; --- Extract 2-bit values from qs ---
    ; qs for sub-block j: 4 bytes at offset 32 + j*4
    ; Broadcast 4 bytes as dword
    vpbroadcastd %4, [%1 + 32 + %2 * 4]

    ; Shift each dword lane by different amount to isolate each 2-bit field
%if %3 == 0
    vpsrlvd   %4, %4, [rel fa2_q3k_shifts_2bit_lo]
%else
    vpsrlvd   %4, %4, [rel fa2_q3k_shifts_2bit_hi]
%endif
    vpand     %4, %4, ymm11                   ; mask to 2 bits each

    ; --- Extract 1-bit values from hmask ---
    ; hmask for sub-block j: 2 bytes at offset 0 + j*2
    ; Broadcast 2 bytes as dword (only low 16 bits matter)
    movzx     eax, word [%1 + %2 * 2]
    vmovd     xmm0, eax
    vpbroadcastd ymm0, xmm0

%if %3 == 0
    vpsrlvd   ymm0, ymm0, [rel fa2_q3k_shifts_1bit_lo]
%else
    vpsrlvd   ymm0, ymm0, [rel fa2_q3k_shifts_1bit_hi]
%endif
    vpand     ymm0, ymm0, ymm12               ; mask to 1 bit each

    ; --- Combine: raw_3bit = qs_2bit | (hmask_bit << 2) ---
    vpslld    ymm0, ymm0, 2                   ; shift hmask bit to position 2
    vpor      %4, %4, ymm0                    ; raw 3-bit values (0-7 as int32)

    ; --- Convert to signed: signed = raw - 4 ---
    vpsubd    %4, %4, ymm13                   ; range -4..+3 as int32
%endmacro

; ============================================================================
; Helper: Dequant + dot one sub-block of 16 values (AVX2, 2 groups of 8)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = ymm register with broadcast eff_scale (d * scale)
; %4 = q_ptr register (points to Q f32 values for this sub-block)
; %5 = acc_ymm (ymm to accumulate dot product into)
;
; Clobbers: ymm0-ymm2, eax
; ============================================================================
%macro DEQUANT_Q3K_SUBBLOCK_DOT 5
    ; Group 0: values 0-7
    EXTRACT_Q3K_8VALUES %1, %2, 0, ymm1
    vcvtdq2ps ymm1, ymm1                    ; int32 -> f32
    vmulps    ymm1, ymm1, %3                ; dequant = eff_scale * signed_value
    vmovups   ymm2, [%4]                    ; Q[0:7]
    vfmadd231ps %5, ymm1, ymm2              ; acc += dequant * Q

    ; Group 1: values 8-15
    EXTRACT_Q3K_8VALUES %1, %2, 1, ymm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, %3
    vmovups   ymm2, [%4 + 32]              ; Q[8:15]
    vfmadd231ps %5, ymm1, ymm2
%endmacro

; ============================================================================
; Helper: Dequant + accumulate one sub-block for V path (AVX2)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = ymm register with broadcast eff_scale (d * scale)
; %4 = prob_ymm (broadcast probability)
; %5 = o_ptr register (points to O accumulator for this sub-block)
;
; Clobbers: ymm0-ymm2, eax
; ============================================================================
%macro DEQUANT_Q3K_SUBBLOCK_V_ACCUM 5
    ; Group 0: values 0-7
    EXTRACT_Q3K_8VALUES %1, %2, 0, ymm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, %3                ; dequant = eff_scale * signed_value
    vmovaps   ymm2, [%5]                    ; O[0:7]
    vfmadd231ps ymm2, %4, ymm1              ; O += prob * dequant
    vmovaps   [%5], ymm2

    ; Group 1: values 8-15
    EXTRACT_Q3K_8VALUES %1, %2, 1, ymm1
    vcvtdq2ps ymm1, ymm1
    vmulps    ymm1, ymm1, %3
    vmovaps   ymm2, [%5 + 32]              ; O[8:15]
    vfmadd231ps ymm2, %4, ymm1
    vmovaps   [%5 + 32], ymm2
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q3_k_block))
;
; %1 = k_block_ptr (register, points to Q3_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product for all 16 sub-blocks:
;   1. Load d (fp16 -> f32) from offset 108
;   2. For each sub-block j=0..15:
;      a. Extract 6-bit signed scale
;      b. Compute eff_scale = d * scale (broadcast to ymm)
;      c. Extract+reconstruct 16 3-bit signed values
;      d. Convert to f32, apply eff_scale
;      e. Dot with Q, accumulate
;
; Clobbers: ymm0-ymm5, eax, ecx, edx
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load super-block scale d (fp16 -> f32) from offset 108
    movzx     eax, word [%1 + 108]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d (f32)

    ; Local accumulator for this block
    vxorps    ymm3, ymm3, ymm3              ; ymm3 = local acc

    ; ------ Sub-block 0 ------
    EXTRACT_Q3K_SCALE %1, 0
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4                    ; float(scale)
    vbroadcastss ymm5, xmm0                 ; broadcast d
    vmulps    ymm4, ymm4, ymm5              ; ymm4 = d * scale (eff_scale)
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 0, ymm4, %2, ymm3

    ; ------ Sub-block 1 ------
    EXTRACT_Q3K_SCALE %1, 1
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 1, ymm4, %2 + 64, ymm3

    ; ------ Sub-block 2 ------
    EXTRACT_Q3K_SCALE %1, 2
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 2, ymm4, %2 + 128, ymm3

    ; ------ Sub-block 3 ------
    EXTRACT_Q3K_SCALE %1, 3
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 3, ymm4, %2 + 192, ymm3

    ; ------ Sub-block 4 ------
    EXTRACT_Q3K_SCALE %1, 4
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 4, ymm4, %2 + 256, ymm3

    ; ------ Sub-block 5 ------
    EXTRACT_Q3K_SCALE %1, 5
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 5, ymm4, %2 + 320, ymm3

    ; ------ Sub-block 6 ------
    EXTRACT_Q3K_SCALE %1, 6
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 6, ymm4, %2 + 384, ymm3

    ; ------ Sub-block 7 ------
    EXTRACT_Q3K_SCALE %1, 7
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 7, ymm4, %2 + 448, ymm3

    ; ------ Sub-block 8 ------
    EXTRACT_Q3K_SCALE %1, 8
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 8, ymm4, %2 + 512, ymm3

    ; ------ Sub-block 9 ------
    EXTRACT_Q3K_SCALE %1, 9
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 9, ymm4, %2 + 576, ymm3

    ; ------ Sub-block 10 ------
    EXTRACT_Q3K_SCALE %1, 10
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 10, ymm4, %2 + 640, ymm3

    ; ------ Sub-block 11 ------
    EXTRACT_Q3K_SCALE %1, 11
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 11, ymm4, %2 + 704, ymm3

    ; ------ Sub-block 12 ------
    EXTRACT_Q3K_SCALE %1, 12
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 12, ymm4, %2 + 768, ymm3

    ; ------ Sub-block 13 ------
    EXTRACT_Q3K_SCALE %1, 13
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 13, ymm4, %2 + 832, ymm3

    ; ------ Sub-block 14 ------
    EXTRACT_Q3K_SCALE %1, 14
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 14, ymm4, %2 + 896, ymm3

    ; ------ Sub-block 15 ------
    EXTRACT_Q3K_SCALE %1, 15
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 15, ymm4, %2 + 960, ymm3

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, ymm3
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q3_k_block)
;
; %1 = v_block_ptr (register, points to Q3_K block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 bytes.
;
; Clobbers: ymm0-ymm5, eax, ecx, edx, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load super-block scale d (fp16 -> f32) from offset 108
    movzx     eax, word [%1 + 108]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d

    ; 2. Compute O base offset = block_idx * 256 * 4 = block_idx * 1024
    mov       rax, %4
    shl       rax, 10                        ; * 1024
    lea       rax, [%3 + rax]                ; rax = o_base + offset
    ; rax now points to O[block_idx * 256]

    ; ------ Sub-block 0 ------
    EXTRACT_Q3K_SCALE %1, 0
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5              ; eff_scale = d * scale
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 0, ymm4, %2, rax

    ; ------ Sub-block 1 ------
    EXTRACT_Q3K_SCALE %1, 1
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 1, ymm4, %2, rax + 64

    ; ------ Sub-block 2 ------
    EXTRACT_Q3K_SCALE %1, 2
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 2, ymm4, %2, rax + 128

    ; ------ Sub-block 3 ------
    EXTRACT_Q3K_SCALE %1, 3
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 3, ymm4, %2, rax + 192

    ; ------ Sub-block 4 ------
    EXTRACT_Q3K_SCALE %1, 4
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 4, ymm4, %2, rax + 256

    ; ------ Sub-block 5 ------
    EXTRACT_Q3K_SCALE %1, 5
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 5, ymm4, %2, rax + 320

    ; ------ Sub-block 6 ------
    EXTRACT_Q3K_SCALE %1, 6
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 6, ymm4, %2, rax + 384

    ; ------ Sub-block 7 ------
    EXTRACT_Q3K_SCALE %1, 7
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 7, ymm4, %2, rax + 448

    ; ------ Sub-block 8 ------
    EXTRACT_Q3K_SCALE %1, 8
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 8, ymm4, %2, rax + 512

    ; ------ Sub-block 9 ------
    EXTRACT_Q3K_SCALE %1, 9
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 9, ymm4, %2, rax + 576

    ; ------ Sub-block 10 ------
    EXTRACT_Q3K_SCALE %1, 10
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 10, ymm4, %2, rax + 640

    ; ------ Sub-block 11 ------
    EXTRACT_Q3K_SCALE %1, 11
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 11, ymm4, %2, rax + 704

    ; ------ Sub-block 12 ------
    EXTRACT_Q3K_SCALE %1, 12
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 12, ymm4, %2, rax + 768

    ; ------ Sub-block 13 ------
    EXTRACT_Q3K_SCALE %1, 13
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 13, ymm4, %2, rax + 832

    ; ------ Sub-block 14 ------
    EXTRACT_Q3K_SCALE %1, 14
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 14, ymm4, %2, rax + 896

    ; ------ Sub-block 15 ------
    EXTRACT_Q3K_SCALE %1, 15
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 15, ymm4, %2, rax + 960
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
