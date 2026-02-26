; ============================================================================
; Flash Attention 2 - Q3_K AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q3_K format (110 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
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
;
; AVX-512 processes 1 group of 16 per sub-block (zmm = 16 f32).
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q3_k_avx512
%define K_BLOCK_BYTES       110
%define V_BLOCK_BYTES       110
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
; Shift amounts for extracting all 16 2-bit values from a 4-byte dword
; Value i uses bits [2*i : 2*i+1], shift right by 2*i then mask with 0x03
fa2_q3k_shifts_2bit_16: dd 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30

; Shift amounts for extracting 16 1-bit hmask values from a 2-byte word
fa2_q3k_shifts_1bit_16: dd 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

; Masks (zmm-width)
fa2_q3k_mask_2bit: times 16 dd 0x03
fa2_q3k_mask_1bit: times 16 dd 0x01

; Subtraction constant for centering: raw_3bit - 4 → signed
fa2_q3k_sub4: times 16 dd 4
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = mask_2bit (times 16 dd 0x03)
; zmm12 = mask_1bit (times 16 dd 0x01)
; zmm13 = sub4 (times 16 dd 4)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu32 zmm11, [rel fa2_q3k_mask_2bit]
    vmovdqu32 zmm12, [rel fa2_q3k_mask_1bit]
    vmovdqu32 zmm13, [rel fa2_q3k_sub4]
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
; Helper: Extract and reconstruct 16 Q3_K values to int32 in zmm
;
; Extracts 2-bit values from qs, 1-bit values from hmask, combines to 3-bit,
; subtracts 4 to get signed values as int32 in zmm.
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = output zmm register (will contain 16 x int32 signed values)
;
; Uses zmm11 (mask_2bit), zmm12 (mask_1bit), zmm13 (sub4)
; Clobbers: zmm0, zmm1, the output register, eax
; ============================================================================
%macro EXTRACT_Q3K_16VALUES 3
    ; --- Extract 2-bit values from qs ---
    ; qs for sub-block j: 4 bytes at offset 32 + j*4
    ; Broadcast 4 bytes as dword to all 16 zmm lanes
    vpbroadcastd %3, [%1 + 32 + %2 * 4]

    ; Shift each dword lane by different amount to isolate each 2-bit field
    vpsrlvd   %3, %3, [rel fa2_q3k_shifts_2bit_16]
    vpandd    %3, %3, zmm11                   ; mask to 2 bits each

    ; --- Extract 1-bit values from hmask ---
    ; hmask for sub-block j: 2 bytes at offset 0 + j*2
    ; Broadcast 2 bytes as dword (only low 16 bits matter)
    movzx     eax, word [%1 + %2 * 2]
    vmovd     xmm0, eax
    vpbroadcastd zmm0, xmm0

    vpsrlvd   zmm0, zmm0, [rel fa2_q3k_shifts_1bit_16]
    vpandd    zmm0, zmm0, zmm12               ; mask to 1 bit each

    ; --- Combine: raw_3bit = qs_2bit | (hmask_bit << 2) ---
    vpslld    zmm0, zmm0, 2                   ; shift hmask bit to position 2
    vpord     %3, %3, zmm0                    ; raw 3-bit values (0-7 as int32)

    ; --- Convert to signed: signed = raw - 4 ---
    vpsubd    %3, %3, zmm13                   ; range -4..+3 as int32
%endmacro

; ============================================================================
; Helper: Dequant + dot one sub-block of 16 values (AVX-512, 1 group of 16)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = zmm register with broadcast eff_scale (d * scale)
; %4 = q_ptr register (points to Q f32 values for this sub-block)
; %5 = acc_zmm (zmm to accumulate dot product into)
;
; Clobbers: zmm0-zmm2, eax
; ============================================================================
%macro DEQUANT_Q3K_SUBBLOCK_DOT 5
    ; Extract all 16 signed 3-bit values as int32 in zmm1
    EXTRACT_Q3K_16VALUES %1, %2, zmm1
    vcvtdq2ps zmm1, zmm1                    ; int32 -> f32
    vmulps    zmm1, zmm1, %3                ; dequant = eff_scale * signed_value
    vmovups   zmm2, [%4]                    ; Q[0:15]
    vfmadd231ps %5, zmm1, zmm2              ; acc += dequant * Q
%endmacro

; ============================================================================
; Helper: Dequant + accumulate one sub-block for V path (AVX-512)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-15)
; %3 = zmm register with broadcast eff_scale (d * scale)
; %4 = prob_zmm (broadcast probability)
; %5 = o_ptr register (points to O accumulator for this sub-block)
;
; Clobbers: zmm0-zmm2, eax
; ============================================================================
%macro DEQUANT_Q3K_SUBBLOCK_V_ACCUM 5
    ; Extract all 16 signed 3-bit values as int32 in zmm1
    EXTRACT_Q3K_16VALUES %1, %2, zmm1
    vcvtdq2ps zmm1, zmm1
    vmulps    zmm1, zmm1, %3                ; dequant = eff_scale * signed_value
    vmovaps   zmm2, [%5]                    ; O[0:15]
    vfmadd231ps zmm2, %4, zmm1              ; O += prob * dequant
    vmovaps   [%5], zmm2
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q3_k_block))
;
; %1 = k_block_ptr (register, points to Q3_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; Dequantization + dot product for all 16 sub-blocks:
;   1. Load d (fp16 -> f32) from offset 108
;   2. For each sub-block j=0..15:
;      a. Extract 6-bit signed scale
;      b. Compute eff_scale = d * scale (broadcast to zmm)
;      c. Extract+reconstruct 16 3-bit signed values
;      d. Convert to f32, apply eff_scale
;      e. Dot with Q, accumulate
;
; Clobbers: zmm0-zmm5, eax, ecx, edx
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load super-block scale d (fp16 -> f32) from offset 108
    movzx     eax, word [%1 + 108]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d (f32)

    ; Local accumulator for this block
    vxorps    zmm3, zmm3, zmm3              ; zmm3 = local acc

    ; ------ Sub-block 0 ------
    EXTRACT_Q3K_SCALE %1, 0
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4                    ; float(scale) broadcast
    vbroadcastss zmm5, xmm0                 ; broadcast d
    vmulps    zmm4, zmm4, zmm5              ; zmm4 = d * scale (eff_scale)
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 0, zmm4, %2, zmm3

    ; ------ Sub-block 1 ------
    EXTRACT_Q3K_SCALE %1, 1
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 1, zmm4, %2 + 64, zmm3

    ; ------ Sub-block 2 ------
    EXTRACT_Q3K_SCALE %1, 2
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 2, zmm4, %2 + 128, zmm3

    ; ------ Sub-block 3 ------
    EXTRACT_Q3K_SCALE %1, 3
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 3, zmm4, %2 + 192, zmm3

    ; ------ Sub-block 4 ------
    EXTRACT_Q3K_SCALE %1, 4
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 4, zmm4, %2 + 256, zmm3

    ; ------ Sub-block 5 ------
    EXTRACT_Q3K_SCALE %1, 5
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 5, zmm4, %2 + 320, zmm3

    ; ------ Sub-block 6 ------
    EXTRACT_Q3K_SCALE %1, 6
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 6, zmm4, %2 + 384, zmm3

    ; ------ Sub-block 7 ------
    EXTRACT_Q3K_SCALE %1, 7
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 7, zmm4, %2 + 448, zmm3

    ; ------ Sub-block 8 ------
    EXTRACT_Q3K_SCALE %1, 8
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 8, zmm4, %2 + 512, zmm3

    ; ------ Sub-block 9 ------
    EXTRACT_Q3K_SCALE %1, 9
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 9, zmm4, %2 + 576, zmm3

    ; ------ Sub-block 10 ------
    EXTRACT_Q3K_SCALE %1, 10
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 10, zmm4, %2 + 640, zmm3

    ; ------ Sub-block 11 ------
    EXTRACT_Q3K_SCALE %1, 11
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 11, zmm4, %2 + 704, zmm3

    ; ------ Sub-block 12 ------
    EXTRACT_Q3K_SCALE %1, 12
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 12, zmm4, %2 + 768, zmm3

    ; ------ Sub-block 13 ------
    EXTRACT_Q3K_SCALE %1, 13
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 13, zmm4, %2 + 832, zmm3

    ; ------ Sub-block 14 ------
    EXTRACT_Q3K_SCALE %1, 14
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 14, zmm4, %2 + 896, zmm3

    ; ------ Sub-block 15 ------
    EXTRACT_Q3K_SCALE %1, 15
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_DOT %1, 15, zmm4, %2 + 960, zmm3

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, zmm3
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q3_k_block)
;
; %1 = v_block_ptr (register, points to Q3_K block)
; %2 = prob_zmm (zmm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 bytes.
;
; Clobbers: zmm0-zmm5, eax, ecx, edx, rax
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
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
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5              ; eff_scale = d * scale
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 0, zmm4, %2, rax

    ; ------ Sub-block 1 ------
    EXTRACT_Q3K_SCALE %1, 1
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 1, zmm4, %2, rax + 64

    ; ------ Sub-block 2 ------
    EXTRACT_Q3K_SCALE %1, 2
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 2, zmm4, %2, rax + 128

    ; ------ Sub-block 3 ------
    EXTRACT_Q3K_SCALE %1, 3
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 3, zmm4, %2, rax + 192

    ; ------ Sub-block 4 ------
    EXTRACT_Q3K_SCALE %1, 4
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 4, zmm4, %2, rax + 256

    ; ------ Sub-block 5 ------
    EXTRACT_Q3K_SCALE %1, 5
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 5, zmm4, %2, rax + 320

    ; ------ Sub-block 6 ------
    EXTRACT_Q3K_SCALE %1, 6
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 6, zmm4, %2, rax + 384

    ; ------ Sub-block 7 ------
    EXTRACT_Q3K_SCALE %1, 7
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 7, zmm4, %2, rax + 448

    ; ------ Sub-block 8 ------
    EXTRACT_Q3K_SCALE %1, 8
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 8, zmm4, %2, rax + 512

    ; ------ Sub-block 9 ------
    EXTRACT_Q3K_SCALE %1, 9
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 9, zmm4, %2, rax + 576

    ; ------ Sub-block 10 ------
    EXTRACT_Q3K_SCALE %1, 10
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 10, zmm4, %2, rax + 640

    ; ------ Sub-block 11 ------
    EXTRACT_Q3K_SCALE %1, 11
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 11, zmm4, %2, rax + 704

    ; ------ Sub-block 12 ------
    EXTRACT_Q3K_SCALE %1, 12
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 12, zmm4, %2, rax + 768

    ; ------ Sub-block 13 ------
    EXTRACT_Q3K_SCALE %1, 13
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 13, zmm4, %2, rax + 832

    ; ------ Sub-block 14 ------
    EXTRACT_Q3K_SCALE %1, 14
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 14, zmm4, %2, rax + 896

    ; ------ Sub-block 15 ------
    EXTRACT_Q3K_SCALE %1, 15
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5
    DEQUANT_Q3K_SUBBLOCK_V_ACCUM %1, 15, zmm4, %2, rax + 960
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
