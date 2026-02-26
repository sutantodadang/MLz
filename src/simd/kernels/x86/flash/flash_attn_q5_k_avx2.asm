; ============================================================================
; Flash Attention 2 - Q5_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q5_K format (176 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q5_K block layout (176 bytes):
;   offset  0: d      (ggml_half, 2 bytes) — super-block scale
;   offset  2: dmin   (ggml_half, 2 bytes) — super-block min
;   offset  4: scales (uint8_t[12], 12 bytes) — packed 4/6-bit sub-block scales/mins
;   offset 16: qh     (uint8_t[32], 32 bytes) — high bits (bit 5 for each value)
;   offset 48: qs     (uint8_t[128], 128 bytes) — 4-bit packed values (256 nibbles)
;
; 8 sub-blocks of 32 values each.
; Dequantization: value = d * sc * (nibble | (qh_bit << 4)) - dmin * m
;
; Scale extraction (get_scale_min_k4 — identical to Q4_K):
;   j < 4:  sc = scales[j] & 0x3F,           m = scales[j+4] & 0x3F
;   j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
;            m = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
;
; High-bit extraction (Q5_K vs Q4_K difference):
;   qh[32] contains 256 bits (one per value), arranged as 32 bytes.
;   For sub-block j (values j*32..j*32+31):
;     qh bytes at offset 16 + j*4 (4 bytes = 32 bits for this sub-block)
;   Each byte covers 8 values; bit i of the byte -> high bit for value i.
;   We use vpsrlvd with shift amounts [0..7] to extract per-value bits.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q5_k_avx2
%define K_BLOCK_BYTES       176
%define V_BLOCK_BYTES       176
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q5k_nibble_mask: times 32 db 0x0F
fa2_q5k_bit_shifts:  dd 0, 1, 2, 3, 4, 5, 6, 7   ; vpsrlvd shift amounts per lane
fa2_q5k_and1:        times 8 dd 1                  ; AND mask for isolating bit 0
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; ymm13 = bit_shifts [0,1,2,3,4,5,6,7] for vpsrlvd high-bit extraction
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q5k_nibble_mask]
    vmovdqu   ymm13, [rel fa2_q5k_bit_shifts]
%endmacro

; ============================================================================
; Helper: Extract Q5_K sub-block scale (sc) and min (m) for sub-block j
;
; Identical to Q4_K — uses get_scale_min_k4 algorithm.
;
; Inputs:
;   %1 = block_ptr register (points to start of Q5_K block)
;   %2 = sub-block index j (immediate 0-7)
;
; Outputs:
;   eax = sc (6-bit scale for this sub-block)
;   edx = m  (6-bit min for this sub-block)
;
; Clobbers: eax, ecx, edx
; ============================================================================
%macro EXTRACT_SCALE_MIN 2
%if %2 < 4
    ; j < 4: sc = scales[j] & 0x3F, m = scales[j+4] & 0x3F
    movzx     eax, byte [%1 + 4 + %2]       ; scales[j]
    and       eax, 0x3F                       ; sc = scales[j] & 63

    movzx     edx, byte [%1 + 4 + %2 + 4]   ; scales[j+4]
    and       edx, 0x3F                       ; m = scales[j+4] & 63
%else
    ; j >= 4:
    ; sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    ; m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)

    ; sc computation
    movzx     eax, byte [%1 + 4 + %2 + 4]   ; scales[j+4]
    movzx     ecx, byte [%1 + 4 + %2 - 4]   ; scales[j-4]
    mov       edx, eax                        ; save scales[j+4] for m computation
    and       eax, 0x0F                       ; low 4 bits of scales[j+4]
    shr       ecx, 6                          ; top 2 bits of scales[j-4]
    shl       ecx, 4
    or        eax, ecx                        ; sc = low4 | (top2 << 4)

    ; m computation
    movzx     ecx, byte [%1 + 4 + %2]       ; scales[j]
    shr       edx, 4                          ; high 4 bits of scales[j+4]
    shr       ecx, 6                          ; top 2 bits of scales[j]
    shl       ecx, 4
    or        edx, ecx                        ; m = high4 | (top2 << 4)
%endif
%endmacro

; ============================================================================
; DEQUANT_SUBBLOCK_DOT: Dequant + dot one sub-block of 32 Q5_K values (AVX2)
;
; Extracts 32 5-bit values (4-bit nibble | qh<<4) from qs+qh, converts to f32,
; applies: result = eff_scale * value + eff_min, dots with Q, accumulates.
;
; Processes 4 groups of 8 values. For each group, one qh byte is loaded,
; broadcast as dword, shifted right by [0..7] via vpsrlvd, AND 1, SHL 4,
; then OR'd with the 4-bit nibble to form the 5-bit value.
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = ymm register with broadcast eff_scale (d * sc)
; %4 = ymm register with broadcast eff_min (negated: -dmin * m)
; %5 = q_ptr register (points to Q f32 values for this sub-block)
; %6 = acc_ymm (ymm to accumulate dot product)
;
; Q5_K offsets:
;   qs at block_ptr + 48 + j * 16  (16 bytes = 32 nibbles per sub-block)
;   qh at block_ptr + 16 + j * 4   (4 bytes = 32 high bits per sub-block)
;
; Clobbers: ymm0-ymm4, ymm7, eax
; ============================================================================
%macro DEQUANT_SUBBLOCK_DOT 6
    ; Load 16 packed bytes (32 nibbles) for sub-block j from qs (offset 48)
    vmovdqu   xmm1, [%1 + 48 + %2 * 16]

    ; Extract low nibbles (even-indexed values: q[0], q[2], ...)
    vpand     xmm2, xmm1, xmm11             ; low nibbles & 0x0F
    ; Extract high nibbles (odd-indexed values: q[1], q[3], ...)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11             ; high nibbles >> 4 & 0x0F

    ; Interleave to get 32 unsigned bytes in original order
    vpunpcklbw xmm0, xmm2, xmm3             ; interleave low halves -> values 0-15
    vpunpckhbw xmm1, xmm2, xmm3             ; interleave high halves -> values 16-31
    vinserti128 ymm0, ymm0, xmm1, 1         ; ymm0 = 32 nibble bytes (4-bit values)

    ; === Group 0: values 0-7 ===
    vpmovzxbd ymm1, xmm0                    ; zero-extend 8 bytes -> 8 int32

    ; Extract high bits for values 0-7 from qh byte 0
    movzx     eax, byte [%1 + 16 + %2 * 4 + 0]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7                 ; all 8 dwords = qh_byte0
    vpsrlvd   ymm7, ymm7, ymm13             ; shift lane i right by i
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]; isolate bit 0
    vpslld    ymm7, ymm7, 4                  ; 0 or 16
    vpor      ymm1, ymm1, ymm7              ; combine: 5-bit value

    vcvtdq2ps ymm1, ymm1                    ; -> f32
    vfmadd132ps ymm1, %4, %3                ; dequant = eff_scale * value + eff_min
    vmovups   ymm2, [%5]                    ; Q[0:7]
    vfmadd231ps %6, ymm1, ymm2              ; acc += dequant * Q

    ; === Group 1: values 8-15 ===
    vextracti128 xmm1, ymm0, 0              ; low 16 bytes
    vpsrldq   xmm1, xmm1, 8                 ; shift right 8 bytes -> bytes 8-15
    vpmovzxbd ymm1, xmm1

    movzx     eax, byte [%1 + 16 + %2 * 4 + 1]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 32]               ; Q[8:15]
    vfmadd231ps %6, ymm1, ymm2

    ; === Group 2: values 16-23 ===
    vextracti128 xmm1, ymm0, 1              ; high 16 bytes
    vpmovzxbd ymm1, xmm1                    ; bytes 0-7 of high -> values 16-23

    movzx     eax, byte [%1 + 16 + %2 * 4 + 2]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 64]               ; Q[16:23]
    vfmadd231ps %6, ymm1, ymm2

    ; === Group 3: values 24-31 ===
    vextracti128 xmm1, ymm0, 1              ; high 16 bytes
    vpsrldq   xmm1, xmm1, 8                 ; bytes 8-15 -> values 24-31
    vpmovzxbd ymm1, xmm1

    movzx     eax, byte [%1 + 16 + %2 * 4 + 3]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 96]               ; Q[24:31]
    vfmadd231ps %6, ymm1, ymm2
%endmacro

; ============================================================================
; DEQUANT_SUBBLOCK_V_ACCUM: Dequant + V accum one sub-block (AVX2)
;
; Same dequant as DEQUANT_SUBBLOCK_DOT, but multiplies by probability
; and accumulates into the O buffer instead of dotting with Q.
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = ymm register with broadcast eff_scale (d * sc)
; %4 = ymm register with broadcast eff_min (negated: -dmin * m)
; %5 = prob_ymm (broadcast probability)
; %6 = o_ptr register (points to O accumulator for this sub-block)
;
; Clobbers: ymm0-ymm4, ymm7, eax
; ============================================================================
%macro DEQUANT_SUBBLOCK_V_ACCUM 6
    ; Load 16 packed bytes (32 nibbles) from qs (offset 48)
    vmovdqu   xmm1, [%1 + 48 + %2 * 16]

    ; Extract nibbles
    vpand     xmm2, xmm1, xmm11
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11

    ; Interleave
    vpunpcklbw xmm0, xmm2, xmm3
    vpunpckhbw xmm1, xmm2, xmm3
    vinserti128 ymm0, ymm0, xmm1, 1

    ; === Group 0: values 0-7 ===
    vpmovzxbd ymm1, xmm0

    movzx     eax, byte [%1 + 16 + %2 * 4 + 0]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3                ; dequant = eff_scale * value + eff_min
    vmovaps   ymm2, [%6]                    ; O[0:7]
    vfmadd231ps ymm2, %5, ymm1              ; O += prob * dequant
    vmovaps   [%6], ymm2

    ; === Group 1: values 8-15 ===
    vextracti128 xmm1, ymm0, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1

    movzx     eax, byte [%1 + 16 + %2 * 4 + 1]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 32]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 32], ymm2

    ; === Group 2: values 16-23 ===
    vextracti128 xmm1, ymm0, 1
    vpmovzxbd ymm1, xmm1

    movzx     eax, byte [%1 + 16 + %2 * 4 + 2]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 64]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 64], ymm2

    ; === Group 3: values 24-31 ===
    vextracti128 xmm1, ymm0, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1

    movzx     eax, byte [%1 + 16 + %2 * 4 + 3]
    vmovd     xmm7, eax
    vpbroadcastd ymm7, xmm7
    vpsrlvd   ymm7, ymm7, ymm13
    vpand     ymm7, ymm7, [rel fa2_q5k_and1]
    vpslld    ymm7, ymm7, 4
    vpor      ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 96]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 96], ymm2
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q5_k_block))
;
; %1 = k_block_ptr (register, points to Q5_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product for all 8 sub-blocks:
;   1. Load d (fp16 -> f32) and dmin (fp16 -> f32, negated)
;   2. For each sub-block j=0..7:
;      a. Extract 6-bit scale (sc) and min (m) via get_scale_min_k4
;      b. Compute eff_scale = d * sc, eff_min = -dmin * m
;      c. Unpack 32 nibbles, add high bits, convert to f32
;      d. Apply: dequant = eff_scale * value + eff_min
;      e. Dot with Q, accumulate
;
; Clobbers: ymm0-ymm5, ymm7, eax, ecx, edx
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load super-block scale d (fp16 -> f32)
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d (f32)

    ; 2. Load super-block min dmin (fp16 -> f32), negate for subtraction
    movzx     eax, word [%1 + 2]
    vmovd     xmm1, eax
    vcvtph2ps xmm1, xmm1                    ; xmm1[0] = dmin (f32)

    ; Negate dmin: we want -dmin * m so that dequant = d*sc*value + (-dmin*m)
    vxorps    xmm2, xmm2, xmm2
    vsubss    xmm1, xmm2, xmm1              ; xmm1[0] = -dmin

    ; Local accumulator for this block
    vxorps    ymm3, ymm3, ymm3              ; ymm3 = local acc

    ; ------ Sub-block 0 ------
    EXTRACT_SCALE_MIN %1, 0
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4                    ; float(sc)
    vbroadcastss ymm5, xmm0                 ; broadcast d
    vmulps    ymm4, ymm4, ymm5              ; ymm4 = d * sc (eff_scale)

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5                    ; float(m)
    vbroadcastss ymm2, xmm1                 ; broadcast -dmin
    vmulps    ymm5, ymm5, ymm2              ; ymm5 = -dmin * m (eff_min)

    DEQUANT_SUBBLOCK_DOT %1, 0, ymm4, ymm5, %2, ymm3

    ; ------ Sub-block 1 ------
    EXTRACT_SCALE_MIN %1, 1
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 1, ymm4, ymm5, %2 + 128, ymm3

    ; ------ Sub-block 2 ------
    EXTRACT_SCALE_MIN %1, 2
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 2, ymm4, ymm5, %2 + 256, ymm3

    ; ------ Sub-block 3 ------
    EXTRACT_SCALE_MIN %1, 3
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 3, ymm4, ymm5, %2 + 384, ymm3

    ; ------ Sub-block 4 ------
    EXTRACT_SCALE_MIN %1, 4
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 4, ymm4, ymm5, %2 + 512, ymm3

    ; ------ Sub-block 5 ------
    EXTRACT_SCALE_MIN %1, 5
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 5, ymm4, ymm5, %2 + 640, ymm3

    ; ------ Sub-block 6 ------
    EXTRACT_SCALE_MIN %1, 6
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 6, ymm4, ymm5, %2 + 768, ymm3

    ; ------ Sub-block 7 ------
    EXTRACT_SCALE_MIN %1, 7
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_DOT %1, 7, ymm4, ymm5, %2 + 896, ymm3

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, ymm3
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q5_k_block)
;
; %1 = v_block_ptr (register, points to Q5_K block)
; %2 = prob_ymm (ymm with broadcast probability)
; %3 = o_base_ptr (register, points to O accumulator base on stack)
; %4 = block_idx (register, block index for offset calculation)
;
; Same dequant as K, then multiply by prob and add to O.
; O offset = block_idx * 256 * 4 = block_idx * 1024 bytes.
;
; Clobbers: ymm0-ymm5, ymm7, eax, ecx, edx, rax
; Preserves: ymm11-ymm13, ymm14-ymm15, r12-r15, rbx
; ============================================================================
%macro DEQUANT_V_ACCUM_BLOCK 4
    ; 1. Load super-block scale d (fp16 -> f32)
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d

    ; 2. Load dmin (fp16 -> f32), negate
    movzx     eax, word [%1 + 2]
    vmovd     xmm1, eax
    vcvtph2ps xmm1, xmm1
    vxorps    xmm2, xmm2, xmm2
    vsubss    xmm1, xmm2, xmm1              ; xmm1[0] = -dmin

    ; 3. Compute O base offset = block_idx * 256 * 4 = block_idx * 1024
    mov       rax, %4
    shl       rax, 10                        ; * 1024
    lea       rax, [%3 + rax]                ; rax = o_base + offset
    ; rax now points to O[block_idx * 256]

    ; ------ Sub-block 0 ------
    EXTRACT_SCALE_MIN %1, 0
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5              ; eff_scale = d * sc

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2              ; eff_min = -dmin * m

    DEQUANT_SUBBLOCK_V_ACCUM %1, 0, ymm4, ymm5, %2, rax

    ; ------ Sub-block 1 ------
    EXTRACT_SCALE_MIN %1, 1
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 1, ymm4, ymm5, %2, rax + 128

    ; ------ Sub-block 2 ------
    EXTRACT_SCALE_MIN %1, 2
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 2, ymm4, ymm5, %2, rax + 256

    ; ------ Sub-block 3 ------
    EXTRACT_SCALE_MIN %1, 3
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 3, ymm4, ymm5, %2, rax + 384

    ; ------ Sub-block 4 ------
    EXTRACT_SCALE_MIN %1, 4
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 4, ymm4, ymm5, %2, rax + 512

    ; ------ Sub-block 5 ------
    EXTRACT_SCALE_MIN %1, 5
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 5, ymm4, ymm5, %2, rax + 640

    ; ------ Sub-block 6 ------
    EXTRACT_SCALE_MIN %1, 6
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 6, ymm4, ymm5, %2, rax + 768

    ; ------ Sub-block 7 ------
    EXTRACT_SCALE_MIN %1, 7
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, xmm0
    vmulps    ymm4, ymm4, ymm5

    vmovd     xmm5, edx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm2, xmm1
    vmulps    ymm5, ymm5, ymm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 7, ymm4, ymm5, %2, rax + 896
%endmacro

; ============================================================================
; Include the shared FA2 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx2.inc"
