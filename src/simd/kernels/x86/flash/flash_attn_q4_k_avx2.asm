; ============================================================================
; Flash Attention 2 - Q4_K AVX2 Kernel
; ============================================================================
; Quantized K/V with Q4_K format (144 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 skeleton.
;
; Q4_K block layout (144 bytes):
;   offset  0: d      (ggml_half, 2 bytes) — super-block scale
;   offset  2: dmin   (ggml_half, 2 bytes) — super-block min
;   offset  4: scales (uint8_t[12], 12 bytes) — packed 4/6-bit sub-block scales/mins
;   offset 16: qs     (uint8_t[128], 128 bytes) — 4-bit packed values (256 nibbles)
;
; 8 sub-blocks of 32 values each.
; Dequantization: value = d * sc * nibble - dmin * m
;
; Scale extraction (get_scale_min_k4):
;   j < 4:  sc = scales[j] & 0x3F,           m = scales[j+4] & 0x3F
;   j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
;            m = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_k_avx2
%define K_BLOCK_BYTES       144
%define V_BLOCK_BYTES       144
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 32
fa2_q4k_nibble_mask: times 32 db 0x0F
%endmacro

; ============================================================================
; Initialize quant constant registers
; ymm11 = nibble_mask (32x 0x0F)
; ymm12 = (unused)
; ymm13 = (unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   ymm11, [rel fa2_q4k_nibble_mask]
%endmacro

; ============================================================================
; Helper: Extract Q4_K sub-block scale (sc) and min (m) for sub-block j
;
; Inputs:
;   %1 = block_ptr register (points to start of Q4_K block)
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
; Helper: Dequant + process one sub-block of 32 values (AVX2)
;
; Extracts 32 nibble values from 16 packed bytes, converts to f32,
; applies: result = eff_scale * nibble + eff_min
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = ymm register with broadcast eff_scale (d * sc)
; %4 = ymm register with broadcast eff_min (negated: -dmin * m)
; %5 = q_ptr register (points to Q f32 values for this sub-block)
; %6 = acc_ymm (ymm to accumulate dot product)
;
; Clobbers: ymm0-ymm4
; ============================================================================
%macro DEQUANT_SUBBLOCK_DOT 6
    ; Load 16 packed bytes (32 nibbles) for sub-block j
    vmovdqu   xmm1, [%1 + 16 + %2 * 16]

    ; Extract low nibbles (even-indexed values)
    vpand     xmm2, xmm1, xmm11             ; low nibbles & 0x0F
    ; Extract high nibbles (odd-indexed values)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11             ; high nibbles >> 4 & 0x0F

    ; Interleave to get 32 unsigned bytes in original order
    vpunpcklbw xmm0, xmm2, xmm3             ; interleave low halves
    vpunpckhbw xmm1, xmm2, xmm3             ; interleave high halves
    vinserti128 ymm0, ymm0, xmm1, 1         ; ymm0 = 32 unsigned nibble bytes

    ; Group 0: bytes 0-7 -> f32, dot with Q
    vpmovzxbd ymm1, xmm0                    ; zero-extend 8 bytes -> 8 int32
    vcvtdq2ps ymm1, ymm1                    ; -> 8 f32 (unsigned nibble values)
    vfmadd132ps ymm1, %4, %3                ; ymm1 = eff_scale * nibble + eff_min
    vmovups   ymm2, [%5]                    ; Q[0:7]
    vfmadd231ps %6, ymm1, ymm2              ; acc += dequant * Q

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm0, 0              ; low 16 bytes
    vpsrldq   xmm1, xmm1, 8                 ; shift right 8 bytes
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 32]               ; Q[8:15]
    vfmadd231ps %6, ymm1, ymm2

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm0, 1              ; high 16 bytes
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 64]               ; Q[16:23]
    vfmadd231ps %6, ymm1, ymm2

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm0, 1              ; high 16 bytes
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovups   ymm2, [%5 + 96]               ; Q[24:31]
    vfmadd231ps %6, ymm1, ymm2
%endmacro

; ============================================================================
; Helper: Dequant + accumulate one sub-block for V path (AVX2)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = ymm register with broadcast eff_scale (d * sc)
; %4 = ymm register with broadcast eff_min (negated: -dmin * m)
; %5 = prob_ymm (broadcast probability)
; %6 = o_ptr register (points to O accumulator for this sub-block)
;
; Clobbers: ymm0-ymm4
; ============================================================================
%macro DEQUANT_SUBBLOCK_V_ACCUM 6
    ; Load 16 packed bytes (32 nibbles)
    vmovdqu   xmm1, [%1 + 16 + %2 * 16]

    ; Extract nibbles
    vpand     xmm2, xmm1, xmm11
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11

    ; Interleave
    vpunpcklbw xmm0, xmm2, xmm3
    vpunpckhbw xmm1, xmm2, xmm3
    vinserti128 ymm0, ymm0, xmm1, 1

    ; Group 0: bytes 0-7
    vpmovzxbd ymm1, xmm0
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3                ; dequant = eff_scale * nibble + eff_min
    vmovaps   ymm2, [%6]                    ; O[0:7]
    vfmadd231ps ymm2, %5, ymm1              ; O += prob * dequant
    vmovaps   [%6], ymm2

    ; Group 1: bytes 8-15
    vextracti128 xmm1, ymm0, 0
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 32]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 32], ymm2

    ; Group 2: bytes 16-23
    vextracti128 xmm1, ymm0, 1
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 64]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 64], ymm2

    ; Group 3: bytes 24-31
    vextracti128 xmm1, ymm0, 1
    vpsrldq   xmm1, xmm1, 8
    vpmovzxbd ymm1, xmm1
    vcvtdq2ps ymm1, ymm1
    vfmadd132ps ymm1, %4, %3
    vmovaps   ymm2, [%6 + 96]
    vfmadd231ps ymm2, %5, ymm1
    vmovaps   [%6 + 96], ymm2
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q4_k_block))
;
; %1 = k_block_ptr (register, points to Q4_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_ymm (ymm register to accumulate dot product into)
;
; Dequantization + dot product for all 8 sub-blocks:
;   1. Load d (fp16 -> f32) and dmin (fp16 -> f32, negated)
;   2. For each sub-block j=0..7:
;      a. Extract 6-bit scale (sc) and min (m) via get_scale_min_k4
;      b. Compute eff_scale = d * sc, eff_min = -dmin * m
;      c. Unpack 32 nibbles, convert to f32
;      d. Apply: dequant = eff_scale * nibble + eff_min
;      e. Dot with Q, accumulate
;
; Clobbers: ymm0-ymm5, eax, ecx, edx
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

    ; Negate dmin: we want -dmin * m so that dequant = d*sc*nibble + (-dmin*m)
    vxorps    xmm2, xmm2, xmm2
    vsubss    xmm1, xmm2, xmm1              ; xmm1[0] = -dmin

    ; Keep d and -dmin in xmm scratch for reuse across sub-blocks
    ; xmm0[0] = d, xmm1[0] = -dmin

    ; Local accumulator for this block
    vxorps    ymm3, ymm3, ymm3              ; ymm3 = local acc

    ; ------ Sub-block 0 ------
    EXTRACT_SCALE_MIN %1, 0
    ; eax = sc, edx = m
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
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_k_block)
;
; %1 = v_block_ptr (register, points to Q4_K block)
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
