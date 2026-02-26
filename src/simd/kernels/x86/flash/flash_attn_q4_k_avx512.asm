; ============================================================================
; Flash Attention 2 - Q4_K AVX-512 Kernel
; ============================================================================
; Quantized K/V with Q4_K format (144 bytes/block, 256 values/block)
; Defines dequantization macros, then includes the shared FA2 AVX-512 skeleton.
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
;
; AVX-512 processes 2 groups of 16 instead of AVX2's 4 groups of 8 per sub-block.
; ============================================================================

; --- Kernel identity ---
%define KERNEL_NAME         simd_flash_attn_q4_k_avx512
%define K_BLOCK_BYTES       144
%define V_BLOCK_BYTES       144
%define K_BLOCK_VALUES      256
%define V_BLOCK_VALUES      256

; ============================================================================
; Quant-specific constants
; ============================================================================
%macro DECLARE_QUANT_CONSTANTS 0
align 64
fa2_q4k_nibble_mask: times 16 db 0x0F      ; 16 bytes for xmm ops
                     times 48 db 0          ; pad to 64 bytes
%endmacro

; ============================================================================
; Initialize quant constant registers
; zmm11 = nibble_mask (only xmm11 used — low 16 bytes)
; zmm12 = (unused)
; zmm13 = (unused)
; ============================================================================
%macro INIT_QUANT_REGS 0
    vmovdqu   xmm11, [rel fa2_q4k_nibble_mask]
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
; Helper: Dequant + process one sub-block of 32 values (AVX-512)
;
; Extracts 32 nibble values from 16 packed bytes, converts to f32,
; applies: result = eff_scale * nibble + eff_min
; Then dots with Q values and accumulates.
;
; Uses 2 groups of 16 (AVX-512 vpmovsxbd zmm, xmm processes 16 at a time).
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = zmm register with broadcast eff_scale (d * sc)
; %4 = zmm register with broadcast eff_min (negated: -dmin * m)
; %5 = q_ptr register (points to Q f32 values for this sub-block)
; %6 = acc_zmm (zmm to accumulate dot product)
;
; Clobbers: zmm0-zmm4
; ============================================================================
%macro DEQUANT_SUBBLOCK_DOT 6
    ; Load 16 packed bytes (32 nibbles) for sub-block j
    vmovdqu   xmm1, [%1 + 16 + %2 * 16]

    ; Extract low nibbles (even-indexed values: q[0], q[2], q[4]...)
    vpand     xmm2, xmm1, xmm11             ; low nibbles & 0x0F

    ; Extract high nibbles (odd-indexed values: q[1], q[3], q[5]...)
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11             ; high nibbles >> 4 & 0x0F

    ; Group 0: low nibbles -> 16 values (0, 2, 4, ..., 30)
    ; We need sequential order, so interleave low and high
    ; But AVX-512 can process 16 at once, so we use the interleaved approach
    ; Interleave to get sequential order in two xmm halves
    vpunpcklbw xmm0, xmm2, xmm3             ; bytes 0-15 interleaved
    vpunpckhbw xmm4, xmm2, xmm3             ; bytes 16-31 interleaved

    ; Group 0: first 16 values (bytes 0-15)
    vpmovzxbd zmm1, xmm0                    ; zero-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1                    ; -> 16 f32 (unsigned nibble values)
    vfmadd132ps zmm1, %4, %3                ; zmm1 = eff_scale * nibble + eff_min
    vmovups   zmm2, [%5]                    ; Q[0:15]
    vfmadd231ps %6, zmm1, zmm2              ; acc += dequant * Q

    ; Group 1: next 16 values (bytes 16-31)
    vpmovzxbd zmm1, xmm4                    ; zero-extend 16 bytes -> 16 int32
    vcvtdq2ps zmm1, zmm1
    vfmadd132ps zmm1, %4, %3
    vmovups   zmm2, [%5 + 64]              ; Q[16:31]
    vfmadd231ps %6, zmm1, zmm2
%endmacro

; ============================================================================
; Helper: Dequant + accumulate one sub-block for V path (AVX-512)
;
; %1 = block_ptr register
; %2 = sub-block index j (immediate 0-7)
; %3 = zmm register with broadcast eff_scale (d * sc)
; %4 = zmm register with broadcast eff_min (negated: -dmin * m)
; %5 = prob_zmm (broadcast probability)
; %6 = o_ptr register (points to O accumulator for this sub-block)
;
; Clobbers: zmm0-zmm4
; ============================================================================
%macro DEQUANT_SUBBLOCK_V_ACCUM 6
    ; Load 16 packed bytes (32 nibbles)
    vmovdqu   xmm1, [%1 + 16 + %2 * 16]

    ; Extract nibbles
    vpand     xmm2, xmm1, xmm11
    vpsrlw    xmm3, xmm1, 4
    vpand     xmm3, xmm3, xmm11

    ; Interleave for sequential order
    vpunpcklbw xmm0, xmm2, xmm3
    vpunpckhbw xmm4, xmm2, xmm3

    ; Group 0: first 16 values
    vpmovzxbd zmm1, xmm0
    vcvtdq2ps zmm1, zmm1
    vfmadd132ps zmm1, %4, %3                ; dequant = eff_scale * nibble + eff_min
    vmovaps   zmm2, [%6]                    ; O[0:15]
    vfmadd231ps zmm2, %5, zmm1              ; O += prob * dequant
    vmovaps   [%6], zmm2

    ; Group 1: next 16 values
    vpmovzxbd zmm1, xmm4
    vcvtdq2ps zmm1, zmm1
    vfmadd132ps zmm1, %4, %3
    vmovaps   zmm2, [%6 + 64]              ; O[16:31]
    vfmadd231ps zmm2, %5, zmm1
    vmovaps   [%6 + 64], zmm2
%endmacro

; ============================================================================
; DEQUANT_K_DOT_BLOCK: dot(Q_f32[0:255], dequant(K_q4_k_block))
;
; %1 = k_block_ptr (register, points to Q4_K block start)
; %2 = q_ptr (register, points to 256 F32 values in Q)
; %3 = acc_zmm (zmm register to accumulate dot product into)
;
; Dequantization + dot product for all 8 sub-blocks:
;   1. Load d (fp16 -> f32) and dmin (fp16 -> f32, negated)
;   2. For each sub-block j=0..7:
;      a. Extract 6-bit scale (sc) and min (m)
;      b. Compute eff_scale = d * sc, eff_min = -dmin * m (broadcast to zmm)
;      c. Unpack 32 nibbles, convert to f32
;      d. Apply: dequant = eff_scale * nibble + eff_min
;      e. Dot with Q, accumulate
;
; Clobbers: zmm0-zmm5, eax, ecx, edx
; Preserves: zmm11-zmm13, zmm30-zmm31, r12-r15, rbx
; ============================================================================
%macro DEQUANT_K_DOT_BLOCK 3
    ; 1. Load super-block scale d (fp16 -> f32)
    movzx     eax, word [%1]
    vmovd     xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; xmm0[0] = d (f32)

    ; 2. Load super-block min dmin (fp16 -> f32), negate
    movzx     eax, word [%1 + 2]
    vmovd     xmm1, eax
    vcvtph2ps xmm1, xmm1                    ; xmm1[0] = dmin (f32)

    ; Negate dmin: we want -dmin * m so that dequant = d*sc*nibble + (-dmin*m)
    vxorps    xmm2, xmm2, xmm2
    vsubss    xmm1, xmm2, xmm1              ; xmm1[0] = -dmin

    ; Keep d and -dmin in xmm scratch for reuse across sub-blocks
    ; xmm0[0] = d, xmm1[0] = -dmin

    ; Local accumulator for this block
    vxorps    zmm3, zmm3, zmm3              ; zmm3 = local acc

    ; ------ Sub-block 0 ------
    EXTRACT_SCALE_MIN %1, 0
    ; eax = sc, edx = m
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4                    ; float(sc) broadcast
    vbroadcastss zmm5, xmm0                 ; broadcast d
    vmulps    zmm4, zmm4, zmm5              ; zmm4 = d * sc (eff_scale)

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5                    ; float(m) broadcast
    vbroadcastss zmm2, xmm1                 ; broadcast -dmin
    vmulps    zmm5, zmm5, zmm2              ; zmm5 = -dmin * m (eff_min)

    DEQUANT_SUBBLOCK_DOT %1, 0, zmm4, zmm5, %2, zmm3

    ; ------ Sub-block 1 ------
    EXTRACT_SCALE_MIN %1, 1
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 1, zmm4, zmm5, %2 + 128, zmm3

    ; ------ Sub-block 2 ------
    EXTRACT_SCALE_MIN %1, 2
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 2, zmm4, zmm5, %2 + 256, zmm3

    ; ------ Sub-block 3 ------
    EXTRACT_SCALE_MIN %1, 3
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 3, zmm4, zmm5, %2 + 384, zmm3

    ; ------ Sub-block 4 ------
    EXTRACT_SCALE_MIN %1, 4
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 4, zmm4, zmm5, %2 + 512, zmm3

    ; ------ Sub-block 5 ------
    EXTRACT_SCALE_MIN %1, 5
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 5, zmm4, zmm5, %2 + 640, zmm3

    ; ------ Sub-block 6 ------
    EXTRACT_SCALE_MIN %1, 6
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 6, zmm4, zmm5, %2 + 768, zmm3

    ; ------ Sub-block 7 ------
    EXTRACT_SCALE_MIN %1, 7
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_DOT %1, 7, zmm4, zmm5, %2 + 896, zmm3

    ; Add local accumulator to output accumulator
    vaddps    %3, %3, zmm3
%endmacro

; ============================================================================
; DEQUANT_V_ACCUM_BLOCK: O[block] += prob * dequant(V_q4_k_block)
;
; %1 = v_block_ptr (register, points to Q4_K block)
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
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5              ; eff_scale = d * sc

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2              ; eff_min = -dmin * m

    DEQUANT_SUBBLOCK_V_ACCUM %1, 0, zmm4, zmm5, %2, rax

    ; ------ Sub-block 1 ------
    EXTRACT_SCALE_MIN %1, 1
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 1, zmm4, zmm5, %2, rax + 128

    ; ------ Sub-block 2 ------
    EXTRACT_SCALE_MIN %1, 2
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 2, zmm4, zmm5, %2, rax + 256

    ; ------ Sub-block 3 ------
    EXTRACT_SCALE_MIN %1, 3
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 3, zmm4, zmm5, %2, rax + 384

    ; ------ Sub-block 4 ------
    EXTRACT_SCALE_MIN %1, 4
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 4, zmm4, zmm5, %2, rax + 512

    ; ------ Sub-block 5 ------
    EXTRACT_SCALE_MIN %1, 5
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 5, zmm4, zmm5, %2, rax + 640

    ; ------ Sub-block 6 ------
    EXTRACT_SCALE_MIN %1, 6
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 6, zmm4, zmm5, %2, rax + 768

    ; ------ Sub-block 7 ------
    EXTRACT_SCALE_MIN %1, 7
    vmovd     xmm4, eax
    vbroadcastss zmm4, xmm4
    vcvtdq2ps zmm4, zmm4
    vbroadcastss zmm5, xmm0
    vmulps    zmm4, zmm4, zmm5

    vmovd     xmm5, edx
    vbroadcastss zmm5, xmm5
    vcvtdq2ps zmm5, zmm5
    vbroadcastss zmm2, xmm1
    vmulps    zmm5, zmm5, zmm2

    DEQUANT_SUBBLOCK_V_ACCUM %1, 7, zmm4, zmm5, %2, rax + 896
%endmacro

; ============================================================================
; Include the shared FA2 AVX-512 skeleton
; ============================================================================
%include "flash_attn_fa2_skeleton_avx512.inc"
