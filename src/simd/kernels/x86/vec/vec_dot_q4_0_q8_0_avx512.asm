;; =============================================================================
;; vec_dot_q4_0_q8_0_avx512.asm - Optimized Q4_0 × Q8_0 dot product kernel (4x Unroll)
;; =============================================================================

section .data
    align 64
    nibble_mask_512: times 64 db 0x0F
    eights_512:      times 64 db 8

section .text

%ifdef AVX512_ENABLED
global simd_vec_dot_q4_0_q8_0_avx512

;; Calling Convention (System V / Windows):
;;   rcx/rdi: n
;;   rdx/rsi: result
;;   r8/rdx:  vx (Q4_0)
;;   r9/rcx:  vy (Q8_0)

%ifdef WINDOWS
    %define ARG1   rcx
    %define ARG1_32 ecx
    %define ARG2   rdx
    %define ARG3   r8
    %define ARG4   r9
%else
    %define ARG1   rdi
    %define ARG1_32 edi
    %define ARG2   rsi
    %define ARG3   rdx
    %define ARG4   rcx
%endif

simd_vec_dot_q4_0_q8_0_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    sub     rsp, 168
    vmovdqu64 [rsp+0],   xmm6
    vmovdqu64 [rsp+16],  xmm7
    vmovdqu64 [rsp+32],  xmm8
    vmovdqu64 [rsp+48],  xmm9
    vmovdqu64 [rsp+64],  xmm10
    vmovdqu64 [rsp+80],  xmm11
    vmovdqu64 [rsp+96],  xmm12
    vmovdqu64 [rsp+112], xmm13
    vmovdqu64 [rsp+128], xmm14
    vmovdqu64 [rsp+144], xmm15
%endif

    ; Setup pointers and counts
    mov     r10d, ARG1_32
    shr     r10d, 5                 ; n / 32 = number of blocks
    test    r10d, r10d
    jz      .done_zero

    mov     r11, ARG3               ; vx
    mov     r12, ARG4               ; vy
    mov     r13, ARG2               ; result

    ; Load constants
    vmovdqa64 zmm30, [rel nibble_mask_512] ; Mask 0x0F
    vmovdqa64 zmm29, [rel eights_512]      ; Value 8 for correction
    
    vxorps    zmm31, zmm31, zmm31   ; Global Accumulator (Float)

    ; Check if enough blocks for 4x loop
    mov     r15d, r10d
    shr     r15d, 2                 ; n_blocks / 4
    jz      .check_remaining

    align 16
.loop_4x:
    ; ==========================================================================
    ; STEP 1: LOAD & PREPARE SCALES
    ; ==========================================================================
    ; Load Q4 Scales (Offsets: 0, 18, 36, 54)
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vpinsrw xmm0, xmm0, [r11 + 18], 1
    vpinsrw xmm0, xmm0, [r11 + 36], 2
    vpinsrw xmm0, xmm0, [r11 + 54], 3
    vcvtph2ps xmm0, xmm0            ; xmm0 = [S_x0, S_x1, S_x2, S_x3]

    ; Load Q8 Scales (Offsets: 0, 34, 68, 102)
    movzx   eax, word [r12]
    vmovd   xmm1, eax
    vpinsrw xmm1, xmm1, [r12 + 34], 1
    vpinsrw xmm1, xmm1, [r12 + 68], 2
    vpinsrw xmm1, xmm1, [r12 + 102], 3
    vcvtph2ps xmm1, xmm1            ; xmm1 = [S_y0, S_y1, S_y2, S_y3]

    ; Combined Scales
    vmulps  xmm0, xmm0, xmm1        ; xmm0 = [S0, S1, S2, S3]

    ; Broadcast to ZMM Lanes
    ; We need: Lane0=S0, Lane1=S1, Lane2=S2, Lane3=S3
    ; Shuffle to get S0, S1, S2, S3 into all positions of 4 XMMs
    vshufps xmm1, xmm0, xmm0, 0x00  ; xmm1 = [S0, S0, S0, S0]
    vshufps xmm2, xmm0, xmm0, 0x55  ; xmm2 = [S1, S1, S1, S1]
    vshufps xmm3, xmm0, xmm0, 0xAA  ; xmm3 = [S2, S2, S2, S2]
    vshufps xmm4, xmm0, xmm0, 0xFF  ; xmm4 = [S3, S3, S3, S3]

    ; Construct ZMM Scale
    vinsertf32x4 zmm2, zmm1, xmm2, 1 ; zmm2 = [S0..|S1..|...]
    vinsertf32x4 zmm2, zmm2, xmm3, 2 ; zmm2 = [S0..|S1..|S2..|...]
    vinsertf32x4 zmm2, zmm2, xmm4, 3 ; zmm2 = [S0..|S1..|S2..|S3..]
    
    ; zmm2 is now the Broadcasted Scale Vector.

    ; ==========================================================================
    ; STEP 2: LOAD Q4 & UNPACK
    ; ==========================================================================
    ; Load Q4 Quants (16 bytes at offset 2 per block)
    ; Offsets: 2, 20, 38, 56
    vmovdqu      xmm3, [r11 + 2]
    vinserti32x4 zmm3, zmm3, [r11 + 20], 1
    vinserti32x4 zmm3, zmm3, [r11 + 38], 2
    vinserti32x4 zmm3, zmm3, [r11 + 56], 3
    ; zmm3 = Packed Q4 [Blk0 | Blk1 | Blk2 | Blk3]

    ; Unpack Nibbles
    vpandd      zmm4, zmm3, zmm30      ; Low nibbles (Even elements)
    vpsrlw      zmm5, zmm3, 4
    vpandd      zmm5, zmm5, zmm30      ; High nibbles (Odd elements)

    ; Interleave to get bytes
    vpunpcklbw  zmm6, zmm4, zmm5       ; Q4 Low Half (Elements 0-15 per block)
    vpunpckhbw  zmm7, zmm4, zmm5       ; Q4 High Half (Elements 16-31 per block)

    ; ==========================================================================
    ; STEP 3: LOAD Q8 & COMPUTE
    ; ==========================================================================
    ; Load Q8 Low Half (Bytes 0-15 per block)
    ; Offsets: 2, 36, 70, 104 (Base + 0)
    vmovdqu      xmm8, [r12 + 2]
    vinserti32x4 zmm8, zmm8, [r12 + 36], 1
    vinserti32x4 zmm8, zmm8, [r12 + 70], 2
    vinserti32x4 zmm8, zmm8, [r12 + 104], 3
    ; zmm8 = Q8 Low Half

    ; Load Q8 High Half (Bytes 16-31 per block)
    ; Offsets: 18, 52, 86, 120 (Base + 16)
    vmovdqu      xmm9, [r12 + 18]
    vinserti32x4 zmm9, zmm9, [r12 + 52], 1
    vinserti32x4 zmm9, zmm9, [r12 + 86], 2
    vinserti32x4 zmm9, zmm9, [r12 + 120], 3
    ; zmm9 = Q8 High Half

    ; Prepare Integer Accumulators (Zero them)
    vxorps      zmm10, zmm10, zmm10    ; Main Accumulator
    vxorps      zmm11, zmm11, zmm11    ; Offset Accumulator

    ; Dot Product: Q4 * Q8
    vpdpbusd    zmm10, zmm6, zmm8      ; Acc += Q4_Lo * Q8_Lo
    vpdpbusd    zmm10, zmm7, zmm9      ; Acc += Q4_Hi * Q8_Hi

    ; Correction: 8 * Q8 (Since Q4 was offset by +8)
    ; We need to SUBTRACT 8 * Q8 from the result.
    ; Calcutate 8 * Q8 first.
    vpdpbusd    zmm11, zmm29, zmm8     ; Off += 8 * Q8_Lo
    vpdpbusd    zmm11, zmm29, zmm9     ; Off += 8 * Q8_Hi

    ; Subtract Correction
    vpsubd      zmm10, zmm10, zmm11    ; Acc = Acc - Off

    ; Convert to Float and Scale
    vcvtdq2ps   zmm10, zmm10           ; Convert int32 to float
    vfmadd231ps zmm31, zmm10, zmm2     ; Global += Acc * Scale

    ; Advance Pointers
    add     r11, 72  ; 18 * 4
    add     r12, 136 ; 34 * 4
    dec     r15d
    jnz     .loop_4x

.check_remaining:
    ; Handle remaining blocks (1-3)
    ; Using the same logic but masking or fallback?
    ; Fallback to single block loop is easiest and safest.
    mov     r15d, r10d
    and     r15d, 3
    jz      .reduce

.loop_1x:
    ; Process 1 block
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm0, xmm0, xmm1
    vbroadcastss zmm2, xmm0         ; Scale

    ; Load Q4
    vmovdqu xmm3, [r11 + 2]         ; 16 bytes
    ; Expand to ZMM just to use the same registers/logic?
    ; We can use XMM/YMM instructions but standardizing on ZMM is fine with AVX512
    vpandd  xmm4, xmm3, xmm30       ; Low
    vpsrlw  xmm5, xmm3, 4
    vpandd  xmm5, xmm5, xmm30       ; High
    vpunpcklbw xmm6, xmm4, xmm5     ; Lo Half (16 bytes) -> Actually xmm6 is 128 bit
    vpunpckhbw xmm7, xmm4, xmm5     ; Hi Half (16 bytes)

    ; Load Q8
    vmovdqu xmm8, [r12 + 2]         ; Lo 16 bytes
    vmovdqu xmm9, [r12 + 18]        ; Hi 16 bytes

    vxorps  xmm10, xmm10, xmm10
    vxorps  xmm11, xmm11, xmm11

    vpdpbusd xmm10, xmm6, xmm8
    vpdpbusd xmm10, xmm7, xmm9
    
    vpdpbusd xmm11, xmm29, xmm8     ; zmm29 is 512, xmm29 is 128 subset
    vpdpbusd xmm11, xmm29, xmm9
    
    vpsubd   xmm10, xmm10, xmm11
    
    vcvtdq2ps xmm10, xmm10
    vfmadd231ps xmm31, xmm10, xmm2  ; Accumulate to ZMM31 (using low part)

    add     r11, 18
    add     r12, 34
    dec     r15d
    jnz     .loop_1x

.reduce:
    ; ZMM31 contains partial float sums.
    ; [Lane0 | Lane1 | Lane2 | Lane3]
    ; Each Lane contains 4 floats.
    ; We need to sum EVERYTHING.
    
    vextractf64x4 ymm0, zmm31, 1
    vaddps  ymm0, ymm0, ymm31       ; Reduce ZMM to YMM
    
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1        ; Reduce YMM to XMM
    
    ; Now xmm0 has 4 floats to sum
    vpermilps xmm1, xmm0, 1         ; Swap adjacents
    vaddps  xmm0, xmm0, xmm1
    vpermilps xmm1, xmm0, 2         ; Swap pairs
    vaddss  xmm0, xmm0, xmm1
    
    vmovss  [r13], xmm0

.epilogue:
    vzeroupper
%ifdef WINDOWS
    vmovdqu64 xmm6,  [rsp+0]
    vmovdqu64 xmm7,  [rsp+16]
    vmovdqu64 xmm8,  [rsp+32]
    vmovdqu64 xmm9,  [rsp+48]
    vmovdqu64 xmm10, [rsp+64]
    vmovdqu64 xmm11, [rsp+80]
    vmovdqu64 xmm12, [rsp+96]
    vmovdqu64 xmm13, [rsp+112]
    vmovdqu64 xmm14, [rsp+128]
    vmovdqu64 xmm15, [rsp+144]
    add     rsp, 168
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

.done_zero:
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0
    jmp     .epilogue

%endif
