;; =============================================================================
;; vec_dot_q8_0_q8_0_avx512.asm - Q8_0 × Q8_0 (VNNI + Vectorized Scales)
;; =============================================================================
;; OPTIMIZATIONS:
;; 1. VNNI (vpdpbusd): Fuses u8*i8 multiply-accumulate, replacing
;;    vpabsb+vpcmpb+vpblendmb+vpmaddubsw+vpmaddwd chain.
;;    Sign trick: x_u8 = x_i8 ^ 0x80; correction = vpdpbusd(0x80_vec, y)
;; 2. Vectorized Scales: Pack 4 FP16 scales via vpinsrw, convert with
;;    vcvtph2ps, broadcast with vpermps. Reduces ~32 scalar instructions to ~14.
;; 3. ZMM16+ for constants: Avoids Windows ABI save overhead.
;; =============================================================================

section .data
    align 64
    ones_16_64:      times 32 dw 1
    offset_128_64:   times 64 db 0x80
    scale_perm_01:   dd 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
    scale_perm_23:   dd 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3

section .text

%ifdef WINDOWS
    %define ARG1    rcx
    %define ARG1_32 ecx
    %define ARG2    rdx
    %define ARG3    r8
    %define ARG4    r9
%else
    %define ARG1    rdi
    %define ARG1_32 edi
    %define ARG2    rsi
    %define ARG3    rdx
    %define ARG4    rcx
%endif

%define BLOCK_SIZE 34

global simd_vec_dot_q8_0_q8_0_avx512

simd_vec_dot_q8_0_q8_0_avx512:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    sub     rsp, 168
    vmovups [rsp+0],   xmm6
    vmovups [rsp+16],  xmm7
    vmovups [rsp+32],  xmm8
    vmovups [rsp+48],  xmm9
    vmovups [rsp+64],  xmm10
    vmovups [rsp+80],  xmm11
    vmovups [rsp+96],  xmm12
    vmovups [rsp+112], xmm13
    vmovups [rsp+128], xmm14
    vmovups [rsp+144], xmm15
%endif

    mov     r10d, ARG1_32
    shr     r10d, 7              ; n / 128 (4 blocks per iter)
    mov     r11, ARG3            ; vx
    mov     r12, ARG4            ; vy
    mov     r13, ARG2            ; result

    ; Load constants into ZMM16+ (no ABI save needed)
    vmovdqa64 zmm16, [rel scale_perm_01]
    vmovdqa64 zmm17, [rel scale_perm_23]
    vmovdqa64 zmm18, [rel offset_128_64]
    vmovdqa64 zmm14, [rel ones_16_64]   ; For remainder loop
    vxorps    zmm15, zmm15, zmm15        ; Float accumulator

    test    r10d, r10d
    jz      .check_remainder

    align 64
.main_loop:
    prefetcht0 [r11 + BLOCK_SIZE*8]
    prefetcht0 [r12 + BLOCK_SIZE*8]

    ; === LOAD X (4 blocks) ===
    vmovdqu     ymm0, [r11 + 2]
    vmovdqu     ymm1, [r11 + BLOCK_SIZE + 2]
    vinserti64x4 zmm0, zmm0, ymm1, 1        ; ZMM0 = [B1 | B0]

    vmovdqu     ymm1, [r11 + BLOCK_SIZE*2 + 2]
    vmovdqu     ymm2, [r11 + BLOCK_SIZE*3 + 2]
    vinserti64x4 zmm1, zmm1, ymm2, 1        ; ZMM1 = [B3 | B2]

    ; === LOAD Y (4 blocks) ===
    vmovdqu     ymm2, [r12 + 2]
    vmovdqu     ymm3, [r12 + BLOCK_SIZE + 2]
    vinserti64x4 zmm2, zmm2, ymm3, 1        ; ZMM2 = [B1 | B0]

    vmovdqu     ymm3, [r12 + BLOCK_SIZE*2 + 2]
    vmovdqu     ymm4, [r12 + BLOCK_SIZE*3 + 2]
    vinserti64x4 zmm3, zmm3, ymm4, 1        ; ZMM3 = [B3 | B2]

    ; === VNNI COMPUTE B0/B1 ===
    vpxord      zmm4, zmm0, zmm18           ; x_u8 = x ^ 0x80
    vpxord      zmm5, zmm5, zmm5            ; zero acc
    vpdpbusd    zmm5, zmm4, zmm2            ; dot(x_biased, y)
    vpxord      zmm4, zmm4, zmm4            ; zero corr
    vpdpbusd    zmm4, zmm18, zmm2           ; 128 * sum(y)
    vpsubd      zmm5, zmm5, zmm4            ; corrected result
    vcvtdq2ps   zmm5, zmm5                  ; -> float

    ; === VNNI COMPUTE B2/B3 ===
    vpxord      zmm4, zmm1, zmm18           ; x_u8 = x ^ 0x80
    vpxord      zmm0, zmm0, zmm0            ; zero acc (reuse zmm0)
    vpdpbusd    zmm0, zmm4, zmm3            ; dot(x_biased, y)
    vpxord      zmm4, zmm4, zmm4            ; zero corr
    vpdpbusd    zmm4, zmm18, zmm3           ; 128 * sum(y)
    vpsubd      zmm0, zmm0, zmm4            ; corrected
    vcvtdq2ps   zmm0, zmm0                  ; -> float

    ; === VECTORIZED SCALE LOADING ===
    ; Pack 4 x-scales into xmm8
    movzx       eax, word [r11]
    vmovd       xmm8, eax
    vpinsrw     xmm8, xmm8, [r11 + BLOCK_SIZE], 1
    vpinsrw     xmm8, xmm8, [r11 + BLOCK_SIZE*2], 2
    vpinsrw     xmm8, xmm8, [r11 + BLOCK_SIZE*3], 3
    vcvtph2ps   xmm8, xmm8                  ; 4 x f32

    ; Pack 4 y-scales into xmm9
    movzx       eax, word [r12]
    vmovd       xmm9, eax
    vpinsrw     xmm9, xmm9, [r12 + BLOCK_SIZE], 1
    vpinsrw     xmm9, xmm9, [r12 + BLOCK_SIZE*2], 2
    vpinsrw     xmm9, xmm9, [r12 + BLOCK_SIZE*3], 3
    vcvtph2ps   xmm9, xmm9                  ; 4 x f32

    ; Multiply and broadcast
    vmulps      xmm8, xmm8, xmm9            ; 4 combined scales
    vpermps     zmm9, zmm16, zmm8            ; [S1... | S0...]
    vfmadd231ps zmm15, zmm5, zmm9            ; Accumulate B0/B1
    vpermps     zmm9, zmm17, zmm8            ; [S3... | S2...]
    vfmadd231ps zmm15, zmm0, zmm9            ; Accumulate B2/B3

    add     r11, BLOCK_SIZE * 4
    add     r12, BLOCK_SIZE * 4
    dec     r10d
    jnz     .main_loop

.check_remainder:
    ; Reduce ZMM15 -> YMM15
    vextractf32x8 ymm0, zmm15, 1
    vaddps      ymm15, ymm15, ymm0

    mov     r10d, ARG1_32
    and     r10d, 127
    shr     r10d, 5
    jz      .horizontal_sum

.remainder_loop:
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    vbroadcastss ymm2, xmm2

    vmovdqu ymm6, [r11 + 2]
    vmovdqu ymm8, [r12 + 2]

    vpsignb ymm9, ymm6, ymm6
    vpsignb ymm10, ymm8, ymm6
    vpmaddubsw ymm9, ymm9, ymm10
    vpmaddwd   ymm9, ymm9, ymm14
    vcvtdq2ps  ymm9, ymm9

    vfmadd231ps ymm15, ymm9, ymm2

    add     r11, BLOCK_SIZE
    add     r12, BLOCK_SIZE
    dec     r10d
    jnz     .remainder_loop

.horizontal_sum:
    vextractf128 xmm1, ymm15, 1
    vaddps    xmm0, xmm15, xmm1
    vhaddps   xmm0, xmm0, xmm0
    vhaddps   xmm0, xmm0, xmm0
    vmovss    [r13], xmm0

.epilogue:
    vzeroupper
%ifdef WINDOWS
    vmovups xmm6,  [rsp+0]
    vmovups xmm7,  [rsp+16]
    vmovups xmm8,  [rsp+32]
    vmovups xmm9,  [rsp+48]
    vmovups xmm10, [rsp+64]
    vmovups xmm11, [rsp+80]
    vmovups xmm12, [rsp+96]
    vmovups xmm13, [rsp+112]
    vmovups xmm14, [rsp+128]
    vmovups xmm15, [rsp+144]
    add     rsp, 168
%endif
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
