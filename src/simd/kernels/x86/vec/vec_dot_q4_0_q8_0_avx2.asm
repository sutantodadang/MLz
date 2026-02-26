;; =============================================================================
;; vec_dot_q4_0_q8_0_avx2.asm - Hand-tuned Q4_0 × Q8_0 dot product kernel
;; =============================================================================
;; 
;; ALGORITHM:
;; For each block pair (32 values):
;;   1. Load 16 bytes from Q4_0 (32 packed 4-bit values)
;;   2. Unpack nibbles to bytes, subtract 8 (unsigned→signed conversion)
;;   3. Load 32 bytes from Q8_0 (32 signed int8 values)
;;   4. Multiply pairs using vpmaddubsw (unsigned * signed → 16-bit)
;;   5. Sum pairs using vpmaddwd (16-bit → 32-bit)
;;   6. Accumulate and scale by (d0 * d1)
;;
;; BLOCK STRUCTURES:
;;   block_q4_0: 2 bytes (d: fp16) + 16 bytes (qs: 32 nibbles) = 18 bytes
;;   block_q8_0: 2 bytes (d: fp16) + 32 bytes (qs: 32 int8)    = 34 bytes
;;
;; CALLING CONVENTION (System V AMD64 / Windows x64 compatible):
;;   void simd_vec_dot_q4_0_q8_0_avx2(
;;       int n,                  ; rdi/rcx - number of elements (must be multiple of 32)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q4_0 blocks
;;       const void* vy          ; rcx/r9  - Q8_0 blocks
;;   );
;;
;; =============================================================================

section .data
    align 32
    ; Low nibble mask: 0x0F repeated 32 times
    nibble_mask:  times 32 db 0x0F
    
    ; Bias for unsigned→signed conversion: 8 repeated 32 times
    q4_bias:      times 32 db 8
    
    ; Ones for horizontal sum: 1 repeated 16 times (16-bit)
    ones_16:      times 16 dw 1

section .text

;; -----------------------------------------------------------------------------
;; Windows x64 calling convention adaptation
;; -----------------------------------------------------------------------------
%ifdef WINDOWS
    %define ARG1   rcx    ; n (full 64-bit)
    %define ARG1_32 ecx   ; n (32-bit version)
    %define ARG2   rdx    ; result
    %define ARG3   r8     ; vx (Q4_0)
    %define ARG4   r9     ; vy (Q8_0)
%else
    ; System V AMD64
    %define ARG1   rdi    ; n (full 64-bit)
    %define ARG1_32 edi   ; n (32-bit version)
    %define ARG2   rsi    ; result
    %define ARG3   rdx    ; vx (Q4_0)
    %define ARG4   rcx    ; vy (Q8_0)
%endif

;; Block sizes
%define Q4_0_BLOCK_SIZE 18   ; 2 + 16 bytes
%define Q8_0_BLOCK_SIZE 34   ; 2 + 32 bytes
%define QK              32   ; Elements per block

global simd_vec_dot_q4_0_q8_0_avx2

simd_vec_dot_q4_0_q8_0_avx2:
    ; Prologue - save callee-saved registers
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WINDOWS
    ; Windows requires shadow space + save XMM6-15
    sub     rsp, 168
    vmovdqu [rsp+0],   xmm6
    vmovdqu [rsp+16],  xmm7
    vmovdqu [rsp+32],  xmm8
    vmovdqu [rsp+48],  xmm9
    vmovdqu [rsp+64],  xmm10
    vmovdqu [rsp+80],  xmm11
    vmovdqu [rsp+96],  xmm12
    vmovdqu [rsp+112], xmm13
    vmovdqu [rsp+128], xmm14
    vmovdqu [rsp+144], xmm15
%endif

    ; Calculate number of blocks: nb = n / 32
    mov     r10d, ARG1_32                   ; n (32-bit)
    shr     r10d, 5                         ; n / 32 = number of blocks
    test    r10d, r10d
    jz      .done_zero                      ; No blocks to process
    
    ; Setup pointers
    mov     r11, ARG3                       ; vx (Q4_0 blocks)
    mov     r12, ARG4                       ; vy (Q8_0 blocks)
    mov     r13, ARG2                       ; result pointer
    
    ; Load constants
    vmovdqa ymm14, [rel nibble_mask]        ; 0x0F mask for nibble extraction
    vmovdqa ymm13, [rel q4_bias]            ; Bias (8) for unsigned→signed
    vmovdqa ymm12, [rel ones_16]            ; Ones for horizontal sum
    
    ; Zero accumulator
    vxorps  ymm15, ymm15, ymm15             ; Float accumulator for final sum
    
    ; Initialize block counter
    xor     r14d, r14d                      ; Block index
    
    ; Calculate unrolled loop count (process 2 blocks per iteration)
    mov     r15d, r10d
    shr     r15d, 1                         ; Pairs of blocks
    test    r15d, r15d
    jz      .single_block_loop

;; -----------------------------------------------------------------------------
;; Main loop: Process 2 blocks per iteration (unrolled)
;; -----------------------------------------------------------------------------
align 16
.main_loop:
    ; Prefetch next blocks
    prefetcht0 [r11 + Q4_0_BLOCK_SIZE*4]
    prefetcht0 [r12 + Q8_0_BLOCK_SIZE*4]
    
    ;; ========== BLOCK 0 ==========
    
    ; Load Q4_0 scale (fp16) and convert to fp32
    movzx   eax, word [r11]                 ; Load fp16 d0
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0                    ; Convert to fp32
    
    ; Load Q8_0 scale (fp16) and convert to fp32
    movzx   ebx, word [r12]                 ; Load fp16 d1  
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1                    ; Convert to fp32
    
    ; Compute scale product: d0 * d1
    vmulss  xmm2, xmm0, xmm1
    
    ; Load Q4_0 packed nibbles (16 bytes → 32 4-bit values)
    vmovdqu xmm3, [r11 + 2]                 ; Load 16 bytes (offset past scale)
    
    ; Unpack lower nibbles (first 16 values)
    vpand   xmm4, xmm3, xmm14               ; Low nibbles: qs[i] & 0x0F
    
    ; Unpack upper nibbles (next 16 values)  
    vpsrlw  xmm5, xmm3, 4                   ; Shift right by 4
    vpand   xmm5, xmm5, xmm14               ; High nibbles: (qs[i] >> 4) & 0x0F
    
    ; Combine into 32 bytes (interleaved order matches Q8_0 layout)
    vpunpcklbw xmm6, xmm4, xmm5             ; Interleave low bytes
    vpunpckhbw xmm7, xmm4, xmm5             ; Interleave high bytes
    vinserti128 ymm6, ymm6, xmm7, 1         ; Combine to 256-bit
    
    ; Convert unsigned [0,15] to signed [-8,7] by subtracting 8
    vpsubb  ymm6, ymm6, ymm13               ; q4_vals -= 8
    
    ; Load Q8_0 values (32 signed int8)
    vmovdqu ymm7, [r12 + 2]                 ; Load 32 bytes (offset past scale)
    
    ; Compute dot product using vpmaddubsw + vpmaddwd
    ; vpmaddubsw: unsigned * signed → 16-bit (takes abs of first, signs of second)
    ; We need signed * signed, so use the sign trick:
    ;   Get absolute values of q4, sign-adjust q8
    vpsignb ymm8, ymm6, ymm6                ; abs(q4_vals)
    vpsignb ymm9, ymm7, ymm6                ; sign_adjust(q8_vals, q4_vals)
    
    ; Multiply unsigned(abs_q4) * signed(adjusted_q8) → 16-bit sums
    vpmaddubsw ymm8, ymm8, ymm9             ; 32x int8 → 16x int16
    
    ; Sum pairs of int16 to int32
    vpmaddwd ymm8, ymm8, ymm12              ; 16x int16 → 8x int32
    
    ; Convert to float and accumulate
    vcvtdq2ps ymm8, ymm8                    ; 8x int32 → 8x float
    vbroadcastss ymm2, xmm2                 ; Broadcast scale product
    vfmadd231ps ymm15, ymm8, ymm2           ; acc += dot * scale
    
    ;; ========== BLOCK 1 ==========
    
    ; Advance pointers to next block
    add     r11, Q4_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    
    ; Load Q4_0 scale (fp16) and convert to fp32
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    
    ; Load Q8_0 scale (fp16) and convert to fp32
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    
    ; Compute scale product
    vmulss  xmm2, xmm0, xmm1
    
    ; Load and unpack Q4_0 nibbles
    vmovdqu xmm3, [r11 + 2]
    vpand   xmm4, xmm3, xmm14
    vpsrlw  xmm5, xmm3, 4
    vpand   xmm5, xmm5, xmm14
    vpunpcklbw xmm6, xmm4, xmm5
    vpunpckhbw xmm7, xmm4, xmm5
    vinserti128 ymm6, ymm6, xmm7, 1
    vpsubb  ymm6, ymm6, ymm13
    
    ; Load Q8_0 values
    vmovdqu ymm7, [r12 + 2]
    
    ; Signed multiplication trick
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    
    ; Accumulate
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    ; Advance pointers and loop
    add     r11, Q4_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    add     r14d, 2
    
    dec     r15d
    jnz     .main_loop

;; -----------------------------------------------------------------------------
;; Handle remaining single block if odd number of blocks
;; -----------------------------------------------------------------------------
.single_block_loop:
    ; Check if there's a remaining block
    test    r10d, 1
    jz      .horizontal_sum
    
    ; Process single remaining block (same as block 0 above)
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu xmm3, [r11 + 2]
    vpand   xmm4, xmm3, xmm14
    vpsrlw  xmm5, xmm3, 4
    vpand   xmm5, xmm5, xmm14
    vpunpcklbw xmm6, xmm4, xmm5
    vpunpckhbw xmm7, xmm4, xmm5
    vinserti128 ymm6, ymm6, xmm7, 1
    vpsubb  ymm6, ymm6, ymm13
    
    vmovdqu ymm7, [r12 + 2]
    
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2

;; -----------------------------------------------------------------------------
;; Horizontal sum of ymm15 (8 floats) → single float result
;; -----------------------------------------------------------------------------
.horizontal_sum:
    ; ymm15 contains 8 accumulated floats
    vextractf128 xmm0, ymm15, 1             ; Get high 128 bits
    vaddps  xmm0, xmm0, xmm15               ; Add high and low halves (4 floats)
    vhaddps xmm0, xmm0, xmm0                ; [a+b, c+d, a+b, c+d]
    vhaddps xmm0, xmm0, xmm0                ; [a+b+c+d, ...]
    
    ; Store result
    vmovss  [r13], xmm0
    jmp     .epilogue

.done_zero:
    ; No blocks - store zero
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0

.epilogue:
    vzeroupper                              ; Clear upper YMM to avoid SSE/AVX transition penalty

%ifdef WINDOWS
    ; Restore XMM registers
    vmovdqu xmm6,  [rsp+0]
    vmovdqu xmm7,  [rsp+16]
    vmovdqu xmm8,  [rsp+32]
    vmovdqu xmm9,  [rsp+48]
    vmovdqu xmm10, [rsp+64]
    vmovdqu xmm11, [rsp+80]
    vmovdqu xmm12, [rsp+96]
    vmovdqu xmm13, [rsp+112]
    vmovdqu xmm14, [rsp+128]
    vmovdqu xmm15, [rsp+144]
    add     rsp, 168
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


;; =============================================================================
;; AVX-512 Version (Optional - for newer CPUs)
;; =============================================================================
%ifdef AVX512_ENABLED
global simd_vec_dot_q4_0_q8_0_avx512

simd_vec_dot_q4_0_q8_0_avx512:
    ; TODO: Implement AVX-512 version with:
    ; - 64-byte vector loads
    ; - VNNI instructions if available (vpdpbusd)
    ; - Process 4 blocks per iteration
    ret
%endif
