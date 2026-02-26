;; =============================================================================
;; vec_dot_q8_0_q8_0_avx2.asm - Hand-tuned Q8_0 × Q8_0 dot product kernel
;; =============================================================================
;;
;; ALGORITHM:
;; For each block pair (32 values):
;;   1. Load 32 int8 values from Q8_0 source x
;;   2. Load 32 int8 values from Q8_0 source y
;;   3. Multiply pairs using sign trick + vpmaddubsw
;;   4. Sum pairs using vpmaddwd
;;   5. Accumulate and scale by (d0 * d1)
;;
;; BLOCK STRUCTURE:
;;   block_q8_0: 2 bytes (d: fp16) + 32 bytes (qs: 32 int8) = 34 bytes
;;
;; CALLING CONVENTION:
;;   void simd_vec_dot_q8_0_q8_0_avx2(
;;       int n,                  ; rdi/rcx - number of elements (must be multiple of 32)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q8_0 blocks (source x)
;;       const void* vy          ; rcx/r9  - Q8_0 blocks (source y)
;;   );
;;
;; =============================================================================

section .data
    align 32
    ; Ones for horizontal sum: 1 repeated 16 times (16-bit)
    q8_ones_16:   times 16 dw 1

section .text

;; -----------------------------------------------------------------------------
;; Windows x64 calling convention adaptation
;; -----------------------------------------------------------------------------
%ifdef WINDOWS
    %define ARG1   rcx    ; n (full 64-bit)
    %define ARG1_32 ecx   ; n (32-bit version)
    %define ARG2   rdx    ; result
    %define ARG3   r8     ; vx
    %define ARG4   r9     ; vy
%else
    ; System V AMD64
    %define ARG1   rdi    ; n (full 64-bit)
    %define ARG1_32 edi   ; n (32-bit version)
    %define ARG2   rsi    ; result
    %define ARG3   rdx    ; vx
    %define ARG4   rcx    ; vy
%endif

%define Q8_0_BLOCK_SIZE 34   ; 2 + 32 bytes

global simd_vec_dot_q8_0_q8_0_avx2

simd_vec_dot_q8_0_q8_0_avx2:
    ; Prologue
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WINDOWS
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

    ; Calculate number of blocks
    mov     r10d, ARG1_32
    shr     r10d, 5                         ; n / 32
    test    r10d, r10d
    jz      .done_zero
    
    ; Setup pointers
    mov     r11, ARG3                       ; vx
    mov     r12, ARG4                       ; vy
    mov     r13, ARG2                       ; result
    
    ; Load constant
    vmovdqa ymm12, [rel q8_ones_16]
    
    ; Zero accumulator
    vxorps  ymm15, ymm15, ymm15
    
    ; Calculate unrolled loop count (4 blocks per iteration)
    mov     r14d, r10d
    shr     r14d, 2                         ; Groups of 4
    test    r14d, r14d
    jz      .remainder_loop

;; -----------------------------------------------------------------------------
;; Main loop: Process 4 blocks per iteration (aggressive unroll)
;; -----------------------------------------------------------------------------
align 16
.main_loop:
    ; Prefetch ahead
    prefetcht0 [r11 + Q8_0_BLOCK_SIZE*8]
    prefetcht0 [r12 + Q8_0_BLOCK_SIZE*8]
    
    ;; ========== BLOCK 0 ==========
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu ymm6, [r11 + 2]                 ; Load 32 int8 from x
    vmovdqu ymm7, [r12 + 2]                 ; Load 32 int8 from y
    
    ; Signed dot product using sign trick
    vpsignb ymm8, ymm6, ymm6                ; abs(x)
    vpsignb ymm9, ymm7, ymm6                ; sign_adjust(y, x)
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    add     r11, Q8_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    
    ;; ========== BLOCK 1 ==========
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu ymm6, [r11 + 2]
    vmovdqu ymm7, [r12 + 2]
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    add     r11, Q8_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    
    ;; ========== BLOCK 2 ==========
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu ymm6, [r11 + 2]
    vmovdqu ymm7, [r12 + 2]
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    add     r11, Q8_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    
    ;; ========== BLOCK 3 ==========
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu ymm6, [r11 + 2]
    vmovdqu ymm7, [r12 + 2]
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    add     r11, Q8_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    
    dec     r14d
    jnz     .main_loop

;; -----------------------------------------------------------------------------
;; Handle remaining 0-3 blocks
;; -----------------------------------------------------------------------------
.remainder_loop:
    mov     r14d, r10d
    and     r14d, 3                         ; Remaining blocks (0-3)
    test    r14d, r14d
    jz      .horizontal_sum

.remainder_single:
    movzx   eax, word [r11]
    vmovd   xmm0, eax
    vcvtph2ps xmm0, xmm0
    movzx   ebx, word [r12]
    vmovd   xmm1, ebx
    vcvtph2ps xmm1, xmm1
    vmulss  xmm2, xmm0, xmm1
    
    vmovdqu ymm6, [r11 + 2]
    vmovdqu ymm7, [r12 + 2]
    vpsignb ymm8, ymm6, ymm6
    vpsignb ymm9, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm9
    vpmaddwd ymm8, ymm8, ymm12
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm2, xmm2
    vfmadd231ps ymm15, ymm8, ymm2
    
    add     r11, Q8_0_BLOCK_SIZE
    add     r12, Q8_0_BLOCK_SIZE
    dec     r14d
    jnz     .remainder_single

;; -----------------------------------------------------------------------------
;; Horizontal sum
;; -----------------------------------------------------------------------------
.horizontal_sum:
    vextractf128 xmm0, ymm15, 1
    vaddps  xmm0, xmm0, xmm15
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss  [r13], xmm0
    jmp     .epilogue

.done_zero:
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0

.epilogue:
    vzeroupper

%ifdef WINDOWS
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
