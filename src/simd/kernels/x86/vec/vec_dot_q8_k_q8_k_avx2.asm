;; =============================================================================
;; vec_dot_q8_k_q8_k_avx2.asm - Optimized Q8_K x Q8_K dot product (AVX2)
;; =============================================================================
;;
;; ALGORITHM:
;; Similar to Q8_0, but with 256-byte blocks and float scales.
;; We use the same sign trick: x*y = abs(x) * (sign(x)*y)
;;
;; BLOCK STRUCTURE:
;;   block_q8_K: 4 bytes (d: float) + 256 bytes (qs: int8) + 32 bytes (bsums) = 292 bytes
;;
;; CALLING CONVENTION:
;;   void simd_vec_dot_q8_k_q8_k_avx2(
;;       int n,                  ; rdi/rcx - number of elements (multiple of 256)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q8_K blocks (source x)
;;       const void* vy          ; rcx/r9  - Q8_K blocks (source y)
;;   );
;; =============================================================================

section .data
    align 32
    ones_16:     times 16 dw 1
    
section .text

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

%define Q8_K_BLOCK_SIZE 292

global simd_vec_dot_q8_k_q8_k_avx2

simd_vec_dot_q8_k_q8_k_avx2:
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

    mov     r10d, ARG1_32
    shr     r10d, 8                 ; n / 256
    mov     r11, ARG3               ; vx
    mov     r12, ARG4               ; vy
    mov     r13, ARG2               ; result
    
    ; Constant
    vmovdqu ymm12, [rel ones_16]
    
    vxorps  ymm15, ymm15, ymm15     ; Accumulator

.main_loop:
    ; Load scales (float)
    vmovss  xmm0, [r11]             ; d_x
    vmovss  xmm1, [r12]             ; d_y
    vmulss  xmm0, xmm0, xmm1        ; scale = d_x * d_y
    vbroadcastss ymm2, xmm0         ; broadcast scale
    
    ; Process 256 bytes in 8 chunks of 32
    xor r14d, r14d                  ; Offset 0..255 (steps of 32)
    
.subblock_loop:
    ; Offsets: 4 + r14
    lea rax, [r11 + 4]
    add rax, r14
    lea rbx, [r12 + 4]
    add rbx, r14
    
    vmovdqu ymm6, [rax]             ; Load 32 x
    vmovdqu ymm7, [rbx]             ; Load 32 y
    
    ; Sign trick
    vpsignb ymm8, ymm6, ymm6        ; abs(x)
    vpsignb ymm9, ymm7, ymm6        ; sign_adjust(y, x)
    
    vpmaddubsw ymm8, ymm8, ymm9     ; dot (16x i16)
    vpmaddwd   ymm8, ymm8, ymm12    ; sum (8x i32)
    vcvtdq2ps  ymm8, ymm8           ; float
    
    vfmadd231ps ymm15, ymm8, ymm2   ; Acc += sum * scale
    
    add r14, 32
    cmp r14, 256
    jl .subblock_loop
    
    ; Next block
    add r11, Q8_K_BLOCK_SIZE
    add r12, Q8_K_BLOCK_SIZE
    dec r10d
    jnz .main_loop
    
.horizontal_sum:
    vextractf128 xmm0, ymm15, 1
    vaddps  xmm0, xmm0, xmm15
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
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
