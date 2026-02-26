;; =============================================================================
;; vec_dot_q8_k_q8_k_avx512.asm - Optimized Q8_K x Q8_K dot product (AVX-512)
;; =============================================================================
;;
;; ALGORITHM:
;; Uses VNNI (vpdpbusd) for efficient 8-bit dot products.
;; Block size: 256 weights (4 x 64 bytes).
;;
;; CALLING CONVENTION:
;;   void simd_vec_dot_q8_k_q8_k_avx512(
;;       int n,                  ; rdi/rcx - number of elements (multiple of 256)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q8_K blocks (source x)
;;       const void* vy          ; rcx/r9  - Q8_K blocks (source y)
;;   );
;; =============================================================================

section .data
    align 64
    offset_128_64:   times 64 db 0x80

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

global simd_vec_dot_q8_k_q8_k_avx512

simd_vec_dot_q8_k_q8_k_avx512:
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
    shr     r10d, 8                 ; n / 256
    mov     r11, ARG3               ; vx
    mov     r12, ARG4               ; vy
    mov     r13, ARG2               ; result
    
    ; Load constant
    vmovdqu64 zmm18, [rel offset_128_64]
    
    vxorps    zmm15, zmm15, zmm15   ; Accumulator

    align 64
.main_loop:
    ; Load scales (float)
    vmovss  xmm0, [r11]             ; d_x
    vmovss  xmm1, [r12]             ; d_y
    vmulss  xmm0, xmm0, xmm1        ; scale = d_x * d_y
    vbroadcastss zmm2, xmm0         ; broadcast scale
    
    ; Load 4 chunks of 64 bytes (256 total)
    ; Offsets: 4, 68, 132, 196
    vmovdqu64 zmm4, [r11 + 4]       ; x0
    vmovdqu64 zmm5, [r11 + 68]      ; x1
    vmovdqu64 zmm6, [r11 + 132]     ; x2
    vmovdqu64 zmm7, [r11 + 196]     ; x3
    
    vmovdqu64 zmm8, [r12 + 4]       ; y0
    vmovdqu64 zmm9, [r12 + 68]      ; y1
    vmovdqu64 zmm10, [r12 + 132]    ; y2
    vmovdqu64 zmm11, [r12 + 196]    ; y3
    
    ; VNNI Computation (4 chunks)
    vxorps    zmm0, zmm0, zmm0      ; chunk acc
    
    ; Chunk 0
    vpxord    zmm12, zmm4, zmm18    ; x_u8
    vpdpbusd  zmm0, zmm12, zmm8     ; dot
    vpxord    zmm12, zmm12, zmm12
    vpdpbusd  zmm12, zmm18, zmm8    ; corr
    vpsubd    zmm0, zmm0, zmm12     ; correct
    
    ; Chunk 1
    vpxord    zmm12, zmm5, zmm18
    vpdpbusd  zmm0, zmm12, zmm9
    vpxord    zmm12, zmm12, zmm12
    vpdpbusd  zmm12, zmm18, zmm9
    vpsubd    zmm0, zmm0, zmm12
    
    ; Chunk 2
    vpxord    zmm12, zmm6, zmm18
    vpdpbusd  zmm0, zmm12, zmm10
    vpxord    zmm12, zmm12, zmm12
    vpdpbusd  zmm12, zmm18, zmm10
    vpsubd    zmm0, zmm0, zmm12
    
    ; Chunk 3
    vpxord    zmm12, zmm7, zmm18
    vpdpbusd  zmm0, zmm12, zmm11
    vpxord    zmm12, zmm12, zmm12
    vpdpbusd  zmm12, zmm18, zmm11
    vpsubd    zmm0, zmm0, zmm12
    
    ; Convert to float and accumulate
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm15, zmm0, zmm2   ; Acc += chunk_sum * scale
    
    ; Next block
    add r11, Q8_K_BLOCK_SIZE
    add r12, Q8_K_BLOCK_SIZE
    dec r10d
    jnz .main_loop
    
.horizontal_sum:
    vextractf32x8 ymm0, zmm15, 1
    vaddps    ymm15, ymm15, ymm0
    
    vextractf128 xmm0, ymm15, 1
    vaddps    xmm0, xmm0, xmm15
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
