;; =============================================================================
;; vec_dot_q2_k_q8_k_avx2.asm - Optimized Q2_K x Q8_K dot product (AVX2)
;; =============================================================================
;;
;; CALLING CONVENTION (System V AMD64 / Windows x64 compatible):
;;   void simd_vec_dot_q2_k_q8_k_avx2(
;;       int n,                  ; rdi/rcx - number of elements (multiple of 256)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q2_K blocks
;;       const void* vy          ; rcx/r9  - Q8_K blocks
;;   );
;; =============================================================================

section .data
    align 32
    mask_low4:   times 32 db 0x0F
    mask_high4:  times 32 db 0xF0
    mask_2bit:   times 32 db 0x03
    ones_16:     times 16 dw 1
    
    ; Shuffle masks for broadcasting scales (byte 0->lo 8, byte 1->hi 8)
    shuf_01: times 8 db 0
             times 8 db 1
    shuf_23: times 8 db 2
             times 8 db 3
    shuf_45: times 8 db 4
             times 8 db 5
    shuf_67: times 8 db 6
             times 8 db 7

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

global simd_vec_dot_q2_k_q8_k_avx2

simd_vec_dot_q2_k_q8_k_avx2:
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
    mov     r11, ARG3               ; vx (Q2_K)
    mov     r12, ARG4               ; vy (Q8_K)
    mov     r13, ARG2               ; result

    vxorps  ymm15, ymm15, ymm15     ; Main Accumulator (Float)
    
    ; Constant masks
    vmovdqa ymm14, [rel mask_low4]
    
.main_loop:
    ; --------------------------------------------------------------------------
    ; STEP 1: LOAD SCALES & MINS
    ; --------------------------------------------------------------------------
    ; Load d (fp16) and dmin (fp16) from Q2_K (offset 80)
    vmovd   xmm0, [r11 + 80]        ; Load d and dmin (4 bytes)
    vcvtph2ps xmm0, xmm0            ; Convert to float: [?, ?, dmin, d]
    
    ; Load d_y (float) from Q8_K (offset 0)
    vmovss  xmm1, [r12]             ; Load d_y
    vbroadcastss ymm1, xmm1
    
    ; Broadcast d and dmin
    vshufps xmm2, xmm0, xmm0, 0x00  ; d
    vshufps xmm3, xmm0, xmm0, 0x55  ; dmin
    vbroadcastss ymm2, xmm2         ; d (256-bit)
    vbroadcastss ymm3, xmm3         ; dmin (256-bit)
    
    ; Calculate Scale = d * d_y
    vmulps  ymm2, ymm2, ymm1        ; Scale (ymm2)
    
    ; Calculate MinScale = dmin * d_y
    vmulps  ymm3, ymm3, ymm1        ; MinScale (ymm3)

    ; Load 16 bytes of scales/mins from Q2_K (bytes 0-15)
    vmovdqu xmm4, [r11]             ; scales (16 bytes)
    vpand   xmm5, xmm4, xmm14       ; scales_low (0-15) = s
    vpsrlw  xmm6, xmm4, 4
    vpand   xmm6, xmm6, xmm14       ; scales_high (0-15) = m
    
    ; --------------------------------------------------------------------------
    ; STEP 2: ACCUMULATE MINS (m * bsum)
    ; --------------------------------------------------------------------------
    ; Load bsums from Q8_K (offset 260, 16 * int16)
    vmovdqu ymm7, [r12 + 260]       ; bsums (16 int16)
    
    ; Expand m (u8) to i16
    vpmovzxbw ymm6, xmm6            ; 16x u8 -> 16x i16
    
    ; Multiply m * bsum (16x i16 -> 8x i32 sums)
    vpmaddwd  ymm6, ymm6, ymm7
    vcvtdq2ps ymm6, ymm6            ; Convert to float
    
    ; Accumulate Term 2: MinScale * sum(m * bsum)
    vfmadd231ps ymm15, ymm6, ymm3   ; Acc += m_sum * MinScale
    
    ; --------------------------------------------------------------------------
    ; STEP 3: ACCUMULATE WEIGHTS (s * q * y)
    ; --------------------------------------------------------------------------
    ; We process 2 halves: 0..127 and 128..255
    ; This corresponds to qs[0..31] and qs[32..63]
    
    xor r14d, r14d ; Half counter (0, 32)
    
.half_loop:
    ; Load 32 bytes of qs (128 weights compressed)
    ; qs offset: 16 + r14d
    lea rax, [r11 + 16]
    add rax, r14
    vmovdqu ymm0, [rax]         ; 32 bytes of qs
    
    ; y offset: 4 + 128 (no, 4 + r14d*4). r14d=32 -> 128.
    lea rbx, [r12 + 4]
    mov ecx, r14d
    shl ecx, 2
    add rbx, rcx
    
    ; Unpack 2-bit weights from ymm0
    vmovdqa ymm1, [rel mask_2bit] ; 0x03
    
    ; Stream 0 (bits 0-1) -> w[0..31]
    vpand   ymm4, ymm0, ymm1
    
    ; Stream 1 (bits 2-3) -> w[32..63]
    vpsrlw  ymm5, ymm0, 2
    vpand   ymm5, ymm5, ymm1
    
    ; Stream 2 (bits 4-5) -> w[64..95]
    vpsrlw  ymm6, ymm0, 4
    vpand   ymm6, ymm6, ymm1
    
    ; Stream 3 (bits 6-7) -> w[96..127]
    vpsrlw  ymm7, ymm0, 6
    vpand   ymm7, ymm7, ymm1
    
    ; Load y (strided)
    vmovdqu ymm8,  [rbx]           ; y[0..31]
    vmovdqu ymm9,  [rbx + 32]      ; y[32..63]
    vmovdqu ymm10, [rbx + 64]      ; y[64..95]
    vmovdqu ymm11, [rbx + 96]      ; y[96..127]
    
    ; Compute dot(w, y) -> i16 (u8 * i8)
    vpmaddubsw ymm4, ymm4, ymm8    ; dot0
    vpmaddubsw ymm5, ymm5, ymm9    ; dot1
    vpmaddubsw ymm6, ymm6, ymm10   ; dot2
    vpmaddubsw ymm7, ymm7, ymm11   ; dot3
    
    ; Get correct 8 scales for this half
    ; Reload scales
    vmovdqu xmm0, [r11]             ; scales
    vpand   xmm0, xmm0, [rel mask_low4]
    
    cmp r14d, 32
    jl .setup_scales
    vpsrldq xmm0, xmm0, 8           ; Shift to get s[8..15]
    
.setup_scales:
    ; Expand scales to match dots
    ; dot0 needs s0, s1 (8x i16 each)
    vpshufb xmm12, xmm0, [rel shuf_01] ; Broadcast s0, s1
    vpmovzxbw ymm12, xmm12             ; u8 -> i16
    vpmullw ymm4, ymm4, ymm12          ; dot0 * scale
    
    vpshufb xmm12, xmm0, [rel shuf_23]
    vpmovzxbw ymm12, xmm12
    vpmullw ymm5, ymm5, ymm12          ; dot1 * scale
    
    vpshufb xmm12, xmm0, [rel shuf_45]
    vpmovzxbw ymm12, xmm12
    vpmullw ymm6, ymm6, ymm12          ; dot2 * scale
    
    vpshufb xmm12, xmm0, [rel shuf_67]
    vpmovzxbw ymm12, xmm12
    vpmullw ymm7, ymm7, ymm12          ; dot3 * scale
    
    ; Sum all dots (i16) -> i32
    vpaddw  ymm4, ymm4, ymm5
    vpaddw  ymm6, ymm6, ymm7
    vpaddw  ymm4, ymm4, ymm6       ; Combined i16 sums
    
    ; Horizontal sum within lanes (i16 -> i32)
    vpmaddwd ymm4, ymm4, [rel ones_16]
    
    ; Convert to float and accumulate
    vcvtdq2ps ymm4, ymm4
    vfmadd231ps ymm15, ymm4, ymm2  ; Acc += Sum * Scale
    
    ; Loop control
    add r14d, 32
    cmp r14d, 64
    jl .half_loop
    
    ; Next block
    add r11, 84
    add r12, 292
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
