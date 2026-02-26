;; =============================================================================
;; vec_dot_q4_k_q8_k_avx2.asm - Optimized Q4_K x Q8_K dot product (AVX2)
;; =============================================================================
;;
;; CALLING CONVENTION (System V AMD64 / Windows x64 compatible):
;;   void simd_vec_dot_q4_k_q8_k_avx2(
;;       int n,                  ; rdi/rcx - number of elements (multiple of 256)
;;       float* result,          ; rsi/rdx - output scalar
;;       const void* vx,         ; rdx/r8  - Q4_K blocks
;;       const void* vy          ; rcx/r9  - Q8_K blocks
;;   );
;; =============================================================================

section .data
    align 32
    mask_low4:   times 32 db 0x0F
    mask_high4:  times 32 db 0xF0
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

global simd_vec_dot_q4_k_q8_k_avx2

simd_vec_dot_q4_k_q8_k_avx2:
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
    mov     r11, ARG3               ; vx (Q4_K)
    mov     r12, ARG4               ; vy (Q8_K)
    mov     r13, ARG2               ; result

    vxorps  ymm15, ymm15, ymm15     ; Main Accumulator (Float)
    
    ; Constant masks
    vmovdqu ymm14, [rel mask_low4]  ; 0x0F mask
    
.main_loop:
    ; --------------------------------------------------------------------------
    ; STEP 1: LOAD SUPER-BLOCK SCALES
    ; --------------------------------------------------------------------------
    ; Load d (fp16) and dmin (fp16) from Q4_K (offset 0)
    vmovd   xmm0, [r11]             ; Load d and dmin (4 bytes)
    vcvtph2ps xmm0, xmm0            ; Convert to float: [?, ?, dmin, d]
    
    ; Load d_y (float) from Q8_K (offset 0)
    vmovss  xmm1, [r12]             ; Load d_y
    vbroadcastss ymm1, xmm1
    
    ; Broadcast d and dmin
    vshufps xmm2, xmm0, xmm0, 0x00  ; d
    vshufps xmm3, xmm0, xmm0, 0x55  ; dmin
    vbroadcastss ymm2, xmm2         ; d (256-bit)
    vbroadcastss ymm3, xmm3         ; dmin (256-bit)
    
    ; Calculate SuperScale = d * d_y
    vmulps  ymm2, ymm2, ymm1        ; SuperScale (ymm2)
    
    ; Calculate SuperMin = dmin * d_y
    vmulps  ymm3, ymm3, ymm1        ; SuperMin (ymm3)

    ; --------------------------------------------------------------------------
    ; STEP 2: PROCESS 8 SUB-BLOCKS
    ; --------------------------------------------------------------------------
    xor r14d, r14d                  ; Sub-block index (0..7)
    
.subblock_loop:
    ; --- Compute Scales & Mins (Scalar) ---
    ; get_scale_min_k4(j, scales, &sc, &m)
    ; scales is at offset 4
    
    cmp r14d, 4
    jge .scales_high
    
    ; j < 4: sc = q[j] & 63, m = q[j+4] & 63
    movzx eax, byte [r11 + 4 + r14]     ; q[j]
    and eax, 63                         ; sc
    
    movzx ebx, byte [r11 + 8 + r14]     ; q[j+4]
    and ebx, 63                         ; m
    jmp .scales_done
    
.scales_high:
    ; j >= 4:
    ; sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    ; m  = (q[j+4] >> 4)  | ((q[j-0] >> 6) << 4)
    ; Note: r11+4 points to scales[0].
    ; q[j+4] is scales[r14+4] (offset 4+r14+4 = 8+r14)
    ; q[j-4] is scales[r14-4] (offset 4+r14-4 = 4+r14) -> wait, r14 is j.
    ; q[j-0] is scales[r14]   (offset 4+r14)
    
    ; sc computation:
    movzx eax, byte [r11 + 4 + r14 + 4] ; q[j+4] (scales[8..11])
    and eax, 0x0F                       ; low 4 bits
    
    movzx ecx, byte [r11 + 4 + r14 - 4] ; q[j-4] (scales[0..3])
    shr ecx, 6
    shl ecx, 4
    or eax, ecx                         ; sc combined
    
    ; m computation:
    movzx ebx, byte [r11 + 4 + r14 + 4] ; q[j+4]
    shr ebx, 4
    
    movzx ecx, byte [r11 + 4 + r14]     ; q[j] (scales[4..7])
    shr ecx, 6
    shl ecx, 4
    or ebx, ecx                         ; m combined

.scales_done:
    ; eax = sc, ebx = m
    vmovd xmm4, eax
    vbroadcastss ymm4, xmm4
    vcvtdq2ps ymm4, ymm4                ; float(sc)
    
    vmovd xmm5, ebx
    vbroadcastss ymm5, xmm5
    vcvtdq2ps ymm5, ymm5                ; float(m)
    
    ; Calculate Effective Scale/Min
    vmulps ymm4, ymm4, ymm2             ; d_eff = SuperScale * sc
    vmulps ymm5, ymm5, ymm3             ; m_eff = SuperMin * m
    
    ; --- Dot Product ---
    ; Load 32 weights (16 bytes packed)
    ; qs offset: 16 + r14*16? No.
    ; 32 values = 16 bytes.
    ; qs offset = 16 + r14 * 16.
    mov eax, r14d
    shl eax, 4                          ; * 16
    lea rax, [r11 + 16 + rax]           ; Address of qs block
    
    vmovdqu xmm6, [rax]                 ; Load 16 bytes (32 nibbles)
    
    ; Unpack nibbles to bytes
    vpsrlw xmm7, xmm6, 4                ; High nibbles
    vpand  xmm6, xmm6, xmm14            ; Low nibbles (mask 0x0F) -> q[0..15] (interleaved?)
    vpand  xmm7, xmm7, xmm14            ; High nibbles -> q[16..31]? No.
    ; In memory: byte 0 = [q1:4 bits][q0:4 bits].
    ; So low nibble is q0, high is q1.
    ; xmm6 has q0, q2, q4...
    ; xmm7 has q1, q3, q5...
    ; We need to unpack them sequentially?
    ; Or does Q8_K match this layout?
    ; Q8_K is linear.
    ; We need to unpack to linear q[0..31].
    ; punpcklbw / punpckhbw?
    
    vpunpcklbw xmm8, xmm6, xmm7         ; q0, q1, q2, q3... (low 128)
    vpunpckhbw xmm9, xmm6, xmm7         ; q16... (high 128)
    vinserti128 ymm6, ymm8, xmm9, 1     ; Combine to ymm6 (32 bytes)
    ; Now ymm6 contains q[0..31] in order.
    
    ; Load activations x (32 bytes)
    ; Q8_K qs offset: 4 + r14*32
    mov eax, r14d
    shl eax, 5                          ; * 32
    lea rax, [r12 + 4 + rax]
    vmovdqu ymm8, [rax]                 ; Load 32 bytes x
    
    ; Dot product: u8 * i8 -> i16
    vpmaddubsw ymm6, ymm6, ymm8         ; dot (16x i16)
    
    ; Horizontal sum to i32
    vpmaddwd ymm6, ymm6, [rel ones_16]  ; 8x i32
    
    ; Convert to float
    vcvtdq2ps ymm6, ymm6
    
    ; Accumulate: Sum += dot * d_eff
    vfmadd231ps ymm15, ymm6, ymm4
    
    ; --- Subtract Min Term ---
    ; term = m_eff * (bsum[2*j] + bsum[2*j+1])
    ; bsums offset: 260 + r14*4 (2 * int16 = 4 bytes)
    lea rdx, [r12 + 260]
    mov ecx, r14d
    shl ecx, 2
    add rdx, rcx
    
    movsx eax, word [rdx]               ; bsum[2*j]
    movsx ebx, word [rdx+2]             ; bsum[2*j+1]
    add eax, ebx                        ; sum(x)
    
    vmovd xmm9, eax
    vbroadcastss ymm9, xmm9
    vcvtdq2ps ymm9, ymm9                ; float(sum_x)
    
    ; Accumulate: Acc -= m_eff * sum_x
    ; vfnmadd231ps is -(a*b)+c? No.
    ; vfnmadd231ps dst, a, b -> dst = -(a*b) + dst
    vfnmadd231ps ymm15, ymm5, ymm9
    
    inc r14d
    cmp r14d, 8
    jl .subblock_loop
    
    ; Next superblock
    add r11, 144                        ; sizeof(block_q4_K)
    add r12, 292                        ; sizeof(block_q8_K)
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
