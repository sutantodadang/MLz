;; =============================================================================
;; vec_dot_q5_k_q8_k_avx2.asm — Handwritten AVX2 implementation of
;; Q5_K x Q8_K dot product.  Bit-for-bit equivalent to upstream ggml's
;; `ggml_vec_dot_q5_K_q8_K_generic` (deps/llama_cpp/ggml/src/ggml-cpu/quants.c).
;;
;; Companion AVX-512 kernel lives in vec_dot_q5_k_q8_k_avx512.asm and uses
;; 512-bit registers (zmm) to fuse two 32-element sub-blocks per iteration.
;; =============================================================================
;;
;; BLOCK LAYOUT
;; ------------
;;   block_q5_K  (176 B, QK_K = 256 elements):
;;       fp16   d           ; +0   super-block scale for quantized scales
;;       fp16   dmin        ; +2   super-block scale for quantized mins
;;       u8     scales[12]  ; +4   6-bit packed scales+mins (8 sub-blocks)
;;       u8     qh[32]      ; +16  high bit per element
;;       u8     qs[128]     ; +48  low 4 bits per element (4 chunks of 32 B)
;;
;;   block_q8_K  (292 B):
;;       f32    d           ; +0
;;       i8     qs[256]     ; +4   activation, signed 8-bit
;;       i16    bsums[16]   ; +260 sums per 16-element sub-sub-block
;;
;; ALGORITHM PER SUPER-BLOCK
;; -------------------------
;;   1. Decode 12-byte packed scales/mins -> 8 u8 scales + 8 u8 mins.
;;   2. sumi_mins = Sigma_{j=0..15} y.bsums[j] * mins[j/2]            (i32)
;;   3. acc_total = 0
;;   4. For each of 8 sub-blocks (S = 0..7):
;;        - 32 elements of x reconstructed as low-nibble (S even) or
;;          high-nibble (S odd) of qs[(S/2)*32], plus +16 where the
;;          corresponding qh bit (1 << S) is set.
;;        - subdot = Sigma x[k] * y.qs[S*32 + k]
;;        - acc_total += scales[S] * subdot
;;   5. sumf += d * y.d * acc_total - dmin * y.d * sumi_mins
;;
;; CALLING CONVENTION (Win64 / SysV both)
;; --------------------------------------
;;   void simd_vec_dot_q5_k_q8_k_avx2(
;;       int          n,        ; ARG1  (must be a multiple of 256)
;;       float      * result,   ; ARG2
;;       const void * vx,       ; ARG3  pointer to N/256 block_q5_K
;;       const void * vy);      ; ARG4  pointer to N/256 block_q8_K
;; =============================================================================

section .data
    align 32
    ones_i16:    times 16 dw 1
    mask_lo4:    times 32 db 0x0F
    const_16:    times 32 db 16

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

%define BS_Q5_K 176
%define BS_Q8_K 292

;; Stack layout (after prologue), 232 bytes total (Win64):
;;   [rsp +   0] xmm6           (callee-saved on Win64)
;;   [rsp +  16] xmm7
;;   [rsp +  32] xmm8
;;   [rsp +  48] xmm9
;;   [rsp +  64] xmm10
;;   [rsp +  80] xmm11
;;   [rsp +  96] xmm12
;;   [rsp + 112] xmm13
;;   [rsp + 128] xmm14
;;   [rsp + 144] xmm15
;;   [rsp + 160] utmp[0..3] = scales[0..7] (8 B) + mins[0..7] (8 B)
;;   [rsp + 176] padding (must end 16-byte aligned)
%define SCALES_OFF 160

;; -----------------------------------------------------------------------------
;; SUB_DOT macro — compute one 32-element sub-block dot and accumulate.
;;
;;   %1 : nibble selector  (lo => raw & 0x0F  ;  hi => (raw >> 4) & 0x0F)
;;   %2 : sub-block index S (0..7)
;;
;; Inputs  : ymm6 = current 32-byte qs chunk
;;           ymm7 = qh (32 B) — same across all 8 sub-blocks
;;           ymm12 = ones_i16, ymm13 = mask_lo4, ymm14 = const_16
;;           r12 = vy super-block base, r14d = acc_total
;;           [rsp + SCALES_OFF + S] = scales[S] (u8)
;; Clobbers: ymm0, ymm1, ymm2, eax, ebx
;; Output  : r14d += scales[S] * subdot
;; -----------------------------------------------------------------------------
%macro SUB_DOT 2
    %ifidn %1, lo
        vpand        ymm0, ymm6, ymm13                  ; nibble = raw & 0x0F
    %else
        vpsrlw       ymm0, ymm6, 4
        vpand        ymm0, ymm0, ymm13                  ; nibble = (raw>>4) & 0x0F
    %endif

    ;; high-bit injection: where qh has bit (1<<S) set, add 16 to the nibble.
    mov              eax, (1 << %2)
    vmovd            xmm1, eax
    vpbroadcastb     ymm1, xmm1                         ; ymm1 = mask byte broadcast
    vpand            ymm2, ymm7, ymm1                   ; qh & mask
    vpcmpeqb         ymm2, ymm2, ymm1                   ; 0xFF where bit is set
    vpand            ymm2, ymm2, ymm14                  ; -> 16 where set, 0 elsewhere
    vpaddb           ymm0, ymm0, ymm2                   ; x_u8 in [0..31]

    ;; load 32 i8 of y for sub-block S
    vmovdqu          ymm1, [r12 + 4 + (%2)*32]

    ;; vpmaddubsw treats first source as u8, second as i8 -> 16 i16 lanes
    ;; vpmaddwd  with ones_i16 just sums adjacent i16 pairs into 8 i32 lanes
    vpmaddubsw       ymm0, ymm0, ymm1
    vpmaddwd         ymm0, ymm0, ymm12

    ;; horizontal-sum 8 i32 -> 1 i32 in eax
    vextracti128     xmm1, ymm0, 1
    vpaddd           xmm0, xmm0, xmm1                   ; 4 i32
    vphaddd          xmm0, xmm0, xmm0                   ; 2 i32 (replicated)
    vphaddd          xmm0, xmm0, xmm0                   ; 1 i32 (replicated)
    vmovd            eax, xmm0                          ; subdot

    movzx            ebx, byte [rsp + SCALES_OFF + (%2)]
    imul             eax, ebx                           ; scales[S] * subdot
    add              r14d, eax                          ; acc_total += ...
%endmacro

global simd_vec_dot_q5_k_q8_k_avx2

simd_vec_dot_q5_k_q8_k_avx2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WINDOWS
    ;; Stack: ret(8) + rbp(8) + 7 GPR pushes(56) = 72.  Need (72 + sub) % 16 == 0.
    ;; Save xmm6..xmm15 (10 * 16 = 160) + utmp scratch (16) + pad (8) = 184.
    ;; (72 + 184) % 16 == 0.
    sub     rsp, 184
    vmovdqu [rsp +   0], xmm6
    vmovdqu [rsp +  16], xmm7
    vmovdqu [rsp +  32], xmm8
    vmovdqu [rsp +  48], xmm9
    vmovdqu [rsp +  64], xmm10
    vmovdqu [rsp +  80], xmm11
    vmovdqu [rsp +  96], xmm12
    vmovdqu [rsp + 112], xmm13
    vmovdqu [rsp + 128], xmm14
    vmovdqu [rsp + 144], xmm15
%else
    ;; SysV: only stack alignment matters (no callee-saved xmm).
    sub     rsp, 184
%endif

    ;; -----------------------------------------------------------------------
    ;; Persistent state across the super-block loop:
    ;;   r10d   = nb (= n / 256)             (loop counter)
    ;;   r11    = vx pointer (advances by 176)
    ;;   r12    = vy pointer (advances by 292)
    ;;   r13    = result pointer
    ;;   xmm15  = sumf accumulator (scalar fp32)
    ;;   ymm12  = ones_i16, ymm13 = mask_lo4, ymm14 = const_16
    ;; -----------------------------------------------------------------------
    mov     r10d, ARG1_32
    shr     r10d, 8                              ; nb = n / 256
    mov     r13, ARG2                             ; result
    vxorps  xmm15, xmm15, xmm15                   ; sumf = 0.0f
    test    r10d, r10d
    jz      .write_result                         ; n == 0 -> *result = 0

    mov     r11, ARG3                             ; vx
    mov     r12, ARG4                             ; vy

    vmovdqa ymm12, [rel ones_i16]
    vmovdqa ymm13, [rel mask_lo4]
    vmovdqa ymm14, [rel const_16]

.main_loop:
    ;; ---- decode d, dmin (fp16) and y_d (f32) ----
    vmovd      xmm0, dword [r11]                  ; lo32 = d_h | (dmin_h << 16)
    vcvtph2ps  xmm0, xmm0                         ; xmm0[0]=d_x, xmm0[1]=dmin_x
    vmovss     xmm1, [r12]                        ; y_d
    vshufps    xmm2, xmm0, xmm0, 0x00             ; xmm2 = d_x broadcast (low)
    vshufps    xmm3, xmm0, xmm0, 0x55             ; xmm3 = dmin_x broadcast (low)
    vmulss     xmm4, xmm2, xmm1                   ; xmm4 = d_x * y_d   (kept)
    vmulss     xmm5, xmm3, xmm1                   ; xmm5 = dmin_x * y_d (kept)

    ;; ---- decode 12-byte scales/mins -> 4 u32 -> stored at SCALES_OFF ----
    ;;   eax = utmp[0]  (raw 0..3)
    ;;   ebx = utmp[1]  (raw 4..7)
    ;;   ecx = utmp[2]  (raw 8..11)
    ;;   edx = utmp[3]  (computed)
    ;; Layout after decode (matches C++ helper, kmask1=0x3f3f3f3f,
    ;;   kmask2=0x0f0f0f0f, kmask3=0x03030303):
    ;;     out[0] = utmp[0] & kmask1
    ;;     out[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4)
    ;;     out[2] = utmp[1] & kmask1                          (saved as 'uaux')
    ;;     out[3] = ((utmp[2] >> 4) & kmask2)
    ;;            | (((utmp[1] >> 6) & kmask3) << 4)
    mov     eax, [r11 + 4 + 0]                    ; utmp[0]
    mov     ebx, [r11 + 4 + 4]                    ; utmp[1]
    mov     ecx, [r11 + 4 + 8]                    ; utmp[2]

    ;; edx = ((ecx >> 4) & kmask2) | (((ebx >> 6) & kmask3) << 4)
    mov     edx, ecx
    shr     edx, 4
    and     edx, 0x0f0f0f0f
    mov     esi, ebx
    shr     esi, 6
    and     esi, 0x03030303
    shl     esi, 4
    or      edx, esi                              ; edx = utmp[3]_decoded

    ;; edi = uaux = ebx & kmask1
    mov     edi, ebx
    and     edi, 0x3f3f3f3f                       ; edi = utmp[2]_decoded

    ;; ebx = (ecx & kmask2) | (((eax >> 6) & kmask3) << 4)
    mov     ebx, ecx
    and     ebx, 0x0f0f0f0f
    mov     esi, eax
    shr     esi, 6
    and     esi, 0x03030303
    shl     esi, 4
    or      ebx, esi                              ; ebx = utmp[1]_decoded

    ;; ecx = uaux
    mov     ecx, edi                              ; ecx = utmp[2]_decoded

    ;; eax = utmp[0] & kmask1
    and     eax, 0x3f3f3f3f                       ; eax = utmp[0]_decoded

    ;; Store decoded utmp[0..3] = scales[0..7] then mins[0..7]
    mov     [rsp + SCALES_OFF +  0], eax          ; scales[0..3]
    mov     [rsp + SCALES_OFF +  4], ebx          ; scales[4..7]
    mov     [rsp + SCALES_OFF +  8], ecx          ; mins[0..3]
    mov     [rsp + SCALES_OFF + 12], edx          ; mins[4..7]

    ;; ---- sumi_mins = Sigma_{j=0..15} bsums[j] * mins[j/2]  (vectorized) ----
    ;; mins doubled to 16 i16: [m0,m0,m1,m1, ..., m7,m7]
    ;; vpmaddwd(bsums, mins_doubled) -> 8 i32, then horizontal sum.
    vmovdqu       ymm0, [r12 + 260]               ; 16 i16 bsums
    vmovq         xmm1, qword [rsp + SCALES_OFF + 8]   ; 8 mins bytes
    vpunpcklbw    xmm1, xmm1, xmm1                ; 16 bytes: m0,m0,m1,m1,...
    vpmovzxbw     ymm1, xmm1                      ; 16 i16
    vpmaddwd      ymm0, ymm0, ymm1                ; 8 i32
    vextracti128  xmm1, ymm0, 1
    vpaddd        xmm0, xmm0, xmm1
    vphaddd       xmm0, xmm0, xmm0
    vphaddd       xmm0, xmm0, xmm0
    vmovd         r15d, xmm0                      ; r15d = sumi_mins (i32)

    ;; ---- load qh once (shared by all 8 sub-blocks) ----
    vmovdqu       ymm7, [r11 + 16]                ; qh = xb + 4 + 12

    ;; ---- accumulate all 8 sub-blocks ----
    xor           r14d, r14d                      ; acc_total = 0

    ;; pair p=0  -> sub-blocks 0,1 sharing qs[0..31]
    vmovdqu       ymm6, [r11 + 48 + 0*32]
    SUB_DOT       lo, 0
    SUB_DOT       hi, 1

    ;; pair p=1  -> sub-blocks 2,3
    vmovdqu       ymm6, [r11 + 48 + 1*32]
    SUB_DOT       lo, 2
    SUB_DOT       hi, 3

    ;; pair p=2  -> sub-blocks 4,5
    vmovdqu       ymm6, [r11 + 48 + 2*32]
    SUB_DOT       lo, 4
    SUB_DOT       hi, 5

    ;; pair p=3  -> sub-blocks 6,7
    vmovdqu       ymm6, [r11 + 48 + 3*32]
    SUB_DOT       lo, 6
    SUB_DOT       hi, 7

    ;; ---- sumf += d_x*y_d * acc_total - dmin_x*y_d * sumi_mins ----
    vcvtsi2ss     xmm0, xmm0, r14d                ; (i32 acc_total) -> f32
    vmulss        xmm0, xmm0, xmm4                ; * (d_x * y_d)
    vcvtsi2ss     xmm1, xmm1, r15d                ; (i32 sumi_mins) -> f32
    vmulss        xmm1, xmm1, xmm5                ; * (dmin_x * y_d)
    vsubss        xmm0, xmm0, xmm1
    vaddss        xmm15, xmm15, xmm0              ; sumf += ...

    ;; advance to next super-block
    add           r11, BS_Q5_K
    add           r12, BS_Q8_K
    dec           r10d
    jnz           .main_loop

.write_result:
    vmovss        [r13], xmm15

.done:
    vzeroupper
%ifdef WINDOWS
    vmovdqu xmm6,  [rsp +   0]
    vmovdqu xmm7,  [rsp +  16]
    vmovdqu xmm8,  [rsp +  32]
    vmovdqu xmm9,  [rsp +  48]
    vmovdqu xmm10, [rsp +  64]
    vmovdqu xmm11, [rsp +  80]
    vmovdqu xmm12, [rsp +  96]
    vmovdqu xmm13, [rsp + 112]
    vmovdqu xmm14, [rsp + 128]
    vmovdqu xmm15, [rsp + 144]
%endif
    add     rsp, 184
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    pop     rbp
    ret
