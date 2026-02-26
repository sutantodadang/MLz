;==============================================================================
;                   FLASH ATTENTION 2 - AVX-512 (F32 Q/K/V)
;==============================================================================
;
; Tiled online-softmax Flash Attention 2 for single-precision F32 tensors.
; Uses AVX-512F (512-bit zmm) instructions for maximum throughput.
;
; Algorithm: Flash Attention 2 (Dao et al.)
;   - Tiled QK computation with online softmax
;   - O(N) auxiliary memory (no full attention matrix materialised)
;   - Grouped Query Attention (GQA) head mapping
;   - Optional additive mask, sliding window
;
; Tile size: Bc = 64 KV positions per tile
; Processes head_dim in chunks of 16 floats (one zmm register)
; Supports head_dim up to FA_MAX_HEAD_DIM (256)
;
; License: MIT
;
;==============================================================================

bits 64
default rel

;------------------------------------------------------------------------------
; Platform detection
;------------------------------------------------------------------------------
%ifidn __OUTPUT_FORMAT__, win64
    %define WIN64
%endif

;------------------------------------------------------------------------------
; Config struct offsets (flash_attn_config_t from flash_attention.h)
; All int64_t = 8 bytes, float = 4 bytes, pointers = 8 bytes on x64
;------------------------------------------------------------------------------
CFG_HEAD_DIM_K    equ 0       ; int64_t
CFG_HEAD_DIM_V    equ 8       ; int64_t
CFG_N_QUERIES     equ 16      ; int64_t
CFG_N_KV          equ 24      ; int64_t
CFG_N_HEAD_Q      equ 32      ; int64_t
CFG_N_HEAD_KV     equ 40      ; int64_t
CFG_BATCH_SIZE    equ 48      ; int64_t
CFG_SCALE         equ 56      ; float
CFG_LOGIT_SOFTCAP equ 60      ; float
CFG_MAX_BIAS      equ 64      ; float
CFG_MODE          equ 68      ; uint32_t
CFG_WINDOW_SIZE   equ 72      ; int64_t
CFG_Q_PTR         equ 80      ; const float*
CFG_K_PTR         equ 88      ; const void*
CFG_V_PTR         equ 96      ; const void*
CFG_MASK_PTR      equ 104     ; const float*
CFG_DST_PTR       equ 112     ; float*
CFG_Q_NB          equ 120     ; size_t[4]  (byte strides for Q dims 0-3)
CFG_K_NB          equ 152     ; size_t[4]
CFG_V_NB          equ 184     ; size_t[4]
CFG_MASK_NB       equ 216     ; size_t[4]
CFG_DST_NB        equ 248     ; size_t[4]
CFG_K_TYPE        equ 280     ; int
CFG_V_TYPE        equ 284     ; int
CFG_ITH           equ 328     ; int  (this thread index)
CFG_NTH           equ 332     ; int  (total threads)

;------------------------------------------------------------------------------
; Algorithm constants
;------------------------------------------------------------------------------
FA_TILE_KV        equ 64      ; KV tile size for AVX-512
FA_MAX_HEAD_DIM   equ 256     ; Maximum supported head dimension

;------------------------------------------------------------------------------
; Mode flags (from flash_attention.h)
;------------------------------------------------------------------------------
FA_MODE_SLIDING_WIN equ 0x01

;------------------------------------------------------------------------------
; Stack frame layout (relative to rsp after allocation and alignment)
; We use `and rsp, ~63` for 64-byte alignment, then save the aligned rsp
; in [rbp - SAVED_RSP_OFFSET] so we can restore it in the epilogue.
;------------------------------------------------------------------------------
STK_O             equ 0       ; O accumulator: 256 floats = 1024 bytes
STK_SCORES        equ 1024    ; Tile scores: 64 floats = 256 bytes
STK_LOCALS        equ 1280    ; Scalar local variables begin here

; Scalar locals (at rsp + STK_LOCALS + offset)
LOC_Q_PTR         equ (STK_LOCALS + 0)    ; current Q row pointer
LOC_K_BASE        equ (STK_LOCALS + 8)    ; K base for current head/batch
LOC_V_BASE        equ (STK_LOCALS + 16)   ; V base for current head/batch
LOC_DST_PTR       equ (STK_LOCALS + 24)   ; current dst row pointer
LOC_MASK_PTR      equ (STK_LOCALS + 32)   ; current mask row pointer (0 if none)
LOC_HEAD_DIM_K    equ (STK_LOCALS + 40)   ; cached head_dim_k
LOC_HEAD_DIM_V    equ (STK_LOCALS + 48)   ; cached head_dim_v
LOC_N_KV          equ (STK_LOCALS + 56)   ; cached n_kv
LOC_HDK_FULL16    equ (STK_LOCALS + 64)   ; head_dim_k / 16
LOC_HDK_REM       equ (STK_LOCALS + 72)   ; head_dim_k % 16
LOC_HDV_FULL16    equ (STK_LOCALS + 80)   ; head_dim_v / 16
LOC_HDV_REM       equ (STK_LOCALS + 88)   ; head_dim_v % 16
LOC_K_STRIDE      equ (STK_LOCALS + 96)   ; k_nb[1] byte stride between KV pos
LOC_V_STRIDE      equ (STK_LOCALS + 104)  ; v_nb[1] byte stride between KV pos
LOC_ROW_MAX       equ (STK_LOCALS + 112)  ; current row max (float, 4 bytes)
LOC_ROW_SUM       equ (STK_LOCALS + 116)  ; current row sum (float, 4 bytes)
LOC_WORK_START    equ (STK_LOCALS + 120)  ; thread work range start
LOC_WORK_END      equ (STK_LOCALS + 128)  ; thread work range end
LOC_N_QUERIES     equ (STK_LOCALS + 136)  ; cached n_queries
LOC_N_HEAD_Q      equ (STK_LOCALS + 144)  ; cached n_head_q
LOC_N_HEAD_KV     equ (STK_LOCALS + 152)  ; cached n_head_kv
LOC_TILE_END      equ (STK_LOCALS + 160)  ; end of current KV tile
LOC_MASK_STRIDE   equ (STK_LOCALS + 168)  ; mask_nb[0] stride per KV pos
LOC_Q_NB1         equ (STK_LOCALS + 176)  ; q_nb[1] stride per query position
LOC_Q_NB2         equ (STK_LOCALS + 184)  ; q_nb[2] stride per head
LOC_Q_NB3         equ (STK_LOCALS + 192)  ; q_nb[3] stride per batch
LOC_K_NB2         equ (STK_LOCALS + 200)  ; k_nb[2] stride per head
LOC_K_NB3         equ (STK_LOCALS + 208)  ; k_nb[3] stride per batch
LOC_V_NB2         equ (STK_LOCALS + 216)  ; v_nb[2] stride per head
LOC_V_NB3         equ (STK_LOCALS + 224)  ; v_nb[3] stride per batch
LOC_DST_NB1       equ (STK_LOCALS + 232)  ; dst_nb[1] stride per query
LOC_DST_NB2       equ (STK_LOCALS + 240)  ; dst_nb[2] stride per head
LOC_DST_NB3       equ (STK_LOCALS + 248)  ; dst_nb[3] stride per batch
LOC_MASK_NB1      equ (STK_LOCALS + 256)  ; mask_nb[1] stride per query
LOC_WINDOW_SIZE   equ (STK_LOCALS + 264)  ; cached window_size
LOC_MODE          equ (STK_LOCALS + 272)  ; cached mode flags (4 bytes)
LOC_BATCH_SIZE    equ (STK_LOCALS + 280)  ; cached batch_size
LOC_NQ_X_NHQ      equ (STK_LOCALS + 288)  ; n_queries * n_head_q (precomputed)
LOC_IQ_SAVED      equ (STK_LOCALS + 296)  ; saved iq for sliding window
LOC_TOTAL_WORK    equ (STK_LOCALS + 304)  ; total work items
LOC_NTH           equ (STK_LOCALS + 312)  ; cached nth

; Windows XMM save area (only used on WIN64)
STK_XMM_SAVE      equ (STK_LOCALS + 320)  ; 10 * 16 = 160 bytes → ends at +480

; Total frame size requirement: 1280 + 480 = 1760, round up to 1856 (mod 64 = 0)
FRAME_SIZE        equ 1856

;------------------------------------------------------------------------------
; Inline macro: fast scalar exp approximation
;------------------------------------------------------------------------------
; %1 = input/output xmm register (contains x, will contain exp(x))
; Clobbers: xmm16, xmm17, xmm18, xmm19
; Uses: Cody-Waite range reduction + degree-4 Horner + vscalefss
;------------------------------------------------------------------------------
%macro FAST_EXP_SS 1
    ; Clamp input to valid range
    vmovss  xmm16, [rel exp_clamp_lo_f32]
    vmaxss  %1, %1, xmm16
    vmovss  xmm16, [rel exp_clamp_hi_f32]
    vminss  %1, %1, xmm16

    ; n = round(x * log2(e))
    vmovss  xmm16, [rel exp_log2e_f32]
    vmulss  xmm16, %1, xmm16                ; x * log2(e)
    vrndscaless xmm17, xmm16, xmm16, 0x08  ; n = round nearest (imm=0x08)

    ; r = x - n * ln(2)  (Cody-Waite reduction)
    vmovss  xmm18, [rel exp_ln2_f32]
    vmulss  xmm18, xmm17, xmm18             ; n * ln(2)
    vsubss  xmm18, %1, xmm18                ; r = x - n*ln(2)

    ; Horner polynomial: p = (((c4*r + c3)*r + c2)*r + c1)*r + 1
    vmovss  xmm19, [rel exp_c4_f32]
    vfmadd213ss xmm19, xmm18, [rel exp_c3_f32]
    vfmadd213ss xmm19, xmm18, [rel exp_c2_f32]
    vfmadd213ss xmm19, xmm18, [rel exp_c1_f32]
    vfmadd213ss xmm19, xmm18, [rel ones_f32]

    ; result = p * 2^n (using vscalefss)
    vscalefss %1, xmm19, xmm17
%endmacro

;------------------------------------------------------------------------------
; Inline macro: fast vectorised exp approximation (16-wide zmm)
;------------------------------------------------------------------------------
; %1 = input/output zmm register
; Clobbers: zmm16, zmm17, zmm18, zmm19
;------------------------------------------------------------------------------
%macro FAST_EXP_ZMM 1
    ; Clamp input to valid range
    vmaxps  %1, %1, [rel exp_clamp_lo_f32]
    vminps  %1, %1, [rel exp_clamp_hi_f32]

    ; n = round(x * log2(e))
    vmulps  zmm16, %1, [rel exp_log2e_f32]
    vrndscaleps zmm17, zmm16, 0x08          ; n = round nearest

    ; r = x - n * ln(2)
    vmulps  zmm18, zmm17, [rel exp_ln2_f32]
    vsubps  zmm18, %1, zmm18                ; r = x - n*ln(2)

    ; Horner polynomial: p = (((c4*r + c3)*r + c2)*r + c1)*r + 1
    vmovaps zmm19, [rel exp_c4_f32]
    vfmadd213ps zmm19, zmm18, [rel exp_c3_f32]
    vfmadd213ps zmm19, zmm18, [rel exp_c2_f32]
    vfmadd213ps zmm19, zmm18, [rel exp_c1_f32]
    vfmadd213ps zmm19, zmm18, [rel ones_f32]

    ; result = p * 2^n
    vscalefps %1, zmm19, zmm17
%endmacro

;------------------------------------------------------------------------------
; Section: Read-only data (constants)
;------------------------------------------------------------------------------
section .rodata
    align 64

; Negative infinity (-inf IEEE 754)
neg_inf_f32:
    times 16 dd 0xFF800000

; All ones (1.0f)
ones_f32:
    times 16 dd 0x3F800000

; Exp approximation constants (Cody-Waite + degree-4 Horner)
exp_log2e_f32:
    times 16 dd 0x3FB8AA3B            ; log2(e) = 1.4426950408889634

exp_ln2_f32:
    times 16 dd 0x3F317218            ; ln(2) = 0.6931471805599453

exp_c1_f32:
    times 16 dd 0x3F800000            ; 1.0

exp_c2_f32:
    times 16 dd 0x3F000000            ; 0.5

exp_c3_f32:
    times 16 dd 0x3E2AAAAB            ; 1/6 ≈ 0.16666667

exp_c4_f32:
    times 16 dd 0x3D2AAAAB            ; 1/24 ≈ 0.041666668

; Exp clamping bounds (avoid overflow/underflow)
exp_clamp_lo_f32:
    times 16 dd 0xC2AEAC50            ; -87.3365

exp_clamp_hi_f32:
    times 16 dd 0x42B17218            ; 88.7228

;------------------------------------------------------------------------------
; Section: Code
;------------------------------------------------------------------------------
section .text

;==============================================================================
; Function: simd_flash_attn_f32_avx512
;==============================================================================
;
; void simd_flash_attn_f32_avx512(const flash_attn_config_t* config)
;
; Implements Flash Attention 2 (tiled online softmax) for F32 Q/K/V tensors
; using AVX-512 instructions (512-bit zmm registers).
;
; Parameters:
;   Windows x64 ABI: RCX = config
;   System V ABI:    RDI = config
;
;==============================================================================

global simd_flash_attn_f32_avx512

simd_flash_attn_f32_avx512:
    ;--------------------------------------------------------------------------
    ; PROLOGUE
    ;--------------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WIN64
    push    rdi
    push    rsi
    ; rbp-8=rbx, rbp-16=r12, rbp-24=r13, rbp-32=r14, rbp-40=r15
    ; rbp-48=rdi, rbp-56=rsi
    ; Allocate frame and align to 64 bytes
    sub     rsp, FRAME_SIZE + 64
    and     rsp, ~63

    ; Save Windows callee-saved XMM6-XMM15
    vmovups [rsp + STK_XMM_SAVE],       xmm6
    vmovups [rsp + STK_XMM_SAVE + 16],  xmm7
    vmovups [rsp + STK_XMM_SAVE + 32],  xmm8
    vmovups [rsp + STK_XMM_SAVE + 48],  xmm9
    vmovups [rsp + STK_XMM_SAVE + 64],  xmm10
    vmovups [rsp + STK_XMM_SAVE + 80],  xmm11
    vmovups [rsp + STK_XMM_SAVE + 96],  xmm12
    vmovups [rsp + STK_XMM_SAVE + 112], xmm13
    vmovups [rsp + STK_XMM_SAVE + 128], xmm14
    vmovups [rsp + STK_XMM_SAVE + 144], xmm15

    ; Remap Windows ABI → R12 = config
    mov     r12, rcx
%else
    ; SysV ABI: rbp-8=rbx, rbp-16=r12, rbp-24=r13, rbp-32=r14, rbp-40=r15
    sub     rsp, FRAME_SIZE + 64
    and     rsp, ~63

    ; R12 = config pointer
    mov     r12, rdi
%endif

    ;--------------------------------------------------------------------------
    ; EARLY EXITS: check for zero dimensions
    ;--------------------------------------------------------------------------
    mov     rax, [r12 + CFG_HEAD_DIM_K]
    test    rax, rax
    jz      .epilogue
    mov     rax, [r12 + CFG_HEAD_DIM_V]
    test    rax, rax
    jz      .epilogue
    mov     rax, [r12 + CFG_N_QUERIES]
    test    rax, rax
    jz      .epilogue
    mov     rax, [r12 + CFG_N_KV]
    test    rax, rax
    jz      .epilogue
    mov     rax, [r12 + CFG_N_HEAD_Q]
    test    rax, rax
    jz      .epilogue
    mov     rax, [r12 + CFG_BATCH_SIZE]
    test    rax, rax
    jz      .epilogue

    ;--------------------------------------------------------------------------
    ; CACHE FREQUENTLY USED CONFIG VALUES ON STACK
    ;--------------------------------------------------------------------------
    mov     rax, [r12 + CFG_HEAD_DIM_K]
    mov     [rsp + LOC_HEAD_DIM_K], rax
    mov     rax, [r12 + CFG_HEAD_DIM_V]
    mov     [rsp + LOC_HEAD_DIM_V], rax
    mov     rax, [r12 + CFG_N_QUERIES]
    mov     [rsp + LOC_N_QUERIES], rax
    mov     rax, [r12 + CFG_N_KV]
    mov     [rsp + LOC_N_KV], rax
    mov     rax, [r12 + CFG_N_HEAD_Q]
    mov     [rsp + LOC_N_HEAD_Q], rax
    mov     rax, [r12 + CFG_N_HEAD_KV]
    mov     [rsp + LOC_N_HEAD_KV], rax
    mov     rax, [r12 + CFG_BATCH_SIZE]
    mov     [rsp + LOC_BATCH_SIZE], rax
    mov     rax, [r12 + CFG_WINDOW_SIZE]
    mov     [rsp + LOC_WINDOW_SIZE], rax
    mov     eax, [r12 + CFG_MODE]
    mov     dword [rsp + LOC_MODE], eax

    ; Precompute n_queries * n_head_q
    mov     rax, [r12 + CFG_N_QUERIES]
    imul    rax, [r12 + CFG_N_HEAD_Q]
    mov     [rsp + LOC_NQ_X_NHQ], rax

    ; Head dimension chunking for K
    mov     rax, [rsp + LOC_HEAD_DIM_K]
    mov     rcx, rax
    shr     rax, 4                          ; head_dim_k / 16
    mov     [rsp + LOC_HDK_FULL16], rax
    and     rcx, 15                         ; head_dim_k % 16
    mov     [rsp + LOC_HDK_REM], rcx

    ; Head dimension chunking for V
    mov     rax, [rsp + LOC_HEAD_DIM_V]
    mov     rcx, rax
    shr     rax, 4                          ; head_dim_v / 16
    mov     [rsp + LOC_HDV_FULL16], rax
    and     rcx, 15                         ; head_dim_v % 16
    mov     [rsp + LOC_HDV_REM], rcx

    ; Byte strides
    mov     rax, [r12 + CFG_Q_NB + 8]       ; q_nb[1]
    mov     [rsp + LOC_Q_NB1], rax
    mov     rax, [r12 + CFG_Q_NB + 16]      ; q_nb[2]
    mov     [rsp + LOC_Q_NB2], rax
    mov     rax, [r12 + CFG_Q_NB + 24]      ; q_nb[3]
    mov     [rsp + LOC_Q_NB3], rax

    mov     rax, [r12 + CFG_K_NB + 8]       ; k_nb[1]
    mov     [rsp + LOC_K_STRIDE], rax
    mov     rax, [r12 + CFG_K_NB + 16]      ; k_nb[2]
    mov     [rsp + LOC_K_NB2], rax
    mov     rax, [r12 + CFG_K_NB + 24]      ; k_nb[3]
    mov     [rsp + LOC_K_NB3], rax

    mov     rax, [r12 + CFG_V_NB + 8]       ; v_nb[1]
    mov     [rsp + LOC_V_STRIDE], rax
    mov     rax, [r12 + CFG_V_NB + 16]      ; v_nb[2]
    mov     [rsp + LOC_V_NB2], rax
    mov     rax, [r12 + CFG_V_NB + 24]      ; v_nb[3]
    mov     [rsp + LOC_V_NB3], rax

    mov     rax, [r12 + CFG_DST_NB + 8]     ; dst_nb[1]
    mov     [rsp + LOC_DST_NB1], rax
    mov     rax, [r12 + CFG_DST_NB + 16]    ; dst_nb[2]
    mov     [rsp + LOC_DST_NB2], rax
    mov     rax, [r12 + CFG_DST_NB + 24]    ; dst_nb[3]
    mov     [rsp + LOC_DST_NB3], rax

    mov     rax, [r12 + CFG_MASK_NB]        ; mask_nb[0]
    mov     [rsp + LOC_MASK_STRIDE], rax
    mov     rax, [r12 + CFG_MASK_NB + 8]    ; mask_nb[1]
    mov     [rsp + LOC_MASK_NB1], rax

    ;--------------------------------------------------------------------------
    ; THREAD PARTITIONING
    ; total_work = n_queries * n_head_q * batch_size
    ; work_start = (total_work * ith) / nth
    ; work_end   = (total_work * (ith + 1)) / nth
    ;--------------------------------------------------------------------------
    mov     rax, [rsp + LOC_NQ_X_NHQ]
    imul    rax, [rsp + LOC_BATCH_SIZE]      ; rax = total_work
    mov     [rsp + LOC_TOTAL_WORK], rax

    movsxd  rcx, dword [r12 + CFG_ITH]      ; ith
    movsxd  r8, dword [r12 + CFG_NTH]       ; nth
    mov     [rsp + LOC_NTH], r8

    ; work_start = (total_work * ith) / nth
    mov     rbx, rax                         ; rbx = total_work
    imul    rax, rcx                         ; rax = total_work * ith
    cqo                                      ; sign-extend rax into rdx:rax
    idiv    r8                               ; rax = (total_work * ith) / nth
    mov     [rsp + LOC_WORK_START], rax

    ; work_end = (total_work * (ith + 1)) / nth
    movsxd  rax, dword [r12 + CFG_ITH]
    inc     rax                              ; ith + 1
    imul    rax, rbx                         ; total_work * (ith + 1)
    cqo
    idiv    r8                               ; rax = (total_work * (ith+1)) / nth
    mov     [rsp + LOC_WORK_END], rax

    ; Check if this thread has work
    mov     r15, [rsp + LOC_WORK_START]
    cmp     r15, [rsp + LOC_WORK_END]
    jge     .epilogue

    ;--------------------------------------------------------------------------
    ; PREPARE REMAINDER OPMASKS (for head_dim not multiple of 16)
    ;--------------------------------------------------------------------------
    ; K remainder mask (k1)
    mov     rcx, [rsp + LOC_HDK_REM]
    test    rcx, rcx
    jz      .no_k_rem_mask
    mov     eax, 1
    shl     eax, cl
    dec     eax                              ; (1 << rem) - 1
    kmovw   k1, eax
    jmp     .k_rem_mask_done
.no_k_rem_mask:
    mov     eax, 0xFFFF
    kmovw   k1, eax
.k_rem_mask_done:

    ; V remainder mask (k2)
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .no_v_rem_mask
    mov     eax, 1
    shl     eax, cl
    dec     eax
    kmovw   k2, eax
    jmp     .v_rem_mask_done
.no_v_rem_mask:
    mov     eax, 0xFFFF
    kmovw   k2, eax
.v_rem_mask_done:

    ;--------------------------------------------------------------------------
    ; LOAD PERSISTENT CONSTANTS INTO ZMM REGISTERS
    ;--------------------------------------------------------------------------
    ; zmm31 = scale (broadcast)
    vbroadcastss zmm31, dword [r12 + CFG_SCALE]
    ; zmm30 = -inf (for max initialisation)
    vmovaps zmm30, [rel neg_inf_f32]

    ;==========================================================================
    ; MAIN WORK LOOP
    ; R15 = current work index, iterates [work_start, work_end)
    ;==========================================================================
    align 64
.work_loop:
    cmp     r15, [rsp + LOC_WORK_END]
    jge     .epilogue

    ;----------------------------------------------------------------------
    ; DECOMPOSE work_idx → (ib, ihq, iq)
    ; work_idx = iq + n_queries * (ihq + n_head_q * ib)
    ; ib  = work_idx / (n_queries * n_head_q)
    ; rem = work_idx % (n_queries * n_head_q)
    ; ihq = rem / n_queries
    ; iq  = rem % n_queries
    ;----------------------------------------------------------------------
    mov     rax, r15
    xor     edx, edx                         ; zero-extend for unsigned div
    mov     rcx, [rsp + LOC_NQ_X_NHQ]
    div     rcx
    ; rax = ib, rdx = remainder
    mov     r13, rax                         ; r13 = ib

    mov     rax, rdx
    xor     edx, edx
    mov     rcx, [rsp + LOC_N_QUERIES]
    div     rcx
    ; rax = ihq, rdx = iq
    mov     r14, rax                         ; r14 = ihq
    mov     rbx, rdx                         ; rbx = iq

    ; Save iq for later sliding window check
    mov     [rsp + LOC_IQ_SAVED], rbx

    ;----------------------------------------------------------------------
    ; GQA HEAD MAPPING: ihkv = ihq * n_head_kv / n_head_q
    ;----------------------------------------------------------------------
    mov     rax, r14                         ; ihq
    imul    rax, [rsp + LOC_N_HEAD_KV]
    xor     edx, edx
    mov     rcx, [rsp + LOC_N_HEAD_Q]
    div     rcx
    ; rax = ihkv
    mov     r8, rax                          ; r8 = ihkv

    ;----------------------------------------------------------------------
    ; COMPUTE POINTERS using byte strides
    ;----------------------------------------------------------------------
    ; Q row pointer: q + iq*q_nb[1] + ihq*q_nb[2] + ib*q_nb[3]
    mov     rax, [r12 + CFG_Q_PTR]
    mov     rcx, rbx
    imul    rcx, [rsp + LOC_Q_NB1]
    add     rax, rcx
    mov     rcx, r14
    imul    rcx, [rsp + LOC_Q_NB2]
    add     rax, rcx
    mov     rcx, r13
    imul    rcx, [rsp + LOC_Q_NB3]
    add     rax, rcx
    mov     [rsp + LOC_Q_PTR], rax

    ; K base pointer: k + ihkv*k_nb[2] + ib*k_nb[3]
    mov     rax, [r12 + CFG_K_PTR]
    mov     rcx, r8
    imul    rcx, [rsp + LOC_K_NB2]
    add     rax, rcx
    mov     rcx, r13
    imul    rcx, [rsp + LOC_K_NB3]
    add     rax, rcx
    mov     [rsp + LOC_K_BASE], rax

    ; V base pointer: v + ihkv*v_nb[2] + ib*v_nb[3]
    mov     rax, [r12 + CFG_V_PTR]
    mov     rcx, r8
    imul    rcx, [rsp + LOC_V_NB2]
    add     rax, rcx
    mov     rcx, r13
    imul    rcx, [rsp + LOC_V_NB3]
    add     rax, rcx
    mov     [rsp + LOC_V_BASE], rax

    ; DST row pointer: dst + iq*dst_nb[1] + ihq*dst_nb[2] + ib*dst_nb[3]
    mov     rax, [r12 + CFG_DST_PTR]
    mov     rcx, rbx
    imul    rcx, [rsp + LOC_DST_NB1]
    add     rax, rcx
    mov     rcx, r14
    imul    rcx, [rsp + LOC_DST_NB2]
    add     rax, rcx
    mov     rcx, r13
    imul    rcx, [rsp + LOC_DST_NB3]
    add     rax, rcx
    mov     [rsp + LOC_DST_PTR], rax

    ; MASK row pointer: mask ? mask + iq*mask_nb[1] : NULL
    mov     rax, [r12 + CFG_MASK_PTR]
    test    rax, rax
    jz      .no_mask_ptr
    mov     rcx, rbx
    imul    rcx, [rsp + LOC_MASK_NB1]
    add     rax, rcx
.no_mask_ptr:
    mov     [rsp + LOC_MASK_PTR], rax

    ;----------------------------------------------------------------------
    ; INITIALIZE O ACCUMULATOR TO ZERO
    ;----------------------------------------------------------------------
    vpxord  zmm0, zmm0, zmm0
    lea     rdi, [rsp + STK_O]
    mov     rcx, [rsp + LOC_HDV_FULL16]
    test    rcx, rcx
    jz      .init_o_rem
.init_o_full:
    vmovaps [rdi], zmm0
    add     rdi, 64
    dec     rcx
    jnz     .init_o_full
.init_o_rem:
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .init_o_done
    vmovaps [rdi], zmm0                      ; safe: buffer has 1024 bytes
.init_o_done:

    ; Initialize row_max = -inf, row_sum = 0
    mov     dword [rsp + LOC_ROW_MAX], 0xFF800000
    mov     dword [rsp + LOC_ROW_SUM], 0x00000000

    ;==========================================================================
    ; FA2 KV TILE LOOP
    ; R13 = tile start (t), reused since ib no longer needed
    ;==========================================================================
    xor     r13, r13                         ; t = 0

    align 32
.tile_loop:
    mov     rax, [rsp + LOC_N_KV]
    cmp     r13, rax
    jge     .tile_loop_done

    ; tile_end = min(t + FA_TILE_KV, n_kv)
    lea     rcx, [r13 + FA_TILE_KV]
    cmp     rcx, rax
    cmovg   rcx, rax
    mov     [rsp + LOC_TILE_END], rcx

    ;------------------------------------------------------------------
    ; STEP A: QK dot products for tile
    ; For each j in [t, tile_end):
    ;   scores[j-t] = dot(Q_row, K_row[j], head_dim_k) * scale
    ;------------------------------------------------------------------
    mov     r14, r13                         ; r14 = j

    align 16
.qk_dot_loop:
    cmp     r14, [rsp + LOC_TILE_END]
    jge     .qk_dot_done

    ; K row pointer: k_base + j * k_stride
    mov     rax, r14
    imul    rax, [rsp + LOC_K_STRIDE]
    add     rax, [rsp + LOC_K_BASE]          ; rax = &K[j]

    ; Q row pointer
    mov     rsi, [rsp + LOC_Q_PTR]

    ; Dot product accumulator
    vpxord  zmm20, zmm20, zmm20

    mov     rcx, [rsp + LOC_HDK_FULL16]
    mov     rdi, rsi                         ; rdi = Q working ptr
    mov     r8, rax                          ; r8 = K working ptr
    test    rcx, rcx
    jz      .qk_dot_rem

    align 16
.qk_dot_inner:
    vmovups zmm21, [rdi]                     ; Q[d..d+15]
    vfmadd231ps zmm20, zmm21, [r8]          ; acc += Q * K
    add     rdi, 64
    add     r8, 64
    dec     rcx
    jnz     .qk_dot_inner

.qk_dot_rem:
    mov     rcx, [rsp + LOC_HDK_REM]
    test    rcx, rcx
    jz      .qk_dot_hsum
    ; Masked load: zero-pad remainder lanes
    vmovups zmm21{k1}{z}, [rdi]
    vmovups zmm22{k1}{z}, [r8]
    vfmadd231ps zmm20, zmm21, zmm22

.qk_dot_hsum:
    ; Horizontal sum: zmm20 → scalar in xmm20[0]
    vextractf32x8 ymm21, zmm20, 1
    vaddps  ymm20, ymm20, ymm21
    vextractf32x4 xmm21, ymm20, 1
    vaddps  xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0x4E
    vaddps  xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0xB1
    vaddps  xmm20, xmm20, xmm21
    ; xmm20[0] = dot(Q, K[j])

    ; score = dot * scale
    vmulss  xmm20, xmm20, xmm31

    ; Store to tile scores buffer
    mov     rax, r14
    sub     rax, r13                         ; j - t
    vmovss  [rsp + STK_SCORES + rax*4], xmm20

    inc     r14
    jmp     .qk_dot_loop

.qk_dot_done:

    ;------------------------------------------------------------------
    ; STEP B: Add mask values (if mask != NULL)
    ;------------------------------------------------------------------
    mov     rax, [rsp + LOC_MASK_PTR]
    test    rax, rax
    jz      .mask_done

    mov     r14, r13                         ; j = t
.mask_add_loop:
    cmp     r14, [rsp + LOC_TILE_END]
    jge     .mask_done

    ; mask value at mask_row + j * mask_stride
    mov     rcx, r14
    imul    rcx, [rsp + LOC_MASK_STRIDE]
    add     rcx, [rsp + LOC_MASK_PTR]
    vmovss  xmm20, [rcx]

    ; scores[j-t] += mask_val
    mov     rax, r14
    sub     rax, r13
    vaddss  xmm20, xmm20, [rsp + STK_SCORES + rax*4]
    vmovss  [rsp + STK_SCORES + rax*4], xmm20

    inc     r14
    jmp     .mask_add_loop
.mask_done:

    ;------------------------------------------------------------------
    ; STEP C: Find tile max
    ;------------------------------------------------------------------
    mov     rcx, [rsp + LOC_TILE_END]
    sub     rcx, r13                         ; tile_size
    lea     rsi, [rsp + STK_SCORES]

    ; Start with -inf
    vmovss  xmm20, [rel neg_inf_f32]

    ; Process full 16-wide chunks
    mov     rax, rcx
    shr     rax, 4                           ; tile_size / 16
    test    rax, rax
    jz      .tile_max_scalar_setup

    vmovaps zmm20, [rel neg_inf_f32]
    mov     rdi, rsi
.tile_max_vec:
    vmaxps  zmm20, zmm20, [rdi]
    add     rdi, 64
    dec     rax
    jnz     .tile_max_vec

    ; Horizontal max: zmm20 → xmm20[0]
    vextractf32x8 ymm21, zmm20, 1
    vmaxps  ymm20, ymm20, ymm21
    vextractf32x4 xmm21, ymm20, 1
    vmaxps  xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0x4E
    vmaxps  xmm20, xmm20, xmm21
    vpermilps xmm21, xmm20, 0xB1
    vmaxps  xmm20, xmm20, xmm21
    jmp     .tile_max_scalar_check

.tile_max_scalar_setup:
    mov     rdi, rsi                         ; rdi = start of scores

.tile_max_scalar_check:
    ; Process remaining (tile_size % 16) scalars
    mov     rax, rcx
    and     rax, 15
    test    rax, rax
    jz      .tile_max_done
.tile_max_scalar:
    vmaxss  xmm20, xmm20, [rdi]
    add     rdi, 4
    dec     rax
    jnz     .tile_max_scalar
.tile_max_done:
    ; xmm20[0] = tile_max

    ;------------------------------------------------------------------
    ; STEP D: Online softmax update
    ;   new_max = max(row_max, tile_max)
    ;   correction = exp(row_max - new_max)
    ;   O *= correction; row_sum *= correction
    ;   row_max = new_max
    ;------------------------------------------------------------------
    vmovss  xmm21, [rsp + LOC_ROW_MAX]      ; old row_max
    vmaxss  xmm22, xmm21, xmm20             ; new_max

    ; correction = exp(old_max - new_max) — always ≤ 0
    vsubss  xmm23, xmm21, xmm22
    FAST_EXP_SS xmm23                        ; xmm23 = correction

    ; row_sum *= correction
    vmovss  xmm24, [rsp + LOC_ROW_SUM]
    vmulss  xmm24, xmm24, xmm23
    vmovss  [rsp + LOC_ROW_SUM], xmm24

    ; row_max = new_max
    vmovss  [rsp + LOC_ROW_MAX], xmm22

    ; O *= correction (vectorised)
    vbroadcastss zmm25, xmm23

    lea     rdi, [rsp + STK_O]
    mov     rcx, [rsp + LOC_HDV_FULL16]
    test    rcx, rcx
    jz      .rescale_o_rem
.rescale_o_full:
    vmulps  zmm26, zmm25, [rdi]
    vmovaps [rdi], zmm26
    add     rdi, 64
    dec     rcx
    jnz     .rescale_o_full
.rescale_o_rem:
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .rescale_o_done
    vmovaps zmm26, [rdi]
    vmulps  zmm26, zmm26, zmm25
    vmovaps [rdi], zmm26
.rescale_o_done:

    ;------------------------------------------------------------------
    ; STEP E: Compute p = exp(score - max) and accumulate O += p * V
    ;------------------------------------------------------------------
    mov     r14, r13                         ; r14 = j (KV position)

    align 16
.pv_accum_loop:
    cmp     r14, [rsp + LOC_TILE_END]
    jge     .pv_accum_done

    ; p = exp(score[j-t] - new_max)
    mov     rax, r14
    sub     rax, r13
    vmovss  xmm23, [rsp + STK_SCORES + rax*4]
    vsubss  xmm23, xmm23, xmm22             ; score - new_max
    FAST_EXP_SS xmm23                        ; xmm23 = p

    ; row_sum += p
    vaddss  xmm24, xmm23, [rsp + LOC_ROW_SUM]
    vmovss  [rsp + LOC_ROW_SUM], xmm24

    ; Broadcast p for PV multiply
    vbroadcastss zmm27, xmm23

    ; V row pointer: v_base + j * v_stride
    mov     rax, r14
    imul    rax, [rsp + LOC_V_STRIDE]
    add     rax, [rsp + LOC_V_BASE]

    ; O += p * V[j] — process head_dim_v in 16-float chunks
    lea     rdi, [rsp + STK_O]
    mov     rcx, [rsp + LOC_HDV_FULL16]
    mov     r8, rax                          ; r8 = V working ptr
    test    rcx, rcx
    jz      .pv_rem

    align 16
.pv_full:
    vmovaps zmm29, [rdi]                     ; O[d..d+15]
    vfmadd231ps zmm29, zmm27, [r8]          ; O += p * V
    vmovaps [rdi], zmm29
    add     rdi, 64
    add     r8, 64
    dec     rcx
    jnz     .pv_full

.pv_rem:
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .pv_next

    ; Masked PV for remainder lanes
    vmovaps zmm29, [rdi]                     ; O (full load from aligned buffer)
    vmovups zmm26{k2}{z}, [r8]              ; V partial (masked, zero-extend)
    vfmadd231ps zmm29, zmm27, zmm26         ; O += p * V
    vmovaps [rdi], zmm29

.pv_next:
    inc     r14
    jmp     .pv_accum_loop
.pv_accum_done:

    ; Advance to next KV tile
    add     r13, FA_TILE_KV
    jmp     .tile_loop

.tile_loop_done:

    ;------------------------------------------------------------------
    ; NORMALIZE: O /= row_sum
    ;------------------------------------------------------------------
    vmovss  xmm24, [rsp + LOC_ROW_SUM]

    ; Guard against zero sum
    vpxord  xmm25, xmm25, xmm25
    vucomiss xmm24, xmm25
    jz      .store_output
    jp      .store_output                    ; NaN check

    ; Reciprocal 1/sum via vrcpss + Newton-Raphson refinement
    vrcp14ss xmm25, xmm25, xmm24            ; ~1/sum (14-bit accuracy)
    ; NR iteration: x' = x * (2 - sum*x)
    vmulss  xmm26, xmm24, xmm25             ; sum * x
    vmovss  xmm27, [rel ones_f32]
    vaddss  xmm27, xmm27, xmm27             ; 2.0
    vsubss  xmm27, xmm27, xmm26             ; 2 - sum*x
    vmulss  xmm25, xmm25, xmm27             ; x * (2 - sum*x)

    ; Broadcast and normalise O
    vbroadcastss zmm25, xmm25

    lea     rdi, [rsp + STK_O]
    mov     rcx, [rsp + LOC_HDV_FULL16]
    test    rcx, rcx
    jz      .norm_rem
.norm_full:
    vmulps  zmm26, zmm25, [rdi]
    vmovaps [rdi], zmm26
    add     rdi, 64
    dec     rcx
    jnz     .norm_full
.norm_rem:
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .store_output
    vmovaps zmm26, [rdi]
    vmulps  zmm26, zmm26, zmm25
    vmovaps [rdi], zmm26

    ;------------------------------------------------------------------
    ; STORE OUTPUT: copy O → dst_row
    ;------------------------------------------------------------------
.store_output:
    lea     rsi, [rsp + STK_O]
    mov     rdi, [rsp + LOC_DST_PTR]

    mov     rcx, [rsp + LOC_HDV_FULL16]
    test    rcx, rcx
    jz      .store_rem
.store_full:
    vmovaps zmm26, [rsi]
    vmovups [rdi], zmm26                     ; dst may not be 64-byte aligned
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .store_full
.store_rem:
    mov     rcx, [rsp + LOC_HDV_REM]
    test    rcx, rcx
    jz      .store_done
    vmovaps zmm26, [rsi]
    vmovups [rdi]{k2}, zmm26                ; masked store for partial tail
.store_done:

    ; Next work item
    inc     r15
    jmp     .work_loop

    ;==========================================================================
    ; EPILOGUE
    ;==========================================================================
.epilogue:
    vzeroupper

%ifdef WIN64
    ; Restore XMM6-15 (saved relative to aligned rsp which hasn't changed)
    vmovups xmm6,  [rsp + STK_XMM_SAVE]
    vmovups xmm7,  [rsp + STK_XMM_SAVE + 16]
    vmovups xmm8,  [rsp + STK_XMM_SAVE + 32]
    vmovups xmm9,  [rsp + STK_XMM_SAVE + 48]
    vmovups xmm10, [rsp + STK_XMM_SAVE + 64]
    vmovups xmm11, [rsp + STK_XMM_SAVE + 80]
    vmovups xmm12, [rsp + STK_XMM_SAVE + 96]
    vmovups xmm13, [rsp + STK_XMM_SAVE + 112]
    vmovups xmm14, [rsp + STK_XMM_SAVE + 128]
    vmovups xmm15, [rsp + STK_XMM_SAVE + 144]

    ; Restore rsp from rbp (undoes alignment + sub)
    lea     rsp, [rbp - 56]                  ; 7 pushes below rbp
    pop     rsi
    pop     rdi
%else
    lea     rsp, [rbp - 40]                  ; 5 pushes below rbp
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
