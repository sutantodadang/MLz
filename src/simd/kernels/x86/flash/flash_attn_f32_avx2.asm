;==============================================================================
;                   FLASH ATTENTION 2 - F32 AVX2 KERNEL
;==============================================================================
;
; High-performance Flash Attention 2 implementation using AVX2 (256-bit) SIMD.
; Implements the full tiled online-softmax algorithm from:
;   "FlashAttention-2: Faster Attention with Better Parallelism and Work
;    Partitioning" (Tri Dao, 2023)
;
; This kernel handles F32 Query, F32 Key, F32 Value tensors.
;
; Algorithm overview:
;   For each query row, iterate over KV tiles (Bc=32):
;     1. Compute QK^T dot products for the tile
;     2. Apply scale and optional mask
;     3. Online softmax: track running max and sum, apply corrections
;     4. Accumulate P*V into output
;     5. Final normalization by row_sum
;
; Supported calling conventions:
;   - System V AMD64 (Linux, macOS, BSD)
;   - Microsoft x64 (Windows) - controlled by WIN64 define
;
; Thread safety: No global mutable state. Thread partitioning via config.
;
; Part of MLz inference engine - replaces C++ intrinsics FA implementation.
;
; License: MIT
;
;==============================================================================

; ── ABI Detection ────────────────────────────────────────────────────────────
%ifidn __OUTPUT_FORMAT__, win64
    %define WIN64
%endif

;==============================================================================
; Config struct field offsets (flash_attn_config_t)
; All int64_t fields are 8 bytes, float fields 4 bytes, pointers 8 bytes.
;==============================================================================
CFG_HEAD_DIM_K    equ 0       ; int64_t  - key/query head dimension
CFG_HEAD_DIM_V    equ 8       ; int64_t  - value head dimension
CFG_N_QUERIES     equ 16      ; int64_t  - number of query positions
CFG_N_KV          equ 24      ; int64_t  - number of key/value positions
CFG_N_HEAD_Q      equ 32      ; int64_t  - number of query heads
CFG_N_HEAD_KV     equ 40      ; int64_t  - number of KV heads (for GQA)
CFG_BATCH_SIZE    equ 48      ; int64_t  - batch size
CFG_SCALE         equ 56      ; float    - attention scale (1/sqrt(dk))
CFG_LOGIT_SOFTCAP equ 60      ; float    - logit soft-capping value
CFG_MAX_BIAS      equ 64      ; float    - ALiBi max bias
CFG_MODE          equ 68      ; uint32_t - attention mode flags
CFG_WINDOW_SIZE   equ 72      ; int64_t  - sliding window size (-1 = none)
CFG_Q_PTR         equ 80      ; const float*  - query tensor base
CFG_K_PTR         equ 88      ; const void*   - key tensor base
CFG_V_PTR         equ 96      ; const void*   - value tensor base
CFG_MASK_PTR      equ 104     ; const float*  - mask tensor base (nullable)
CFG_DST_PTR       equ 112     ; float*        - output tensor base
CFG_Q_NB          equ 120     ; size_t[4]     - Q byte strides [elem, row, head, batch]
CFG_K_NB          equ 152     ; size_t[4]     - K byte strides
CFG_V_NB          equ 184     ; size_t[4]     - V byte strides
CFG_MASK_NB       equ 216     ; size_t[4]     - mask byte strides
CFG_DST_NB        equ 248     ; size_t[4]     - dst byte strides
CFG_K_TYPE        equ 280     ; int           - K element type enum
CFG_V_TYPE        equ 284     ; int           - V element type enum
CFG_ITH           equ 328     ; int           - this thread index
CFG_NTH           equ 332     ; int           - total thread count

;==============================================================================
; Algorithm constants
;==============================================================================
FA_TILE_KV        equ 32      ; KV tile size (Bc) for tiled attention
FA_MAX_HEAD_DIM   equ 256     ; Maximum supported head dimension

;==============================================================================
; Stack frame layout (offsets from rsp after allocation)
; Total frame is 32-byte aligned for vmovaps compatibility.
;==============================================================================
STK_O             equ 0       ; float[256] - output accumulator (1024 bytes)
STK_SCORES        equ 1024    ; float[32]  - QK scores for current tile (128 bytes)
STK_PROBS         equ 1152    ; float[32]  - softmax probabilities (128 bytes)
; ── Saved pointers & scalars ─────────────────────────────────────────────────
STK_Q_ROW         equ 1280    ; const float* - current Q row pointer
STK_K_BASE        equ 1288    ; const void*  - K base for current head/batch
STK_V_BASE        equ 1296    ; const void*  - V base for current head/batch
STK_DST_ROW       equ 1304    ; float*       - current output row pointer
STK_HEAD_DK       equ 1312    ; int64        - head_dim_k
STK_HEAD_DV       equ 1320    ; int64        - head_dim_v
STK_N_KV          equ 1328    ; int64        - n_kv
STK_MASK_PTR      equ 1336    ; const float* - mask pointer (or NULL)
STK_MASK_NB0      equ 1344    ; size_t       - mask stride dim 0
STK_MASK_NB1      equ 1352    ; size_t       - mask stride dim 1
STK_CONFIG        equ 1360    ; const flash_attn_config_t* - config pointer
STK_WORK_END      equ 1368    ; int64        - last work item (exclusive)
STK_N_QUERIES     equ 1376    ; int64
STK_N_HEAD_Q      equ 1384    ; int64
STK_N_HEAD_KV     equ 1392    ; int64
STK_BATCH_SIZE    equ 1400    ; int64
; ── Q strides ────────────────────────────────────────────────────────────────
STK_Q_NB1         equ 1408    ; size_t - Q row stride (bytes)
STK_Q_NB2         equ 1416    ; size_t - Q head stride (bytes)
STK_Q_NB3         equ 1424    ; size_t - Q batch stride (bytes)
; ── K strides ────────────────────────────────────────────────────────────────
STK_K_NB1         equ 1432    ; size_t - K row stride (bytes)
STK_K_NB2         equ 1440    ; size_t - K head stride (bytes)
STK_K_NB3         equ 1448    ; size_t - K batch stride (bytes)
; ── V strides ────────────────────────────────────────────────────────────────
STK_V_NB1         equ 1456    ; size_t - V row stride (bytes)
STK_V_NB2         equ 1464    ; size_t - V head stride (bytes)
STK_V_NB3         equ 1472    ; size_t - V batch stride (bytes)
; ── Dst strides ──────────────────────────────────────────────────────────────
STK_DST_NB1       equ 1480    ; size_t - dst row stride (bytes)
STK_DST_NB2       equ 1488    ; size_t - dst head stride (bytes)
STK_DST_NB3       equ 1496    ; size_t - dst batch stride (bytes)
; ── Work decomposition ───────────────────────────────────────────────────────
STK_IQ            equ 1504    ; int64 - current query index
STK_IHQ           equ 1512    ; int64 - current query head index
STK_IB            equ 1520    ; int64 - current batch index
STK_SCALE_VEC     equ 1536    ; float[8] - broadcast scale (32 bytes, aligned)
STK_RBP_SAVE      equ 1568    ; original rbp before alignment
FRAME_SIZE        equ 1600    ; total frame (multiple of 32)

;==============================================================================
; Section: Read-only data constants (32-byte aligned for AVX2)
;==============================================================================
section .data
    align 32
    ; Negative infinity for initial row_max
    neg_inf:        times 8 dd 0xFF800000       ; -INF as IEEE 754

    align 32
    ; exp() approximation constants (Cody-Waite + Horner polynomial)
    ; Using: exp(x) = 2^n * P(r), where x = n*ln2 + r
    exp_log2e:      times 8 dd 1.4426950408889634     ; log2(e)
    align 32
    exp_half:       times 8 dd 0.5                     ; 0.5 for rounding
    align 32
    exp_ln2_hi:     times 8 dd 0.693145751953125       ; ln(2) high part
    align 32
    exp_ln2_lo:     times 8 dd 1.428606765330187e-06   ; ln(2) low part
    align 32
    ; Horner coefficients for exp(r) ≈ 1 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    exp_c1:         times 8 dd 1.0                     ; c1 = 1
    align 32
    exp_c2:         times 8 dd 0.5                     ; c2 = 1/2!
    align 32
    exp_c3:         times 8 dd 0.16666666666666666     ; c3 = 1/3!
    align 32
    exp_c4:         times 8 dd 0.041666666666666664    ; c4 = 1/4!
    align 32
    exp_c5:         times 8 dd 0.008333333333333333    ; c5 = 1/5!
    align 32
    ; Clamping range for exp input (avoid overflow/underflow)
    exp_clamp_hi:   times 8 dd 88.3762626647949        ; max before overflow
    align 32
    exp_clamp_lo:   times 8 dd -87.3365447504      ; min before underflow
    align 32
    ; Magic number for float→int rounding: 2^23 + 2^22 (round-to-nearest)
    exp_magic_bias: times 8 dd 12582912.0              ; 1.5 * 2^23
    align 32
    ; For constructing 2^n from integer n via bit shift
    exp_127:        times 8 dd 127                     ; IEEE 754 exponent bias
    align 32
    ; Constants for general use
    ones_f32:       times 8 dd 1.0                     ; broadcast 1.0
    align 32
    zeros_f32:      times 8 dd 0.0                     ; broadcast 0.0

;==============================================================================
; Section: Code
;==============================================================================
section .text

;==============================================================================
; Function: simd_flash_attn_f32_avx2
;==============================================================================
;
; void simd_flash_attn_f32_avx2(const flash_attn_config_t* config)
;
; Parameters:
;   System V:  rdi = config pointer
;   Windows:   rcx = config pointer
;
; Implements Flash Attention 2 with online softmax for F32 Q/K/V tensors.
; Thread-safe: uses config->ith / config->nth for work partitioning.
;
;==============================================================================

global simd_flash_attn_f32_avx2

simd_flash_attn_f32_avx2:
    ;--------------------------------------------------------------------------
    ; PROLOGUE: Save callee-saved registers, set up stack frame
    ;--------------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

%ifdef WIN64
    ; Windows x64: RDI and RSI are callee-saved
    push    rdi
    push    rsi
    ; Save XMM6-XMM15 (callee-saved on Windows)
    sub     rsp, 160
    vmovaps [rsp +   0], xmm6
    vmovaps [rsp +  16], xmm7
    vmovaps [rsp +  32], xmm8
    vmovaps [rsp +  48], xmm9
    vmovaps [rsp +  64], xmm10
    vmovaps [rsp +  80], xmm11
    vmovaps [rsp +  96], xmm12
    vmovaps [rsp + 112], xmm13
    vmovaps [rsp + 128], xmm14
    vmovaps [rsp + 144], xmm15
    ; Remap Windows arg to System V register
    mov     rdi, rcx            ; config pointer → rdi
%endif

    ;--------------------------------------------------------------------------
    ; STACK FRAME ALLOCATION
    ; Align rsp to 32 bytes, then allocate FRAME_SIZE for locals.
    ;--------------------------------------------------------------------------
    mov     [rsp - 8], rbp      ; save original rbp for later (temp)
    mov     r12, rdi            ; r12 = config pointer (preserved across all loops)
    and     rsp, ~31            ; align to 32 bytes
    sub     rsp, FRAME_SIZE     ; allocate local frame
    mov     [rsp + STK_RBP_SAVE], rbp  ; save original rbp in frame
    mov     [rsp + STK_CONFIG], r12     ; save config pointer in frame

    ;--------------------------------------------------------------------------
    ; LOAD CONFIG: Read all needed fields from config struct into locals
    ;--------------------------------------------------------------------------
    ; Dimensions
    mov     rax, [r12 + CFG_HEAD_DIM_K]
    mov     [rsp + STK_HEAD_DK], rax
    mov     rax, [r12 + CFG_HEAD_DIM_V]
    mov     [rsp + STK_HEAD_DV], rax
    mov     rax, [r12 + CFG_N_QUERIES]
    mov     [rsp + STK_N_QUERIES], rax
    mov     rax, [r12 + CFG_N_KV]
    mov     [rsp + STK_N_KV], rax
    mov     rax, [r12 + CFG_N_HEAD_Q]
    mov     [rsp + STK_N_HEAD_Q], rax
    mov     rax, [r12 + CFG_N_HEAD_KV]
    mov     [rsp + STK_N_HEAD_KV], rax
    mov     rax, [r12 + CFG_BATCH_SIZE]
    mov     [rsp + STK_BATCH_SIZE], rax

    ; Scale: broadcast to 8-wide vector on stack (32-byte aligned at STK_SCALE_VEC)
    vbroadcastss ymm0, [r12 + CFG_SCALE]
    vmovaps [rsp + STK_SCALE_VEC], ymm0

    ; Mask pointer (may be NULL)
    mov     rax, [r12 + CFG_MASK_PTR]
    mov     [rsp + STK_MASK_PTR], rax
    ; Mask strides (only meaningful if mask != NULL)
    mov     rax, [r12 + CFG_MASK_NB]          ; mask_nb[0] = element stride
    mov     [rsp + STK_MASK_NB0], rax
    mov     rax, [r12 + CFG_MASK_NB + 8]      ; mask_nb[1] = row stride
    mov     [rsp + STK_MASK_NB1], rax

    ; Q strides (byte strides): nb[1]=row, nb[2]=head, nb[3]=batch
    mov     rax, [r12 + CFG_Q_NB + 8]
    mov     [rsp + STK_Q_NB1], rax
    mov     rax, [r12 + CFG_Q_NB + 16]
    mov     [rsp + STK_Q_NB2], rax
    mov     rax, [r12 + CFG_Q_NB + 24]
    mov     [rsp + STK_Q_NB3], rax

    ; K strides
    mov     rax, [r12 + CFG_K_NB + 8]
    mov     [rsp + STK_K_NB1], rax
    mov     rax, [r12 + CFG_K_NB + 16]
    mov     [rsp + STK_K_NB2], rax
    mov     rax, [r12 + CFG_K_NB + 24]
    mov     [rsp + STK_K_NB3], rax

    ; V strides
    mov     rax, [r12 + CFG_V_NB + 8]
    mov     [rsp + STK_V_NB1], rax
    mov     rax, [r12 + CFG_V_NB + 16]
    mov     [rsp + STK_V_NB2], rax
    mov     rax, [r12 + CFG_V_NB + 24]
    mov     [rsp + STK_V_NB3], rax

    ; Dst strides
    mov     rax, [r12 + CFG_DST_NB + 8]
    mov     [rsp + STK_DST_NB1], rax
    mov     rax, [r12 + CFG_DST_NB + 16]
    mov     [rsp + STK_DST_NB2], rax
    mov     rax, [r12 + CFG_DST_NB + 24]
    mov     [rsp + STK_DST_NB3], rax

    ;--------------------------------------------------------------------------
    ; THREAD PARTITIONING
    ; total_work = n_queries * n_head_q * batch_size
    ; Each thread gets a contiguous range [work_start, work_end).
    ; Work item i maps to (iq, ihq, ib) triple via modular decomposition.
    ;--------------------------------------------------------------------------
    mov     rax, [rsp + STK_N_QUERIES]
    imul    rax, [rsp + STK_N_HEAD_Q]
    imul    rax, [rsp + STK_BATCH_SIZE]
    ; rax = total_work

    ; If total_work == 0, nothing to do
    test    rax, rax
    jz      .epilogue

    mov     rcx, rax                ; rcx = total_work
    movsxd  r8, dword [r12 + CFG_NTH]  ; r8 = nth (total threads)
    movsxd  r9, dword [r12 + CFG_ITH]  ; r9 = ith (this thread index)

    ; work_per_thread = total_work / nth (integer division)
    ; remainder = total_work % nth
    ; work_start = ith * work_per_thread + min(ith, remainder)
    ; work_end   = work_start + work_per_thread + (ith < remainder ? 1 : 0)
    xor     edx, edx
    mov     rax, rcx
    div     r8                      ; rax = work_per_thread, rdx = remainder

    mov     r10, rax                ; r10 = work_per_thread
    mov     r11, rdx                ; r11 = remainder

    ; work_start = ith * work_per_thread + min(ith, remainder)
    mov     rax, r9
    imul    rax, r10                ; rax = ith * work_per_thread
    cmp     r9, r11
    jge     .thread_no_extra_start
    add     rax, r9                 ; + ith (since ith < remainder)
    jmp     .thread_start_done
.thread_no_extra_start:
    add     rax, r11                ; + remainder
.thread_start_done:
    mov     r13, rax                ; r13 = work_start (current work item)

    ; work_end = work_start + work_per_thread + (ith < remainder ? 1 : 0)
    mov     r14, rax
    add     r14, r10                ; + work_per_thread
    cmp     r9, r11
    jge     .thread_no_extra_end
    inc     r14                     ; + 1 if ith < remainder
.thread_no_extra_end:
    mov     [rsp + STK_WORK_END], r14

    ; If this thread has no work, bail out
    cmp     r13, r14
    jge     .epilogue

    ;==========================================================================
    ; MAIN WORK ITEM LOOP
    ; r13 = current work item index, incremented each iteration
    ; Each work item corresponds to one (iq, ihq, ib) triple.
    ;==========================================================================
    align 16
.work_loop:
    cmp     r13, [rsp + STK_WORK_END]
    jge     .epilogue

    ;----------------------------------------------------------------------
    ; Decompose work item index → (iq, ihq, ib)
    ; item = ib * (n_head_q * n_queries) + ihq * n_queries + iq
    ; So: ib = item / (n_head_q * n_queries)
    ;     rem = item % (n_head_q * n_queries)
    ;     ihq = rem / n_queries
    ;     iq  = rem % n_queries
    ;----------------------------------------------------------------------
    mov     rax, r13
    mov     rcx, [rsp + STK_N_HEAD_Q]
    imul    rcx, [rsp + STK_N_QUERIES]  ; rcx = n_head_q * n_queries
    xor     edx, edx
    div     rcx                         ; rax = ib, rdx = rem
    mov     [rsp + STK_IB], rax

    mov     rax, rdx                    ; rax = rem
    xor     edx, edx
    div     qword [rsp + STK_N_QUERIES] ; rax = ihq, rdx = iq
    mov     [rsp + STK_IHQ], rax
    mov     [rsp + STK_IQ], rdx

    ;----------------------------------------------------------------------
    ; GQA head mapping: ihkv = ihq * n_head_kv / n_head_q
    ; This maps multiple query heads to the same KV head.
    ;----------------------------------------------------------------------
    mov     rax, [rsp + STK_IHQ]
    imul    rax, [rsp + STK_N_HEAD_KV]
    xor     edx, edx
    div     qword [rsp + STK_N_HEAD_Q]  ; rax = ihkv
    mov     rbx, rax                     ; rbx = ihkv

    ;----------------------------------------------------------------------
    ; POINTER COMPUTATION for current work item
    ; All strides are in BYTES — use byte-level pointer arithmetic.
    ;----------------------------------------------------------------------
    mov     r12, [rsp + STK_CONFIG]

    ; Q_row = q_ptr + iq * q_nb[1] + ihq * q_nb[2] + ib * q_nb[3]
    mov     rax, [r12 + CFG_Q_PTR]      ; rax = q base pointer
    mov     rcx, [rsp + STK_IQ]
    imul    rcx, [rsp + STK_Q_NB1]
    add     rax, rcx
    mov     rcx, [rsp + STK_IHQ]
    imul    rcx, [rsp + STK_Q_NB2]
    add     rax, rcx
    mov     rcx, [rsp + STK_IB]
    imul    rcx, [rsp + STK_Q_NB3]
    add     rax, rcx
    mov     [rsp + STK_Q_ROW], rax

    ; K_base = k_ptr + ihkv * k_nb[2] + ib * k_nb[3]
    mov     rax, [r12 + CFG_K_PTR]
    mov     rcx, rbx                    ; ihkv
    imul    rcx, [rsp + STK_K_NB2]
    add     rax, rcx
    mov     rcx, [rsp + STK_IB]
    imul    rcx, [rsp + STK_K_NB3]
    add     rax, rcx
    mov     [rsp + STK_K_BASE], rax

    ; V_base = v_ptr + ihkv * v_nb[2] + ib * v_nb[3]
    mov     rax, [r12 + CFG_V_PTR]
    mov     rcx, rbx                    ; ihkv
    imul    rcx, [rsp + STK_V_NB2]
    add     rax, rcx
    mov     rcx, [rsp + STK_IB]
    imul    rcx, [rsp + STK_V_NB3]
    add     rax, rcx
    mov     [rsp + STK_V_BASE], rax

    ; dst_row = dst_ptr + iq * dst_nb[1] + ihq * dst_nb[2] + ib * dst_nb[3]
    mov     rax, [r12 + CFG_DST_PTR]
    mov     rcx, [rsp + STK_IQ]
    imul    rcx, [rsp + STK_DST_NB1]
    add     rax, rcx
    mov     rcx, [rsp + STK_IHQ]
    imul    rcx, [rsp + STK_DST_NB2]
    add     rax, rcx
    mov     rcx, [rsp + STK_IB]
    imul    rcx, [rsp + STK_DST_NB3]
    add     rax, rcx
    mov     [rsp + STK_DST_ROW], rax

    ;----------------------------------------------------------------------
    ; INITIALIZE per-query state
    ; O[0..head_dim_v) = 0.0
    ; row_max = -FLT_MAX (we use -infinity)
    ; row_sum = 0.0
    ;----------------------------------------------------------------------
    ; Zero out O accumulator: loop d = 0..head_dim_v step 8
    vxorps  ymm0, ymm0, ymm0
    mov     rcx, [rsp + STK_HEAD_DV]
    test    rcx, rcx
    jz      .skip_o_init
    xor     rax, rax                    ; d = 0
    align 16
.zero_o_loop:
    vmovaps [rsp + STK_O + rax*4], ymm0
    add     rax, 8
    cmp     rax, rcx
    jl      .zero_o_loop
.skip_o_init:

    ; row_max = -infinity (scalar in xmm15, kept across tile loop)
    vmovss  xmm15, [rel neg_inf]        ; xmm15[0] = -inf
    ; row_sum = 0.0 (scalar in xmm14)
    vxorps  xmm14, xmm14, xmm14         ; xmm14[0] = 0.0

    ;==========================================================================
    ; KV TILE LOOP
    ; Process KV positions in tiles of FA_TILE_KV (32).
    ; r15 = tile start position (t), incremented by Bc each iteration.
    ;==========================================================================
    xor     r15d, r15d                  ; t = 0
    mov     rcx, [rsp + STK_N_KV]
    test    rcx, rcx
    jz      .normalize_output           ; n_kv == 0 → skip to output

    align 16
.tile_loop:
    mov     rcx, [rsp + STK_N_KV]
    cmp     r15, rcx
    jge     .normalize_output           ; done all KV tiles

    ; tile_len = min(Bc, n_kv - t)
    mov     rax, rcx
    sub     rax, r15                    ; remaining = n_kv - t
    cmp     rax, FA_TILE_KV
    jle     .tile_len_ok
    mov     rax, FA_TILE_KV             ; clamp to Bc
.tile_len_ok:
    mov     r14, rax                    ; r14 = tile_len

    ;------------------------------------------------------------------
    ; STEP 1: QK DOT PRODUCTS
    ; For each j in [0, tile_len):
    ;   K_row = K_base + (t+j) * k_nb[1]
    ;   score[j] = dot(Q_row, K_row, head_dim_k)
    ; Score values stored in STK_SCORES on stack.
    ;------------------------------------------------------------------
    xor     ebx, ebx                    ; j = 0
    align 16
.qk_dot_loop:
    cmp     rbx, r14
    jge     .qk_dot_done

    ; Compute K_row = K_base + (t + j) * k_nb[1]
    mov     rax, r15
    add     rax, rbx                    ; t + j
    imul    rax, [rsp + STK_K_NB1]      ; * k_nb[1] (byte stride per KV row)
    add     rax, [rsp + STK_K_BASE]     ; + K_base
    mov     rsi, rax                    ; rsi = K_row

    ; Prefetch next K row (t + j + 1)
    mov     rax, r15
    lea     rax, [rax + rbx + 1]
    imul    rax, [rsp + STK_K_NB1]
    add     rax, [rsp + STK_K_BASE]
    prefetcht0 [rax]

    ; dot(Q_row, K_row, head_dim_k):
    ; Accumulate in ymm0, loop d=0..head_dim_k step 8
    mov     rdi, [rsp + STK_Q_ROW]      ; rdi = Q_row
    vxorps  ymm0, ymm0, ymm0            ; accumulator = 0
    mov     rcx, [rsp + STK_HEAD_DK]
    xor     edx, edx                     ; d = 0
    test    rcx, rcx
    jz      .qk_hsum                     ; head_dim_k == 0

    align 16
.qk_dim_loop:
    vmovups ymm1, [rdi + rdx*4]         ; Q[d..d+7]
    vfmadd231ps ymm0, ymm1, [rsi + rdx*4] ; acc += Q[d]*K[d]
    add     rdx, 8
    cmp     rdx, rcx
    jl      .qk_dim_loop

.qk_hsum:
    ; Horizontal sum of ymm0 → xmm0[0]
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1           ; [a7+a3, a6+a2, a5+a1, a4+a0]
    vpermilps xmm1, xmm0, 0x4E         ; swap high/low 64-bit lanes
    vaddps  xmm0, xmm0, xmm1
    vpermilps xmm1, xmm0, 0xB1         ; swap adjacent 32-bit elements
    vaddps  xmm0, xmm0, xmm1           ; xmm0[0] = total dot product

    ; Store score[j]
    vmovss  [rsp + STK_SCORES + rbx*4], xmm0

    inc     rbx
    jmp     .qk_dot_loop
.qk_dot_done:

    ;------------------------------------------------------------------
    ; STEP 2: SCALE scores
    ; score[j] *= scale for all j in tile
    ; Process 8 scores at a time with ymm.
    ;------------------------------------------------------------------
    vmovaps ymm7, [rsp + STK_SCALE_VEC]  ; ymm7 = broadcast(scale)
    xor     eax, eax                      ; j = 0 (process in groups of 8)
    align 16
.scale_loop:
    cmp     rax, r14
    jge     .scale_done
    ; How many to process? min(8, tile_len - j)
    mov     rcx, r14
    sub     rcx, rax
    cmp     rcx, 8
    jge     .scale_full_vec
    ; Partial: process remaining scalars one by one
.scale_scalar:
    vmovss  xmm0, [rsp + STK_SCORES + rax*4]
    vmulss  xmm0, xmm0, xmm7           ; xmm7[0] has scale
    vmovss  [rsp + STK_SCORES + rax*4], xmm0
    inc     rax
    cmp     rax, r14
    jl      .scale_scalar
    jmp     .scale_done
.scale_full_vec:
    vmovups ymm0, [rsp + STK_SCORES + rax*4]
    vmulps  ymm0, ymm0, ymm7
    vmovups [rsp + STK_SCORES + rax*4], ymm0
    add     rax, 8
    jmp     .scale_loop
.scale_done:

    ;------------------------------------------------------------------
    ; STEP 3: APPLY MASK (if mask != NULL)
    ; mask value for (iq, j_global) =
    ;   *(float*)((char*)mask + iq*mask_nb[0] + j_global*mask_nb[1])
    ; score[j] += mask_value
    ;------------------------------------------------------------------
    mov     rax, [rsp + STK_MASK_PTR]
    test    rax, rax
    jz      .mask_done                  ; mask is NULL, skip

    ; Precompute mask base for this query: mask + iq * mask_nb[0]
    mov     rcx, [rsp + STK_IQ]
    imul    rcx, [rsp + STK_MASK_NB0]   ; iq * mask_nb[0]
    add     rax, rcx                     ; rax = mask_base for this query row
    mov     rdi, rax                     ; rdi = mask_base

    mov     rsi, [rsp + STK_MASK_NB1]   ; rsi = mask_nb[1] (stride per KV position)

    xor     ebx, ebx                     ; j = 0
    align 16
.mask_loop:
    cmp     rbx, r14
    jge     .mask_done
    ; j_global = t + j
    mov     rax, r15
    add     rax, rbx                     ; j_global
    imul    rax, rsi                      ; j_global * mask_nb[1]
    vmovss  xmm0, [rdi + rax]            ; load mask value (float)
    vaddss  xmm0, xmm0, [rsp + STK_SCORES + rbx*4]  ; score[j] + mask
    vmovss  [rsp + STK_SCORES + rbx*4], xmm0
    inc     rbx
    jmp     .mask_loop
.mask_done:

    ;------------------------------------------------------------------
    ; STEP 4: TILE SOFTMAX (online algorithm)
    ;
    ; (a) tile_max = max(score[0..tile_len))
    ; (b) new_max = max(row_max, tile_max)
    ; (c) correction = exp(row_max - new_max)
    ; (d) O *= correction, row_sum *= correction
    ; (e) p[j] = exp(score[j] - new_max) for each j
    ; (f) tile_sum = sum(p[0..tile_len))
    ; (g) row_sum += tile_sum
    ; (h) row_max = new_max
    ;
    ; xmm15 = row_max (scalar), xmm14 = row_sum (scalar)
    ;------------------------------------------------------------------

    ; ── (a) Find tile_max ────────────────────────────────────────────
    vmovss  xmm0, [rel neg_inf]          ; tile_max = -inf
    xor     ebx, ebx
    align 16
.tile_max_loop:
    cmp     rbx, r14
    jge     .tile_max_done
    vmaxss  xmm0, xmm0, [rsp + STK_SCORES + rbx*4]
    inc     rbx
    jmp     .tile_max_loop
.tile_max_done:
    ; xmm0 = tile_max

    ; ── (b) new_max = max(row_max, tile_max) ─────────────────────────
    vmaxss  xmm1, xmm15, xmm0          ; xmm1 = new_max

    ; ── (c) correction = exp(row_max - new_max) ─────────────────────
    vmovss  xmm0, xmm15                 ; old row_max
    ; Use vmovaps to copy xmm1 to a safe register
    vmovaps xmm13, xmm1                 ; save new_max in xmm13
    vsubss  xmm0, xmm0, xmm13           ; row_max - new_max (≤ 0)
    ; Scalar exp: we need exp(xmm0[0])
    ; Broadcast to ymm, compute, extract
    vbroadcastss ymm0, xmm0
    call    .fast_exp_avx2               ; ymm0 = exp(ymm0)
    ; Extract scalar correction from xmm0[0]
    ; xmm0[0] = correction factor

    ; ── (d) Rescale O and row_sum by correction ──────────────────────
    vbroadcastss ymm6, xmm0             ; ymm6 = broadcast(correction)
    vmulss  xmm14, xmm14, xmm0          ; row_sum *= correction

    ; O[d] *= correction for d=0..head_dim_v step 8
    mov     rcx, [rsp + STK_HEAD_DV]
    xor     eax, eax
    test    rcx, rcx
    jz      .rescale_o_done
    align 16
.rescale_o_loop:
    vmovaps ymm1, [rsp + STK_O + rax*4]
    vmulps  ymm1, ymm1, ymm6
    vmovaps [rsp + STK_O + rax*4], ymm1
    add     rax, 8
    cmp     rax, rcx
    jl      .rescale_o_loop
.rescale_o_done:

    ; ── (e) p[j] = exp(score[j] - new_max) for j in tile ────────────
    ; Process in chunks of 8 for vectorized exp
    xor     eax, eax                     ; j = 0
    align 16
.compute_probs_loop:
    cmp     rax, r14
    jge     .compute_probs_done

    mov     rcx, r14
    sub     rcx, rax                     ; remaining
    cmp     rcx, 8
    jl      .compute_probs_scalar

    ; Vector path: 8 scores at once
    vmovups ymm0, [rsp + STK_SCORES + rax*4]
    vbroadcastss ymm1, xmm13            ; new_max
    vsubps  ymm0, ymm0, ymm1            ; score[j] - new_max
    call    .fast_exp_avx2               ; ymm0 = exp(ymm0)
    vmovups [rsp + STK_PROBS + rax*4], ymm0
    add     rax, 8
    jmp     .compute_probs_loop

.compute_probs_scalar:
    ; Scalar path for remaining < 8 elements
    vbroadcastss ymm1, xmm13            ; new_max
.compute_probs_scalar_inner:
    cmp     rax, r14
    jge     .compute_probs_done
    vmovss  xmm0, [rsp + STK_SCORES + rax*4]
    vsubss  xmm0, xmm0, xmm13           ; score[j] - new_max
    ; Scalar exp via broadcast + vector exp + extract
    vbroadcastss ymm0, xmm0
    ; Save rax across call
    mov     [rsp + STK_IQ], rax          ; temp save (we'll restore IQ later)
    push    rax
    call    .fast_exp_avx2
    pop     rax
    vmovss  [rsp + STK_PROBS + rax*4], xmm0
    inc     rax
    jmp     .compute_probs_scalar_inner
.compute_probs_done:

    ; Restore IQ (in case scalar path clobbered it — recompute from work item)
    ; IQ is still valid from work decomposition; scalar path only used temp save
    ; for rax which was restored. Re-derive IQ for safety:
    mov     rax, r13                     ; work item index
    mov     rcx, [rsp + STK_N_HEAD_Q]
    imul    rcx, [rsp + STK_N_QUERIES]
    xor     edx, edx
    div     rcx
    mov     [rsp + STK_IB], rax
    mov     rax, rdx
    xor     edx, edx
    div     qword [rsp + STK_N_QUERIES]
    mov     [rsp + STK_IHQ], rax
    mov     [rsp + STK_IQ], rdx

    ; ── (f) tile_sum = sum(p[0..tile_len)) ───────────────────────────
    vxorps  xmm0, xmm0, xmm0            ; tile_sum = 0
    xor     ebx, ebx
    align 16
.tile_sum_loop:
    cmp     rbx, r14
    jge     .tile_sum_done
    vaddss  xmm0, xmm0, [rsp + STK_PROBS + rbx*4]
    inc     rbx
    jmp     .tile_sum_loop
.tile_sum_done:

    ; ── (g) row_sum += tile_sum ──────────────────────────────────────
    vaddss  xmm14, xmm14, xmm0

    ; ── (h) row_max = new_max ────────────────────────────────────────
    vmovaps xmm15, xmm13                ; row_max = new_max

    ;------------------------------------------------------------------
    ; STEP 5: PV ACCUMULATION
    ; For each j in tile:
    ;   V_row = V_base + (t+j) * v_nb[1]
    ;   O[d] += p[j] * V_row[d]  for d in 0..head_dim_v
    ;------------------------------------------------------------------
    xor     ebx, ebx                     ; j = 0
    align 16
.pv_accum_loop:
    cmp     rbx, r14
    jge     .pv_accum_done

    ; Load p[j] and broadcast
    vbroadcastss ymm6, [rsp + STK_PROBS + rbx*4]  ; ymm6 = p[j] broadcast

    ; Compute V_row = V_base + (t+j) * v_nb[1]
    mov     rax, r15
    add     rax, rbx                     ; t + j
    imul    rax, [rsp + STK_V_NB1]       ; * v_nb[1]
    add     rax, [rsp + STK_V_BASE]      ; + V_base
    mov     rsi, rax                     ; rsi = V_row

    ; Prefetch next V row
    mov     rax, r15
    lea     rax, [rax + rbx + 1]
    imul    rax, [rsp + STK_V_NB1]
    add     rax, [rsp + STK_V_BASE]
    prefetcht0 [rax]

    ; O[d] += p[j] * V_row[d] for d=0..head_dim_v step 8
    mov     rcx, [rsp + STK_HEAD_DV]
    xor     edx, edx                     ; d = 0
    test    rcx, rcx
    jz      .pv_dim_done
    align 16
.pv_dim_loop:
    vmovaps ymm0, [rsp + STK_O + rdx*4] ; O[d..d+7]
    vmovups ymm1, [rsi + rdx*4]          ; V_row[d..d+7]
    vfmadd231ps ymm0, ymm6, ymm1        ; O += p[j] * V[d]
    vmovaps [rsp + STK_O + rdx*4], ymm0
    add     rdx, 8
    cmp     rdx, rcx
    jl      .pv_dim_loop
.pv_dim_done:

    inc     rbx
    jmp     .pv_accum_loop
.pv_accum_done:

    ;------------------------------------------------------------------
    ; Advance tile pointer and loop back
    ;------------------------------------------------------------------
    add     r15, FA_TILE_KV
    jmp     .tile_loop

    ;==========================================================================
    ; NORMALIZE OUTPUT: O[d] /= row_sum for d=0..head_dim_v
    ;==========================================================================
    align 16
.normalize_output:
    ; Check row_sum > 0 to avoid division by zero
    vxorps  xmm0, xmm0, xmm0
    vucomiss xmm14, xmm0
    jbe     .store_output               ; row_sum <= 0, skip normalization

    ; Compute 1.0 / row_sum and broadcast
    vmovss  xmm0, [rel ones_f32]
    vdivss  xmm0, xmm0, xmm14           ; 1.0 / row_sum
    vbroadcastss ymm6, xmm0             ; ymm6 = broadcast(1/row_sum)

    mov     rcx, [rsp + STK_HEAD_DV]
    xor     eax, eax
    test    rcx, rcx
    jz      .store_output
    align 16
.norm_loop:
    vmovaps ymm0, [rsp + STK_O + rax*4]
    vmulps  ymm0, ymm0, ymm6
    vmovaps [rsp + STK_O + rax*4], ymm0
    add     rax, 8
    cmp     rax, rcx
    jl      .norm_loop

    ;==========================================================================
    ; STORE OUTPUT: Copy O to dst_row
    ;==========================================================================
    align 16
.store_output:
    mov     rdi, [rsp + STK_DST_ROW]
    mov     rcx, [rsp + STK_HEAD_DV]
    xor     eax, eax
    test    rcx, rcx
    jz      .next_work_item
    align 16
.store_loop:
    vmovaps ymm0, [rsp + STK_O + rax*4]
    vmovups [rdi + rax*4], ymm0          ; dst may not be 32-aligned
    add     rax, 8
    cmp     rax, rcx
    jl      .store_loop

    ;==========================================================================
    ; ADVANCE to next work item
    ;==========================================================================
.next_work_item:
    inc     r13
    jmp     .work_loop

    ;==========================================================================
    ; EPILOGUE: Restore registers and return
    ;==========================================================================
    align 16


.epilogue:
    ; Restore rbp from frame (it was saved before alignment)
    mov     rbp, [rsp + STK_RBP_SAVE]

    ; Now unwind the prologue pushes in reverse order.
    ; rbp was set to rsp right after push rbp in the prologue.
    ; Callee-saved regs were pushed AFTER that, at rbp-8, rbp-16, etc.

%ifdef WIN64
    ; Restore XMM6-XMM15
    ; Prologue push sequence after mov rbp,rsp:
    ;   push rbx (rbp-8), push r12 (rbp-16), push r13 (rbp-24),
    ;   push r14 (rbp-32), push r15 (rbp-40), push rdi (rbp-48),
    ;   push rsi (rbp-56), sub rsp,160 → XMMs at rbp-216
    lea     rax, [rbp - 216]
    vmovaps xmm6,  [rax +   0]
    vmovaps xmm7,  [rax +  16]
    vmovaps xmm8,  [rax +  32]
    vmovaps xmm9,  [rax +  48]
    vmovaps xmm10, [rax +  64]
    vmovaps xmm11, [rax +  80]
    vmovaps xmm12, [rax +  96]
    vmovaps xmm13, [rax + 112]
    vmovaps xmm14, [rax + 128]
    vmovaps xmm15, [rax + 144]
    lea     rsp, [rbp - 56]              ; point to pushed rsi
    pop     rsi
    pop     rdi
%else
    ; System V: 5 callee-saved regs pushed after mov rbp,rsp
    lea     rsp, [rbp - 40]              ; point to pushed r15
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret

;==============================================================================
; SUBROUTINE: .fast_exp_avx2
;==============================================================================
;
; Fast vectorized exp(x) for 8 floats using AVX2 + FMA.
; Uses Cody-Waite range reduction and Horner polynomial approximation.
;
; Input:  ymm0 = x[0..7]
; Output: ymm0 = exp(x[0..7])
; Clobbers: ymm1, ymm2, ymm3, ymm4, ymm5 (ymm6-ymm15 preserved)
;
; Algorithm:
;   1. Clamp x to [-87.33, 88.37] to avoid overflow/underflow
;   2. n = round(x * log2(e))     — integer part for 2^n
;   3. r = x - n * ln2_hi - n * ln2_lo  — reduced argument (Cody-Waite)
;   4. p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))  — Horner
;   5. result = p * 2^n  — reconstruct via exponent bit manipulation
;
;==============================================================================
    align 16
.fast_exp_avx2:
    ; Step 1: Clamp input to valid range
    vmaxps  ymm0, ymm0, [rel exp_clamp_lo]  ; max(x, -87.33)
    vminps  ymm0, ymm0, [rel exp_clamp_hi]  ; min(x, 88.37)

    ; Step 2: Compute n = round(x * log2(e))
    ; Use the magic number trick: add 2^23+2^22, then subtract, for rounding
    vmulps  ymm1, ymm0, [rel exp_log2e]     ; ymm1 = x * log2(e)
    vaddps  ymm2, ymm1, [rel exp_magic_bias] ; add magic bias for rounding
    vsubps  ymm3, ymm2, [rel exp_magic_bias] ; ymm3 = n (rounded, as float)
    ; ymm2 still has integer bits of n in low mantissa bits

    ; Step 3: Cody-Waite range reduction: r = x - n*ln2_hi - n*ln2_lo
    vmulps  ymm4, ymm3, [rel exp_ln2_hi]    ; n * ln2_hi
    vsubps  ymm0, ymm0, ymm4                ; x - n*ln2_hi
    vmulps  ymm4, ymm3, [rel exp_ln2_lo]    ; n * ln2_lo
    vsubps  ymm0, ymm0, ymm4                ; r = x - n*ln2_hi - n*ln2_lo

    ; Step 4: Polynomial approximation using Horner's method
    ; p(r) = c5*r^5 + c4*r^4 + c3*r^3 + c2*r^2 + c1*r + 1
    ;       = 1 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    ;
    ; Evaluate inner-to-outer:
    vmovaps ymm1, [rel exp_c5]               ; p = c5
    vfmadd213ps ymm1, ymm0, [rel exp_c4]    ; p = c5*r + c4
    vfmadd213ps ymm1, ymm0, [rel exp_c3]    ; p = (c5*r+c4)*r + c3
    vfmadd213ps ymm1, ymm0, [rel exp_c2]    ; p = ((c5*r+c4)*r+c3)*r + c2
    vfmadd213ps ymm1, ymm0, [rel exp_c1]    ; p = (((c5*r+c4)*r+c3)*r+c2)*r + c1
    vfmadd213ps ymm1, ymm0, [rel ones_f32]  ; p = ((((c5*r+c4)*r+c3)*r+c2)*r+c1)*r + 1

    ; Step 5: Reconstruct exp(x) = p * 2^n
    ; 2^n is constructed by adding n to the IEEE 754 exponent bias (127)
    ; and shifting into the exponent field (bits 23..30).
    ; ymm2 contains n + magic_bias as integer bits; the low 23 bits = n as int.
    ; We need: (n + 127) << 23
    vpaddd  ymm2, ymm2, [rel exp_127]       ; int(n) + 127 (in low bits)
    vpslld  ymm2, ymm2, 23                  ; shift to exponent position
    ; ymm2 now represents 2^n as float bit pattern

    ; Multiply polynomial by 2^n
    vmulps  ymm0, ymm1, ymm2                ; result = p * 2^n

    ret
