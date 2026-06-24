# MLz Roadmap — vLLM/SGLang throughput, LM Studio ease, config-first tuning

## Positioning

MLz = thin, fast Zig serving layer over llama.cpp (GGML compute + GPU backends)
with custom CPU SIMD kernels. **Do not reinvent GGML.** Differentiate on three
axes only:

1. **Performance** — continuous batching + prefix-sharing serving loop (the
   thing llama.cpp gives you primitives for but MLz does not yet use).
2. **Ease of use** — config-file-first, sane auto-tuned defaults, model
   management.
3. **Tunability** — one declarative config (`mlz.toml`) controlling every knob,
   layered with env + CLI override.

Honest scope note: vLLM/SGLang win throughput via PagedAttention/RadixAttention
on their *own* CUDA kernels. MLz rides llama.cpp's KV cache + batch API, which
already supports multi-sequence decode and slot scheduling (llama-server proves
it). The win is wiring that up — not writing CUDA.

---

## Current state (baseline)

| Area | Status |
|---|---|
| Compute | llama.cpp + custom SIMD (AVX2/512, NEON) for vec_dot/silu/rope/layernorm/quantize/FA |
| Batching | **None** — `n_seq_max=1`, `engine.mutex` serializes server requests |
| Prefix cache | Single sequential `cached_tokens` longest-common-prefix reuse |
| Spec decode | Implemented (draft model), single-sequence |
| Server | OpenAI `/v1/chat/completions` (SSE + WebSocket), `/v1/models` |
| Config | CLI flags only (`src/config.zig`) |
| Observability | Per-turn stdout stats; no `/metrics`, no structured logs |

---

## Phase 0 — Config-first foundation (ease + tunability) `~1 wk`

The "easy to tune by config" ask. Ship before perf work so every later knob has
a home.

- [x] `mlz.toml` loader. Layer precedence: **defaults < file < env (`MLZ_*`) <
      CLI flags**. Hand-rolled flat TOML subset parser (no dep). File/env strings
      owned by a `Config` arena; CLI strings borrow argv. `--config <path>` +
      auto-discover `./mlz.toml`.
- [x] `mlz --print-config` dumps the fully-resolved effective config.
- [x] `mlz --init` writes a commented starter `mlz.toml` (refuses to overwrite).
      Source: `mlz.toml.example` / `config.starter_toml`.
- [x] Auto-tune: `threads = "auto"` → CPU count (engine default when null);
      `n_ctx = "auto"` → model's `n_ctx_train` (resolved in `engine.init` via
      sentinel 0). `n_gpu_layers = "auto"` → 999 = offload all layers (llama
      clamps to model layer count).
- [ ] **Deferred:** VRAM-aware *partial* GPU offload (probe free VRAM, fit a
      subset of layers). Needs a cross-platform backend VRAM query llama.cpp
      doesn't expose cleanly. "auto" = offload-all is the correct default until
      then; revisit when serving models larger than VRAM.

Config surface (additive to today's flags):
```toml
[model]
path = "models/qwen2.5-7b-q4.gguf"
n_ctx = 8192
n_gpu_layers = "auto"   # auto = probe VRAM
threads = "auto"

[serve]
host = "127.0.0.1"
port = 8080
max_concurrent = 8      # NEW: slot count (phase 1)
prefix_cache = true     # NEW: phase 2

[sampling]
temp = 0.8
top_k = 40
top_p = 0.95
min_p = 0.05

[speculative]
draft_model = ""        # optional
draft_tokens = 5
```

---

## Phase 1 — Continuous batching (THE perf win) `~3-4 wk`

Replace single-slot + global mutex with a multi-slot scheduler decoding N
sequences per `llama_decode`. This is what closes the throughput gap to
vLLM/SGLang on the same hardware.

- [x] Set `cparams.n_seq_max = max_concurrent` (engine.init); scheduler batch
      sized to `n_batch` (1024).
- [x] **Slot pool** (`src/scheduler.zig`): each slot = one `seq_id` with
      prefill/decode state + n_past; per-request sampler keeps sampling state
      independent across interleaved sequences.
- [x] **Scheduler loop**: single owner thread. Per step: admit queued requests
      into idle slots, pack one continuing-decode token per active slot + prefill
      chunks into one batch, `llama_decode`, sample each slot at its logits
      index, emit to its sink, free seq KV (`seq_rm`) on finish.
- [x] Engine routes `chat()` to the scheduler when `max_concurrent > 1` (no
      engine mutex in that path); single-stream path unchanged at N=1. Server
      needed no changes — concurrency comes from concurrent handler threads.
- [x] **Chunked prefill** (budget split: decode tokens first, then prefill
      chunks fill the rest).
- [x] Backpressure: bounded queue (`n_seq_max * 4`), `error.QueueFull` when full.
- [x] Prefix reuse (per-slot) — moved to **opt-in** in Phase 2 (see below). It
      was unsafe by default: correct on plain transformer KV caches but corrupts
      recurrent/hybrid models.
- [x] **Reliable benchmark** (`bench/bench_serve.py`, stdlib threads):
      sequential vs concurrent tok/s + the prefix-cache A/B.
- [ ] **Deferred:** speculative decoding in the batched path (single-stream
      keeps it); CI gating of the benchmark (Phase 3). Slow-client head-of-line
      blocking (owner thread does socket I/O) — upgrade to per-slot writer threads
      if it bites.

Validated end-to-end (gemma-3-4b Q6_K, `--max-concurrent 4`):
- correctness: 6 concurrent → 6 correct distinct answers, 2 queued past 4 slots;
  no crash/deadlock/leak.
- **throughput: 3.16x** (9.9 → 31.3 tok/s, 4 slots) via `bench_serve.py`.

Two pre-existing server bugs fixed while validating: (1) `openai.writeJson` never
flushed the `adaptToNewApi` adapter → every HTTP body/SSE chunk came out empty;
(2) `readHttpRequest` held method/path/header slices into a buffer that the body
read reallocated → any request whose body spanned multiple `recv()`s 404'd.

---

## Phase 2 — Prefix sharing / RadixAttention-lite `done`

SGLang's edge: reuse KV across requests with a common prefix (system prompts,
few-shot, multi-turn). Delivered as a **cross-slot prefix cache**
(`src/prefix_cache.zig`) — a request on ANY slot reuses a prefix that a request
on a DIFFERENT slot prefilled.

Two findings made it work. (1) A KV-lifecycle bug — `finishSlot` wiped a slot's
KV while the cache still claimed those tokens were live — was the sole cause of
the "model corrupts" symptoms (not architecture). (2) The defining cross-slot
primitive: a `--seqcp-test` experiment proved that **full-sequence
`seq_cp(src,dst,0,-1)` succeeds** on both Llama and gemma, while **sub-range
`seq_cp` aborts** (`GGML_ASSERT is_full`). So sharing is built on full copies +
end-aligned `seq_rm` truncation only.

- [x] **Cross-slot cache pool** (`PrefixCache`): a pool of dedicated cache
      sequences (ids past the serving slots, `n_cache = max_concurrent`), each
      holding one cached prompt prefix. `acquire` picks the longest-matching
      entry, truncates it to the matched length (end-aligned `seq_rm`), and
      full-copies it into the slot; `store` full-copies a slot's prefilled prompt
      KV into a free/LRU entry. `seq_cp` shares cells, so it's cheap. LRU
      eviction; graceful skip when a backend can't truncate.
- [x] Scheduler integration: `admit` calls `acquire` (clean clear on miss),
      snapshot calls `store`, `finishSlot` frees the slot (cache is independent).
      `n_seq_max = max_concurrent + n_cache`.
- [x] **Hit-rate metric** (`hits` / `reused_tokens` / `prefilled_tokens`) logged
      on shutdown; wire into `/metrics` in Phase 5.
- [x] `prefix_cache` **on by default** (`serve.prefix_cache` / `MLZ_PREFIX_CACHE`
      / `--prefix-cache`; disable with `--no-prefix-cache`).
- [ ] **Deferred:** split cached prefixes at branch points into a true radix
      *tree* (multi-node assembly via several full-copies) for finer-grained
      sharing; per-prefix ref-counting. The flat LRU pool already captures the
      dominant shared-system / multi-turn cases.

Validated (`--max-concurrent 4 --prefix-cache`, greedy) on Llama-3.2-1B
(transformer) **and** gemma-Q6 (hybrid):
- correctness: distinct, 4-way concurrent, multi-turn recall, and a 12-way
  concurrent stress (queue overflow) — all correct, 0 errors.
- **cross-slot reuse**: a cold request caches a long shared prefix (Llama 904 ms
  / gemma 4963 ms); the next *different* request that shares it reuses cross-slot
  (Llama 113 ms ≈ 8x / gemma 451 ms ≈ 11x).
- continuous-batching throughput unchanged: 4.19x on Llama-3.2-1B.

---

## Phase 3 — CPU SIMD kernel expansion (perf, existing strength) `ongoing`

Keep the custom-kernel differentiator for the CPU-only path (where llama.cpp is
the actual compute and SIMD matters).

- [x] **Bench/correctness regression gate.** `bench_simd --json` emits NDJSON
      per kernel; `tools/bench_gate.py` records a baseline (`bench/baseline.json`)
      and fails on >20% regression (`--update` / default `--check`) or, for CI
      where absolute GFLOPS vary per runner, a `--smoke` liveness check (every
      kernel finite > 0). `.github/workflows/simd.yml` runs `test-simd`
      (golden-vector correctness vs scalar refs) + the smoke on every push/PR
      (Windows runner — kernels use the Win64 ABI). `test-simd` is green:
      pass=78, fail=0.
- [x] **Golden-vector correctness** vs scalar reference already covers the
      shipping kernels (`test_simd.zig`): vec_dot q4_0..q8_k, rms_norm, rope_neox
      — all pass across sizes on AVX2 + AVX-512.
- [x] **New unary/vec kernels integrated + fixed** (layer_norm, quantize
      q8_0/q8_k, silu, rope_standard, vec_dot_f32; AVX2/AVX-512 + NEON). Wired
      into `build.zig`; **all 12/12 x86 variants pass golden-vector correctness**
      (test-simd pass=138, fail=0). Bugs fixed:
  - nasm assembly: 32-bit dest ← 64-bit `ARG_N`; `cmovg` immediate; bogus `_t`
    register suffix; vpermd rodata 32-byte alignment.
  - **silu** segfault — the scalar tail clamped the exponent through `eax`, which
    aliases `rax` = the source pointer, corrupting it after the first element;
    moved to a non-pointer scratch (`r10d`). Loosened the silu test tolerance
    (it's a degree-4 poly approximation of exp, ~5e-5 rel — exact for an
    activation).
  - **rope_standard avx2** — `vpermpd` duplicates the low lane into the high
    lane, so `vunpcklps` alone wrote the high half wrong; combine `vunpcklps` +
    `vunpckhps` + `vinsertf128`.
  - **quantize_q8_0 avx512** — used the 14-bit `vrcp14ps` reciprocal for
    `127/amax` → off-by-one quant; switched to exact `vdivss`.
  - **quantize_q8_k avx2** over-saturated batches 2..8 to -128: pass-2 pack
    reused `ymm13` (the int32 `127` clamp) and `ymm10/11/12/14` (consts +
    accumulators) as scratch, clobbering the clamp after the first batch. Not an
    smax bug as first thought — rewrote pack to use only `xmm7` scratch + reuse
    `ymm0..3`, leaving constants intact. Now passes all sizes.
  - Build gotcha found: zig caches nasm output by argv only, so editing a `.asm`
    does NOT re-assemble in some cases — clear `.zig-cache` to force it.
- [x] **INT8 GEMM microkernel** (`gemm_s8s8s32`, AVX2 + AVX512-VNNI fast path).
      `C[M,N] = A[M,K]·B[N,K]ᵀ`, s8·s8→s32. Uses ggml's `vpsignb` sign trick
      (x86 has no s8·s8 mul): AVX2 path `vpmaddubsw`+`vpmaddwd`, VNNI path one
      `vpdpbusd`. Runtime gate `simd_check_avx512_vnni()` (CPUID leaf-7
      VNNI bit 11 + VL bit 31). Golden-vector correctness vs scalar triple loop,
      incl. K%32 scalar tail (test-simd pass=152, fail=0). Benched ~136 GOPS
      (AVX2) / ~139 GOPS (VNNI) at 64×64×512.
- [x] **MR×NR GEMM tiling** (`gemm_s8s8s32_avx512vnni_t`, 4×2 register-blocked).
      Each A/B load is reused across the tile instead of re-read per (m,n),
      fixing the bandwidth bound. Uses the unsigned-offset trick (A XOR 0x80 →
      u8; correct with `128·ΣB[n]`, the column sum accumulated by `vpdpbusd`
      against a u8=1 vector) so `vpdpbusd` needs no per-row sign fold — VNNI-only
      because the AVX2 `vpmaddubsw` path would overflow int16 with u8∈[0,255].
      C dispatcher `simd_gemm_s8s8s32()` routes aligned shapes (M%4,N%2,K%32) to
      the tile, else the naive AVX2 kernel. Validated (test-simd pass=164 fail=0,
      incl. dispatcher on arbitrary shapes). **~359 GOPS vs ~123 naive VNNI =
      2.9×** at 64×64×512.
      - further levers: larger MR/NR (more reuse, more reg pressure); K-panel
        packing.
- [x] **AVX2 sign-trick GEMM tile** (`gemm_s8s8s32_avx2_t`, 4×2). AVX2 can't use
      the unsigned-offset trick (`vpmaddubsw` would overflow int16 at u8∈[0,255]),
      so it keeps the `|a|` sign trick — the fold is per (m,n) but the A/B loads
      are still shared across the tile. ~156 GOPS vs ~135 naive AVX2 = 1.22× (the
      extra sign folds eat into the reuse win; VNNI's offset trick is cleaner).
      Dispatcher now picks vnni-tile → avx2-tile → naive. test-simd pass=174/0.
- [x] **Fused RoPE + attention** (`simd_fused_rope_attn_f32`, f32 single head).
      Applies rotary embedding to Q and K *inline* inside the online-softmax
      (flash) loop, so rotated Q/K are never materialised — removes the separate
      RoPE pass and the read-back of the rotated tensors. GGML "standard"
      adjacent-pair RoPE; numerically matches a two-pass reference (rotate→attend)
      to ~1e-7 (test-simd pass=169 fail=0). Decode bench (n_q=1, n_kv=4096,
      D=128) **1.27× over the unfused two-pass** — the rotated-K memory
      round-trip; both still pay the per-element libm cos/sin, so precomputing
      per-position sin/cos tables is the next lever (would widen the gap), then a
      hand-tuned AVX/VNNI inner loop. C++ (consistent with the existing
      function-pointer flash-attn path), not asm.
- [x] **RoPE sin/cos tables + AVX2 fused inner loop.** Precompute per-(position,
      pair) cos/sin once (kill inner libm; reused across queries), and vectorise
      rotation + score dot + V accumulation with AVX2/FMA. Rotation uses the
      pair-swap identity `out = x*cos + sign*swap(x)*sin` (vpermilps 0xB1). Also
      fixed the algorithm: the fused kernel now rotates K **once** for n_q>1
      (re-rotating per query was n_q-fold waste) and inlines only at n_q==1.
      Speedup vs the pure-scalar baseline: **decode (n_q=1) 1.31×**,
      **prefill 2.5× (n_q=32) – 3.5× (n_q=128)**. Decode stays rotation-libm-
      bound; a vectorised sin/cos approximation (with mod-2π range reduction) is
      the remaining lever there. Why not asm: see the asm note below.
- [x] **Vectorised sin/cos for decode RoPE — asm vs C++, measured.** Decode was
      rotation-libm-bound; replaced the per-element `cosf/sinf` with a Cephes
      single-precision polynomial sin/cos (Pommier `sincos_ps`), 8 angles/instr,
      computed in BOTH a C++ AVX2-intrinsic path (`fra_rope_row_vec` /
      `simd_rope_row_cpp`) and a hand-written NASM kernel
      (`simd_rope_row_avx2`). Both match the scalar libm reference to ~1e-7
      (test-simd pass=180 fail=0). Decode (n_q=1, n_kv=4096, D=128) is now
      **~32× over the scalar baseline**.
  - **Head-to-head (rotate 4096×64 pairs ×4000): C++ intrinsics 1385 Mpair/s vs
    asm 1118 Mpair/s — intrinsics 1.24× faster.** The poly needs ~11 constants;
    the compiler keeps them in registers and schedules across iterations, while
    the asm re-broadcasts them from `.rodata` each use (too few ymm to cache them
    alongside the ~10 live sincos temps), and there is no control-flow edge for
    asm here. So the **C++ intrinsic path is the dispatch path**; the asm kernel
    is kept only as the reproducible comparison. Confirms the earlier call: keep
    RoPE+attention in C++ intrinsics, `.asm` for pure-data GEMM only.
- [ ] **Remaining:** AMX dispatch (needs Sapphire Rapids — untestable on this
      Zen4 box, deferred).

---

## Phase 4 — Ease-of-use / LM Studio parity `~2-3 wk`

- [x] **Model management** (`src/models.zig`): `mlz models list|pull|rm|dir`.
      Registry at `%LOCALAPPDATA%\mlz\models` (Win) / `$XDG_DATA_HOME|~/.local/
      share/mlz/models` (else). `pull` accepts a full URL or HuggingFace
      shorthand `owner/repo/file.gguf` (→ `…/resolve/main/…`), downloads via
      `std.http.Client` to a `.part` file, **resumable** (HTTP Range; restarts if
      the server ignores Range), atomic rename on completion. `list` shows sizes,
      `dir` prints the path. Unit-tested (resolveSource, list/resolve/remove);
      live pull+rm smoke-tested.
- [x] **Bare-name model resolution**: if `<model_path>` / `model =` isn't an
      existing file, it's resolved against the registry — so `mlz qwen2.5-0.5b`
      and `model = "qwen2.5-0.5b"` work for pulled models (`main.zig`).
- [x] **`/v1/completions`** (legacy text completion): prompt wrapped as a user
      message through the same engine path, response reshaped to the completion
      schema. `/v1/chat/completions`, `/v1/models`, `/health` already present.
      Parse unit-tested.
- [x] **Auto multi-model LRU load/unload** (`model_manager.zig` +
      `EngineManager`): the startup model is always resident; a request naming a
      different model (registry name or file path) loads it on demand into a
      refcount-pinned LRU pool (`server.max_loaded_models`, default 1 extra),
      evicting the least-recently-used *unpinned* engine at capacity. Pinning
      keeps an engine alive for the whole request even under concurrent eviction.
      Generic LRU core has 4 unit tests; live-verified: alt model by path loaded
      + generated, default still served afterward, unknown model → 404.
- [x] **`/v1/embeddings`** (`embeddings.zig`): model loaded in embedding mode
      (mean pooling), tokenize → decode → pooled vector, L2-normalised. Accepts a
      string or array of strings; lazy `EmbeddingService` caches one embedder and
      reloads on model change. Live-verified (returns normalised vectors).
- [x] **SSE streaming for `/v1/completions`** (`CompletionSseSink`): `text_
      completion` chunks + `[DONE]`. Live-verified (token chunks → finish → DONE).
- [ ] **Deferred:** TUI dashboard.

---

## Phase 5 — Observability & ops `~1 wk`

- [ ] `/metrics` Prometheus: tok/s, TTFT, queue depth, slot utilization, KV
      usage, prefix hit rate, accept rate (spec decode).
- [ ] Structured JSON logs (level via config).
- [ ] `/health` + `/health/ready` (model loaded & slots warm).

---

## Sequencing rationale

```
Phase 0 (config) ─┬─> Phase 1 (batching) ──> Phase 2 (prefix share)
                  └─> Phase 3 (SIMD, parallel track)
Phase 1 done ─────────> Phase 4 (UX) + Phase 5 (metrics)
```

Phase 1 is the single highest-leverage item — without continuous batching MLz
cannot approach vLLM/SGLang throughput regardless of kernel speed. Phase 0 is
cheap and unblocks tuning every later phase. Phases 3/4/5 are parallelizable.

## Explicit non-goals (YAGNI)

- No custom CUDA/Metal kernels — llama.cpp backends already cover GPU.
- No PagedAttention block allocator — llama.cpp KV cache + `seq_cp` is enough at
  this scale; revisit only if KV fragmentation measurably caps concurrency.
- No web GUI — TUI + OpenAI API is the surface. LM Studio's GUI is not the moat.
- No training / fine-tuning — inference only.
