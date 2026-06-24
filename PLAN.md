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
      into `build.zig`; **11 of 12 x86 variants pass golden-vector correctness**
      (test-simd pass=135, fail=0). Bugs fixed:
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
  - Build gotcha found: zig caches nasm output by argv only, so editing a `.asm`
    does NOT re-assemble in some cases — clear `.zig-cache` to force it.
- [ ] **Remaining:** `quantize_q8_k_f32 avx2` over-saturates some elements to
      -128 (an `smax`/iscale extraction bug; the avx512 variant is correct). It
      is the only skipped variant (`if (false)` in test-simd, documented inline).
      Then: INT8 GEMM microkernels, AVX512-VNNI / AMX dispatch, fused
      RoPE+attention.

---

## Phase 4 — Ease-of-use / LM Studio parity `~2-3 wk`

- [ ] **Model management**: `mlz models list|pull|rm`. `pull` from HuggingFace
      (resolve GGUF, resumable download). Local registry under `~/.mlz/models`.
- [ ] **Auto model load**: server resolves `model` field of request against
      registry; lazy-load + LRU-unload to fit memory (multi-model serving).
- [ ] **TUI dashboard** (optional, not GUI): live slots, tok/s, KV usage, queue
      depth. Reuse `terminal.zig`.
- [ ] More OpenAI endpoints: `/v1/completions`, `/v1/embeddings` (embedding
      models), `/v1/models` already present.

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
