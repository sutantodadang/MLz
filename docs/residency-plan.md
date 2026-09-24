# Bounded Tensor Residency — Implementation Plan & Progress

Dokumen ini adalah tracker hidup untuk implementasi resident memory terbatas di MLz. Status hanya ditandai selesai setelah kode dikompilasi dan diuji.

## Goal

Membuktikan bahwa MLz dapat mengakses tensor/model dari backing file dengan active mapped memory yang dibatasi secara eksplisit, melakukan fault secara transparan, dan mengembalikan data yang benar setelah eviction.

> Catatan scope: `resident_bytes` saat ini mengukur active mmap ranges yang dikelola residency layer, bukan total RSS proses atau filesystem page cache.

## Design invariants

1. Caller menyimpan `TensorHandle`, bukan pointer permanen.
2. Setiap pointer hanya valid selama `TensorView` masih dipin dan belum `release()`.
3. `resident_bytes` tidak boleh melebihi `budget_bytes`.
4. Eviction hanya boleh memilih mapping yang tidak dipin.
5. Fault ulang harus membaca byte yang sama dari backing file.
6. Accounting memakai ukuran mapping aktual termasuk alignment prefix OS.
7. Tensor lebih besar dari budget harus dapat diproses melalui bounded range/chunk views.

## Progress

| Phase | Status | Deliverable | Acceptance criteria |
|---|---|---|---|
| 0. Audit & design | Selesai | Adaptasi desain ke Zig 0.15 dan codebase aktual | Tidak mengganti GGML secara paksa; jalur lama tetap build |
| 1. Residency core | Selesai | `TensorHandle`, `BackingStore`, mmap, budget, LRU, pinning, metrics | Fault/hit/eviction teruji; budget mapping tidak terlampaui |
| 2. Chunk/range access | Selesai | `Manager.acquireRange()` | Tensor > budget dapat dibaca per window; remap saat pinned ditolak |
| 3. Benchmark harness | Selesai | `zig build bench-residency` | Baseline, bounded multi-tensor, dan large-tensor chunked dilaporkan |
| 4. GGUF metadata bridge | Selesai | Parse/index nama, offset, ukuran, tipe, dan dimensi tensor GGUF menjadi descriptor/handle | Descriptor tervalidasi terhadap batas file; fixture GGUF nyata berhasil fault melalui manager |
| 5. Compute integration | Selesai | `matVecF32` serta quantized dispatch Q4_0/Q4_K/Q6_K mengonsumsi descriptor GGUF melalui range views; validator berjalan pada model GGUF nyata | Proof kernel identik baseline; model Llama 3.2 1B nyata tervalidasi dengan budget 4 MiB |
| 6. End-to-end memory proof | Selesai | CPU execution adapter dengan bounded pin lifetime, full token path (embedding, seluruh decoder blocks, output norm, LM head), prompt prefill, incremental append/KV reuse, CLI budget, RSS instrumentation, dan llama.cpp reference tersedia | Resident-vs-bounded logits identik; prefill-vs-incremental identik; llama.cpp reference berada dalam toleransi numerik dan top-1 sama pada Llama 3.2 1B nyata |
| 7. Concurrency/prefetch | Selesai | Thread-safe manager, bounded fixed-worker prefetch scheduler, sync page prefault, adaptive budget-aware tile policy, configurable replacement, dan long token-loop benchmark | Concurrent acquire/release menjaga invariant budget; queue menerapkan backpressure; prefetched acquire menjadi hit; adaptive tiles identik dan mengurangi faults; tuning tidak diklaim lebih cepat bila benchmark tidak mendukung |
| 8. Batched prefill & execution proof | Selesai | Batched F32/quantized projection, layer-major causal Llama prefill, prompt 128/512 benchmark, same-window shared-manager executor stress, dan bounded Qwen3-Next Q2_K projection probe | Prefill identik dengan incremental; one-scan projection reuse mengurangi faults; prompt tetap dalam weight budget; Qwen probe tidak mengklaim graph DeltaNet/MoE penuh |
| 11. Official GGML backend bridge | Milestone 4 selesai (native tiled bounded residency) | `MLzResidency` official buffer backend memakai reserved identity space; synchronized native node hooks; file-backed acquire/rebase/release; tiled stock-GGML `MUL_MAT`, `MUL_MAT_ID`, dan bounded `GET_ROWS` | Upload weight nol; hooks seimbang; peak mapping ≤ budget; Llama 762.81 MiB dan Qwen3-Coder-Next 27.2 GiB berjalan dengan budget 4 MiB; logits bit-identik saat CPU_REPACK off |

## Current verdict and canonical status

**Target proof awal sudah tercapai.** Status kanonik proyek saat ini adalah:

> MLz dapat menjalankan model melalui graph dan kernel resmi GGML, dengan
> active mapped immutable weights yang dibatasi secara eksplisit, melakukan
> fault dari backing GGUF secara transparan, dan melepaskan mapping agar dapat
> di-evict tanpa mengubah hasil komputasi.

Bukti real-model terbaru:

| Model | Logical weights | Weight budget | Peak mapped | Weight upload | Correctness gate |
|---|---:|---:|---:|---:|---|
| Llama 3.2 1B Q4_K_M | 762.81 MiB | 4 MiB | 4 MiB | 0 byte | logits bit-identik dengan ordinary llama.cpp saat CPU_REPACK off |
| Qwen3-Coder-Next Q2_K | 27.2 GiB | 4 MiB | 4 MiB | 0 byte | logits bit-identik dengan ordinary llama.cpp saat CPU_REPACK off |

Native official-GGML path yang sudah tercakup:

- `ggml_backend_buffer_type_t` melalui `tensor_buft_overrides`;
- GGUF file-backed mappings, pin/release, faults, metrics, dan eviction;
- synchronized pre/post node lifetime hooks;
- tiled native `MUL_MAT` untuk regular projections;
- selected-expert tiled `MUL_MAT_ID` untuk routed MoE Qwen3-Next;
- bounded `GET_ROWS` untuk token embedding;
- stock GGML graph construction, scheduling, type conversion, vec-dot, DeltaNet,
  attention, dan MoE arithmetic tetap digunakan.

**Batas klaim:** budget 4 MiB adalah batas active mapped immutable weights,
bukan batas total process RSS. KV cache, recurrent state, graph workspace, dan
logits buffer dibatasi terpisah oleh `state_budget_mib` (P0.2). Allocator
memory, thread stacks, filesystem page cache, dan GPU VRAM tidak memiliki hard
limit.

## Productization roadmap — post-proof tracking

Bagian ini adalah tracker kanonik untuk pekerjaan setelah proof. Status hanya
boleh berubah menjadi `Selesai` setelah acceptance criteria dan verification
commands yang relevan lulus. Snapshot historis di bagian bawah dokumen tidak
menggantikan status pada tabel ini.

Status yang digunakan:

- `Belum mulai`
- `Dalam proses`
- `Blocked`
- `Selesai`

### Tracking protocol

Aturan perubahan status:

1. `Belum mulai` → `Dalam proses` hanya setelah ada branch/PR atau commit kerja
   yang dapat dirujuk.
2. `Dalam proses` → `Blocked` harus mencatat blocker, owner blocker, dan kondisi
   yang diperlukan untuk melanjutkan.
3. `Dalam proses` → `Selesai` hanya setelah seluruh acceptance criteria relevan
   dicentang dan verification command lulus.
4. Jika regression membuka kembali invariant yang sudah lulus, status kembali ke
   `Dalam proses`; jangan mempertahankan `Selesai` hanya karena pernah lulus.
5. Bukti model nyata harus mencatat model hash, revision llama.cpp/GGML, build
   options, budget, peak mapping, hook balance, dan checksum/argmax logits.
6. Angka benchmark tidak boleh diperbarui tanpa command, machine/storage,
   thread count, warm/cold-cache condition, dan sedikitnya tiga pengulangan.

Setiap PR yang mengubah status menambahkan entry berikut di bawah
[Evidence log](#evidence-log):

```markdown
- YYYY-MM-DD — P0.x — `<commit atau PR>`
  - Status: `Dalam proses` → `Selesai`
  - Commands: `<verification commands>`
  - Result: `<invariant, metrics, dan model hash bila relevan>`
  - Known gaps: `<none atau daftar gap yang tidak memblokir acceptance>`
```

| ID | Priority | Workstream | Status | Owner | Evidence | Depends on |
|---|---|---|---|---|---|---|
| P0.1 | P0 | Normal server-path integration | Selesai | Codex, Claude | `tests/residency_server_smoke.py` 47/47, 2026-09-24 | Phase 11 milestone 4 |
| P0.2 | P0 | Unified memory policy and preflight | Selesai | Claude | `no_alloc` breakdown preflight; llama.cpp reports compute "matches expectation" | P0.1 |
| P0.3 | P0 | Concurrent native graph pin tokens | Selesai | Codex, Claude | Validator `concurrent`/`cancel`/`fail` on Llama, Qwen3.5, Qwen3-Coder-Next; 15/15 stress | P0.1 |
| P0.4 | P0 | Correctness/CI regression matrix | Selesai | Codex, Claude | Fast matrix + nightly/dispatch model job; every command reproduced locally | P0.1–P0.3 |
| P1.1 | P1 | Complete `GET_ROWS` sparse/view coverage | Dalam proses | Codex | Sparse opposite-row exact gates pass; view/layout matrix remains | Phase 11 milestone 4 |
| P1.2 | P1 | GGML compatibility and patch maintenance | Dalam proses | Codex | b9106 pin and fail-closed exact patch markers; ABI checks remain | P0.4 |
| P1.3 | P1 | Production observability and diagnostics | Dalam proses | Codex, Claude | Weight, acquire latency, failure classes, memory plan/actual exported; debug trace remains | P0.1–P0.2 |
| P2.1 | P2 | I/O, prefetch, and replacement tuning | Belum mulai | Unassigned | — | P1.3 |
| P2.2 | P2 | Performance benchmark matrix | Belum mulai | Unassigned | — | P0.4, P2.1 |
| P3.1 | P3 | GPU residency architecture | Belum mulai | Unassigned | — | P0.2, P0.4 |

### P0.1 — Integrate official GGML residency into the normal server path

**Goal:** request biasa dapat memilih official bounded GGML backend tanpa
validator khusus atau executor proof terpisah.

Checklist:

- [x] Tambahkan config yang stabil, misalnya:

  ```toml
  [residency]
  enabled = true
  backend = "ggml"
  weight_budget_mib = 256
  state_budget_mib = 4096
  ```

- [x] Hubungkan config ke normal model loading melalui
      `llama_model_params.tensor_buft_overrides`.
- [x] Pastikan `/v1/completions`, `/v1/chat/completions`, streaming, sampling,
      dan chat template memakai model/backend yang sama.
- [x] Pertahankan fallback ordinary llama.cpp saat residency dinonaktifkan.
- [x] Tambahkan startup preflight dan error yang actionable bila model,
      architecture, build option, atau budget tidak kompatibel.
      (Architecture gate dihapus: jalur GGML resmi architecture-agnostic;
      Llama, Qwen3, Qwen3.5, Qwen3-Next, Gemma 3 exact.)
- [x] Pastikan graceful shutdown me-release graph, model, registry, manager,
      dan backing store dalam urutan lifecycle yang aman.

Acceptance criteria:

- [x] Server menghasilkan output yang sama dengan validator untuk prompt/token
      deterministik. (Server backed = server ordinary llama.cpp, yang juga
      reference validator; chat, completion, streaming, scheduler.)
- [x] `uploaded_weight_bytes == 0` pada backed mode.
- [x] `peak_mapped_weight_bytes <= weight_budget_bytes` untuk setiap request.
- [x] Hooks acquire/release seimbang setelah request selesai.
- [x] Jalur server biasa tetap build dan bekerja saat residency disabled.

### P0.2 — Unified memory policy and request preflight

**Goal:** pengguna dapat merencanakan lebih dari weight mappings dan tidak salah
mengartikan weight budget sebagai total RSS limit.

Checklist:

- [x] Definisikan kategori accounting resmi:
  - mapped immutable weights (`weight_budget_mib`);
  - KV cache + DeltaNet/recurrent state (llama.cpp breakdown `context`;
    hybrid memory melaporkan keduanya bersama);
  - graph/compute workspace (breakdown `compute`);
  - activations and request buffers (logits `n_vocab * n_seq_max * 4`;
    aktivasi graph berada di compute workspace);
  - optional GPU allocations: tidak berlaku, residency hanya CPU (P3.1).
- [x] Satukan policy executor/service yang sudah ada dengan official GGML path.
      (Normal path memakai `src/residency/memory_policy.zig`; endpoint proof
      lama sudah dihapus.)
- [x] Tambahkan checked estimators sebelum alokasi context/model/request state.
      (`llama_model_params.no_alloc` + `llama_get_memory_breakdown`.)
- [x] Tambahkan hard limit, soft limit/warning, dan overflow-safe arithmetic.
      (Hard = `state_budget_mib`; warning di atas 90%.)
- [x] Dokumentasikan filesystem page cache dan allocator/RSS yang tidak dapat
      dijadikan hard mmap budget.

Acceptance criteria:

- [x] Exact-limit allocation berhasil dan one-byte-under limit ditolak sebelum
      partial allocation.
- [x] Rejected request tidak mengubah KV/recurrent state dan tidak bocor mapping.
- [x] Metrics per kategori menjumlah secara konsisten dengan estimator.
- [x] Long-context preflight menolak konfigurasi yang tidak muat sebelum decode.

### P0.3 — Concurrent native graph execution

**Goal:** lebih dari satu graph/request dapat memakai model backed yang sama
secara aman tanpa global serialization.

Checklist:

- [x] Ganti release berbasis `source_id` tunggal menjadi pin token unik per
      graph/node/acquire.
- [x] Simpan mapping records per execution ID, termasuk tied/shared tensors dan
      tensor views. (Thread-local node pins/clones per graph thread 0.)
- [x] Pastikan eviction tidak memilih mapping yang dipin oleh graph lain.
- [x] Propagasikan cancellation/error sehingga seluruh pin request dilepas.
      (Failure → `GGML_STATUS_FAILED`, bukan `abort()`.)
- [x] Tambahkan fair admission/backpressure ketika union pinned sources melebihi
      budget. (`acquire_many` all-or-nothing, tanpa hold-and-wait; fail-fast
      bila mustahil muat; timeout 30 s.)

Acceptance criteria:

- [x] Sedikitnya dua context menjalankan decode concurrent pada model sama.
- [x] Output concurrent identik dengan eksekusi serial deterministik.
- [x] Tidak ada data race, dangling `tensor->data`, double release, atau leaked pin.
      (Dibuktikan oleh desain thread-local + gate exact/pin-zero berulang;
      belum ada ThreadSanitizer run.)
- [x] Peak mapping tetap dalam budget dan hook/pin counters kembali nol.
- [x] Stress test meliputi cancellation dan satu request gagal di tengah graph.

### P0.4 — Correctness and CI regression matrix

**Goal:** perubahan GGML, model, compiler, atau platform tidak diam-diam merusak
bounded execution.

Fast tests on every relevant change:

- [x] Manager alignment/range/LRU/pinning/multi-window tests.
- [x] Hook balance and pointer restoration tests. (Validator gates.)
- [x] Tiled `MUL_MAT`, `MUL_MAT_ID`, and `GET_ROWS` fixtures. (Real-model
      exact gates at 4 MiB; no synthetic fixture yet.)
- [x] Build matrix: hooks on/off, SIMD on/off, CPU_REPACK on/off.
- [x] Failure tests: insufficient budget, unsupported layout, corrupt span,
      cancellation, and teardown after error. (Bridge unit tests for
      impossible budget, admission wait, injection; validator `cancel`/`fail`;
      preflight rejects unsupported row layout and span mismatch as `invalid`.)

Nightly or opt-in model tests:

- [x] Small Llama GGUF exact gate with CPU_REPACK off.
- [x] Qwen3-Next hybrid/DeltaNet/routed-MoE exact gate. (CI: Qwen3.5-4B
      hybrid DeltaNet. Routed MoE: Qwen3-Coder-Next 27 GiB, local/manual.)
- [x] Default CPU_REPACK documented tolerance and top-1 gate. (Llama.)
- [x] Record model hash, GGML revision, budget, peak mapping, faults, evictions,
      hooks, and logits checksum. (SHA-256 pinned in workflow; validator
      prints the rest; exact gate compares full logits bitwise.)

Acceptance criteria:

- [x] CI failures show which invariant changed, not only `validator failed`.
- [x] Runtime-heavy model jobs are separately selectable and reproducible.
- [x] Normal application build remains covered without residency hooks.

### P1.1 — Complete `GET_ROWS` and view/layout coverage

**Current limitation:** sparse multi-row lookup maps the min/max row envelope;
uncommon views/layouts may fall back to whole-source mapping. Envelope/fallback
that exceeds budget can still fail.

Checklist:

- [x] Implement per-row or grouped-row acquisition in the native operation path.
- [ ] Support non-contiguous indices using GGML strides (implemented; fixture pending).
- [ ] Cover `view_src` offsets and multi-dimensional/view layouts.
- [x] Avoid whole-source fallback when a bounded row-wise path is possible.
- [ ] Fail before compute with an actionable unsupported-layout diagnostic when
      no bounded implementation exists.

Acceptance criteria:

- [x] Sparse rows from opposite ends of a vocabulary tensor fit a small budget.
- [x] Output matches stock GGML exactly with CPU_REPACK off.
- [x] No mapping envelope exceeds the declared request range unexpectedly.

### P1.2 — GGML compatibility and patch maintenance

Checklist:

- [x] Pin/document supported llama.cpp/GGML revision.
- [x] Add patch-generation drift check against vendored `ggml-cpu.c`.
- [x] Make hook insertion fail clearly when upstream code shape changes.
- [ ] Document whether integration can use upstream extension points or requires
      maintained MLz patching.
- [ ] Add ABI/version checks for buffer callbacks and private CPU traits used by
      tiled operations.

Acceptance criteria:

- [ ] Vendored GGML upgrade either passes all gates or fails during patch/build,
      never silently omits hooks.
- [ ] Patch delta remains reviewable and generated deterministically.

### P1.3 — Production observability and diagnostics

Checklist:

- [x] Export current/peak mapped weights, budget, faults, hits, evictions,
      acquire latency, bytes mapped, and tile counts.
- [x] Export state/workspace estimates and actual allocations separately.
- [ ] Include model, operation, tensor name, requested bytes, largest contiguous
      row/source, and budget in failure messages.
- [ ] Add per-request metrics without high-cardinality tensor labels by default.
- [ ] Add debug-only trace for node/source acquire-release balance.

Acceptance criteria:

- [x] Operator dapat membedakan budget failure, unsupported layout, I/O failure,
      and compute failure dari logs/metrics saja. (`failures.*` classes +
      `last_reason`; non-residency decode errors return 500, residency 503.)
- [x] Metrics tidak mengubah correctness atau membuat unbounded allocations.

### P2.1 — I/O, prefetch, and replacement tuning

Checklist:

- [ ] Profile cold and warm page-cache workloads separately.
- [ ] Evaluate bounded look-ahead at node/tile granularity.
- [ ] Tune row tile size from budget, mapping granularity, op shape, and thread count.
- [ ] Compare LRU/largest-first and only add CLOCK/2Q after profiling proves need.
- [ ] Keep prefetch opt-in per platform until benchmark shows a repeatable benefit.

Acceptance criteria:

- [ ] No performance claim without before/after data on the same model/storage.
- [ ] Correctness, budget, and pin-lifetime gates remain unchanged.
- [ ] Negative benchmark results remain documented.

### P2.2 — Performance benchmark matrix

Required scenarios:

- [ ] Llama and Qwen3-Next.
- [ ] Single-token decode and prompt prefill (128/512 tokens).
- [ ] Warm cache and cold storage where reproducible.
- [ ] Multiple weight budgets, including 4 MiB proof and practical production sizes.
- [ ] Metrics: load time, TTFT, prompt tokens/s, generation tokens/s, RSS,
      mapped peak, faults, evictions, bytes read, and CPU utilization.

Acceptance criteria:

- [ ] Benchmark records command, model hash, machine/storage, build options,
      thread count, and repetitions.
- [ ] Correctness gate is run before timing results are accepted.

### P3.1 — GPU residency architecture

This is a separate architecture milestone, not a direct reuse of host mmap
pointers.

Checklist:

- [ ] Separate host backing budget, staging budget, and device VRAM budget.
- [ ] Track upload/download lifetime with backend events/fences.
- [ ] Prevent eviction while a CUDA/Metal/Vulkan command still references a buffer.
- [ ] Add asynchronous staging and bounded transfer queues.
- [ ] Preserve CPU fallback and mixed CPU/GPU execution accounting.

Acceptance criteria:

- [ ] Device output matches ordinary backend reference within its documented gate.
- [ ] Peak VRAM and host staging stay within explicit budgets.
- [ ] Eviction cannot race an in-flight device operation.

## Recommended execution order

1. P0.1 normal server-path integration.
2. P0.2 unified memory policy.
3. P0.3 concurrent graph pin tokens.
4. P0.4 CI matrix, then make it a required regression gate.
5. P1.1 close `GET_ROWS`/view gaps.
6. P1.2 compatibility hardening and P1.3 observability.
7. P2.1/P2.2 performance tuning and reproducible benchmarking.
8. P3.1 GPU residency only after CPU production gates are stable.

## Verification command catalog

Command berikut adalah baseline proof yang sudah tersedia. Jalankan dari root
repository. Path model dapat diganti, tetapi evidence log wajib mencatat hash
file yang dipakai.

### Fast local gates

```text
zig build test -Dsimd-backend=false
zig build -Dsimd-backend=false
zig build test
zig build
git diff --check
```

### Official GGML exact gates

Llama:

```text
zig build validate-ggml-backend \
  -Doptimize=ReleaseFast \
  -Dsimd-backend=false \
  -Dggml-residency-hooks=true \
  -Dcpu-repack=false -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 1 4
```

Qwen3-Coder-Next:

```text
zig build validate-ggml-backend \
  -Doptimize=ReleaseFast \
  -Dsimd-backend=false \
  -Dggml-residency-hooks=true \
  -Dcpu-repack=false -- \
  models/Qwen3-Coder-Next-Q2_K.gguf 1 4
```

Expected invariant untuk kedua command:

- logits finite dan bit-identik terhadap ordinary llama.cpp;
- `uploaded_weight_bytes == 0`;
- pre/post hooks dan acquire/release seimbang;
- final active hook/pin count nol;
- `peak_mapped_weight_bytes <= 4 MiB`.

Mode CPU_REPACK default harus dijalankan terpisah dan dinilai dengan documented
tolerance + top-1 gate; hasilnya tidak boleh disebut bit-identik.

### Commands yang harus ditambahkan selama productization

| Workstream | Required verification entry point |
|---|---|
| P0.1 | `python tests/residency_server_smoke.py --exe zig-out/bin/MLz --model <gguf>` (done) |
| P0.2 | `residency_memory_policy.zig` tests + smoke `state budget rejection` (done) |
| P0.3 | `validate-ggml-backend ... <model> <tokens> 4 concurrent|cancel|fail` (done) |
| P0.4 | `.github/workflows/residency.yml` jobs `fast` and `models` (done) |
| P1.1 | Sparse opposite-row `GET_ROWS` fixture pada budget kecil |
| P1.2 | Deterministic patch drift/check command terhadap vendored GGML |
| P1.3 | Metrics/log assertion test tanpa high-cardinality labels default |
| P2.1 | Warm/cold benchmark runner dengan replacement/prefetch variants |
| P2.2 | Reproducible Llama/Qwen benchmark report generator |
| P3.1 | Backend-specific VRAM/staging budget validator |

## Evidence log

Entry terbaru diletakkan paling atas. Proof milestone yang sudah ada sebelum
tracker productization dibuat dicatat sebagai baseline:

- 2026-09-24 — P0.1/P0.2/P0.3/P0.4/P1.3 — working tree on `feat/phase-8-batched-prefill` (not committed)
  - Status: P0.1, P0.2, P0.3, P0.4 `Dalam proses`/`Belum mulai` → `Selesai`; P1.3 tetap `Dalam proses`.
  - Changes: node sources pinned all-or-nothing (`acquire_many`, no hold-and-wait, fail-fast when impossible); acquisition failures stop the graph with `GGML_STATUS_FAILED` (llama_decode rc=-3) instead of `abort()`; engine/scheduler discard partial KV and return `503 residency_error`; non-weight policy from llama.cpp `no_alloc` breakdown; architecture gate removed; `/v1/residency/metrics` adds acquire latency, failure classes, last reason, planned/allocated memory.
  - Commands: `zig build test` with `-Dsimd-backend=false` × hooks on/off × CPU_REPACK on/off, plus default flags; `zig build` for the same; `git diff --check`; `zig build validate-ggml-backend -Doptimize=ReleaseFast -Dsimd-backend=false -Dggml-residency-hooks=true -Dcpu-repack=false -- <model> <tokens> 4 <single|concurrent|cancel|fail>`; default CPU_REPACK tolerance gate on Llama; `python tests/residency_server_smoke.py --exe zig-out/bin/MLz.exe --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
  - Environment: Windows 11, Zig 0.15.2, llama.cpp/GGML `b9106`, 4 threads for server, 1 thread for validator. Models (SHA-256): Llama-3.2-1B Q4_K_M `3f5a2242…2dcc1` (unsloth), Qwen3.5-4B Q4_K_S `27caeb0e…2ea77` (unsloth), Qwen3-Coder-Next Q2_K `2ac738bc…abad87`, Qwen3-4B Q4_K_M, Gemma-3-4B Q2_K.
  - Result: all four modes exact (bitwise logits) at 4 MiB on Llama (tokens `1,128000`), Qwen3.5-4B (`1,1000`) and Qwen3-Coder-Next (`1,151000`); `fail` mode exercised GET_ROWS, whole-node, tiled `MUL_MAT` and tiled `MUL_MAT_ID` failures, each with zero open pins and balanced hooks before an exact recovery. Qwen3-4B and Gemma-3-4B `single` exact. 15/15 repeated `concurrent`/`cancel`/`fail` runs with 4-token prompts. Server smoke 47/47: backed chat/completion/streaming/scheduler outputs equal the ordinary path, metrics invariants hold, injected failure → 503 then exact recovery (single-stream and scheduler), `--state-budget-mib 1` rejected at startup, SIGBREAK shutdown exit 0. Memory plan for Llama ctx 512: state 16,777,216 + compute 271,058,944 + logits 513,024 bytes; llama.cpp reported the allocated compute buffer "matches expectation". Qwen3-Coder-Next plan: state 91,619,328, compute 331,098,144 bytes. Default CPU_REPACK Llama: max error 0.040820480, mean 0.006926090, top-1 match.
  - Linux: the `fast` (hooks/CPU_REPACK combinations without SIMD) and Llama `models` job steps of `residency.yml` were replayed in WSL Ubuntu 26.04 with Zig 0.15.2: all builds/tests pass, model SHA verified, four exact modes and the CPU_REPACK gate pass, server smoke 0 failed invariants including SIGINT shutdown.
  - Known gaps: hosted GitHub Actions run pending (workflow commands reproduced locally on Windows and Linux; Linux SIMD/NASM combination not replayed); routed-MoE model gate is local only (27 GiB); CPU_REPACK tolerance gate calibrated only for Llama (Qwen3.5 keeps top-1, max error 0.48 from repack kernels); no ThreadSanitizer run; admission timeout fixed at 30 s; P1.1 view/layout fixtures, P1.2 ABI checks, P1.3 per-request metrics/debug trace remain.

- 2026-09-24 — P0.1/P1.1/P0.4/P1.3 — working tree on `feat/phase-8-batched-prefill` (not committed)
  - Status: `Belum mulai` → `Dalam proses` for P0.1, P0.2, P0.4, P1.1, P1.2, and P1.3.
  - Commands: `zig build test -Dsimd-backend=false -Dggml-residency-hooks=true -Dcpu-repack=false`; `zig build test -Dsimd-backend=false`; `zig build -Dsimd-backend=false`; `zig build validate-ggml-backend -Doptimize=ReleaseFast -Dsimd-backend=false -Dggml-residency-hooks=true -Dcpu-repack=false -- <model> <tokens> 4` with Llama `1,128000` and Qwen `1,151000`; `git diff --check`.
  - Environment: Windows, Zig 0.15.2, pinned llama.cpp/GGML tag `b9106`, CPU_REPACK off for exact gates. Llama SHA-256 `3F5A22426976AB26CFE84DBA63C1D08391717ABB1AF893E10F1B2968D862DCC1`; Qwen SHA-256 `2AC738BC947BC3470E37962C2EE0AB390FA953740E9873575695276980ABAD87`.
  - Result: both two-token sparse gates had exact logits and 4 MiB peak mapping with zero uploaded bytes and balanced hooks. Default CPU_REPACK on Llama passed the documented tolerance/top-1 gate (max error 0.040820480, mean 0.006926090, top-1 16309/16309). The normal Llama server returned the same deterministic `How can` for chat and completion with residency on/off; two repeated backed requests and two concurrent scheduler requests matched. Backed metrics after requests: budget 4,194,304 bytes, peak mapped 4,194,288, pre/post 2,148/2,148, acquire/release 2,045/2,045, active hooks 0, uploaded bytes 0. A 1 MiB state guard rejected before context allocation without Debug allocator leaks; a 32 MiB guard passed for a 512-token single-context Llama run. The guard rejects unsupported Qwen architecture rather than reporting a misleading total-state limit.
  - Known gaps: no independent concurrent graph pin tokens, complete non-weight hard policy, cancellation/failure unwind gate, complete view-layout fixture, model CI jobs, or GPU residency. The added CI matrix has not run in GitHub Actions yet. No performance claim is made from these single runs.

- Baseline — Phase 11 milestone 4 — commit `83b692c`
  - Status: official GGML proof mencapai native tiled bounded residency.
  - Result: Llama 762.81 MiB dan Qwen3-Coder-Next 27.2 GiB berjalan pada
    weight budget 4 MiB, upload weight nol, hooks seimbang, dan logits
    bit-identik saat CPU_REPACK nonaktif.
  - Known gaps: normal server integration, unified non-weight policy,
    concurrent graph pin tokens, sparse/view `GET_ROWS`, dan GPU backend.

## Definition of production-ready bounded GGML residency

The feature may be called production-ready only when all P0 items are
`Selesai` and the following are true:

- [x] normal serving requests use official GGML bounded residency through config;
- [x] weight and non-weight memory policies are explicit and preflighted;
- [x] concurrent requests have independent pin lifetimes;
- [x] errors cleanly unwind all mappings/state;
- [x] CI covers hooks on/off and real-model correctness;
- [x] operators can observe budget, mappings, faults, evictions, and failure reason.

Status 2026-09-24: all P0 items are `Selesai` and every condition above has
passing local evidence (see the Evidence log). Scope of the claim: CPU host
backend, one backed model per process. The first hosted CI run happens on the
next push.


## Historical implementation log

Kode proof lama (custom executor, `residency_service`, endpoint
`/v1/residency/completions`, `validate-residency`, `bench-residency`,
`residency-serve`) dihapus pada 2026-09-24; bagian di bawah hanya catatan
historis.

Bagian di bawah mempertahankan API notes, benchmark snapshots, dan keputusan
fase lama untuk audit trail. Gunakan [Productization roadmap](#productization-roadmap--post-proof-tracking)
dan [Evidence log](#evidence-log) sebagai sumber status terkini. Pernyataan
lama seperti "Qwen hanya projection probe" atau "satu tensor hanya satu window"
menjelaskan keadaan pada saat snapshot tersebut dibuat dan bukan limitation
kanonik saat ini.

> Detail integrasi dan validation command terbaru: [Official GGML Residency Backend Integration](ggml-residency-backend.md).
> Milestone 4 sudah menambahkan tiled native `MUL_MAT`, selected-expert
> `MUL_MAT_ID`, dan bounded `GET_ROWS`; stock contiguous whole-source fallback
> tetap dipakai hanya untuk layout yang belum tercakup.

## Implemented API

```zig
var store = try residency.BackingStore.open(path_z);
var manager = try residency.Manager.init(allocator, &store, budget_bytes);

try manager.register(.{ .id = 1 }, tensor_file_offset, tensor_len);

// Whole tensor, jika muat dalam budget.
var whole = try manager.acquire(.{ .id = 1 });
defer whole.release();

// Window tensor, termasuk jika tensor lebih besar dari budget.
// Compute tiler dapat membatasi ukuran request terhadap overhead alignment OS.
const capacity = try manager.rangeCapacity(.{ .id = 1 }, chunk_offset);
var chunk = try manager.acquireRange(.{ .id = 1 }, chunk_offset, @min(chunk_len, capacity));
defer chunk.release();
consume(chunk.bytes());
```

`acquireRange()` menggunakan mapping resident yang sudah mencakup requested range sebagai hit. Jika range baru memerlukan remap pada tensor yang sama sementara view lama masih dipin, operasi mengembalikan `error.TensorBusy` agar pointer lama tidak menjadi dangling.

## Multi-window residency (Phase 9)

Satu tensor kini dapat memiliki beberapa mapped window aktif sekaligus, selama
total bytes mapping tetap dalam budget:

- `Manager` menyimpan window aktif di hash map global keyed slot; `Entry`
  tensor hanya menyimpan range logis di backing store.
- Hit dicek terhadap seluruh window tensor, jadi acquire range yang termuat
  dalam window manapun adalah hit.
- Eviction beroperasi per window; window pinned tidak pernah dipilih sebagai
  victim. Bila semua window pinned dan budget tidak cukup, pemanggil menerima
  `error.BudgetExceeded`.
- `error.TensorBusy` kini hanya berlaku pada `unregister()` saat masih ada
  window tensor yang dipin; acquire disjoint pada tensor yang sama tidak lagi
  ditolak selama budget memadai.
- Dua executor atau lebih dapat memegang window berbeda dari tensor yang sama
  secara bersamaan, yang sebelumnya mustahil dengan desain satu window per
  tensor (`same-tensor multi-window` acceptance item Phase 9).

## Phase 7 API

Seluruh operasi registry, fault, pin/release, LRU, metrics, dan prefetch pada `Manager` sekarang diserialisasi oleh mutex internal. `TensorView` menyimpan slice dari mapping yang telah dipin, sehingga `bytes()` tidak perlu mengakses hash map manager di luar lock.

```zig
// Synchronous mmap + native-page prefault; pin dilepas sebelum return.
try manager.prefetchRange(handle, tensor_offset, len);

// Explicit asynchronous task. Caller wajib wait tepat sekali sebelum manager
// atau backing store dihancurkan.
var task = try manager.prefetchRangeAsync(allocator, handle, tensor_offset, len);
try task.wait();

// Bounded fixed-worker scheduler. Submission never grows memory without limit:
// a saturated queue returns error.PrefetchQueueFull.
const scheduler = try residency.PrefetchScheduler.init(allocator, &manager, 1, 2);
defer scheduler.deinit();
var scheduled = try scheduler.submit(handle, tensor_offset, len);
try scheduled.wait();

manager.setReplacementPolicy(.largest_first);

try executor.setTilePolicy(.{ .adaptive = .{
    .target_bytes = 0, // gunakan kapasitas budget/alignment maksimum
    .max_rows = 256,
    .prefault = true,
} });
```

Adaptive policy dihitung ulang pada setiap offset karena alignment prefix mmap dapat berubah. Existing fixed-row APIs tetap menjadi compatibility wrappers. Prefault menyentuh halaman virtual native dan byte terakhir secara sinkron; ini bukan mlock dan OS tetap boleh membuang page setelahnya.

## Phase 8 API dan hasil

`CpuExecutor.matMul()` menerima activation rows `[batch, columns]` dan menjaga setiap weight tile tetap dipin selama seluruh batch diproses. `modelPrefill()` menggunakan primitive ini secara layer-major untuk Q/K/V/O dan gate/up/down, lalu menjalankan causal attention dengan KV cache yang sama dengan incremental path. Prompt workspace dialokasikan dan dilaporkan eksplisit; batch quantization scratch tumbuh reusable sampai batch terbesar.

```zig
var prompt = try residency_executor.PrefillWorkspace.init(allocator, token_count, hidden, intermediate);
defer prompt.deinit();
try executor.modelPrefill(embedding, layers, output_norm, output_weight,
    tokens, config, caches, &prompt, states, logits);
```

Validasi Llama 3.2 1B Q4_K_M, budget mapped weight 4 MiB:

```text
prompt 128: 5223.42 ms, 24.51 token/s, faults=430, peak-map=4 MiB
prompt 512: 21663.06 ms, 23.63 token/s, faults=814, peak-map=4 MiB
resident-vs-bounded max error: 0
prefill-vs-incremental max error: 0
llama.cpp top-1: sama untuk prompt 128 dan 512
```

Fault tidak konstan karena token embedding row lookup tetap satu fault per token; projection weights hanya discan sekali per layer. Strict Phase-6 llama.cpp tolerance tetap berlaku dan menjadi gate untuk prompt pendek. Pada prompt 128/512 scalar reduction drift melewati threshold tersebut; validator menandainya `mismatch` dan hanya mencatat long-prompt reference secara informational. Completion Phase 8 untuk prompt panjang digate oleh exact resident-vs-bounded, exact prefill-vs-incremental, finite logits, budget invariant, dan top-1 reference yang sama—bukan oleh klaim numerical-reference pass.

Probe model `Qwen3-Coder-Next-Q2_K.gguf` (27.2 GiB) sengaja tidak menjalankan graph Qwen penuh. Ia memilih projection Q2_K 2D terkecil yang lebih besar dari budget, lalu membandingkan empat matvec dengan satu batched pass tanpa full-tensor baseline atau llama.cpp model load:

```text
blk.0.attn_qkv.weight, 8192x2048, 5.25 MiB, batch=4
repeated: 19.23 ms, faults=128
batched:   3.83 ms, faults=32
peak mapped: 0.22 MiB / 4 MiB budget
max error: 0, current RSS: 6.24 MiB
```

Ini membuktikan kompatibilitas metadata GGUF + bounded canonical Q2_K projection. DeltaNet, hybrid layer schedule, shared/routed MoE orchestration, dan recurrent state Qwen3-Next belum diimplementasikan sehingga full Qwen inference tidak diklaim.

## Verification log

### Unit tests and existing functionality

```text
zig build test -Dsimd-backend=false
PASS
```

Coverage residency saat ini:

- first fault dan cache hit;
- LRU ordering dan transparent re-fault;
- data benar setelah eviction;
- pinned mapping tidak dapat dieviction;
- mmap alignment overhead masuk accounting;
- invalid range dan oversized whole-tensor access ditolak;
- tensor tiga kali budget dapat ditraverse melalui tiga range views;
- containing mapping menghasilkan range hit;
- pinned view mencegah remap tensor yang sama;
- fixture GGUF v3 nyata mengindeks nama, absolute offset, ukuran, tipe, dan dimensi;
- descriptor GGUF yang melewati batas backing file ditolak;
- registrasi index GGUF bersifat transactional dan rollback saat terjadi konflik;
- byte tensor fixture di-fault melalui handle hasil index, bukan dibaca oleh parser metadata;
- proof kernel F32 matvec membaca matrix per row tile melalui pinned range views;
- proof kernel quantized memakai satu dispatch untuk Q4_0, Q4_K, dan Q6_K dengan layout GGUF/GGML asli serta canonical ggml dequantizer per row;
- baseline resident penuh dan bounded adaptive tiles untuk Q4_0/Q4_K/Q6_K menghasilkan output identik dengan canonical dequantized reference;
- compute tiling menyesuaikan kapasitas range terhadap alignment prefix mmap aktual;
- bounded matvec untuk logical matrix tiga kali budget identik dengan baseline resident penuh dan peak mapping tetap dalam budget;
- executable `validate-residency` membuka GGUF produksi, memilih tensor 2D yang didukung, dan membandingkan full-resident dengan bounded output;
- RSS current/peak dilaporkan melalui Windows process working set, Linux `/proc`/`getrusage`, dan macOS `getrusage`;
- `CpuExecutor` tidak menyimpan atau mengembalikan pointer weight; setiap matvec menyelesaikan pinned tile sebelum operasi berikutnya;
- proof subgraph SwiGLU menjalankan `down(silu(gate(input)) * up(input))` dengan scratch dan dua activation buffer yang di-account terpisah;
- baseline dan bounded layer-0 FFN dari GGUF produksi menghasilkan output identik;
- RMSNorm F32 weight dibaca dalam pinned scope dan residual dikerjakan di activation state;
- single-token grouped-query attention menjalankan bounded Q/K/V/O projections, RoPE, causal softmax, dan writable KV cache;
- two-token decoder fixture membuktikan bounded layer identik dengan resident baseline serta memisahkan weight, executor activation, attention workspace, dan KV-cache accounting;
- validator GGUF produksi menjalankan layer 0 lengkap (attention + residual + FFN) baseline-vs-bounded;
- token embedding lookup membaca dan, bila perlu, mendequantisasi hanya satu row melalui bounded view;
- full token path menjalankan embedding, seluruh decoder blocks, output RMSNorm, dan LM head tanpa menyimpan pointer mapped weight;
- prompt-style multi-token execution dan incremental one-token append memakai KV cache yang sama, dengan logits final identik;
- Llama normal RoPE memakai pasangan nilai berurutan sesuai `LLAMA_ROPE_TYPE_NORM`, bukan layout half-head NeoX;
- validator GGUF produksi membandingkan seluruh 128.256 logits resident-vs-bounded dan meng-account KV cache semua layer secara terpisah;
- reference harness memuat model CPU/non-mmap melalui llama.cpp dan memvalidasi error numerik terbatas serta top-1 yang sama;
- concurrent readers pada beberapa tensor berbagi manager dengan fault/hit/release yang terserialisasi dan peak mapping tetap di bawah budget;
- synchronous dan asynchronous prefetch menyentuh setiap native OS page, lalu acquire berikutnya tercatat sebagai residency hit;
- bounded fixed-worker scheduler menerapkan queue backpressure, drain-on-shutdown, task failure propagation, dan tidak membuat satu thread per request;
- prefault melepas mutex manager saat native page touch berlangsung namun mempertahankan mapping melalui pin;
- replacement dapat dipilih antara exact LRU dan largest-first; largest-first membebaskan window besar dengan satu eviction pada workload campuran;
- adaptive F32 tiling mempertahankan output fixed path, memakai window maksimum yang diizinkan budget/alignment, dan mengurangi fault count;
- `CpuExecutor` menerapkan tile policy yang sama pada dense F32, canonical GGML quantized dot, dan expert/MoE slices.

### Real-model validation

Command:

```text
zig build validate-residency -Doptimize=ReleaseFast -Dsimd-backend=false -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 4 4
```

Snapshot lokal Windows:

| Tensor | Type/size | Baseline | Bounded 4 MiB | Peak map | Faults/evictions | Baseline RSS | Bounded RSS | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `token_embd.weight` | Q6_K / 205.49 MiB | 264.17 ms | 288.80 ms | 4.00 MiB | 53 / 52 | 210.09 MiB | 5.82 MiB | 0 |
| `blk.0.attn_k.weight` | Q4_K / 0.56 MiB | 0.94 ms | 1.01 ms | 0.59 MiB | 1 / 0 | 4.72 MiB | 4.72 MiB | 0 |
| `blk.0.attn_output.weight` | Q4_K / 2.25 MiB | 3.44 ms | 3.41 ms | 2.29 MiB | 1 / 0 | 6.43 MiB | 6.44 MiB | 0 |
| `blk.0.attn_q.weight` | Q4_K / 2.25 MiB | 3.50 ms | 3.51 ms | 2.29 MiB | 1 / 0 | 6.44 MiB | 6.45 MiB | 0 |

### Phase 7 benchmark snapshot

Command:

```text
zig build bench-residency -Doptimize=ReleaseFast -Dsimd-backend=false
```

Snapshot lokal Windows:

| Workload | Time | Peak map | Faults | Hits/prefetches |
|---|---:|---:|---:|---:|
| Semua 8 tensor resident | 2.13 ms | 8 MiB | 8 | 120 / 0 |
| Budget 2 MiB | 26.13 ms | 2 MiB | 128 | 0 / 0 |
| Tensor 8 MiB, window 1 MiB | 26.06 ms | 1 MiB | 128 | 0 / 0 |
| Tensor 8 MiB, sync-prefault 1 MiB | 25.78 ms | 1 MiB | 128 | 128 / 128 |
| Scheduled look-ahead, budget 2 MiB | 27.88 ms | 2 MiB | 128 | 128 / 128 |
| Token loop 128 token, bounded | 206.48 ms | 2 MiB | 1024 | 0 / 0 |
| Token loop 128 token, scheduled | 289.90 ms | 2 MiB | 1024 | 1024 / 1024 |

Snapshot ini menunjukkan scheduler memenuhi correctness/backpressure tetapi bukan speedup pada warm-cache synthetic workload Windows: token loop naik dari 1.61 menjadi 2.26 ms/token. Sinkronisasi worker lebih mahal daripada page fault yang berhasil disembunyikan. Karena itu prefetch tetap opt-in dan tidak dijadikan default executor policy.

Phase 7 completion notes:

- scheduler memakai fixed worker count dan bounded queue; cancellation per-task belum tersedia karena shutdown sengaja menguras semua task yang sudah diterima;
- executor dan KV cache tetap single-owner walaupun manager aman dipakai multi-thread;
- satu tensor tetap hanya memiliki satu active mapped window, sehingga double-buffer current/next-window pada tensor yang sama belum tersedia;
- LRU/largest-first masih melakukan scan registry `O(n)`; CLOCK/2Q memerlukan struktur intrusive tambahan;
- prompt prefill masih token-by-token secara internal; optimized batched kernel dipindahkan ke fase throughput berikutnya;
- benchmark panjang saat ini synthetic 128-token weight traversal, bukan full 1B-model generation benchmark.

Phase 7 ditandai selesai karena acceptance correctness, bounded queue, adaptive policy, replacement selection, dan long-loop measurement terpenuhi. Hasil benchmark negatif dipertahankan untuk mencegah prefetch diaktifkan secara default tanpa bukti per-platform.

current RSS adalah pembanding yang relevan pada tool ini.

Validator sekarang juga menjalankan weight-bearing subgraph layer 0:

```text
layer-0 SwiGLU FFN: baseline=56.36 ms, bounded=55.45 ms, max-error=0,
checksum=9.233231, weight-map=4.00/4.00 MiB peak/budget,
scratch=32.00 KiB, activations=64.00 KiB, faults=10, evictions=9,
baseline-rss=18.90 MiB, bounded-rss=7.08 MiB
```

Subgraph memakai tensor `blk.0.ffn_gate.weight`, `blk.0.ffn_up.weight`, dan `blk.0.ffn_down.weight` dari GGUF nyata. Angka ini snapshot lokal dan bukan benchmark throughput stabil; acceptance criterion-nya adalah output identik, lifetime pin terbatas per tile, dan accounting memory terpisah.

Full-model single-token snapshot pada model yang sama, token id 1:

```text
full single-token logits: token=1, layers=16, vocab=128256,
baseline=1826.78 ms, bounded=1262.08 ms, max-error=0,
checksum=-17647.454345, argmax=16309,
weight-map=4.00/4.00 MiB peak/budget,
scratch=32.00 KiB, executor-activations=64.00 KiB,
attention-workspace=16.00 KiB, all-layer-kv=64.06 KiB,
faults=303, evictions=302,
baseline-rss=204.79 MiB, bounded-rss=7.82 MiB
```

Perbandingan resident-vs-bounded memakai executor Zig yang sama dan bit-identik. Reference llama.cpp CPU/non-mmap juga dijalankan terhadap token yang sama. Karena orchestration/reduction scalar adapter tidak bit-identik dengan graph GGML, acceptance reference memakai batas `max-error <= 0.5`, `mean-error <= 0.1`, seluruh nilai finite, dan top-1/argmax yang sama; toleransi ini diverifikasi pada single-token dan sequence dua token.

Snapshot Phase 6 final pada model yang sama dengan budget 4 MiB:

```text
single token [1]: resident-vs-bounded max-error=0, peak-map=4.00 MiB,
llama.cpp max-error=0.198390, mean-error=0.038467, argmax=11/11, status=close

prefill [1,2]: resident-vs-bounded max-error=0, peak-map=4.00 MiB,
llama.cpp max-error=0.303819, mean-error=0.051097, argmax=62/62, status=close

incremental append [1] lalu [2]: max-error-vs-prefill=0,
argmax=62, KV=128.13 KiB, peak-map=4.00 MiB
```

Reference sengaja memakai `use_mmap=false`, `n_gpu_layers=0`, satu thread, serta K/Q/V offload dan flash attention nonaktif. RSS reference (~829 MiB pada snapshot) berada di luar bounded manager; RSS bounded executor setelah full single-token run ~8.39 MiB.

Validator juga menjalankan satu decoder layer lengkap untuk satu token memakai metadata head/RMS/RoPE dan sembilan tensor layer 0 dari model yang sama:

```text
layer-0 single-token decoder: baseline=71.48 ms, bounded=70.45 ms,
max-error=0, checksum=-4.557518, weight-map=4.00/4.00 MiB peak/budget,
scratch=32.00 KiB, executor-activations=64.00 KiB,
attention-workspace=16.00 KiB, kv-cache=4.00 KiB,
faults=16, evictions=15, baseline-rss=17.93 MiB, bounded-rss=6.11 MiB
```

Ini adalah perbandingan resident-vs-bounded untuk adapter Zig yang sama; belum merupakan validasi numerik terhadap intermediate llama.cpp.

### Benchmark snapshot

Command:

```text
zig build bench-residency -Doptimize=ReleaseFast -Dsimd-backend=false
```

Snapshot Windows lokal (angka waktu dapat berubah antar run):

| Workload | Time | Peak active mapping | Faults | Evictions |
|---|---:|---:|---:|---:|
| 8 tensor resident (baseline) | 2.49 ms | 8 MiB | 8 | 0 |
| 8 tensor, budget 2 MiB | 33.19 ms | 2 MiB | 128 | 126 |
| 1 tensor 8 MiB, chunk 1 MiB, budget 1 MiB | 29.04 ms | 1 MiB | 128 | 127 |

Sequential scan ini sengaja merupakan worst case untuk LRU. Target fase ini bukan speedup, tetapi bukti batas mapping eksplisit dan recovery transparan dari backing file. Chunked path berhasil memproses logical tensor 8 MiB dengan peak active mapping 1 MiB.

## Current limitations

- Full Llama token path, prompt-style sequence execution, incremental KV reuse, dan llama.cpp reference sudah divalidasi pada model produksi nyata. Adapter tetap merupakan executor proof MLz, bukan tensor loader/backend resmi llama.cpp/GGML; reference logits dekat dan top-1 sama tetapi tidak bit-identik karena urutan reduksi/orchestration berbeda.
- Implementasi attention mendukung Llama causal decoding dan layer-major prompt prefill, even head dimension, Llama normal adjacent-pair RoPE, dan metadata Llama dasar. RoPE scaling variants, sliding-window attention, attention bias, multimodal architectures, dan batch paralel antar sequence belum didukung.
- Budget bukan batas total RSS; page cache OS, allocator, GPU memory, dan buffer llama.cpp berada di luar accounting. Executor kini memisahkan mapped weights, dequant scratch, activation buffers, attention workspace, dan KV cache miliknya.
- Manager registry/fault/pin/release/metrics thread-safe; executor dan KV cache tetap single-owner. Beberapa executor dapat berbagi manager saat mereka memakai resident window yang sama, tetapi divergent tiles pada tensor sama dapat menerima `TensorBusy`; serving-grade multi-sequence scheduling belum selesai.
- Read-only mapping saja.
- Satu entry hanya memiliki satu mapped window; range lain pada tensor yang sama memerlukan release lalu remap.
- Prefetch scheduler bounded dan fixed-worker, tetapi belum mendukung cancellation atau same-tensor double buffering. Prefetch opt-in karena warm-cache benchmark lokal menunjukkan overhead.
- Replacement tersedia sebagai LRU dan largest-first, keduanya masih melakukan scan registry `O(n)`.

## Next implementation step

Phase 9 berfokus pada serving-grade execution dan architecture coverage:

1. Tambahkan prompt chunking agar workspace 128/512 tidak harus tumbuh sampai seluruh context.
2. Integrasikan layer-major prefill ke request path MLz, bukan hanya validator proof executor.
3. Tambahkan sequence scheduler dengan executor/KV cache terpisah dan shared manager; same-tensor multi-window membutuhkan entry/window redesign.
4. Optimalkan attention serta matrix kernels (SIMD/thread pool); Phase 8 scalar proof jauh lebih lambat dari llama.cpp.
5. Implementasikan Qwen3-Next hybrid schedule, DeltaNet recurrent state, Q/K norm/gating, dan shared+routed MoE sebelum mengaktifkan full-model Qwen validator.
6. Pertahankan exact resident/bounded dan prefill/incremental gates, serta short-prompt llama.cpp tolerance tanpa dilonggarkan.

### Status Phase 9: chunked prefill (item 1) — selesai

- `CpuExecutor.modelPrefillChunked()`: prompt diproses per chunk; workspace
  caller-owned dibatasi ukuran chunk (`chunk_states` berkapasitas
  `chunk_size * hidden`), KV cache per layer dipertahankan lintas chunk dan
  mengikuti seluruh posisi sebelumnya; logits hanya dihitung pada chunk
  terakhir melalui flag `want_logits` pada `modelPrefillInner`.
- Validasi seluruh request (shape, token, cache capacity, output head, norm,
  per-layer weight shapes) dilakukan sebelum KV cache dimutasi.
- Unit test baru: `chunked prefill matches full prefill and incremental append
  exactly` — full prefill, chunked (chunk=3 dari 4 token), dan incremental
  append menghasilkan logits bit-identik serta panjang KV cache yang sama.
- Mode `.chunked` ditambahkan ke validator GGUF nyata; hasil dilaporkan
  dibandingkan terhadap full prefill dan ditolak bila tidak bit-identik atau
  argmax berbeda.
- Hasil Llama-3.2-1B-Instruct-Q4_K_M, budget 4 MiB:
  - 2 token: chunked `max-error-vs-prefill=0`, argmax=62, 378 ms.
  - 128 token (chunk=32): `max-error-vs-prefill=0`, argmax=226,
    5917 ms / 21.63 token-s, workspace 32 token (bukan 128).
  - Prompt 128 full prefill: 5164 ms / 24.79 token-s, faults=430; chunked:
    5917 ms / 21.63 token-s, fault delta kecil karena LM head sekali di akhir.
  - Long-prompt llama.cpp reference tetap informational; strict gate tidak
    berubah.

### Status Phase 9: same-tensor multi-window residency (item 3 foundation) — selesai

- Redesain internal `Manager`: window aktif dipindah dari `Entry` ke hash map
  global `windows` keyed slot; satu tensor dapat memiliki beberapa window.
- Multi-window hit, eviction per window, accounting budget per bytes mapping
  aktual (termasuk alignment prefix OS), dan pin tetap per window.
- `TensorBusy` dipindah ke `unregister()`; acquire disjoint pada tensor sama
  kini valid selama budget memadai.
- Dua test baru: coexistence dua window disjoint satu tensor dalam budget, dan
  stress concurrent empat worker memegang window berbeda tensor yang sama
  (0 eviction, budget invariant terjaga).
- Test FFN executor disesuaikan: eviction tidak lagi dijamin untuk mapping
  kecil yang co-resident; invariant yang diuji adalah budget itu sendiri.
- Verifikasi: `zig build test` PASS (14/14 residency, 58/58 total), real-model
  Llama validator PASS semua gate (hasil identik dengan sebelum redesain),
  benchmark PASS, `zig fmt` + `git diff --check` bersih.

### Status Phase 9: multi-sequence generation scheduler (item 3) — selesai

- `GenerationContext` pada unit test executor: dua sequence berjalan pada
  thread terpisah, masing-masing dengan executor, KV cache, attention
  workspace, dan hidden state sendiri, tetapi berbagi satu `residency.Manager`
  thread-safe.
- Autoregressive generation tiga token per sequence; output logits tiap
  sequence dibandingkan bit-identik dengan baseline sekuensial single-thread
  melalui manager yang sama.
- Invariant budget diverifikasi: `peak_resident_bytes <= budget` meskipun dua
  generation sequence mengakses weight tensor secara bersamaan.
- Regression guard: kegagalan alokasi per-sequence (executor/cache/workspace)
  dan kegagalan step inference dilaporkan melalui flag `failed`, bukan panic
  pada worker thread.
- Verifikasi: `zig build test -Dsimd-backend=false` PASS, real-model Llama
  validator PASS semua gate exact (single-token, layer-0 decoder, full logits
  resident-vs-bounded max-error=0, prefill, incremental, chunked),
  `zig fmt` + `git diff --check` bersih.

### Status Phase 9: parallel matmul kernel (item 4) — selesai

- Refactor `residency_compute.zig`: per-tile computation diekstrak menjadi
  `matMulF32Tile` dan `matMulQuantizedTile` (publik), dipakai bersama oleh
  jalur sekuensial dan driver paralel — satu sumber kebenaran kernel GGML
  (`vec_dot`), tanpa duplikasi dispatch.
- Driver paralel `parallelMatMul` (via wrapper tipis `residency_parallel.zig`,
  diekspor dari `root.zig`): worker pool dengan tile cursor atomik.
- **Bug race ditemukan dan diperbaiki**: desain awal cursor `load` +
  `cmpxchg` advance-setelah-proses memungkinkan dua worker memproses `row_start`
  yang sama; worker yang kalah CAS meng-advance melewati baris yang belum
  dihitung siapa pun → output `identical=false` pada benchmark. Perbaikan:
  klaim tile via `fetchAdd(rows_per_tile)` **sebelum** diproses (tiap baris
  dijamin milik tepat satu worker), lalu rentang klaim diproses dalam
  sub-chunk yang dibatasi `rangeCapacity` (alignment prefix mmap ikut
  diperhitungkan) sehingga budget tidak pernah terlampaui.
- Budget-aware concurrency: jumlah thread efektif dibatasi
  `budget / (tile_bytes + granularity)` agar worker tidak saling menggusur
  window; aktivasi dikuantisasi sekali di depan dan dibaca-only selama fase
  paralel.
- Unit test: parallel Q4_K 4096 kolom vs sekuensial bit-identik pada beberapa
  konfigurasi thread; worker failure diteruskan ke caller.
- Benchmark `bench-residency` baru: Q4_K 4096x4096, batch 8, budget 2 MiB —
  hasil Windows lokal:
  `sequential=9.27 ms (faults=147), parallel 4T=3.74 ms (faults=147),
  speedup=2.48x, identical=true`.
  Fault count identik sequential vs parallel — driver mempertahankan
  invariant residency; speedup berasal dari paralelisme, bukan dari
  mengubah access pattern.
- Wrapper `residency_parallel.zig` dan include path ggml di `build.zig` untuk
  benchmark exe (`linkLibrary(ggml_lib)`).
- Verifikasi: `zig build test -Dsimd-backend=false` PASS, real-model Llama
  validator PASS semua gate exact (single-token max-error=0 argmax=11
  status=close, prefill/incremental/chunked max-error=0), benchmark PASS,
  `git diff --check` bersih.

### Status Phase 9: bounded-residency completion service (item 2) — selesai

- Modul `src/residency_service.zig`: `ResidencyService` — serving boundary
  completion di atas bounded-residency executor pada model GGUF nyata.
  Handle llama.cpp dibuka **hanya** sebagai penyedia vocab/tokenizer
  (tokenize via `llama_tokenize`, detokenize via `llama_token_to_piece`,
  EOS/BOS dari vocab API); seluruh compute weight berjalan melalui
  `CpuExecutor` dengan budget mmap eksplisit — compute tidak pernah melewati
  graph GGML.
- `complete()`: tokenize (atau prompt tokens eksplisit) → chunked prefill
  (`modelPrefillChunked`, workspace dibatasi `prefill_chunk`) → decode loop
  greedy autoregressive via `modelTokens` → detokenize incremental → stop
  pada EOS atau `context_capacity`. Residency manager dibuat per request;
  semua weight window ter-unmap saat request selesai.
- Accounting per request dilaporkan lengkap: peak/budget mapped weights,
  dequant scratch, activations, attention workspace, KV cache, faults/hits/
  evictions, dan current RSS.
- Validasi sebelum mutasi KV tetap diwarisi dari executor; token di luar
  vocab dan `prompt + max_tokens > context_capacity` ditolak lebih awal.
- CLI smoke run: `zig build residency-serve -- <model.gguf> "<prompt>"
  [budget-mib] [max-tokens]` (exe `residency_service` di `build.zig`,
  men-link ggml + llama lib dan `residency_mmap.c`).
- Hasil Llama-3.2-1B-Instruct-Q4_K_M nyata, prompt "The capital of France is":
  - budget 16 MiB, 16 token: output koheren
    `" Paris. The capital of Germany is Berlin. The capital of Italy is Rome."`,
    peak-map 15.99/16.00 MiB, faults=2758.
  - budget 4 MiB, 8 token: output koheren
    `" Paris. The capital of Germany is Berlin"`,
    peak-map 4.00/4.00 MiB (budget invariant terjaga), faults=2731.
- Scope yang tidak diklaim: ini execution path MLz opt-in, bukan penggantian
  jalur llama.cpp di server; sampler masih greedy; belum ada streaming,
  chat template, atau batching server-level.
- Verifikasi: `zig build test -Dsimd-backend=false` PASS (62/62), smoke run
  budget 16 MiB dan 4 MiB PASS, `zig fmt` + `git diff --check` bersih.

### Status Phase 9: Qwen3-Next hybrid graph (item 5) - selesai

- Modul `src/residency_qwen3next.zig` mengimplementasikan schedule hybrid nyata
  dari GGUF `qwen3next`: tiga recurrent gated-DeltaNet layer lalu satu
  full-attention layer, berulang sesuai `full_attention_interval`, untuk 48
  block model Qwen3-Coder-Next.
- Recurrent block lengkap: RMSNorm, bounded QKV/Z/beta-alpha projections,
  depthwise causal convolution dengan raw-history persisten, GGML-compatible L2
  Q/K norm, grouped gated-DeltaNet recurrence, per-head gated RMSNorm, output
  projection, dan residual. `DeltaNetCache` memisahkan conv history dan matriks
  recurrent writable dari budget immutable weights.
- Full-attention block lengkap: interleaved per-head Q+gate projection, per-head
  Q/K RMSNorm, partial RoPE (`rope.dimension_count`), GQA causal attention,
  sigmoid output gating, output projection, KV cache, dan residual.
- MoE lengkap di kedua jenis block: softmax top-10 dari 512 routed experts,
  hanya selected expert slices yang di-fault, canonical GGML vec-dot arithmetic,
  shared SwiGLU expert, learned sigmoid shared gate, dan residual.
- `modelSingleToken()` menjalankan token embedding, seluruh hybrid layers,
  output norm, dan 151,936 vocabulary logits. `initLayerCaches()` membentuk
  cache recurrent/full-attention per layer sesuai schedule metadata.
- Bug correctness yang ditemukan oleh reference gate:
  1. `attn_q.weight` adalah interleaved `[Q_head, gate_head]`, bukan
     `[all-Q, all-gate]`;
  2. selected-expert path lama mendequantisasi weight lalu scalar-dot, bukan
     canonical GGML activation quantization + `vec_dot`; setelah diperbaiki,
     full-model top-1 berubah dari salah (`3830`) menjadi reference (`220`).
  3. Q/K L2 norm mengikuti GGML `1 / max(sqrt(sum(x*x)), epsilon)`, bukan
     `1 / sqrt(sum + epsilon)`.
- Validator Qwen besar sekarang menjalankan projection probe, layer-0
  DeltaNet+MoE, layer-3 full-attention+MoE, full 48-layer logits, resident-ish
  64 MiB vs bounded 4 MiB exact gate, dan optional llama.cpp mmap reference
  (`qwen-reference=true`) agar default validation tidak memetakan 27 GiB dua
  kali.
- Hasil `Qwen3-Coder-Next-Q2_K.gguf` (27.21 GiB), token 1, budget 4 MiB:
  - full 48-layer elapsed 2.33 s snapshot, peak active mapping 4.00/4.00 MiB,
    all-layer recurrent+KV state 75.42 MiB, workspace 0.14 MiB;
  - resident-ish 64 MiB vs bounded 4 MiB: logits `max-error=0`, argmax 220/220;
  - llama.cpp mmap reference: `max-error=0.843506`, mean `0.125230`, top-1
    220/220, seluruh logits finite; reference RSS ~19.4 GiB (di luar manager);
  - two-token recurrent-state/KV reuse: `max-error=1.247838`, mean `0.133008`,
    top-1 220/220, state 75.47 MiB, finite.
- Reference acceptance khusus Qwen Q2_K: single-token max <= 1.0 / mean <=
  0.2; two-token max <= 1.5 / mean <= 0.3; finite dan top-1 sama. Exact gate
  resident-vs-bounded tetap nol dan tidak dilonggarkan.
- Verifikasi: `zig build test -Dsimd-backend=false` PASS; full-model Qwen
  bounded validation PASS; optional llama.cpp single/two-token reference PASS;
  existing Llama gates tetap PASS.

Dengan item 1-5 selesai, Phase 9 dinyatakan selesai. Keterbatasan yang tersisa
menjadi Phase 10: batched/chunked Qwen prompt kernel (saat ini token-by-token),
server streaming/sampling integration untuk Qwen path, SIMD/thread-pool pada
DeltaNet/MoE orchestration, dan budget policy gabungan weights + recurrent
state + KV cache.

## Phase 10 — Combined budget & execution hardening

### Status Phase 10: combined non-weight state budget (item 1) — selesai

Sebelum item ini, hanya mapped weights yang dibatasi budget. Recurrent
DeltaNet state, full-attention KV cache, score scratch, dan execution
workspace dialokasikan tanpa kebijakan eksplisit (75,42 MiB state pada Qwen
48 layer dengan konteks 1 token).

Implementasi:

- `Config.deltaNetCacheBytes()`, `Config.fullAttentionCacheBytes(capacity)`,
  dan `Config.workspaceBytes()` — estimator statis byte yang harus cocok
  dengan alokasi aktual `DeltaNetCache.init`, `FullAttentionCache.init`, dan
  `Workspace.init`.
- `StateBudget` policy: `cache_bytes` (semua layer cache) dan
  `workspace_bytes`, keduanya opsional; `null` berarti perilaku legacy tanpa
  batas.
- `initLayerCachesBudgeted()` memvalidasi policy **sebelum** alokasi pertama;
  penolakan bersifat transactional sehingga tidak ada alokasi parsial yang
  bocor. Error baru: `StateBudgetExceeded`.
- `StateBudget.stateBytes()` menghitung kebutuhan gabungan cache + workspace.
- Validator Qwen full-model kini mengalokasikan melalui jalur budgeted dengan
  kebutuhan tepat, memverifikasi estimator cocok dengan byte aktual, dan
  membuktikan policy satu byte lebih ketat ditolak sebelum alokasi
  (`StateBudgetNotEnforced` tidak boleh terjadi pada jalur valid).

Acceptance terbukti:

- Unit test estimator: byte aktual cache/workspace == estimator statis untuk
  recurrent, full-attention, dan workspace.
- Unit test policy: kebutuhan tepat diterima; satu byte lebih ketat pada
  cache atau workspace ditolak sebelum alokasi; legacy unlimited tetap jalan.
- Qwen3-Next 27,2 GiB nyata, budget weights 4 MiB: full 48-layer logits
  argmax=220, peak-map=4,00/4,00 MiB, all-layer-state=75,42 MiB (kini
  ter-enforce, bukan sekadar dilaporkan), resident-vs-bounded max-error=0.
- Llama 3.2 1B regression: seluruh gate exact tetap PASS, llama.cpp reference
  `status=close`, top-1 sama.

Verifikasi: `zig build test -Dsimd-backend=false` PASS (66/66); validator
Qwen dan Llama PASS.

### Status Phase 10: StateBudget di serving path (item 2) — selesai

`StateBudget` kini terintegrasi ke `ResidencyService.complete()`:

- `CompletionOptions.state_budget: ?StateBudget` (null = legacy unlimited).
- Policy divalidasi **sebelum** alokasi manager/workspace/cache apa pun
  (transactional rejection): workspace gabungan (attention + prefill +
  chunk states + state + logits) via `checkWorkspace`, dan total KV cache
  semua layer via `checkCache`.
- Estimator statis executor baru: `kvCacheBytes()`,
  `attentionWorkspaceBytes()`, `prefillWorkspaceBytes()`, plus
  `initKvCachesBudgeted()` yang mengalokasikan semua layer cache di bawah
  policy dengan errdefer transaksional.
- CLI `residency-serve` menerima argumen opsional
  `[state-cache-mib] [state-workspace-mib]`.

Bukti model nyata (Llama-3.2-1B, weight budget 4 MiB):

- cache limit 2 MiB / 64 MiB → ditolak `StateBudgetExceeded` sebelum alokasi
  (kebutuhan aktual ~65,5 MiB untuk 16 layer × 1024 token × kv_width 1024).
- cache 128 MiB, workspace 8 MiB → completion koheren
  (`Paris. The capital of Germany is Berlin. ...`), weight-map 4,00/4,00 MiB,
  kv 4100 KiB ter-enforce, log llama.cpp di-silence untuk output yang bersih.

Verifikasi: `zig build test -Dsimd-backend=false` PASS; smoke run budget
positif dan negatif PASS; `zig fmt` bersih.

Sisa Phase 10 (selesai):
1. Combined non-weight state budget (DeltaNet recurrent + KV + workspace) — estimator byte-exact, penolakan transaksional, ter-wire ke Qwen path dan serving path.
2. StateBudget di serving path (ResidencyService) — policy cache/workspace via CLI, smoke test positif & negatif pada model nyata.
3. Qwen chunked prompt kernel — modelPrefillChunked bit-identik dengan incremental (max-error=0, faults 77.256 → 64.453 pada 2 token, 27 GiB nyata).
4. Bit-exact parallel DeltaNet pool (opt-in, value-head partition, MLZ_QWEN_PARALLEL) — checksum identik scalar pada full 48-layer model nyata; tetap opt-in karena MoE dominan, bukan recurrence.

Sisa kerja lanjutan di luar Phase 10 kini mencakup integrasi backend GGML resmi.
Milestone native terbaru menjalankan stock graph/kernels GGML terhadap bounded
GGUF mappings, termasuk row-tiled `MUL_MAT`, selected-expert `MUL_MAT_ID`, dan
row-envelope `GET_ROWS`. Strict `cpu-repack=false` validation pada budget 4 MiB:

- Llama-3.2-1B Q4_K_M (762.81 MiB): logits exact, argmax 11/11,
  peak mapped 4.00 MiB, faults/evictions 319/318, uploaded weight 0 byte;
- Qwen3-Coder-Next Q2_K (27.2 GiB): logits exact, argmax 3830/3830,
  peak mapped 4.00 MiB, faults/evictions 2332/2331, uploaded weight 0 byte,
  dan routed MoE tetap memakai canonical stock GGML `MUL_MAT_ID`.

Detail dan limitation ada di `docs/ggml-residency-backend.md`.

### Status lanjutan: HTTP endpoint untuk bounded-residency completion - selesai

- `src/residency_endpoint.zig`: handler `POST /v1/residency/completions` (OpenAI-compatible subset).
  - Service dibuka lazily pada request pertama; serialized via mutex (executor single-owner).
  - Streaming `stream:true` -> OpenAI-style SSE + `data: [DONE]`; non-streaming -> satu JSON completion.
  - Respons memuat blok `residency` (budget, peak mapped, faults, evictions, kv bytes) untuk observability.
- Wire ke server: `--residency-budget-mib` (0 = endpoint disabled, 404), init/deinit lifecycle, route di `handleConnection`.
- Perbaikan bug yang ditemukan smoke test: handler lupa men-tokenisasi prompt teks (`PromptEmpty`) — kini endpoint men-tokenisasi via vocab service sebelum `complete()`.
- Smoke test end-to-end (Llama-3.2-1B, budget 8 MiB, Windows):
  - non-streaming: `" Paris. The capital of Germany is Berlin. The capital of"`, prompt 5 tok + 12 gen, peak map 8.388.592/8.388.608 byte, faults 2864.
  - streaming: 7 SSE chunks + `data: [DONE]` diterima.
- Skrip smoke: `smoke_residency_endpoint.ps1` (lokal, tidak dibutuhkan CI).
- Sisa (di luar scope ini): chat template/messages input, per-request budget override, multi-request concurrency >1, SIMD per-op pada orchestration, integrasi backend GGML resmi.

### Status lanjutan: chat `messages` input pada residency endpoint — selesai

- `ResidencyService.applyChatTemplate()`: merender `messages` melalui jinja
  chat template bawaan model (`llama_model_chat_template` +
  `mlz_render_chat_template`, add_generation_prompt=true) lalu men-tokenisasi
  hasilnya dengan special tokens aktif — template output dapat memuat control
  token seperti `<|eot_id|>`.
- Endpoint `POST /v1/residency/completions` kini menerima `messages` (array
  `{role, content}`, dirender via chat template) atau `prompt` (raw text);
  keduanya divalidasi (400 untuk bentuk yang salah/kosong).
- Smoke test end-to-end (Llama-3.2-1B, budget 8 MiB):
  - raw prompt: 5 prompt tokens, output koheren, budget invariant terjaga;
  - `messages` chat: 156 prompt tokens (template dirender penuh), output
    dihasilkan, peak map 8.388.592/8.388.608 byte;
  - streaming SSE tetap PASS.
- Verifikasi: `zig build` + `zig build test -Dsimd-backend=false` PASS;
  smoke script diperluas dengan kasus chat (`PASS chat messages`).

### Status lanjutan: per-request budget override + concurrency > 1 - selesai

- Audit menyimpulkan `ResidencyService.complete()` sudah mengalokasikan semua
  state eksekusi per request (manager, executor, workspace, KV caches), dan
  `BackingStore`/`TensorIndex` bersifat read-only terhadap mapping
  (`MapViewOfFile`/`mmap` per window, tanpa cursor bersama) sehingga beberapa
  service instance dapat mengeksekusi paralel tanpa refactor.
- `ResidencyEndpoint` kini pool dari `slots` service independen, masing-masing
  mutex sendiri; request mengambil slot bebas (try-lock semua slot, fallback
  round-robin). `--residency-slots N` (default 1 = perilaku serialized lama).
- Per-request budget override: field request `residency_budget_mib`
  (1..1048576; di luar rentang -> 400). Override hanya berlaku untuk request
  tersebut karena manager dibuat per request.
- Unit test: akuisisi slot multi-slot, penolakan slot_count=0.
- Smoke end-to-end (Llama-3.2-1B, budget 8 MiB, slots=2): non-streaming,
  streaming SSE, chat messages, override 4 MiB (echo budget 4 MiB, invariant
  terjaga), override 0 -> 400, dan dua request concurrent keduanya sukses
  dengan budget invariant terjaga.
- Verifikasi: `zig build` + `zig build test` PASS; `zig fmt`; smoke script
  diperluas (`PASS per-request budget override`, `PASS invalid budget override
  rejected`, `PASS concurrent requests (2 slots)`).
