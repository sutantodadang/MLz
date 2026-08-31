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

Sisa kerja lanjutan (di luar Phase 10): streaming SSE/sampling non-greedy pada service, endpoint HTTP, SIMD per-op, dan integrasi backend GGML resmi.

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
