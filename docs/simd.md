# Custom SIMD Backend

MLz ships an optional custom CPU SIMD backend that intercepts a small set of
ggml ops (`MUL_MAT`, `FLASH_ATTN_EXT`) and routes them through hand-written
AVX2 / AVX-512 / NEON kernels living under [src/simd/kernels](../src/simd/kernels).

The backend is **opt-in at build time** and **rollback-safe at runtime**.

## Build

```sh
zig build -Dsimd-backend=true -Doptimize=ReleaseFast
```

When the flag is set, [build.zig](../build.zig) swaps the upstream
`ggml/src/ggml-cpu/ggml-cpu.c` for our patched copy
[src/simd/ggml-cpu-simd.c](../src/simd/ggml-cpu-simd.c) and links in
[src/simd/ggml_simd_hook.cpp](../src/simd/ggml_simd_hook.cpp).

`zig build` (no flag) builds against unmodified ggml — there is no SIMD code
in the binary.

## Runtime controls

| Flag                | Env var                  | Default | Effect                                                    |
|---------------------|--------------------------|---------|-----------------------------------------------------------|
| `--no-simd`         | `MLZ_SIMD=0`             | off     | Force every hook to return 0 → ggml default path          |
| `--simd-trace`      | `MLZ_SIMD_TRACE=1`       | off     | Print every dispatched op (type, M, N, K) to stderr       |
| `--simd-flash-attn` | `MLZ_SIMD_FLASH_ATTN=1`  | off     | Opt in to the flash-attention hook (off until E2 passes)  |

CLI flags set the corresponding env vars before any model is loaded.  Env vars
read directly also work — useful for `MLZ_SIMD=0 ./MLz.exe ...` rollback
without rebuild.

## Dispatch contract

```
ggml_compute_forward_mul_mat()
  └── ggml_simd_try_mul_mat(params, dst)
         ├── if MLZ_SIMD=0                      → return 0 (fallback)
         ├── if dst->type != F32                → return 0
         ├── if src0->extra || src1->extra      → return 0   (Repack owns it)
         ├── if non-contiguous 2D layout        → return 0
         ├── lookup vec_dot kernel by src0 type
         ├── thread 0 quantizes src1 (cached on data ptr/K/N)
         └── per-thread tile loop dispatches the kernel
       returns 1 on success
  └── if try returned 0 → ggml default code runs
```

`FLASH_ATTN_EXT` parallels this, but is gated additionally on
`MLZ_SIMD_FLASH_ATTN=1` because the historical Q8_0 register-clobber crash has
not been re-validated end-to-end (see PLAN-ASSEMBLY-REWRITE).

## Why we defer to Repack

`-Dcpu-repack` (default `true`) builds with `GGML_USE_CPU_REPACK`, which
repacks K-quant weights into SIMD-friendly interleaved layouts at model load.
The dispatch gate (`src0->extra != nullptr`) checks if Repack has claimed the
tensor and yields to it.  Net effect: Repack handles the bulk of MUL_MAT for
loaded weights; our hook handles the rest (intermediate F32, non-repacked
shapes, and any quant Repack doesn't cover).

## Tests

Two end-to-end gates live under [tests/](../tests):

- **[e2e_token_diff.ps1](../tests/e2e_token_diff.ps1)** — same prompt + seed
  + temperature 0 with and without `--no-simd`; output must hash-match.
- **[e2e_long_ctx.ps1](../tests/e2e_long_ctx.ps1)** — 2048-token generation
  at ctx=4096 must exit cleanly under simd-off, simd-on, and `--simd-flash-attn`.

Per-kernel correctness/perf is exercised by `zig build bench -Dsimd-backend=true`.

## Adding a new kernel (current process)

1. Drop the `.asm` (x86, NASM) or `.S` (aarch64, GAS) under
   `src/simd/kernels/{x86,aarch64}/...`
2. Add the source path to the corresponding loop in
   [build.zig](../build.zig) (search for `kernels/x86/vec`).
3. Declare the symbol in the `extern "C"` block at the top of
   [src/simd/ggml_simd_hook.cpp](../src/simd/ggml_simd_hook.cpp).
4. Wire the dispatch in `ggml_simd_try_mul_mat` (the `else if (src0->type == ...)`
   chain).
5. Add a correctness test in [src/bench_simd.zig](../src/bench_simd.zig).
6. Run `tests/e2e_token_diff.ps1` to confirm no end-to-end divergence.

A future improvement (tracked in PLAN-ASSEMBLY-REWRITE) is to drive steps 2–4
from a `kernels/manifest.txt` file via build-time codegen, eliminating the
hand-maintained extern "C" list.

## Known limitations

The U1 validator (`zig build test-simd`) compares every built `vec_dot` kernel
against a scalar `dequantize → dot` reference (rel-tol 1e-3).  Per the latest
run on x86_64 (Ryzen 5 7500F, AVX2 + AVX-512), the following kernels are
**verified incorrect** and the dispatch in
[src/simd/ggml_simd_hook.cpp](../src/simd/ggml_simd_hook.cpp) routes them to
upstream `ggml_vec_dot_*` instead.  We still amortize the Q8_0/Q8_K activation
quantization cache across the M rows of each matmul, which is the dominant cost.

| Kernel              | AVX2 | AVX-512 | Action                              |
|---------------------|------|---------|-------------------------------------|
| `q4_0_q8_0`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx512.asm)) on x86 + NEON .S on aarch64 |
| `q8_0_q8_0`         |  ✓   |   ✓     | Use handwritten kernel              |
| `q2_K_q8_K`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx512.asm)) on x86 + NEON .S on aarch64 |
| `q3_K_q8_K`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx512.asm)) on x86 + NEON .S on aarch64 |
| `q4_K_q8_K`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx512.asm)) on x86 + NEON .S ([neon](../src/simd/kernels/aarch64/vec/vec_dot_q4_k_q8_k_neon.S)) on aarch64 |
| `q5_K_q8_K`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx512.asm)) on x86 + NEON .S ([neon](../src/simd/kernels/aarch64/vec/vec_dot_q5_k_q8_k_neon.S)) on aarch64 |
| `q6_K_q8_K`         |  ✓   |   ✓     | Handwritten NASM ([avx2](../src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx2.asm), [avx512](../src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx512.asm)) on x86 + NEON .S ([neon](../src/simd/kernels/aarch64/vec/vec_dot_q6_k_q8_k_neon.S)) on aarch64 |
| `q8_K_q8_K`         |  ✓   |   ✓     | Use handwritten kernel              |

End-to-end correctness is gated by [tests/e2e_token_diff.ps1](../tests/e2e_token_diff.ps1)
which asserts that `--no-simd` and full-SIMD generations produce SHA256-identical
output for a fixed prompt and seed.

- **Flash attention is opt-in.**  Until [tests/e2e_long_ctx.ps1](../tests/e2e_long_ctx.ps1)
  passes with `MLZ_SIMD_FLASH_ATTN=1` reliably across multiple long-context
  runs, the hook is off by default.  Enable per-process via the env var or
  the `--simd-flash-attn` CLI flag.
- **Q5_0 / Q5_1 not yet covered** in the hook (legacy quants — not in any
  shipping model the project targets).
- **`softmax` / RoPE not yet hooked.**  Per the plan, these are
  gated on profiler evidence (≥ 8% e2e gain from steps 1–2 first).
- **`rms_norm_f32` is opt-in.**  Hook is implemented (NASM AVX2 +
  AVX-512 at [rms_norm_f32_avx2.asm](../src/simd/kernels/x86/unary/rms_norm_f32_avx2.asm)
  and [rms_norm_f32_avx512.asm](../src/simd/kernels/x86/unary/rms_norm_f32_avx512.asm))
  but defaults OFF so E1 SHA256-identity is never at risk.  Enable per-process
  via `MLZ_SIMD_RMS_NORM=1`.  U1 confirms bit-exact match with the f64 scalar
  reference at n ∈ {7, 64, 256, 1024, 4096, 8193}; E1 stays SHA-identical
  with the hook on for both Llama-3.2-1B Q4_K_M and Gemma-3-4B Q2_K.

## Validating kernels (U1)

```sh
zig build test-simd -Dsimd-backend=true -Doptimize=ReleaseFast
```

Generates random F32 vectors at K ∈ {32, 256, 1024, 4096}, quantizes via
ggml's reference (`quantize_row_q*_ref`), calls each compiled kernel, and
asserts the result is within `1e-3` relative error of the
`dequantize → scalar dot` reference.  Exit non-zero on any failure.

When you add or fix a kernel, U1 must still pass before you flip dispatch in
the hook from "delegate to upstream" to "call our kernel".
