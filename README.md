# MLz — fast, tunable LLM inference in Zig

MLz is a Zig serving layer over `llama.cpp` focused on **performance**, **ease of
use**, and **easy tuning by config**. It pairs a continuous-batching scheduler and
cross-slot prefix sharing with an OpenAI-compatible server, a local model
registry (LM Studio–style `pull`), and an optional hand-written CPU SIMD backend
(AVX2 / AVX-512 / NEON).

Targets the niche between `vllm` / `sglang` (throughput) and LM Studio (ease of
use), on CPU and GPU.

## Highlights

- **Continuous batching** — multi-slot scheduler decodes N sequences per step;
  chunked prefill, backpressure. 3–4× throughput vs serial under concurrency.
- **Prefix sharing (RadixAttention-lite)** — cross-slot KV prefix cache reuses a
  shared prompt prefix across requests (big TTFT win on shared system prompts).
- **Config-first** — layered `mlz.toml` + env + CLI; `--init` writes a starter
  file; `--print-config` shows the resolved settings.
- **Model registry** — `mlz models pull|list|rm` from a URL or HuggingFace
  shorthand; resumable downloads; run models by bare name.
- **Auto multi-model serving** — a request's `model` field loads another model on
  demand into a refcount-pinned LRU pool (startup model always resident).
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models`, `/health`, plus a WebSocket streaming endpoint.
- **Speculative decoding** — optional draft model (`--draft-model`).
- **Grammar-constrained output** — GBNF grammars (`--grammar`).
- **Custom CPU SIMD backend** — hand-written AVX2/AVX-512/NEON kernels (vec-dot,
  quantize, SiLU, RoPE, INT8 GEMM, fused RoPE+attention), runtime-dispatched and
  rollback-safe.
- **GPU acceleration** — CUDA, Vulkan, Metal (via llama.cpp backends).

## Quick start

```bash
# Build (Release recommended)
zig build -Doptimize=ReleaseFast

# Pull a model into the local registry, then run it by name
.\zig-out\bin\MLz.exe models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
.\zig-out\bin\MLz.exe qwen2.5-0.5b-instruct-q8_0.gguf

# Or run a model file directly
.\zig-out\bin\MLz.exe model.gguf

# One-shot prompt
.\zig-out\bin\MLz.exe model.gguf --prompt "Explain quantum computing"

# OpenAI-compatible server
.\zig-out\bin\MLz.exe model.gguf --server --port 8080
```

## Configuration

MLz is tuned by a TOML file, environment variables, and CLI flags, applied in
this order (later wins):

```
built-in defaults  <  mlz.toml  <  MLZ_* env vars  <  CLI flags
```

```bash
.\zig-out\bin\MLz.exe --init            # write a starter mlz.toml to the cwd
.\zig-out\bin\MLz.exe --config mlz.toml # load a config (./mlz.toml is auto-loaded)
.\zig-out\bin\MLz.exe model.gguf --print-config   # show resolved settings and exit
```

`mlz.toml` sections: `[model]` (path, `n_ctx`, `n_gpu_layers`, `threads`),
`[serve]` (enabled, host, port, api_key), `[sampling]` (temp, top_k, top_p,
min_p, seed), `[chat]` (stream, system, template, grammar), `[speculative]`
(draft_model).

### Common CLI flags

| Flag | Purpose |
|---|---|
| `--prompt <s>` | one-shot, non-interactive |
| `--temp / --top-k / --top-p / --min-p / --seed` | sampling |
| `--ctx <n>` | context size |
| `--ngl <n>` | GPU layers to offload |
| `--threads <n>` | CPU threads |
| `--system <s>` | system prompt |
| `--chat-template <name>` | override chat template (e.g. `gemma`) |
| `--grammar <file>` / `--grammar-root <rule>` | GBNF-constrained output |
| `--load-chat / --save-chat <file>` | persist conversation (JSON) |
| `--draft-model <file>` | speculative decoding |
| `--server` `--host` `--port` `--api-key` | server mode |
| `--max-concurrent <n>` | continuous-batching slots (server) |
| `--prefix-cache` / `--no-prefix-cache` | toggle prefix sharing (on when `--max-concurrent > 1`) |

> Tip: with `--save-chat`, Ctrl+C exits cleanly and saves.

## Model registry

Models live under a per-user data dir (`%LOCALAPPDATA%\mlz\models` on Windows,
`~/.local/share/mlz/models` elsewhere). Any `.gguf` there is runnable by bare
name.

```bash
mlz models list                       # list downloaded models + sizes
mlz models pull <url>                  # download a .gguf from a direct URL
mlz models pull owner/repo/file.gguf   # HuggingFace shorthand (resolve/main)
mlz models pull <src> <name>           # save under a custom name
mlz models rm <name>                   # delete
mlz models dir                         # print the registry path
```

`pull` streams to a `.part` file and is **resumable** (HTTP Range), with an
atomic rename on completion. A `<model_path>` that isn't a file is resolved
against the registry, so `mlz qwen2.5-0.5b` and `model = "qwen2.5-0.5b"` both
work.

## Server mode

```bash
.\zig-out\bin\MLz.exe model.gguf --server --host 0.0.0.0 --port 8080 \
    --api-key secret --ctx 8192 --max-concurrent 4
```

### Endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming (SSE) & blocking |
| `POST /v1/completions` | legacy text completion, streaming & blocking |
| `POST /v1/embeddings` | mean-pooled, L2-normalised; string or array input |
| `GET /v1/models` | startup model + registry models |
| `GET /health` | liveness |
| `GET /v1/chat/completions/ws` | WebSocket streaming |

- **Continuous batching**: concurrent requests share the scheduler's decode
  steps (`--max-concurrent`).
- **Prefix caching**: shared prompt prefixes are reused across requests to cut
  TTFT.
- **Auto multi-model**: set `model` to another registry name / file path and the
  server loads it on demand (LRU-evicts the least-recently-used extra model;
  the startup model is always resident).

```bash
# Stream a chat completion
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

# Embeddings
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":["hello","world"]}'
```

## Hardware acceleration

GPU backends are enabled at build time:

```bash
zig build -Dcuda=true   -Doptimize=ReleaseFast   # NVIDIA (CUDA Toolkit 12.x + MSVC on Windows)
zig build -Dvulkan=true -Doptimize=ReleaseFast   # cross-platform (Vulkan SDK)
zig build               -Doptimize=ReleaseFast   # Metal is default on macOS/iOS
```

CUDA auto-detects `CUDA_PATH` / `CUDA_HOME`.

### CPU SIMD

AVX-512 is enabled by default on x86_64. For older CPUs (pre-Skylake-X Intel /
pre-Zen4 AMD):

```bash
zig build -Dno-avx512=true -Doptimize=ReleaseFast
```

### Custom SIMD backend (optional)

A hand-written kernel backend (NASM `.asm` for x86, `.S` for aarch64) with
runtime CPU dispatch and rollback-safe fallback to ggml:

```bash
# Requires NASM: scoop/choco install nasm  •  apt install nasm
zig build -Dsimd-backend=true -Doptimize=ReleaseFast
```

Kernels include vec-dot (Q4_0…Q8_K), `quantize_q8_0/q8_k`, SiLU, RoPE
(standard + NeoX), layer/rms-norm, an **INT8 GEMM microkernel** (AVX2 + a
register-tiled AVX-512-VNNI fast path), and a **fused RoPE + attention** path
(vectorised sin/cos). Every kernel is validated against a scalar reference
(`zig build test-simd`) and benchmarked (`zig build bench`).

Runtime controls:

| Flag | Env var | Effect |
|---|---|---|
| `--no-simd` | `MLZ_SIMD=0` | disable hooks, use the ggml default path |
| `--simd-trace` | `MLZ_SIMD_TRACE=1` | print every dispatched op to stderr |
| `--simd-flash-attn` | `MLZ_SIMD_FLASH_ATTN=1` | opt in to the flash-attention hook |

See [docs/simd.md](docs/simd.md) for the dispatch contract, kernel inventory, and
how to add kernels.

## Building from source

Requirements: **Zig 0.15.x** and a C/C++ toolchain (Zig drives it for the
`llama.cpp` dependency). NASM only for `-Dsimd-backend=true`.

```bash
zig build                      # fetch deps + build
zig build -Doptimize=ReleaseFast
zig build -Doptimize=ReleaseSmall
```

## Testing & benchmarks

```bash
zig build test                          # unit tests (config, registry, server, LRU, openai)
zig build test-simd -Dsimd-backend=true # validate every SIMD kernel vs a scalar reference
zig build bench     -Dsimd-backend=true # kernel GFLOPS/GOPS benchmarks
```

## Project structure

```
src/
├── main.zig           # CLI entry (interactive / prompt / server / `models`)
├── config.zig         # layered TOML + env + CLI config
├── models.zig         # local model registry + `mlz models` (pull/list/rm)
├── model_manager.zig  # generic refcount-pinned LRU (multi-model serving)
├── server.zig         # HTTP/WebSocket server, OpenAI endpoints, EngineManager
├── engine.zig         # inference engine (sampling, chat, KV management)
├── scheduler.zig      # continuous-batching multi-slot scheduler
├── prefix_cache.zig   # cross-slot prefix (RadixAttention-lite) cache
├── embeddings.zig     # embedding-mode model + /v1/embeddings service
├── inference.zig      # prompt building, token generation
├── openai.zig         # OpenAI request/response types + JSON
├── chat.zig           # chat history / templating
├── llama_cpp.zig      # idiomatic llama.cpp wrapper
├── simd/              # custom SIMD backend (kernels, hooks, dispatch)
└── root.zig           # library root
```

## Model compatibility

Most GGUF models for recent `llama.cpp` work (Llama 3.x, Qwen2.5/3, Gemma, …).
Pull from HuggingFace:

```bash
mlz models pull unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```
