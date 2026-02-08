# MLz - LLaMA Inference in Zig

MLz is a Zig implementation for running LLaMA language models, primarily utilizing fine-tuned bindings to `llama.cpp` for high-performance inference. It supports hardware acceleration via CUDA and Vulkan, provides a modern chat interface, and includes an OpenAI-compatible API server.

## Quick Start

```bash
# Build the project (Release mode recommended)
zig build -Doptimize=ReleaseFast

# Run interactive chat
.\zig-out\bin\MLz.exe Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Run as OpenAI-compatible server
.\zig-out\bin\MLz.exe Llama-3.2-3B-Instruct-Q4_K_M.gguf --server --port 8080

# Run non-interactive prompt
.\zig-out\bin\MLz.exe model.gguf --prompt "Explain quantum computing"
```

### CLI Chat Options
```bash
# Run with custom system prompt and disabled streaming
.\zig-out\bin\MLz.exe model.gguf --system "You are a helpful Zig assistant." --stream false

# Load/save chat history (JSON)
.\zig-out\bin\MLz.exe model.gguf --load-chat chat.json --save-chat chat.json

# Run single prompt (non-interactive mode)
.\zig-out\bin\MLz.exe model.gguf --prompt "Write a hello world in Zig"

# Tip: when --save-chat is set, Ctrl+C exits cleanly and saves.
```

## Server Mode

MLz includes a high-performance, OpenAI-compatible HTTP server.

```bash
.\zig-out\bin\MLz.exe model.gguf --server --host 0.0.0.0 --port 8080 --api-key secret --ctx 8192
```

### Features
*   **OpenAI-Compatible API**: Supports `/v1/chat/completions` (streaming & blocking) and `/v1/models`.
*   **Context Caching**: Automatically caches conversation history to dramatically reduce latency (TTFT) for long multi-turn chats.
*   **Multi-threaded Handling**: Handles multiple concurrent connections without blocking (though inference is serialized per model).
*   **WebSocket Interface**: Custom WebSocket endpoint at `/v1/chat/completions/ws` for low-latency streaming.

## Hardware Acceleration

MLz supports GPU acceleration out of the box. You can enable it during the build step:

### CUDA (NVIDIA)
```bash
zig build -Dcuda=true -Doptimize=ReleaseFast
```
*Note: On Windows, this requires the CUDA Toolkit (v12.x) and MSVC. The build system will automatically detect the CUDA path if `CUDA_PATH` or `CUDA_HOME` is set.*

### Vulkan (Cross-Platform)
```bash
zig build -Dvulkan=true -Doptimize=ReleaseFast
```
*Requires Vulkan SDK to be installed.*

### Metal (macOS)
Metal support is enabled by default on macOS/iOS builds.
```bash
zig build -Doptimize=ReleaseFast
```

### CPU SIMD Optimizations (x86_64)
AVX512 SIMD instructions are enabled by default on x86_64 for maximum performance. If you need to run on older CPUs that don't support AVX512 (pre-Skylake-X Intel or pre-Zen4 AMD):
```bash
zig build -Dno-avx512=true -Doptimize=ReleaseFast
```

## Project Structure

```
src/
├── main.zig          # CLI Entry point (Interactive/Prompt/Server)
├── server.zig        # HTTP/WebSocket Server implementation
├── inference.zig     # Core inference logic (Prompt building, token generation)
├── openai.zig        # OpenAI API data structures and serialization
├── chat.zig          # Chat history and context management
├── llama_cpp.zig     # Zig idiomatic wrapper for llama.cpp
├── root.zig          # Library export root
└── ggml_shim.h       # C header shim for GGML/llama.cpp
```

## Testing

MLz includes a comprehensive test suite covering the server, inference logic, and data models.

```bash
# Run all tests
zig build test
```

## Building from Source

Requirements:
- Zig 0.15.x
- C++ compiler (automatically handled by Zig build system for dependencies)

```bash
# Fetch dependencies and build
zig build

# Build with specific optimization
zig build -Doptimize=ReleaseSmall
```

## Model Compatibility

MLz is tested with **Llama 3.2 3B Instruct** in GGUF format (specifically `Q4_K_M` quantization). Most GGUF models compatible with recent `llama.cpp` versions should work.

Download models from Hugging Face:
```bash
# Example: Llama 3.2 3B Instruct
# Download from unsloth or similar providers
wget https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```
