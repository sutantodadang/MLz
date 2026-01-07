# MLz - LLaMA Inference in Zig

MLz is a Zig implementation for running LLaMA language models. It includes three implementations:

1. **llama.cpp bindings** - Fast, production-ready (uses C++ llama.cpp)
2. **Native Zig + GGML** - Experimental Zig inference using GGML for tensor ops
3. **Pure Zig with SIMD** - Educational pure Zig with native `@Vector` SIMD ⭐ NEW

## Quick Start

```bash
# Build all executables
zig build

# Option 1: Fast inference (llama.cpp bindings)
.\zig-out\bin\MLz.exe --gguf Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Option 2: Pure Zig with native SIMD (educational)
.\zig-out\bin\MLz-pure.exe Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Option 3: Native Zig + GGML (experimental)
.\zig-out\bin\MLz-native.exe Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

## Project Structure

```
src/
├── main.zig          # Entry point (llama.cpp bindings)
├── llama.zig         # ✅ WORKING: LLaMA using GGML
├── llama_cpp.zig     # C bindings to llama.cpp
│
├── main_pure.zig     # ⭐ Pure Zig chat interface
├── llama_pure.zig    # ⭐ Pure Zig LLaMA inference
├── tensor.zig        # ⭐ SIMD tensor operations (@Vector)
│
├── main_native.zig   # ⚠️ EXPERIMENTAL: Native + GGML
├── llama_native.zig  # ⚠️ EXPERIMENTAL: WIP
│
├── ggml.zig          # GGML tensor library bindings
├── gguf.zig          # GGUF file format parser
└── tokenizer.zig     # BPE tokenizer implementation
```

## Pure Zig SIMD Implementation ⭐

The `tensor.zig` module demonstrates how to use Zig's native SIMD:

```zig
const std = @import("std");

// SIMD vector type - maps to CPU registers (AVX = 256-bit)
pub const Vec = @Vector(8, f32);

/// Load 8 floats from memory into SIMD register
pub inline fn simdLoad(ptr: [*]const f32) Vec {
    return ptr[0..8].*;
}

/// SIMD-optimized dot product
fn dotProduct(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    var sum: Vec = @splat(0);
    var i: usize = 0;
    
    // Process 8 elements at a time with SIMD
    while (i + 8 <= len) : (i += 8) {
        const va = simdLoad(a + i);
        const vb = simdLoad(b + i);
        sum += va * vb;  // Fused multiply-add
    }
    
    return @reduce(.Add, sum) + scalarRemainder(a, b, i, len);
}
```

### Key Operations (Pure Zig)

| Operation | File | Description |
|-----------|------|-------------|
| `matmul` | tensor.zig | SIMD matrix multiplication |
| `matmulTransB` | tensor.zig | Cache-friendly A @ B^T |
| `rmsNorm` | tensor.zig | RMS normalization |
| `softmax` | tensor.zig | Attention softmax |
| `applyRope` | tensor.zig | Rotary position embedding |
| `siluInplace` | tensor.zig | SiLU activation |

### Performance Note

The pure Zig implementation is **slower** than GGML because:
- GGML has hand-optimized assembly kernels
- GGML uses threading (OpenMP)
- We dequantize weights on-demand

However, it's fully **educational** - you can read every line of code!

## How It Works

### 1. Model Loading (gguf.zig)

The GGUF format stores:
- Model metadata (dimensions, vocab size, etc.)
- Tensor information (shapes, types, offsets)
- Tensor data (quantized weights)

```zig
// Parse GGUF and load tensors
var model = try Model.init(allocator, "model.gguf");
defer model.deinit();
```

### 2. Tokenization (tokenizer.zig)

Converts text to token IDs using BPE (Byte-Pair Encoding):

```zig
var tok = try Tokenizer.initFromGguf(allocator, "model.gguf");
const tokens = try tok.encode("Hello, world!");
const text = try tok.decode(tokens);
```

### 3. Inference (llama.zig)

The transformer forward pass:

```
Input Tokens
     │
     ▼
┌─────────────────┐
│  Token Embed   │  Look up embedding vectors
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Transformer   │  × N layers
│    Layer       │
│  ┌───────────┐ │
│  │ RMS Norm  │ │
│  │ Attention │ │  Multi-head attention with KV cache
│  │ + Residual│ │
│  ├───────────┤ │
│  │ RMS Norm  │ │
│  │ FFN(SwiGLU)│  Feed-forward with gated activation
│  │ + Residual│ │
│  └───────────┘ │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  RMS Norm      │
│  Output Proj   │  Project to vocabulary size
└─────────────────┘
     │
     ▼
   Logits (vocab_size)
```

### 4. Sampling

Convert logits to probability distribution and sample next token:

```zig
// Top-K sampling
const next_token = sampleTopK(logits, k: 40);
```

## Key Concepts

### Grouped Query Attention (GQA)

LLaMA uses GQA where multiple query heads share the same key/value heads:
- Reduces memory usage for KV cache
- n_heads = 24 (queries), n_kv_heads = 8 (shared)
- Each KV head serves 3 query heads (24/8 = 3)

### RoPE (Rotary Position Embedding)

Position information is encoded by rotating the embedding vectors:
```
x'[2d] = x[2d] * cos(θ) - x[2d+1] * sin(θ)
x'[2d+1] = x[2d+1] * cos(θ) + x[2d] * sin(θ)
```

### KV Cache

Stores computed key/value pairs from previous tokens to avoid recomputation:
```
First forward: compute K,V for all tokens, store in cache
Next forward: only compute K,V for new token, read past from cache
```

## GGML Dependency

This project uses GGML (the tensor library from llama.cpp) for:
- Matrix multiplication (optimized SIMD/BLAS)
- Tensor operations (reshape, transpose, etc.)
- Quantized tensor support (Q4_K_M, Q8, etc.)

GGML is NOT llama.cpp - it's the low-level tensor library that llama.cpp uses.

## Building from Source

Requirements:
- Zig 0.15.x
- C++ compiler (for GGML)

```bash
# Fetch dependencies and build
zig build

# Build specific targets
zig build MLz        # Main executable
zig build MLz-native # Experimental native build
```

## Model Compatibility

Tested with:
- Llama-3.2-3B-Instruct (Q4_K_M quantization)
- Other GGUF format models should work

Download models from Hugging Face:
```bash
# Example: Download from TheBloke
wget https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! The native implementation (`llama_native.zig`) needs help with:
- Fixing the attention computation for correct output
- Properly implementing GQA (Grouped Query Attention)  
- KV cache layout optimization
