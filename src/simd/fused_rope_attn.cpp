// ----------------------------------------------------------------------------
// fused_rope_attn.cpp — fused RoPE + flash attention (f32, single head)
// ----------------------------------------------------------------------------
//
// Applies rotary position embedding to Q and K *inline* inside the online
// (flash) softmax loop, so the rotated Q/K are never materialised to memory.
// In the autoregressive decode case (n_q small, large KV cache) this removes a
// full RoPE pass over Q/K plus the read-back of the rotated tensors — the win
// the "fused" framing is about.
//
// RoPE: GGML "standard" / NORMAL convention — adjacent pairs (x[2i], x[2i+1])
// rotated by theta_i = pos * base^(-2i/D).
//
// Attention: O[i] = softmax_j( scale * dot(rope(Q_i), rope(K_j)) ) . V_j, with
// online rescaling (numerically identical to a batched softmax up to fp order).
//
// ponytail: plain C++ loops (autovectorise under ReleaseFast + native). Rotated
// rows live in small stack buffers, not the heap. A hand-tuned AVX/VNNI variant
// is the next lever; the per-row RoPE recompute (cos/sin via libm) is the
// obvious cost to hoist (precompute per-position tables) if this lands on the
// hot path. K is re-rotated per query — optimal at n_q==1 (decode); for prefill,
// caching rotated K is the upgrade.
//
// void simd_fused_rope_attn_f32(int n_q, int n_kv, int D,
//      const float* Q, const float* K, const float* V,
//      const int* q_pos, const int* k_pos,
//      float scale, float base, int causal, float* O);
// ----------------------------------------------------------------------------

#include <cmath>
#include <cfloat>
#include <cstring>
#include <algorithm>

#ifndef FRA_MAX_D
#define FRA_MAX_D 1024
#endif

// Rotate one row (D floats) by position `pos`, writing D floats to `out`.
static inline void fra_rope_row(const float* __restrict x, float* __restrict out,
                                int D, int pos, float base) {
    const int half = D / 2;
    const float fpos = (float)pos;
    for (int i = 0; i < half; i++) {
        const float freq = powf(base, -2.0f * (float)i / (float)D);
        const float theta = fpos * freq;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float a = x[2 * i];
        const float b = x[2 * i + 1];
        out[2 * i]     = a * c - b * s;
        out[2 * i + 1] = a * s + b * c;
    }
    // odd tail (D not even): pass through
    if (D & 1) out[D - 1] = x[D - 1];
}

static inline float fra_dot(const float* __restrict a, const float* __restrict b, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; d++) s += a[d] * b[d];
    return s;
}

extern "C" void simd_fused_rope_attn_f32(
    int n_q, int n_kv, int D,
    const float* Q, const float* K, const float* V,
    const int* q_pos, const int* k_pos,
    float scale, float base, int causal, float* O)
{
    if (D <= 0 || D > FRA_MAX_D || n_q <= 0 || n_kv <= 0) return;

    float qrot[FRA_MAX_D];
    float krot[FRA_MAX_D];
    float acc[FRA_MAX_D];

    for (int i = 0; i < n_q; i++) {
        fra_rope_row(Q + (size_t)i * D, qrot, D, q_pos[i], base);

        float m = -FLT_MAX;   // running max
        float l = 0.0f;       // running denom
        memset(acc, 0, sizeof(float) * (size_t)D);

        const int qp = q_pos[i];
        for (int j = 0; j < n_kv; j++) {
            if (causal && k_pos[j] > qp) continue;

            fra_rope_row(K + (size_t)j * D, krot, D, k_pos[j], base);
            const float s = scale * fra_dot(qrot, krot, D);

            const float mnew = std::max(m, s);
            const float corr = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
            const float p = expf(s - mnew);

            const float* __restrict Vj = V + (size_t)j * D;
            for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * Vj[d];
            l = l * corr + p;
            m = mnew;
        }

        float* __restrict Oi = O + (size_t)i * D;
        if (l > 0.0f) {
            const float inv = 1.0f / l;
            for (int d = 0; d < D; d++) Oi[d] = acc[d] * inv;
        } else {
            memset(Oi, 0, sizeof(float) * (size_t)D);
        }
    }
}

// Unfused two-pass baseline (for benchmarking the fusion win): materialise the
// rotated K cache to `kbuf` first, then attend. `kbuf` must hold n_kv*D floats.
// Demonstrates the memory round-trip the fused kernel avoids.
extern "C" void simd_unfused_rope_attn_f32(
    int n_q, int n_kv, int D,
    const float* Q, const float* K, const float* V,
    const int* q_pos, const int* k_pos,
    float scale, float base, int causal, float* O, float* kbuf)
{
    if (D <= 0 || D > FRA_MAX_D || n_q <= 0 || n_kv <= 0) return;

    // Pass 1: rotate all K rows into kbuf.
    for (int j = 0; j < n_kv; j++) {
        fra_rope_row(K + (size_t)j * D, kbuf + (size_t)j * D, D, k_pos[j], base);
    }

    float qrot[FRA_MAX_D];
    float acc[FRA_MAX_D];
    for (int i = 0; i < n_q; i++) {
        fra_rope_row(Q + (size_t)i * D, qrot, D, q_pos[i], base);
        float m = -FLT_MAX, l = 0.0f;
        memset(acc, 0, sizeof(float) * (size_t)D);
        const int qp = q_pos[i];
        for (int j = 0; j < n_kv; j++) {
            if (causal && k_pos[j] > qp) continue;
            const float s = scale * fra_dot(qrot, kbuf + (size_t)j * D, D);
            const float mnew = std::max(m, s);
            const float corr = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
            const float p = expf(s - mnew);
            const float* __restrict Vj = V + (size_t)j * D;
            for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * Vj[d];
            l = l * corr + p;
            m = mnew;
        }
        float* __restrict Oi = O + (size_t)i * D;
        if (l > 0.0f) { const float inv = 1.0f / l; for (int d = 0; d < D; d++) Oi[d] = acc[d] * inv; }
        else memset(Oi, 0, sizeof(float) * (size_t)D);
    }
}
