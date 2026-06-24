// ----------------------------------------------------------------------------
// fused_rope_attn.cpp — fused RoPE + flash attention (f32, single head)
// ----------------------------------------------------------------------------
//
// Applies rotary position embedding to Q and K *inline* inside the online
// (flash) softmax loop, so the rotated Q/K are never materialised to memory.
// In autoregressive decode (small n_q, large KV) this removes a full RoPE pass
// over Q/K plus the read-back of the rotated tensors.
//
// Two performance levers vs the first scalar version:
//   1. sin/cos TABLES — cos/sin per (position, pair) are precomputed once from
//      libm, then the inner loop is pure FMA. For n_q>1 the table is reused
//      across all queries, so the libm cost is paid once instead of per query.
//   2. AVX2 fused inner loop — rotation, score dot, and the V accumulation all
//      run 8 floats/instr. Rotation uses the pair-swap identity:
//         out = x*cos + sign * swap(x) * sin,
//      where swap() exchanges the two halves of each adjacent pair (vpermilps
//      0xB1) and sign = {-1,+1,...}, so cos/sin can be applied lane-wise.
//
// RoPE: GGML "standard" / NORMAL convention — adjacent pairs (x[2i], x[2i+1])
// rotated by theta_i = pos * base^(-2i/D).
//
// ponytail: table is built over the position RANGE [minpos, maxpos]; if that
// range is implausibly large it falls back to the scalar inline path (no table,
// libm per element). The next lever for decode (n_q==1, where the table build
// is not amortised) is a vectorised sin/cos approximation in the table build.
// ----------------------------------------------------------------------------

#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdint>
#include <algorithm>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FRA_HAVE_AVX2 1
#endif

#ifndef FRA_MAX_D
#define FRA_MAX_D 1024
#endif

// Scalar inline rotation (fallback / odd tail).
static inline void fra_rope_row_scalar(const float* __restrict x, float* __restrict out,
                                       int D, int pos, float base) {
    const int half = D / 2;
    const float fpos = (float)pos;
    for (int i = 0; i < half; i++) {
        const float freq = powf(base, -2.0f * (float)i / (float)D);
        const float theta = fpos * freq;
        const float c = cosf(theta), s = sinf(theta);
        const float a = x[2 * i], b = x[2 * i + 1];
        out[2 * i]     = a * c - b * s;
        out[2 * i + 1] = a * s + b * c;
    }
    if (D & 1) out[D - 1] = x[D - 1];
}

#ifdef FRA_HAVE_AVX2
// Vectorised sin/cos (Cephes single-precision polynomial, after Pommier's
// sse_mathfun sincos_ps, AVX2 port). Accurate to <~1e-6 abs for |x| up to a few
// thousand radians (single Cody-Waite reduction) — enough for RoPE thetas at
// realistic context lengths. Writes sin(x), cos(x).
static inline void fra_sincos256(__m256 x, __m256* s_out, __m256* c_out) {
    const __m256 FOPI = _mm256_set1_ps(1.27323954473516f);     // 4/pi
    const __m256 DP1  = _mm256_set1_ps(-0.78515625f);
    const __m256 DP2  = _mm256_set1_ps(-2.4187564849853515625e-4f);
    const __m256 DP3  = _mm256_set1_ps(-3.77489497744594108e-8f);
    const __m256 sc0  = _mm256_set1_ps(-1.9515295891e-4f);
    const __m256 sc1  = _mm256_set1_ps( 8.3321608736e-3f);
    const __m256 sc2  = _mm256_set1_ps(-1.6666654611e-1f);
    const __m256 cc0  = _mm256_set1_ps( 2.443315711809948e-5f);
    const __m256 cc1  = _mm256_set1_ps(-1.388731625493765e-3f);
    const __m256 cc2  = _mm256_set1_ps( 4.166664568298827e-2f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    __m256 sign_bit_sin = _mm256_and_ps(x, sign_mask);
    x = _mm256_andnot_ps(sign_mask, x);                        // abs(x)

    __m256 y = _mm256_mul_ps(x, FOPI);
    __m256i emm2 = _mm256_cvttps_epi32(y);
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);
    __m256i emm4 = emm2;

    __m256i emm0 = _mm256_and_si256(emm2, _mm256_set1_epi32(4));
    emm0 = _mm256_slli_epi32(emm0, 29);
    __m256 swap_sign_sin = _mm256_castsi256_ps(emm0);

    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(2));
    __m256 poly_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256()));

    // Cody-Waite range reduction
    x = _mm256_fmadd_ps(y, DP1, x);
    x = _mm256_fmadd_ps(y, DP2, x);
    x = _mm256_fmadd_ps(y, DP3, x);

    emm4 = _mm256_sub_epi32(emm4, _mm256_set1_epi32(2));
    emm4 = _mm256_andnot_si256(emm4, _mm256_set1_epi32(4));
    emm4 = _mm256_slli_epi32(emm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(emm4);
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_sin);

    __m256 z = _mm256_mul_ps(x, x);
    // cos poly
    __m256 yc = _mm256_fmadd_ps(cc0, z, cc1);
    yc = _mm256_fmadd_ps(yc, z, cc2);
    yc = _mm256_mul_ps(_mm256_mul_ps(yc, z), z);
    yc = _mm256_fnmadd_ps(z, half, yc);                        // yc - 0.5*z
    yc = _mm256_add_ps(yc, one);
    // sin poly
    __m256 ys = _mm256_fmadd_ps(sc0, z, sc1);
    ys = _mm256_fmadd_ps(ys, z, sc2);
    ys = _mm256_mul_ps(_mm256_mul_ps(ys, z), x);
    ys = _mm256_add_ps(ys, x);

    // select per poly_mask: sin = mask? ys : yc ; cos = mask? yc : ys
    __m256 sin = _mm256_blendv_ps(yc, ys, poly_mask);
    __m256 cos = _mm256_blendv_ps(ys, yc, poly_mask);
    *s_out = _mm256_xor_ps(sin, sign_bit_sin);
    *c_out = _mm256_xor_ps(cos, sign_bit_cos);
}

// Rotate one row with on-the-fly vectorised sin/cos. freq[i] = base^(-2i/D)
// (precomputed once by the caller). theta_i = pos * freq_i.
static inline void fra_rope_row_vec(const float* __restrict x, float* __restrict out,
                                    int D, float pos, const float* __restrict freq) {
    const int half = D / 2;
    const __m256 vpos = _mm256_set1_ps(pos);
    const __m256i idx_lo = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i idx_hi = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
    const __m256 sign = _mm256_setr_ps(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
    int i = 0;
    for (; i + 8 <= half; i += 8) {
        __m256 th = _mm256_mul_ps(vpos, _mm256_loadu_ps(freq + i));
        __m256 sc, cc;
        fra_sincos256(th, &sc, &cc);
        // expand cos/sin: each pair value duplicated to its two slots
        __m256 c_lo = _mm256_permutevar8x32_ps(cc, idx_lo);
        __m256 c_hi = _mm256_permutevar8x32_ps(cc, idx_hi);
        __m256 s_lo = _mm256_permutevar8x32_ps(sc, idx_lo);
        __m256 s_hi = _mm256_permutevar8x32_ps(sc, idx_hi);
        // lo 4 pairs (8 floats)
        __m256 xl = _mm256_loadu_ps(x + 2 * i);
        __m256 tl = _mm256_mul_ps(_mm256_mul_ps(_mm256_permute_ps(xl, 0xB1), s_lo), sign);
        _mm256_storeu_ps(out + 2 * i, _mm256_fmadd_ps(xl, c_lo, tl));
        // hi 4 pairs
        __m256 xh = _mm256_loadu_ps(x + 2 * i + 8);
        __m256 th2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_permute_ps(xh, 0xB1), s_hi), sign);
        _mm256_storeu_ps(out + 2 * i + 8, _mm256_fmadd_ps(xh, c_hi, th2));
    }
    // scalar tail pairs
    for (; i < half; i++) {
        const float th = pos * freq[i];
        const float c = cosf(th), s = sinf(th);
        const float a = x[2 * i], b = x[2 * i + 1];
        out[2 * i] = a * c - b * s;
        out[2 * i + 1] = a * s + b * c;
    }
    if (D & 1) out[D - 1] = x[D - 1];
}
#endif // FRA_HAVE_AVX2

static inline float fra_dot_scalar(const float* __restrict a, const float* __restrict b, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; d++) s += a[d] * b[d];
    return s;
}

// Rotate row using a precomputed table row (cos/sin duplicated to both pair
// slots: ct[2i]=ct[2i+1]=cos_i, st likewise). out[d] = x*ct + sign*swap(x)*st.
static inline void fra_rope_row_table(const float* __restrict x, float* __restrict out,
                                      const float* __restrict ct, const float* __restrict st,
                                      int D) {
    int d = 0;
#ifdef FRA_HAVE_AVX2
    // sign = {-1,+1,-1,+1,...} : negate even lanes, keep odd.
    const __m256 sign = _mm256_setr_ps(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
    for (; d + 8 <= D; d += 8) {
        __m256 vx = _mm256_loadu_ps(x + d);
        __m256 vc = _mm256_loadu_ps(ct + d);
        __m256 vs = _mm256_loadu_ps(st + d);
        // swap adjacent pairs: [1,0,3,2,5,4,7,6]
        __m256 sw = _mm256_permute_ps(vx, 0xB1);
        __m256 t  = _mm256_mul_ps(_mm256_mul_ps(sw, vs), sign);
        __m256 r  = _mm256_fmadd_ps(vx, vc, t);
        _mm256_storeu_ps(out + d, r);
    }
#endif
    for (; d + 1 < D; d += 2) {
        const float c = ct[d], s = st[d];
        const float a = x[d], b = x[d + 1];
        out[d]     = a * c - b * s;
        out[d + 1] = a * s + b * c;
    }
    if (D & 1) out[D - 1] = x[D - 1];
}

static inline float fra_dot(const float* __restrict a, const float* __restrict b, int D) {
#ifdef FRA_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; d < D; d++) s += a[d] * b[d];
    return s;
#else
    return fra_dot_scalar(a, b, D);
#endif
}

// acc[d] = acc[d]*corr + p*V[d]
static inline void fra_axpy(float* __restrict acc, const float* __restrict V,
                            float corr, float p, int D) {
    int d = 0;
#ifdef FRA_HAVE_AVX2
    const __m256 vc = _mm256_set1_ps(corr);
    const __m256 vp = _mm256_set1_ps(p);
    for (; d + 8 <= D; d += 8) {
        __m256 a = _mm256_loadu_ps(acc + d);
        a = _mm256_mul_ps(a, vc);
        a = _mm256_fmadd_ps(vp, _mm256_loadu_ps(V + d), a);
        _mm256_storeu_ps(acc + d, a);
    }
#endif
    for (; d < D; d++) acc[d] = acc[d] * corr + p * V[d];
}

// Build cos/sin table over positions [minpos, maxpos], each row length D with
// cos/sin duplicated to both pair slots. Returns false if the range is too big.
static bool fra_build_table(int minpos, int maxpos, int D, float base,
                            float** ct_out, float** st_out, int* span_out) {
    const int span = maxpos - minpos + 1;
    if (span <= 0 || (int64_t)span * D > (int64_t)64 * 1024 * 1024) return false;
    const int half = D / 2;
    float* ct = new (std::nothrow) float[(size_t)span * D];
    float* st = new (std::nothrow) float[(size_t)span * D];
    if (!ct || !st) { delete[] ct; delete[] st; return false; }
    for (int p = 0; p < span; p++) {
        const float fpos = (float)(p + minpos);
        float* cr = ct + (size_t)p * D;
        float* sr = st + (size_t)p * D;
        for (int i = 0; i < half; i++) {
            const float freq = powf(base, -2.0f * (float)i / (float)D);
            const float th = fpos * freq;
            const float c = cosf(th), s = sinf(th);
            cr[2 * i] = cr[2 * i + 1] = c;
            sr[2 * i] = sr[2 * i + 1] = s;
        }
        if (D & 1) { cr[D - 1] = 1.0f; sr[D - 1] = 0.0f; }
    }
    *ct_out = ct; *st_out = st; *span_out = span;
    return true;
}

// Thin wrapper exposing the C++ intrinsic rotation for the asm-vs-intrinsics
// comparison (mirrors simd_rope_row_avx2). pos as int, freq[half] precomputed.
extern "C" void simd_rope_row_cpp(int half, int pos, const float* freq,
                                  const float* x, float* out) {
#ifdef FRA_HAVE_AVX2
    fra_rope_row_vec(x, out, half * 2, (float)pos, freq);
#else
    (void)half; (void)pos; (void)freq; (void)x; (void)out;
#endif
}

extern "C" void simd_fused_rope_attn_f32(
    int n_q, int n_kv, int D,
    const float* Q, const float* K, const float* V,
    const int* q_pos, const int* k_pos,
    float scale, float base, int causal, float* O)
{
    if (D <= 0 || D > FRA_MAX_D || n_q <= 0 || n_kv <= 0) return;

    // Position range across Q and K (for the shared sin/cos table).
    int minp = q_pos[0], maxp = q_pos[0];
    for (int i = 0; i < n_q; i++) { minp = std::min(minp, q_pos[i]); maxp = std::max(maxp, q_pos[i]); }
    for (int j = 0; j < n_kv; j++) { minp = std::min(minp, k_pos[j]); maxp = std::max(maxp, k_pos[j]); }

    // Rotating K once and reusing it is optimal whenever n_q>1 (rotating K per
    // query would redo n_q-fold work). At n_q==1 (decode) rotate K inline — no
    // intermediate buffer. The sin/cos table amortises libm across the >1
    // rotations of the n_q>1 case; at n_q==1 it is not worth building.
    const bool multi_q = (n_q >= 2);
    float* ct = nullptr; float* st = nullptr; int span = 0;
    const bool have_table = multi_q && fra_build_table(minp, maxp, D, base, &ct, &st, &span);

#ifdef FRA_HAVE_AVX2
    // Decode path (no table): vectorised sin/cos. Precompute freq[half] once
    // (the only libm left), then rotations are pure SIMD.
    float freq[FRA_MAX_D / 2];
    const bool use_vec = !have_table;
    if (use_vec) {
        const int half = D / 2;
        for (int i = 0; i < half; i++) freq[i] = powf(base, -2.0f * (float)i / (float)D);
    }
#endif

    // n_q>1: rotate all K once into a cache, then attend.
    float* kcache = multi_q ? new (std::nothrow) float[(size_t)n_kv * D] : nullptr;
    if (multi_q && kcache) {
        for (int j = 0; j < n_kv; j++) {
            if (have_table) fra_rope_row_table(K + (size_t)j * D, kcache + (size_t)j * D, ct + (size_t)(k_pos[j] - minp) * D, st + (size_t)(k_pos[j] - minp) * D, D);
            else            fra_rope_row_scalar(K + (size_t)j * D, kcache + (size_t)j * D, D, k_pos[j], base);
        }
    }

#ifdef FRA_HAVE_AVX2
    #define FRA_ROPE(xp, op, pos) do { \
        if (have_table) fra_rope_row_table((xp), (op), ct + (size_t)((pos) - minp) * D, st + (size_t)((pos) - minp) * D, D); \
        else if (use_vec) fra_rope_row_vec((xp), (op), D, (float)(pos), freq); \
        else fra_rope_row_scalar((xp), (op), D, (pos), base); \
    } while (0)
#else
    #define FRA_ROPE(xp, op, pos) do { \
        if (have_table) fra_rope_row_table((xp), (op), ct + (size_t)((pos) - minp) * D, st + (size_t)((pos) - minp) * D, D); \
        else fra_rope_row_scalar((xp), (op), D, (pos), base); \
    } while (0)
#endif

    float qrot[FRA_MAX_D];
    float krot[FRA_MAX_D];
    float acc[FRA_MAX_D];

    for (int i = 0; i < n_q; i++) {
        FRA_ROPE(Q + (size_t)i * D, qrot, q_pos[i]);

        float m = -FLT_MAX, l = 0.0f;
        memset(acc, 0, sizeof(float) * (size_t)D);
        const int qp = q_pos[i];

        for (int j = 0; j < n_kv; j++) {
            if (causal && k_pos[j] > qp) continue;

            const float* Kj;
            if (kcache) {
                Kj = kcache + (size_t)j * D;          // rotated once, reuse
            } else {
                FRA_ROPE(K + (size_t)j * D, krot, k_pos[j]);  // decode: inline (vectorised)
                Kj = krot;
            }

            const float s = scale * fra_dot(qrot, Kj, D);
            const float mnew = std::max(m, s);
            const float corr = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
            const float p = expf(s - mnew);
            fra_axpy(acc, V + (size_t)j * D, corr, p, D);
            l = l * corr + p;
            m = mnew;
        }

        float* __restrict Oi = O + (size_t)i * D;
        if (l > 0.0f) { const float inv = 1.0f / l; for (int d = 0; d < D; d++) Oi[d] = acc[d] * inv; }
        else memset(Oi, 0, sizeof(float) * (size_t)D);
    }

    delete[] kcache;
    delete[] ct; delete[] st;
}

// Pure-scalar baseline (original approach: libm rotation, scalar dot/axpy,
// rotate K once into kbuf then attend). Kept for benchmarking the table+AVX2
// speedup of the optimised kernel above. Not the dispatch path.
extern "C" void simd_unfused_rope_attn_f32(
    int n_q, int n_kv, int D,
    const float* Q, const float* K, const float* V,
    const int* q_pos, const int* k_pos,
    float scale, float base, int causal, float* O, float* kbuf)
{
    if (D <= 0 || D > FRA_MAX_D || n_q <= 0 || n_kv <= 0) return;

    for (int j = 0; j < n_kv; j++)
        fra_rope_row_scalar(K + (size_t)j * D, kbuf + (size_t)j * D, D, k_pos[j], base);

    float qrot[FRA_MAX_D];
    float acc[FRA_MAX_D];
    for (int i = 0; i < n_q; i++) {
        fra_rope_row_scalar(Q + (size_t)i * D, qrot, D, q_pos[i], base);
        float m = -FLT_MAX, l = 0.0f;
        memset(acc, 0, sizeof(float) * (size_t)D);
        const int qp = q_pos[i];
        for (int j = 0; j < n_kv; j++) {
            if (causal && k_pos[j] > qp) continue;
            float s = 0.0f;
            const float* kr = kbuf + (size_t)j * D;
            for (int d = 0; d < D; d++) s += qrot[d] * kr[d];
            s *= scale;
            const float mnew = std::max(m, s);
            const float corr = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
            const float p = expf(s - mnew);
            const float* Vj = V + (size_t)j * D;
            for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * Vj[d];
            l = l * corr + p;
            m = mnew;
        }
        float* __restrict Oi = O + (size_t)i * D;
        if (l > 0.0f) { const float inv = 1.0f / l; for (int d = 0; d < D; d++) Oi[d] = acc[d] * inv; }
        else memset(Oi, 0, sizeof(float) * (size_t)D);
    }
}
