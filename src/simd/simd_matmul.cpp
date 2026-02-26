/**
 * @file simd_matmul.cpp
 * @brief SIMD Matrix Multiplication Implementation
 *
 * CPU feature detection and dispatch logic for AVX2/AVX-512 matrix
 * multiplication.
 *
 * Based on: https://github.com/sutantodadang/assembly-simd
 * License: MIT
 */

#include "simd_matmul.h"
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
// ARM AArch64 — NEON is always available
#else
// x86_64 — need CPUID for feature detection
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

/*============================================================================
 * CPU FEATURE DETECTION
 *============================================================================*/

#if defined(__aarch64__) || defined(_M_ARM64)

// NEON is always available on AArch64
int simd_check_neon(void) { return 1; }

// Stubs for x86 functions on ARM
int simd_check_avx2(void) { return 0; }
int simd_check_avx512(void) { return 0; }
int simd_check_f16c(void) { return 0; }
int simd_check_avx512_fp16(void) { return 0; }

#else  // x86_64

// NEON stub for x86
int simd_check_neon(void) { return 0; }

// CPUID helper
static void cpuid(int info[4], int leaf, int subleaf = 0) {
#ifdef _WIN32
  __cpuidex(info, leaf, subleaf);
#else
  __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
#endif
}

// Check XGETBV for OS support of extended registers
static unsigned long long xgetbv(unsigned int index) {
#ifdef _WIN32
  return _xgetbv(index);
#else
  unsigned int eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((unsigned long long)edx << 32) | eax;
#endif
}

// Cached feature detection results
static int g_avx2_checked = 0;
static int g_avx2_supported = 0;
static int g_avx512_checked = 0;
static int g_avx512_supported = 0;

int simd_check_avx2(void) {
  if (g_avx2_checked) {
    return g_avx2_supported;
  }
  g_avx2_checked = 1;

  int info[4] = {0};

  // Check for CPUID support and get max leaf
  cpuid(info, 0);
  int max_leaf = info[0];
  if (max_leaf < 7) {
    return (g_avx2_supported = 0);
  }

  // Check leaf 1 for AVX and OSXSAVE
  cpuid(info, 1);
  bool has_avx = (info[2] & (1 << 28)) != 0;     // AVX
  bool has_osxsave = (info[2] & (1 << 27)) != 0; // OSXSAVE
  bool has_fma = (info[2] & (1 << 12)) != 0;     // FMA

  if (!has_avx || !has_osxsave || !has_fma) {
    return (g_avx2_supported = 0);
  }

  // Check OS support for YMM registers
  unsigned long long xcr0 = xgetbv(0);
  bool ymm_enabled = (xcr0 & 6) == 6; // XMM and YMM state enabled
  if (!ymm_enabled) {
    return (g_avx2_supported = 0);
  }

  // Check leaf 7 for AVX2
  cpuid(info, 7, 0);
  bool has_avx2 = (info[1] & (1 << 5)) != 0; // AVX2

  return (g_avx2_supported = has_avx2 ? 1 : 0);
}

int simd_check_avx512(void) {
  if (g_avx512_checked) {
    return g_avx512_supported;
  }
  g_avx512_checked = 1;

  // AVX2 is a prerequisite
  if (!simd_check_avx2()) {
    return (g_avx512_supported = 0);
  }

  int info[4] = {0};

  // Check OS support for ZMM registers
  unsigned long long xcr0 = xgetbv(0);
  bool zmm_enabled = (xcr0 & 0xE6) == 0xE6; // XMM, YMM, ZMM, and opmask
  if (!zmm_enabled) {
    return (g_avx512_supported = 0);
  }

  // Check leaf 7 for AVX-512F
  cpuid(info, 7, 0);
  bool has_avx512f = (info[1] & (1 << 16)) != 0; // AVX-512F

  return (g_avx512_supported = has_avx512f ? 1 : 0);
}

// Cached F16C detection
static int g_f16c_checked = 0;
static int g_f16c_supported = 0;

int simd_check_f16c(void) {
  if (g_f16c_checked) {
    return g_f16c_supported;
  }
  g_f16c_checked = 1;

  int info[4] = {0};
  cpuid(info, 1);
  bool has_f16c = (info[2] & (1 << 29)) != 0; // F16C bit in ECX

  return (g_f16c_supported = has_f16c ? 1 : 0);
}

// Cached AVX-512 FP16 detection (Intel Sapphire Rapids+)
static int g_avx512_fp16_checked = 0;
static int g_avx512_fp16_supported = 0;

int simd_check_avx512_fp16(void) {
  if (g_avx512_fp16_checked) {
    return g_avx512_fp16_supported;
  }
  g_avx512_fp16_checked = 1;

  if (!simd_check_avx512()) {
    return (g_avx512_fp16_supported = 0);
  }

  int info[4] = {0};
  cpuid(info, 7, 0);
  bool has_avx512_fp16 = (info[3] & (1 << 23)) != 0; // AVX-512 FP16 in EDX

  return (g_avx512_fp16_supported = has_avx512_fp16 ? 1 : 0);
}

#endif  // __aarch64__ / x86_64

/*============================================================================
 * MATRIX MULTIPLICATION DISPATCH
 *============================================================================*/

// External assembly functions (linked from .asm/.S files)
extern "C" {
#if defined(__aarch64__) || defined(_M_ARM64)
void matrix_mult_neon(const float *A, const float *B, float *C, size_t M,
                      size_t N, size_t K);
#else
void matrix_mult_avx2(const float *A, const float *B, float *C, size_t M,
                      size_t N, size_t K);
void matrix_mult_avx512(const float *A, const float *B, float *C, size_t M,
                        size_t N, size_t K);
#endif
}

// Scalar fallback implementation
static void matrix_mult_scalar(const float *A, const float *B, float *C,
                               size_t M, size_t N, size_t K) {
  // Zero the output matrix
  memset(C, 0, M * N * sizeof(float));

  // Standard i-k-j loop order (cache-friendly)
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      float a_ik = A[i * K + k];
      const float *B_row = B + k * N;
      float *C_row = C + i * N;
      for (size_t j = 0; j < N; j++) {
        C_row[j] += a_ik * B_row[j];
      }
    }
  }
}

void simd_matmul_f32(const float *A, const float *B, float *C, size_t M,
                     size_t N, size_t K, int ith, int nth) {
  // For very small matrices, use scalar to avoid SIMD overhead
  if (M * N * K < 512) {
    // Only one thread should handle small matrices to avoid contention
    if (ith == 0) {
      matrix_mult_scalar(A, B, C, M, N, K);
    }
    return;
  }

  // Calculate thread partition
  size_t rows_per_thread = (M + nth - 1) / nth;
  size_t start_row = ith * rows_per_thread;
  size_t end_row = start_row + rows_per_thread;
  if (end_row > M) end_row = M;

  if (start_row >= end_row) {
    return;
  }

  size_t local_M = end_row - start_row;
  const float *local_A = A + start_row * K;
  float *local_C = C + start_row * N;

  // Dispatch to best available implementation
#if defined(__aarch64__) || defined(_M_ARM64)
    matrix_mult_neon(local_A, B, local_C, local_M, N, K);
#else
  if (simd_check_avx512()) {
    matrix_mult_avx512(local_A, B, local_C, local_M, N, K);
  } else if (simd_check_avx2()) {
    matrix_mult_avx2(local_A, B, local_C, local_M, N, K);
  } else {
    matrix_mult_scalar(local_A, B, local_C, local_M, N, K);
  }
#endif
}


/*============================================================================
 * FP16 SUPPORT
 *============================================================================*/

#if defined(__aarch64__) || defined(_M_ARM64)
// ARM: Use NEON fcvtl/fcvtn for FP16<->FP32 conversion
#include <arm_neon.h>

static void convert_f16_to_f32(const uint16_t *src, float *dst, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    float16x4_t f16 = vld1_f16((const __fp16 *)(src + i));
    float32x4_t f32 = vcvt_f32_f16(f16);
    vst1q_f32(dst + i, f32);
  }
  for (; i < count; i++) {
    __fp16 h;
    memcpy(&h, src + i, sizeof(h));
    dst[i] = (float)h;
  }
}

static void convert_f32_to_f16(const float *src, uint16_t *dst, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    float32x4_t f32 = vld1q_f32(src + i);
    float16x4_t f16 = vcvt_f16_f32(f32);
    vst1_f16((__fp16 *)(dst + i), f16);
  }
  for (; i < count; i++) {
    __fp16 h = (__fp16)src[i];
    memcpy(dst + i, &h, sizeof(h));
  }
}

// Use posix_memalign on Linux/ARM
#include <cstdlib>
static inline void* simd_aligned_alloc(size_t size, size_t align) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, align, size) != 0) return nullptr;
  return ptr;
}
static inline void simd_aligned_free(void* ptr) { free(ptr); }

#else  // x86_64

#ifdef _WIN32
#include <immintrin.h>
#else
#include <immintrin.h>
#endif

// Convert FP16 array to FP32 using F16C instructions (8 elements at a time)
static void convert_f16_to_f32(const uint16_t *src, float *dst, size_t count) {
  size_t i = 0;

  if (simd_check_f16c()) {
    // Use F16C SIMD conversion (vcvtph2ps)
    for (; i + 8 <= count; i += 8) {
      __m128i f16_vec = _mm_loadu_si128((const __m128i *)(src + i));
      __m256 f32_vec = _mm256_cvtph_ps(f16_vec);
      _mm256_storeu_ps(dst + i, f32_vec);
    }
  }

  // Scalar fallback for remaining elements
  for (; i < count; i++) {
    // IEEE 754 half-precision to single-precision conversion
    uint16_t h = src[i];
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
      // Denormalized or zero
      if (mant == 0) {
        // Zero
        uint32_t result = sign;
        memcpy(dst + i, &result, sizeof(float));
      } else {
        // Denormalized - convert to normalized
        while ((mant & 0x400) == 0) {
          mant <<= 1;
          exp--;
        }
        exp++;
        mant &= 0x3FF;
        uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
        memcpy(dst + i, &result, sizeof(float));
      }
    } else if (exp == 31) {
      // Inf or NaN
      uint32_t result = sign | 0x7F800000 | (mant << 13);
      memcpy(dst + i, &result, sizeof(float));
    } else {
      // Normalized
      uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
      memcpy(dst + i, &result, sizeof(float));
    }
  }
}

// Convert FP32 array to FP16 using F16C instructions (8 elements at a time)
static void convert_f32_to_f16(const float *src, uint16_t *dst, size_t count) {
  size_t i = 0;

  if (simd_check_f16c()) {
    // Use F16C SIMD conversion (vcvtps2ph)
    for (; i + 8 <= count; i += 8) {
      __m256 f32_vec = _mm256_loadu_ps(src + i);
      __m128i f16_vec = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
      _mm_storeu_si128((__m128i *)(dst + i), f16_vec);
    }
  }

  // Scalar fallback for remaining elements
  for (; i < count; i++) {
    float f = src[i];
    uint32_t f32;
    memcpy(&f32, &f, sizeof(float));

    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f32 >> 13) & 0x3FF;

    if (exp <= 0) {
      // Denormalized or zero
      if (exp < -10) {
        dst[i] = (uint16_t)sign; // Too small, flush to zero
      } else {
        mant = (mant | 0x400) >> (1 - exp);
        dst[i] = (uint16_t)(sign | mant);
      }
    } else if (exp >= 31) {
      // Overflow to Inf
      dst[i] = (uint16_t)(sign | 0x7C00);
    } else {
      dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
  }
}

// x86 aligned allocation helpers
static inline void* simd_aligned_alloc(size_t size, size_t align) {
#ifdef _WIN32
  return _aligned_malloc(size, align);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, align, size) != 0) return nullptr;
  return ptr;
#endif
}
static inline void simd_aligned_free(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

#endif  // __aarch64__ / x86_64

void simd_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     size_t M, size_t N, size_t K, int ith, int nth) {
  // Calculate thread partition for M rows
  size_t rows_per_thread = (M + nth - 1) / nth;
  size_t start_row = ith * rows_per_thread;
  size_t end_row = start_row + rows_per_thread;
  if (end_row > M) end_row = M;

  if (start_row >= end_row) {
    return;
  }

  size_t local_M = end_row - start_row;
  const uint16_t *local_A = A + start_row * K;
  uint16_t *local_C = C + start_row * N;

  // Allocate temporary F32 buffers
  // Note: B is fully duplicated per thread because we can't easily share it here.
  // A and C are sliced.
  size_t A_size = local_M * K;
  size_t B_size = K * N;
  size_t C_size = local_M * N;

  float *A_f32 = (float *)simd_aligned_alloc(A_size * sizeof(float), 64);
  float *B_f32 = (float *)simd_aligned_alloc(B_size * sizeof(float), 64);
  float *C_f32 = (float *)simd_aligned_alloc(C_size * sizeof(float), 64);

  if (!A_f32 || !B_f32 || !C_f32) {
    if (A_f32)
      simd_aligned_free(A_f32);
    if (B_f32)
      simd_aligned_free(B_f32);
    if (C_f32)
      simd_aligned_free(C_f32);
    return; // Allocation failed
  }

  // Convert inputs FP16 -> FP32
  convert_f16_to_f32(local_A, A_f32, A_size);
  convert_f16_to_f32(B, B_f32, B_size);

  // Perform F32 matrix multiplication using optimized kernels
  // We act as a single thread for this slice
  simd_matmul_f32(A_f32, B_f32, C_f32, local_M, N, K, 0, 1);

  // Convert output FP32 -> FP16
  convert_f32_to_f16(C_f32, local_C, C_size);

  // Cleanup
  simd_aligned_free(A_f32);
  simd_aligned_free(B_f32);
  simd_aligned_free(C_f32);
}

