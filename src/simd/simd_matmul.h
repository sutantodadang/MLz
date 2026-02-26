/**
 * @file simd_matmul.h
 * @brief SIMD Matrix Multiplication API for MLz
 *
 * Public C interface for high-performance matrix multiplication
 * using AVX2 and AVX-512 SIMD instructions.
 *
 * Based on: https://github.com/sutantodadang/assembly-simd
 * License: MIT
 */

#ifndef SIMD_MATMUL_H
#define SIMD_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CPU FEATURE DETECTION
 *============================================================================*/

/**
 * @brief Check if CPU supports AVX2 and FMA instructions
 * @return Non-zero if AVX2+FMA is supported, zero otherwise
 */
int simd_check_avx2(void);

/**
 * @brief Check if CPU supports AVX-512F instructions
 * @return Non-zero if AVX-512F is supported, zero otherwise
 */
int simd_check_avx512(void);

/**
 * @brief Check if CPU supports F16C (FP16 conversion) instructions
 * @return Non-zero if F16C is supported, zero otherwise
 */
int simd_check_f16c(void);

/**
 * @brief Check if CPU supports AVX-512 FP16 instructions (native FP16 compute)
 * @return Non-zero if AVX-512 FP16 is supported, zero otherwise
 */
int simd_check_avx512_fp16(void);

/**
 * @brief Check if CPU supports ARM NEON instructions
 * @return Non-zero if NEON is supported (always true on AArch64), zero otherwise
 */
int simd_check_neon(void);

/*============================================================================
 * MATRIX MULTIPLICATION FUNCTIONS
 *============================================================================*/

/**
 * @brief Matrix multiplication using AVX2 instructions
 *
 * Computes C = A × B for single-precision float matrices.
 *
 * @param A Pointer to matrix A (M × K), row-major, 32-byte aligned recommended
 * @param B Pointer to matrix B (K × N), row-major, 32-byte aligned recommended
 * @param C Pointer to result matrix C (M × N), row-major, 32-byte aligned
 * recommended
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 */
void matrix_mult_avx2(const float *A, const float *B, float *C, size_t M,
                      size_t N, size_t K);

/**
 * @brief Matrix multiplication using AVX-512 instructions
 *
 * Computes C = A × B for single-precision float matrices.
 * Requires 64-byte alignment for optimal performance.
 *
 * @param A Pointer to matrix A (M × K), row-major, 64-byte aligned recommended
 * @param B Pointer to matrix B (K × N), row-major, 64-byte aligned recommended
 * @param C Pointer to result matrix C (M × N), row-major, 64-byte aligned
 * recommended
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 */
void matrix_mult_avx512(const float *A, const float *B, float *C, size_t M,
                        size_t N, size_t K);

/**
 * @brief Matrix multiplication using ARM NEON instructions
 *
 * Computes C = A × B for single-precision float matrices.
 * Requires AArch64 with NEON support (always available).
 *
 * @param A Pointer to matrix A (M × K), row-major, 16-byte aligned recommended
 * @param B Pointer to matrix B (K × N), row-major, 16-byte aligned recommended
 * @param C Pointer to result matrix C (M × N), row-major, 16-byte aligned
 * recommended
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 */
void matrix_mult_neon(const float *A, const float *B, float *C, size_t M,
                      size_t N, size_t K);

/**
 * @brief Automatic dispatch matrix multiplication
 *
 * Automatically selects the best SIMD implementation based on CPU features.
 * Falls back to AVX2 if AVX-512 is not available.
 *
 * @param A Pointer to matrix A (M × K), row-major
 * @param B Pointer to matrix B (K × N), row-major
 * @param C Pointer to result matrix C (M × N), row-major
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 */
void simd_matmul_f32(const float *A, const float *B, float *C, size_t M,
                     size_t N, size_t K, int ith, int nth);

/**
 * @brief Automatic dispatch matrix multiplication for FP16 (half-precision)
 *
 * Automatically selects the best SIMD implementation:
 * - AVX-512 FP16 (native) on Sapphire Rapids+ CPUs
 * - F16C conversion + AVX2/AVX-512 compute on older CPUs
 *
 * @param A Pointer to matrix A (M × K), row-major, FP16 format (uint16_t)
 * @param B Pointer to matrix B (K × N), row-major, FP16 format (uint16_t)
 * @param C Pointer to result matrix C (M × N), row-major, FP16 format
 * (uint16_t)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 */
void simd_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     size_t M, size_t N, size_t K, int ith, int nth);


/*============================================================================
 * QUANTIZED DOT PRODUCT FUNCTIONS
 *============================================================================*/

/**
 * @brief Q4_0 × Q8_0 dot product using AVX2
 *
 * Computes the dot product between Q4_0 and Q8_0 quantized vectors.
 * Used for quantized matrix multiplication (weights × activations).
 *
 * @param n Number of elements (must be multiple of 32)
 * @param result Pointer to store scalar result
 * @param vx Pointer to Q4_0 quantized blocks
 * @param vy Pointer to Q8_0 quantized blocks
 */
void simd_vec_dot_q4_0_q8_0_avx2(int n, float *result, const void *vx,
                                 const void *vy);

/**
 * @brief Q8_0 × Q8_0 dot product using AVX2
 *
 * Computes the dot product between two Q8_0 quantized vectors.
 * Used for quantized matrix multiplication (activations × activations).
 *
 * @param n Number of elements (must be multiple of 32)
 * @param result Pointer to store scalar result
 * @param vx Pointer to first Q8_0 quantized blocks
 * @param vy Pointer to second Q8_0 quantized blocks
 */
void simd_vec_dot_q8_0_q8_0_avx2(int n, float *result, const void *vx,
                                 const void *vy);

/**
 * @brief Q4_0 × Q8_0 dot product using AVX-512
 */
void simd_vec_dot_q4_0_q8_0_avx512(int n, float *result, const void *vx,
                                   const void *vy);

/**
 * @brief Q8_0 × Q8_0 dot product using AVX-512
 */
void simd_vec_dot_q8_0_q8_0_avx512(int n, float *result, const void *vx,
                                   const void *vy);

/**
 * @brief Q2_K × Q8_K dot product using AVX2
 */
void simd_vec_dot_q2_k_q8_k_avx2(int n, float *result, const void *vx,
                                 const void *vy);

/**
 * @brief Q6_K × Q8_K dot product using AVX2
 */
void simd_vec_dot_q6_k_q8_k_avx2(int n, float *result, const void *vx,
                                 const void *vy);

/**
 * @brief Q2_K × Q8_K dot product using AVX-512
 */
void simd_vec_dot_q2_k_q8_k_avx512(int n, float *result, const void *vx,
                                   const void *vy);

/**
 * @brief Q6_K × Q8_K dot product using AVX-512
 */
void simd_vec_dot_q6_k_q8_k_avx512(int n, float *result, const void *vx,
                                    const void *vy);

/**
 * @brief Q3_K × Q8_K dot product using AVX2
 *
 * Computes the dot product between Q3_K (3-bit quantized weights) and Q8_K
 * (8-bit quantized activations).
 *
 * Q3_K block structure: 126 bytes for 256 elements
 *   - d: 2 bytes (fp16 super-block scale)
 *   - scales: 12 bytes (6-bit scales)
 *   - hmask: 16 bytes (sign mask)
 *   - qs: 96 bytes (3-bit weights packed)
 *
 * @param n Number of elements (must be multiple of 256)
 * @param result Pointer to store scalar result
 * @param vx Pointer to Q3_K quantized blocks
 * @param vy Pointer to Q8_K quantized blocks
 */
void simd_vec_dot_q3_k_q8_k_avx2(int n, float *result, const void *vx,
                                  const void *vy);

/**
 * @brief Q3_K × Q8_K dot product using AVX-512
 *
 * AVX-512 version with 512-bit vectors for 2x throughput over AVX2.
 *
 * @param n Number of elements (must be multiple of 256)
 * @param result Pointer to store scalar result
 * @param vx Pointer to Q3_K quantized blocks
 * @param vy Pointer to Q8_K quantized blocks
 */
void simd_vec_dot_q3_k_q8_k_avx512(int n, float *result, const void *vx,
                                    const void *vy);

/**
 * @brief Q4_0 × Q8_0 1x8 GEMM microkernel using AVX-512

 *
 * Computes 1 row of activations x 8 rows of weights.
 *
 * @param K Inner dimension
 * @param dst Destination pointer (f32, 8 elements)
 * @param src0 Weights (Q4_0), pointer to first row
 * @param src1 Activations (Q8_0), pointer to row
 * @param src0_row_stride Stride of src0 in bytes
 */
void simd_gemm_q4_0_q8_0_avx512_1x8(int K, float *dst, const void *src0,
                                    const void *src1, size_t src0_row_stride);

/*============================================================================
 * ARM NEON QUANTIZED DOT PRODUCT FUNCTIONS (AArch64)
 *============================================================================*/

/** Q4_0 × Q8_0 dot product using NEON */
void simd_vec_dot_q4_0_q8_0_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q8_0 × Q8_0 dot product using NEON */
void simd_vec_dot_q8_0_q8_0_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q2_K × Q8_K dot product using NEON */
void simd_vec_dot_q2_k_q8_k_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q3_K × Q8_K dot product using NEON */
void simd_vec_dot_q3_k_q8_k_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q4_K × Q8_K dot product using NEON */
void simd_vec_dot_q4_k_q8_k_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q6_K × Q8_K dot product using NEON */
void simd_vec_dot_q6_k_q8_k_neon(int n, float *result, const void *vx,
                                 const void *vy);

/** Q8_K × Q8_K dot product using NEON */
void simd_vec_dot_q8_k_q8_k_neon(int n, float *result, const void *vx,
                                 const void *vy);

/*============================================================================
 * ALIGNMENT HELPERS
 *============================================================================*/

#define SIMD_ALIGN_AVX2 32
#define SIMD_ALIGN_AVX512 64
#define SIMD_ALIGN_NEON 16

/**
 * @brief Check if pointer is properly aligned for AVX2
 */
static inline int simd_is_aligned_avx2(const void *ptr) {
  return ((uintptr_t)ptr & (SIMD_ALIGN_AVX2 - 1)) == 0;
}

/**
 * @brief Check if pointer is properly aligned for AVX-512
 */
static inline int simd_is_aligned_avx512(const void *ptr) {
  return ((uintptr_t)ptr & (SIMD_ALIGN_AVX512 - 1)) == 0;
}

#ifdef __cplusplus
}
#endif

#endif /* SIMD_MATMUL_H */
