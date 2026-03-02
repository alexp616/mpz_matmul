/*
  GPU polynomial recomposition (zz_recompose) — chunked CUDA implementation.

  Reconstructs a big integer from N signed coefficients (each num_primes limbs)
  by evaluating  sum_{i=0}^{N-1} coeff[i] * 2^{i*r}.

  Input layout matches cu_zz_crt_batch output:
    d_crt[j * num_primes + k] is limb k of coefficient j.
  Output:
    d_out has rn limbs (uint64_t), holds the 2's-complement result.

  This chunked version mirrors crt_recompose.c: each chunk handles a window of
  coefficients and produces a partial sum plus a spill buffer. A second pass
  propagates spill carries across chunk boundaries. The chunk-internal work is
  serial inside the block; blocks run in parallel across chunks.
*/

#ifndef CURECOMPOSE_CUH
#define CURECOMPOSE_CUH

#include <stdint.h>
#include <stddef.h>
#include "mpzfft_moduli.h"  /* for ZZ_MAX_PRIMES */

#ifdef __cplusplus
extern "C" {
#endif

/* Launch chunked GPU recomposition.
 *
 * d_out    : device pointer to rn limbs (uint64_t) for the result. Must be zeroed.
 * rn       : number of limbs in output (host computed).
 * r        : bit spacing between coefficients (host computed params->r).
 * d_crt    : device pointer to CRT coefficients (N * num_primes limbs).
 * N        : number of coefficients.
 * num_primes : limbs per coefficient (<= ZZ_MAX_PRIMES).
 */
void cu_zz_recompose_chunked(uint64_t* d_out,
                             size_t rn,
                             unsigned r,
                             const uint64_t* d_crt,
                             size_t N,
                             unsigned num_primes);

/* Batched version: recompose `entries` matrix entries in parallel.
 * d_out_all : [entries * rn] device output.
 * d_crt_all : [entries * N * num_primes] CRT coefficients.
 */
void cu_zz_recompose_batch(uint64_t* d_out_all,
                           size_t rn,
                           unsigned r,
                           const uint64_t* d_crt_all,
                           size_t N,
                           unsigned num_primes,
                           int entries);

#ifdef __cplusplus
}
#endif

#endif /* CURECOMPOSE_CUH */
