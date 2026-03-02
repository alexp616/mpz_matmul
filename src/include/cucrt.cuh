/*
  GPU CRT (Chinese Remainder Theorem) — Garner's algorithm on CUDA.

  Translates zz_crt() from crt.c into a massively-parallel GPU kernel.
  Each CUDA thread reconstructs one coefficient from its residues modulo
  num_primes primes, producing num_primes mp_limb_t output limbs in
  signed 2's-complement form (same semantics as the CPU version).

  The kernel is fully parameterised by num_primes (up to ZZ_MAX_PRIMES).
*/

#ifndef CUCRT_CUH
#define CUCRT_CUH

#include <stdint.h>
#include <stddef.h>
#include "mpzfft_moduli.h"   /* ZZ_MAX_PRIMES, zz_moduli_t */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned num_primes;

    uint64_t p[ZZ_MAX_PRIMES];
    uint64_t pinvb[ZZ_MAX_PRIMES];

    uint64_t s[ZZ_MAX_PRIMES];
    uint64_t spinv[ZZ_MAX_PRIMES];

    uint64_t u    [ZZ_MAX_PRIMES + 1][ZZ_MAX_PRIMES];
    uint64_t uhalf[ZZ_MAX_PRIMES + 1][ZZ_MAX_PRIMES];
} cu_crt_constants_t;

/* Fill *ctx from CPU-side moduli (host-only, no CUDA calls). */
void cu_crt_constants_init(cu_crt_constants_t* ctx,
                           const zz_moduli_t* moduli,
                           unsigned num_primes);

/* Allocate + initialise (for opaque usage from plain C code). */
cu_crt_constants_t* cu_crt_constants_create(const zz_moduli_t* moduli,
                                            unsigned num_primes);

/* Free a context allocated by cu_crt_constants_create(). */
void cu_crt_constants_destroy(cu_crt_constants_t* ctx);

/* ── Batch CRT kernel launcher ─────────────────────────────────────────────── */
/*
   d_out   – device output buffer, datasz * num_primes uint64_t's.
             Layout: coefficient j occupies d_out[j*num_primes .. j*num_primes+num_primes-1].
             Each coefficient is in signed 2's-complement (same as zz_crt).
   d_data  – device input (e.g. IFFT output), num_primes * datasz uint64_t's.
             Layout: prime i's values sit at d_data[i*datasz .. (i+1)*datasz-1].
             Values are in [0, 4p) as produced by the inverse NTT.
   datasz  – number of coefficients per prime (= total coefficients to CRT).
   h_ctx   – host pointer to pre-initialised constants (copied to kernel args).
*/
void cu_zz_crt_batch(uint64_t*       d_out,
                     const uint64_t* d_data,
                     size_t          datasz,
                     const cu_crt_constants_t* h_ctx);

#ifdef __cplusplus
}
#endif

#endif /* CUCRT_CUH */
