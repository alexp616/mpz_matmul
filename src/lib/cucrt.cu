/*
  GPU CRT (Chinese Remainder Theorem) — Garner's algorithm on CUDA.

  This is a faithful translation of zz_crt() (crt.c) to a CUDA kernel.
  One thread handles one coefficient; all coefficients are independent.

  See cucrt.cuh for the public interface.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cucrt.cuh"

// direct translations of modular arithmetic primitives in crt.c

__device__ static inline void mul_wide(uint64_t &hi, uint64_t &lo,
                                           uint64_t a, uint64_t b)
{
    lo = a * b;
    hi = __umul64hi(a, b);
}

__device__ static inline uint64_t mul_hi(uint64_t a, uint64_t b)
{
    return __umul64hi(a, b);
}

__device__ static inline uint64_t mod62_reduce4(uint64_t x, uint64_t m)
{
    uint64_t t = x - ((x >= 2 * m) ? (2 * m) : 0);
    return t - ((t >= m) ? m : 0);
}

__device__ static inline uint64_t mod62_mul_ypinv(uint64_t x, uint64_t y,
                                                      uint64_t ypinv, uint64_t p)
{
    uint64_t q = mul_hi(x, ypinv);
    uint64_t r = x * y - q * p;
    return r - ((r >= p) ? p : 0);
}

__device__ static inline uint64_t mpn_addmul_1(uint64_t *rp,
                                                   const uint64_t *up,
                                                   unsigned n, uint64_t s)
{
    uint64_t cy = 0;
    for (unsigned k = 0; k < n; k++) {
        uint64_t lo, hi;
        mul_wide(hi, lo, up[k], s);
        lo += cy;
        hi += (lo < cy);
        uint64_t prev = rp[k];
        lo += prev;
        hi += (lo < prev);
        rp[k] = lo;
        cy = hi;
    }
    return cy;
}

/*  Lexicographic compare of {a,n} vs {b,n} (most significant limb first).
    Returns  1 if a > b,  0 if equal,  -1 if a < b.                           */
__device__ static inline int mpn_cmp(const uint64_t *a,
                                         const uint64_t *b,
                                         unsigned n)
{
    for (int i = (int)n - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

/*  rp = ap - bp  (n limbs).  Returns borrow (0 or 1).                        */
__device__ static inline uint64_t mpn_sub_n(uint64_t *rp,
                                                const uint64_t *ap,
                                                const uint64_t *bp,
                                                unsigned n)
{
    uint64_t borrow = 0;
    for (unsigned k = 0; k < n; k++) {
        uint64_t a = ap[k], b = bp[k];
        uint64_t diff = a - b - borrow;
        borrow = (a < b + borrow) || (borrow && b == UINT64_MAX);
        rp[k] = diff;
    }
    return borrow;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   CRT kernel — one thread per coefficient
   ═══════════════════════════════════════════════════════════════════════════════ */

__global__ void cu_zz_crt_kernel(uint64_t       *rp,
                                 const uint64_t *d_data,
                                 size_t          datasz,
                                 cu_crt_constants_t ctx)
{
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= datasz)
        return;

    const unsigned np = ctx.num_primes;

    /* Register-local accumulator for this coefficient's CRT result. */
    uint64_t dest[ZZ_MAX_PRIMES];

    /* ── Special case: single prime ──────────────────────────────────────── */
    if (np == 1) {
        uint64_t p0 = ctx.p[0];
        uint64_t x  = mod62_reduce4(d_data[j], p0);
        dest[0] = x - ((x > p0 / 2) ? p0 : 0);
        rp[j] = dest[0];
        return;
    }

    /* ── First pass: CRT from first two primes ──────────────────────────── */
    {
        uint64_t p0    = ctx.p[0];
        uint64_t p1    = ctx.p[1];
        uint64_t s_1   = ctx.s[1];
        uint64_t sp1   = ctx.spinv[1];

        uint64_t x = mod62_reduce4(d_data[0 * datasz + j], p0);
        uint64_t y = d_data[1 * datasz + j];                  /* [0, 4·p1) */
        uint64_t r = x - y + ((x < y) ? (4 * p1) : 0);
        r = mod62_mul_ypinv(r, s_1, sp1, p1);

        uint64_t s1, s0;
        mul_wide(s1, s0, r, p0);
        s0 += x;
        dest[0] = s0;
        dest[1] = s1 + (s0 < x);
    }

    /* ── Remaining primes (i = 2 … np-1) ────────────────────────────────── */
    for (unsigned i = 2; i < np; i++) {
        uint64_t p     = ctx.p[i];
        uint64_t pinvb = ctx.pinvb[i];
        uint64_t s_i   = ctx.s[i];
        uint64_t sp_i  = ctx.spinv[i];

        uint64_t x = dest[0];
        uint64_t y = d_data[i * datasz + j];                  /* [0, 4·p) */
        uint64_t r = x - y + ((x < y) ? (4 * p) : 0);

        for (unsigned k = 1; k < i; k++) {
            uint64_t h  = mul_hi(r * pinvb, p);
            uint64_t xk = dest[k];
            r = xk - h + ((xk < h) ? p : 0);
        }

        r = mod62_mul_ypinv(r, s_i, sp_i, p);
        dest[i] = mpn_addmul_1(dest, ctx.u[i], i, r);
    }

    /* ── Signed reduction: if value > P/2, subtract P ────────────────────── */
    if (mpn_cmp(dest, ctx.uhalf[np], np) > 0)
        mpn_sub_n(dest, dest, ctx.u[np], np);

    /* ── Write output (interleaved: np limbs per coefficient) ────────────── */
    uint64_t *out = rp + (size_t)j * np;
    for (unsigned k = 0; k < np; k++)
        out[k] = dest[k];
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Host-side functions
   ═══════════════════════════════════════════════════════════════════════════════ */

/* Populate cu_crt_constants_t from CPU-side zz_moduli_t.
   Pure host code — no CUDA calls.                                              */
void cu_crt_constants_init(cu_crt_constants_t *ctx,
                           const zz_moduli_t  *moduli,
                           unsigned            num_primes)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->num_primes = num_primes;

    for (unsigned i = 0; i < num_primes; i++) {
        ctx->p[i]     = moduli->p[i];
        ctx->pinvb[i] = moduli->pinvb[i];
        ctx->s[i]     = moduli->s[i];
        ctx->spinv[i] = moduli->spinv[i];
    }
    for (unsigned i = 0; i <= num_primes; i++)
        for (unsigned k = 0; k < num_primes; k++) {
            ctx->u[i][k]     = moduli->u[i][k];
            ctx->uhalf[i][k] = moduli->uhalf[i][k];
        }
}

/* Allocate + initialise (for opaque usage from C code). */
cu_crt_constants_t* cu_crt_constants_create(const zz_moduli_t *moduli,
                                            unsigned num_primes)
{
    cu_crt_constants_t *ctx =
        (cu_crt_constants_t*)malloc(sizeof(cu_crt_constants_t));
    cu_crt_constants_init(ctx, moduli, num_primes);
    return ctx;
}

void cu_crt_constants_destroy(cu_crt_constants_t *ctx)
{
    free(ctx);
}

/* Launch CRT kernel for datasz coefficients.
   Blocks until the kernel completes (synchronous).                             */
void cu_zz_crt_batch(uint64_t             *d_out,
                     const uint64_t       *d_data,
                     size_t                datasz,
                     const cu_crt_constants_t *h_ctx)
{
    if (datasz == 0)
        return;

    const int threads = 256;
    const int blocks  = (int)((datasz + threads - 1) / threads);

    cu_zz_crt_kernel<<<blocks, threads>>>(d_out, d_data, datasz, *h_ctx);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cu_zz_crt_batch: CUDA error: %s\n",
                cudaGetErrorString(err));
        exit(1);
    }
}
