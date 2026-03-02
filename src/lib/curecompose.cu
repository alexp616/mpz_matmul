#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "curecompose.cuh"

static inline void cuda_check_recompose() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "curecompose: CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Direct translation of zz_recompose (recompose.c).
__global__ void recompose_serial_kernel(uint64_t*       d_out_all,
                                        size_t          rn,
                                        unsigned        r,
                                        const uint64_t* d_crt_all,
                                        size_t          N,
                                        unsigned        s,      /* num_primes */
                                        int             entries)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= entries) return;

    uint64_t* rp       = d_out_all  + (size_t)eid * rn;
    const uint64_t* up = d_crt_all  + (size_t)eid * N * s;

    size_t u = r / 64;

    ptrdiff_t dst = 0;
    unsigned  dst_bit = 0;
    uint64_t  rp_cy = 0;

    uint64_t temp[ZZ_MAX_PRIMES + 1];

    size_t src = 0;
    while (src < N && dst < (ptrdiff_t)(rn - s - 1))
    {
        const uint64_t* coeff = up + src * s;
        uint64_t up_cy = coeff[s - 1] >> 63;

        if (dst_bit != 0) {
            uint64_t carry = 0;
            for (unsigned k = 0; k < s; k++) {
                uint64_t v = coeff[k];
                temp[k] = (v << dst_bit) | carry;
                carry   = v >> (64 - dst_bit);
            }
            temp[s] = carry + (((uint64_t)(-(int64_t)up_cy)) << dst_bit);
        } else {
            for (unsigned k = 0; k < s; k++) temp[k] = coeff[k];
            temp[s] = (uint64_t)(-(int64_t)up_cy);
        }

        uint64_t cy = 0;
        for (unsigned k = 0; k < s - u + 1; k++) {
            uint64_t a   = rp[dst + k];
            uint64_t b   = temp[k];
            uint64_t sum = a + b + cy;
            cy = (sum < a) || (cy && sum == a);
            rp[dst + k]  = sum;
        }

        if (u > 0) {
            for (unsigned k = 0; k < u; k++) {
                uint64_t sum = temp[s - u + 1 + k] + cy;
                cy = (sum < cy) ? 1ULL : 0ULL;
                rp[dst + s - u + 1 + k] = sum;
            }
            uint64_t borrow = rp_cy;
            for (unsigned k = 0; k < u; k++) {
                uint64_t val  = rp[dst + s - u + 1 + k];
                uint64_t diff = val - borrow;
                borrow = (diff > val) ? 1ULL : 0ULL;
                rp[dst + s - u + 1 + k] = diff;
            }
            cy -= borrow;
        } else {
            cy -= rp_cy;
        }
        cy -= up_cy;
        rp[dst + s + 1] = cy;
        rp_cy = (uint64_t)(-(int64_t)cy);

        dst_bit += r;
        dst += dst_bit / 64;
        dst_bit &= 63;
        src++;
    }

    while (src < N && dst < (ptrdiff_t)rn)
    {
        const uint64_t* coeff = up + src * s;
        uint64_t up_cy = coeff[s - 1] >> 63;

        if (dst_bit != 0) {
            uint64_t carry = 0;
            for (unsigned k = 0; k < s; k++) {
                uint64_t v = coeff[k];
                temp[k] = (v << dst_bit) | carry;
                carry   = v >> (64 - dst_bit);
            }
            temp[s] = carry + (((uint64_t)(-(int64_t)up_cy)) << dst_bit);
        } else {
            for (unsigned k = 0; k < s; k++) temp[k] = coeff[k];
            temp[s] = (uint64_t)(-(int64_t)up_cy);
        }

        uint64_t cy = 0;
        {
            size_t n_add = s - u + 1;
            if ((size_t)dst + n_add > rn) n_add = rn - (size_t)dst;
            for (size_t k = 0; k < n_add; k++) {
                uint64_t a   = rp[dst + k];
                uint64_t b   = temp[k];
                uint64_t sum = a + b + cy;
                cy = (sum < a) || (cy && sum == a);
                rp[dst + k]  = sum;
            }
        }

        if (u > 0 && (size_t)(dst + s - u + 1) < rn) {
            size_t kmax = u;
            if ((size_t)(dst + s - u + 1) + kmax > rn)
                kmax = rn - (size_t)(dst + s - u + 1);
            for (size_t k = 0; k < kmax; k++) {
                uint64_t sum = temp[s - u + 1 + k] + cy;
                cy = (sum < cy) ? 1ULL : 0ULL;
                rp[dst + s - u + 1 + k] = sum;
            }
            uint64_t borrow = rp_cy;
            for (size_t k = 0; k < kmax; k++) {
                uint64_t val  = rp[dst + s - u + 1 + k];
                uint64_t diff = val - borrow;
                borrow = (diff > val) ? 1ULL : 0ULL;
                rp[dst + s - u + 1 + k] = diff;
            }
            cy -= borrow;
        } else {
            cy -= rp_cy;
        }
        cy -= up_cy;
        rp_cy = (uint64_t)(-(int64_t)cy);

        dst_bit += r;
        dst += dst_bit / 64;
        dst_bit &= 63;
        src++;
    }

    /* propagate final borrow to rest of output */
    uint64_t fill = (uint64_t)(-(int64_t)rp_cy);
    for (ptrdiff_t i = dst + (ptrdiff_t)s - (ptrdiff_t)u + 1;
         i < (ptrdiff_t)rn; i++) {
        if (i >= 0) rp[i] = fill;
    }
}

/* ── host wrappers ─────────────────────────────────────────────────────────── */

void cu_zz_recompose_chunked(uint64_t*       d_out,
                             size_t          rn,
                             unsigned        r,
                             const uint64_t* d_crt,
                             size_t          N,
                             unsigned        num_primes)
{
    if (N == 0 || num_primes == 0) return;

    cudaMemset(d_out, 0, rn * sizeof(uint64_t));
    cuda_check_recompose();

    recompose_serial_kernel<<<1, 1>>>(d_out, rn, r, d_crt, N, num_primes, 1);
    cuda_check_recompose();
}

void cu_zz_recompose_batch(uint64_t*       d_out_all,
                           size_t          rn,
                           unsigned        r,
                           const uint64_t* d_crt_all,
                           size_t          N,
                           unsigned        num_primes,
                           int             entries)
{
    if (N == 0 || num_primes == 0 || entries == 0) return;

    cudaMemset(d_out_all, 0, (size_t)entries * rn * sizeof(uint64_t));
    cuda_check_recompose();

    int threads = 256;
    int blocks  = (entries + threads - 1) / threads;
    recompose_serial_kernel<<<blocks, threads>>>(d_out_all, rn, r,
                                                 d_crt_all, N,
                                                 num_primes, entries);
    cuda_check_recompose();
}
