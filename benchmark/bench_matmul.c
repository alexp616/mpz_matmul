/*
 * benchmark/bench_matmul.c
 *
 * Benchmarks naive, CPU-FFT, and GPU-FFT integer matrix multiplication
 * across a range of bit sizes and matrix dimensions.
 *
 * Matrix dimensions : 2x2, 5x5, 10x10
 * Bit sizes         : 50 000, 100 000, 200 000
 *
 * Compile the library with -DBENCH_PROFILE to enable per-step timing
 * printouts that are emitted directly from inside mpz_rmatrix_mult_fft()
 * and cu_mpz_rmatrix_mult_fft().  This file only measures and prints
 * the total wall-clock time for each call.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmp.h>

#include "hwmpz.h"
#include "hwmem.h"
#include "cufft62_thresholds.h"

/* Simple wall-clock timer (local to this file) */
static double wall_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* -- Benchmark parameters -------------------------------------------------- */

// static const long BIT_SIZES[] = { 50000, 100000, 500000, 1000000, 10000000, 100000000 };
static const long BIT_SIZES[] = { 10000000, 100000000 };
static const int  DIM_SIZES[] = { 2, 5 };

#define N_BITS ((int)(sizeof(BIT_SIZES) / sizeof(BIT_SIZES[0])))
#define N_DIMS ((int)(sizeof(DIM_SIZES) / sizeof(DIM_SIZES[0])))

int main(void)
{
    hw_mem_init(0);

    /* One-time FFT/moduli initialisation.
     * When -DBENCH_PROFILE is set, hw_mpz_setup() itself prints the time
     * taken by zz_moduli_init (CPU) and cu_zz_moduli_init (GPU). */
    printf("============================================================\n");
    printf("  Moduli initialisation\n");
    printf("============================================================\n");
    double t_setup = wall_sec();
    hw_mpz_setup();
    printf("  hw_mpz_setup total          : %.6f s\n\n", wall_sec() - t_setup);

    /* Reproducible random seed */
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, 42UL);

    /* ====================================================================
       Main loop
       ==================================================================== */
    for (int bi = 0; bi < N_BITS; bi++) {
        long bits = BIT_SIZES[bi];

        for (int di = 0; di < N_DIMS; di++) {
            int d = DIM_SIZES[di];

            printf("============================================================\n");
            printf("  bits = %ld,  %d x %d  *  %d x %d\n", bits, d, d, d, d);
            printf("============================================================\n");

            mpz_t *A     = mpz_vec_alloc_and_init(d * d);
            mpz_t *B     = mpz_vec_alloc_and_init(d * d);
            mpz_t *C_ref = mpz_vec_alloc_and_init(d * d);
            mpz_t *C_cpu = mpz_vec_alloc_and_init(d * d);
            mpz_t *C_gpu = mpz_vec_alloc_and_init(d * d);

            for (int i = 0; i < d*d; i++) mpz_urandomb(A[i], rng, bits);
            for (int i = 0; i < d*d; i++) mpz_urandomb(B[i], rng, bits);

            mpz_t w;
            mpz_init(w);

            /* ---- naive (reference) ---- */
            double t0 = wall_sec();
            mpz_rmatrix_mult_naive(C_ref, A, d, B, d, w);
            printf("  Naive            total: %.6f s\n", wall_sec() - t0);
            printf("\n");

            /* ---- CPU FFT ----
             * When -DBENCH_PROFILE is active, per-step times are printed
             * from inside mpz_rmatrix_mult_fft(). */
            t0 = wall_sec();
            mpz_rmatrix_mult_fft(C_cpu, A, d, B, d, w);
            printf("  CPU FFT          total: %.6f s\n", wall_sec() - t0);
            printf("\n");

            int ok = 1;
            for (int i = 0; i < d*d; i++)
                if (mpz_cmp(C_cpu[i], C_ref[i]) != 0) { ok = 0; break; }
            if (!ok)
                printf("  ** CPU FFT CORRECTNESS MISMATCH **\n");

            /* ---- GPU FFT ----
             * When -DBENCH_PROFILE is active, per-step times are printed
             * from inside cu_mpz_rmatrix_mult_fft(). */
            if (!mpz_rmatrix_gpu_fft_suitable(A, d, B, d)) {
                printf("  GPU FFT          skipped (lgN outside [%d,%d])\n",
                       GPU_MIN_THRESHOLD, GPU_MAX_THRESHOLD);
            } else {
                t0 = wall_sec();
                cu_mpz_rmatrix_mult_fft(C_gpu, A, d, B, d, w);
                printf("  GPU FFT          total: %.6f s\n", wall_sec() - t0);

                ok = 1;
                for (int i = 0; i < d*d; i++)
                    if (mpz_cmp(C_gpu[i], C_ref[i]) != 0) { ok = 0; break; }
                if (!ok)
                    printf("  ** GPU FFT CORRECTNESS MISMATCH **\n");
            }

            printf("\n");

            mpz_clear(w);
            mpz_vec_clear_and_free(A,     d * d);
            mpz_vec_clear_and_free(B,     d * d);
            mpz_vec_clear_and_free(C_ref, d * d);
            mpz_vec_clear_and_free(C_cpu, d * d);
            mpz_vec_clear_and_free(C_gpu, d * d);
        }
    }

    hw_mpz_clear();
    hw_mem_clear();
    gmp_randclear(rng);
    return 0;
}
