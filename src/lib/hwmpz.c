#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>
#include "mpzfft.h"
#include "recompose.h"
#include "zzmem.h"
#include "hwmem.h"
#include "hwmpz.h"
#include "cufft62_thresholds.h"

#define HW_ZZ_PRIMES            4

int hw_disable_fft;
int mpzfft_threads = 1;

static zz_moduli_t zz_moduli;
extern cu_zz_moduli_t* cu_zz_moduli;
extern cu_zz_moduli_t* create_cu_zz_moduli_t();
static cu_crt_constants_t* cu_crt_ctx;
static int mpzfft_initialized;

static inline void hw_mpzfft_setup ()
{
    if ( hw_disable_fft || mpzfft_initialized ) return;
    _BENCH_TICK(t_cpu_init);
    zz_moduli_init (&zz_moduli, ZZ_MAX_PRIMES);
    _BENCH_PRINT("zz_moduli_init (CPU)", _BENCH_TOCK(t_cpu_init));
    _BENCH_TICK(t_gpu_init);
    cu_zz_moduli = create_cu_zz_moduli_t();
    cu_zz_moduli_init (cu_zz_moduli, ZZ_MAX_PRIMES);
    _BENCH_PRINT("cu_zz_moduli_init (GPU)", _BENCH_TOCK(t_gpu_init));
    cu_crt_ctx = cu_crt_constants_create(&zz_moduli, HW_ZZ_PRIMES);
#if HW_MEM_TRACKING
    extern unsigned long zz_overhead;
    zz_overhead = zz_mem_peak;
#endif
    mpzfft_initialized = 1;
}

static inline void hw_mpzfft_clear ()
{
    if ( hw_disable_fft || ! mpzfft_initialized ) return;
    zz_moduli_clear (&zz_moduli);
    cu_zz_moduli_clear (cu_zz_moduli);
    if (cu_crt_ctx) { cu_crt_constants_destroy(cu_crt_ctx); cu_crt_ctx = NULL; }
    mpzfft_initialized = 0;
}

void hw_mpz_setup (void) { hw_mpzfft_setup ();}
void hw_mpz_clear (void) { hw_mpzfft_clear (); }

static inline size_t mpz_rmatrix_product_max_bits (mpz_t *A, int r, mpz_t *B, int d)
{
    register int i;
    register size_t m, n;
    
    m = 0;
    for ( i = 0 ; i < r*d ; i++ ) { n = mpz_size(A[i]); if ( n > m ) m = n; }
    for ( i = 0 ; i < d*d ; i++ ) { n = mpz_size(B[i]); if ( n > m ) m = n; }
    return GMP_NUMB_BITS*(2*m+1);
}

void mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT;
    assert ( mpzfft_initialized );

    _BENCH_TICK(t_all);

    _BENCH_TICK(t_params);
    mpzfft_params_init (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);
    #ifdef BENCH_PROFILE
        printf("  [profile] FFT lgN: %d\n", params.lgN);
    #endif
    _BENCH_PRINT("[CPU FFT] params init", _BENCH_TOCK(t_params));

    AT = hw_malloc (r*d*sizeof(mpzfft_t));
    BT = hw_malloc (d*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < r*d; i++) zz_mpnfft_poly_init(AT[i], &params);
    for ( int i = 0 ; i < d*d; i++) zz_mpnfft_poly_init(BT[i], &params);

    _BENCH_TICK(t_int2poly);
    for ( int i = 0 ; i < r*d; i++)
        zz_mpnfft_mpn_to_poly(AT[i], A[i]->_mp_d, mpz_size(A[i]), mpz_sgn(A[i]), 0, 0, mpzfft_threads);
    for ( int i = 0 ; i < d*d; i++)
        zz_mpnfft_mpn_to_poly(BT[i], B[i]->_mp_d, mpz_size(B[i]), mpz_sgn(B[i]), 0, 0, mpzfft_threads);
    _BENCH_PRINT("[CPU FFT] int -> poly", _BENCH_TOCK(t_int2poly));

    _BENCH_TICK(t_fwd);
    for ( int i = 0 ; i < r*d; i++) zz_mpnfft_poly_fft(AT[i], AT[i], mpzfft_threads);
    for ( int i = 0 ; i < d*d; i++) zz_mpnfft_poly_fft(BT[i], BT[i], mpzfft_threads);
    _BENCH_PRINT("[CPU FFT] forward FFT", _BENCH_TOCK(t_fwd));

    _BENCH_TICK(t_mm);
    zz_mpnfft_poly_matrix_mul(AT, AT, BT, (unsigned)r, (unsigned)d, (unsigned)d, mpzfft_threads);
    _BENCH_PRINT("[CPU FFT] freq matmul", _BENCH_TOCK(t_mm));

    for ( int i = 0 ; i < d*d ; i++) mpzfft_clear (BT[i]);
    hw_free (BT, d*d*sizeof(mpzfft_t));

    _BENCH_TICK(t_inv);
    for ( int i = 0 ; i < r*d ; i++) zz_mpnfft_poly_ifft(AT[i], AT[i], 1, mpzfft_threads);
    _BENCH_PRINT("[CPU FFT] inverse FFT", _BENCH_TOCK(t_inv));

    _BENCH_TICK(t_crt);
    for ( int i = 0 ; i < r*d ; i++) {
        if (AT[i]->size == 0) {
            mpz_set_ui(C[i], 0);
        } else {
            size_t n = ((AT[i]->size - 1) * params.r + 62 * params.num_primes + 2) / 64 + 1;
            mpz_realloc(C[i], n);
            zz_mpnfft_poly_to_mpn(C[i]->_mp_d, n, AT[i], mpzfft_threads);
            int neg = 0;
            if (C[i]->_mp_d[n - 1] >> 63) {
                mpn_neg(C[i]->_mp_d, C[i]->_mp_d, n);
                neg = 1;
            }
            while (n > 0 && C[i]->_mp_d[n - 1] == 0) n--;
            C[i]->_mp_size = neg ? -(int)n : (int)n;
        }
        mpzfft_clear (AT[i]);
    }
    _BENCH_PRINT("[CPU FFT] poly -> int (CRT)", _BENCH_TOCK(t_crt));

    hw_free (AT, r*d*sizeof(mpzfft_t));
    mpzfft_params_clear (&params);
}

void cu_mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    assert ( mpzfft_initialized );

    _BENCH_TICK(t_all);

    _BENCH_TICK(t_params);
    cu_mpzfft_params_init (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);
    _BENCH_PRINT("[GPU FFT] params init", _BENCH_TOCK(t_params));

    if (params.lgN < GPU_MIN_THRESHOLD || params.lgN > GPU_MAX_THRESHOLD) {
        fprintf(stderr, "cu_mpz_rmatrix_mult_fft: lgN = %u is outside GPU range [%d, %d]\n",
                params.lgN, GPU_MIN_THRESHOLD, GPU_MAX_THRESHOLD);
        exit(1);
    }

    size_t Asz_0 = r * d * params.N;
    size_t Bsz_0 = d * d * params.N;

    uint64_t* Aptr = (uint64_t*)cu_malloc_pinned(Asz_0 * params.num_primes * sizeof(uint64_t));
    uint64_t* Bptr = (uint64_t*)cu_malloc_pinned(Bsz_0 * params.num_primes * sizeof(uint64_t));

    uint64_t* Adata[HW_ZZ_PRIMES];
    uint64_t* Bdata[HW_ZZ_PRIMES];

    _BENCH_TICK(t_int2poly);
    for ( int i = 0 ; i < r*d; i++) {
        size_t start = i * params.N;
        for (int j = 0; j < HW_ZZ_PRIMES; ++j) {
            Adata[j] = Aptr + start + j * Asz_0;
        }
        cu_zz_mpnfft_mpn_to_poly(Adata, &params, A[i]->_mp_d, mpz_size(A[i]), mpz_sgn(A[i]), 0, 0, 1);
    }
    for ( int i = 0 ; i < d*d; i++) {
        size_t start = i * params.N;
        for (int j = 0; j < HW_ZZ_PRIMES; ++j) {
            Bdata[j] = Bptr + start + j * Bsz_0;
        }
        cu_zz_mpnfft_mpn_to_poly(Bdata, &params, B[i]->_mp_d, mpz_size(B[i]), mpz_sgn(B[i]), 0, 0, 1);
    }
    _BENCH_PRINT("[GPU FFT] int -> poly", _BENCH_TOCK(t_int2poly));

    for (int i = 0; i < HW_ZZ_PRIMES; ++i) {
        Adata[i] = Aptr + i * Asz_0;
        Bdata[i] = Bptr + i * Bsz_0;
    }

    uint64_t *d_Aptr, *d_Bptr;
#ifdef BENCH_PROFILE
    for (int _ri = 0; _ri < (int)params.num_primes; _ri++) {
        uint64_t  _p  = params.moduli->p[_ri];
        uint64_t *_pA = Aptr + (size_t)_ri * Asz_0;
        uint64_t *_pB = Bptr + (size_t)_ri * Bsz_0;
        for (size_t _j = 0; _j < (size_t)Asz_0; _j++) if (_pA[_j] >= _p) _pA[_j] -= _p;
        for (size_t _j = 0; _j < (size_t)Bsz_0; _j++) if (_pB[_j] >= _p) _pB[_j] -= _p;
    }
    _BENCH_TICK(th2d);
    d_Aptr = cu_h2d_batch_only(Aptr, (int)params.num_primes, (int)Asz_0);
    d_Bptr = cu_h2d_batch_only(Bptr, (int)params.num_primes, (int)Bsz_0);
    _BENCH_PRINT("[GPU FFT] H2D copy", _BENCH_TOCK(th2d));
    _BENCH_TICK(tntt);
    cu_ntt_batch_only(d_Aptr, (int)params.num_primes, params.lgN, (int)Asz_0);
    cu_ntt_batch_only(d_Bptr, (int)params.num_primes, params.lgN, (int)Bsz_0);
    _BENCH_PRINT("[GPU FFT] forward NTT", _BENCH_TOCK(tntt));
#else
    d_Aptr = cu_zz_mpnfft_poly_fft(Aptr, &params, Asz_0);
    d_Bptr = cu_zz_mpnfft_poly_fft(Bptr, &params, Bsz_0);
#endif

    cu_free_pinned(Bptr);

    _BENCH_TICK(t_mm);
    cu_mpzfft_matrix_mul(d_Aptr, d_Aptr, d_Bptr, r, d, d, HW_ZZ_PRIMES, params.N);
    _BENCH_PRINT("[GPU FFT] freq matmul", _BENCH_TOCK(t_mm));

    _BENCH_TICK(tintt);
    cu_intt_batch_only(d_Aptr, (int)params.num_primes, params.lgN, (int)Asz_0);
    _BENCH_PRINT("[GPU FFT] inverse NTT", _BENCH_TOCK(tintt));

    _BENCH_TICK(t_gpu_crt);
    uint64_t* d_crt_out = (uint64_t*)cu_device_malloc(
            (size_t)Asz_0 * params.num_primes * sizeof(uint64_t));
    cu_zz_crt_batch(d_crt_out, d_Aptr, Asz_0, cu_crt_ctx);
    cu_device_free(d_Aptr);
    _BENCH_PRINT("[GPU FFT] GPU CRT kernel", _BENCH_TOCK(t_gpu_crt));

    _BENCH_TICK(t_crt_d2h);
    cu_memcpy_d2h(Aptr, d_crt_out,
                  (size_t)Asz_0 * params.num_primes * sizeof(uint64_t));
    cu_device_free(d_crt_out);
    _BENCH_PRINT("[GPU FFT] CRT D2H copy", _BENCH_TOCK(t_crt_d2h));

    _BENCH_TICK(t_recompose);
    for (int i = 0; i < r * d; ++i) {
        size_t rn = ((size_t)params.N * params.r + 62 * params.num_primes + 2) / 64 + 1;
        mpz_realloc(C[i], rn);

        mp_limb_t* crt_data = (mp_limb_t*)(Aptr + (size_t)i * params.N * params.num_primes);

        zz_recompose(C[i]->_mp_d, rn, crt_data, params.N,
                     params.num_primes, params.r, 0);

        int neg = 0;
        if (C[i]->_mp_d[rn - 1] >> 63) {
            mpn_neg(C[i]->_mp_d, C[i]->_mp_d, rn);
            neg = 1;
        }
        while (rn > 0 && C[i]->_mp_d[rn - 1] == 0) rn--;
        C[i]->_mp_size = neg ? -(int)rn : (int)rn;
    }
    _BENCH_PRINT("[GPU FFT] CPU recompose", _BENCH_TOCK(t_recompose));

    cu_free_pinned(Aptr);
    mpzfft_params_clear (&params);
}

int mpz_rmatrix_gpu_fft_suitable(mpz_t *A, int r, mpz_t *B, int d)
{
    mpzfft_params_t params;
    assert ( mpzfft_initialized );
    cu_mpzfft_params_init(&params, mpz_rmatrix_product_max_bits(A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);
    return (params.lgN >= GPU_MIN_THRESHOLD && params.lgN <= GPU_MAX_THRESHOLD);
}
