#ifndef _HWMPZ_INCLUDE_
#define _HWMPZ_INCLUDE_

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include "hwmem.h"

// Per-step wall-clock profiling.  Compile with -DBENCH_PROFILE to enable.
#ifdef BENCH_PROFILE
#include <time.h>
static inline double _hwmpz_wall_sec(void) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return _ts.tv_sec + _ts.tv_nsec * 1e-9;
}
#define _BENCH_TICK(v)         double _bench_t_##v = _hwmpz_wall_sec()
#define _BENCH_TOCK(v)         (_hwmpz_wall_sec() - _bench_t_##v)
#define _BENCH_PRINT(label, s) fprintf(stdout, "  [profile] %-30s %.6f s\n", (label), (double)(s))
#else
#define _BENCH_TICK(v)         ((void)0)
#define _BENCH_TOCK(v)         (0.0)
#define _BENCH_PRINT(label, s) ((void)(s))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int hw_disable_fft;

typedef struct cu_zz_moduli_t cu_zz_moduli_t;
extern void cu_zz_moduli_init(cu_zz_moduli_t* moduli, int numPrimes);
extern void cu_zz_moduli_clear(cu_zz_moduli_t* moduli);

// pinned memory helpers
extern void* cu_malloc_pinned(size_t n);
extern void  cu_free_pinned(void* p);

// device memory helpers
extern void* cu_device_malloc(size_t n);
extern void  cu_device_free(void* p);
extern void  cu_memcpy_d2h(void* dst, const void* src, size_t n);

typedef struct cu_crt_constants_t cu_crt_constants_t;
extern cu_crt_constants_t* cu_crt_constants_create(const void* moduli,
                                                   unsigned num_primes);
extern void cu_crt_constants_destroy(cu_crt_constants_t* ctx);
extern void cu_zz_crt_batch(uint64_t* d_out,
                            const uint64_t* d_data,
                            size_t datasz,
                            const cu_crt_constants_t* h_ctx);

extern void cu_zz_recompose_chunked(uint64_t* d_out,
                                    size_t rn,
                                    unsigned r,
                                    const uint64_t* d_crt,
                                    size_t N,
                                    unsigned num_primes);
extern void cu_zz_recompose_batch(uint64_t* d_out_all,
                                  size_t rn,
                                  unsigned r,
                                  const uint64_t* d_crt_all,
                                  size_t N,
                                  unsigned num_primes,
                                  int entries);
// used for benching
extern uint64_t* cu_h2d(uint64_t* host_data, int num_primes, int datasz);
extern void      cu_ntt_batch_only(uint64_t* d_data, int num_primes, unsigned lgN, int datasz);
extern void      cu_intt_batch_only(uint64_t* d_data, int num_primes, unsigned lgN, int datasz);
extern void      cu_d2h_batch_only(uint64_t* host_dst, uint64_t* d_data, int num_primes, int datasz);

void hw_mpz_setup ();
void hw_mpz_clear ();

// handy mpz inlines
static inline int mpz_is_zero (mpz_t x) { return mpz_sgn(x) ? 0 : 1; }
static inline void mpz_set_zero (mpz_t o) { mpz_set_ui (o,0); }
static inline int mpz_is_one (mpz_t x) { return mpz_cmp_ui(x,1) == 0 ? 1 : 0; }
static inline void mpz_set_one (mpz_t o) { mpz_set_ui (o,1); }
static inline int mpz_equal (mpz_t a, mpz_t b) { return mpz_cmp(a,b) == 0; }
static inline long mpz_bits (mpz_t a) { return mpz_sizeinbase(a,2); }

// vector versions of basic mpz_t functions
static inline mpz_t *mpz_vec_alloc (long n) { return (mpz_t*)hw_malloc (n*sizeof(mpz_t)); }
static inline mpz_t *mpz_vec_init (mpz_t *A, long n) { long i; for ( i = 0 ; i < n ; i++ ) { mpz_init (A[i]); } return A; }
static inline mpz_t *mpz_vec_alloc_and_init (long n) { mpz_t *A = (mpz_t*)hw_malloc (n*sizeof(mpz_t)); mpz_vec_init (A,n); return A; }
static inline mpz_t *mpz_vec_set (mpz_t *A, mpz_t *B, long n) { if ( A == B ) return A; for ( long i = 0 ; i < n ; i++ ) { mpz_set (A[i], B[i]); } return A; }
static inline mpz_t *mpz_vec_set_zero (mpz_t *A, long n) { long i; for ( i = 0 ; i < n ; i++ ) { mpz_set_zero (A[i]); } return A; }
static inline int mpz_vec_equal (mpz_t *A, mpz_t *B, long n) { if ( A == B ) return 1; for ( long i = 0 ; i < n ; i++ ) if ( ! mpz_equal (A[i], B[i]) ) return 0;  return 1; }
static inline long mpz_vec_max_size (mpz_t *A, long n) { long s, t;  s = 0; for ( long i = 0 ; i < n ; i++ ) { t = mpz_size(A[i]); if ( t > s ) s = t; } return s; }
static inline long mpz_vec_total_size (mpz_t *A, long n) { long s;  s = 0; for ( long i = 0 ; i < n ; i++ ) s += mpz_size(A[i]);   return s;}
static inline mpz_t *mpz_vec_mod_naive (mpz_t *A, mpz_t *B, long n, mpz_t m) { for (  long i = 0 ; i < n ; i++ ) { mpz_fdiv_r (A[i], B[i], m); } return A; }
static inline void mpz_vec_clear (mpz_t *A, long n) {  int i; for ( i = 0 ; i < n ; i++ ) mpz_clear (A[i]); }
static inline void mpz_vec_free (mpz_t *A, long n) { hw_free (A,n*sizeof(mpz_t) ); }
static inline void mpz_vec_clear_and_free (mpz_t *A, long n) { mpz_vec_clear (A, n);  mpz_vec_free (A, n); }

// matrix alloc/init/clear helpers
static inline mpz_t *mpz_matrix_alloc (int d) { return mpz_vec_alloc (d*d); }
static inline void mpz_matrix_init (mpz_t *M, int d) { mpz_vec_init (M, d*d); }
static inline mpz_t *mpz_matrix_alloc_and_init (int d) { return mpz_vec_alloc_and_init (d*d); }
static inline void mpz_matrix_set_zero (mpz_t *M, int d) { mpz_vec_set_zero (M, d*d); }
static inline void mpz_matrix_set_one (mpz_t *M, int d) { mpz_vec_set_zero (M, d*d); for ( int i = 0 ; i < d ; i++ ) mpz_set_one (M[i*d+i]); }
static inline void mpz_matrix_clear (mpz_t *M, int d) { mpz_vec_clear (M, d*d); }
static inline void mpz_matrix_clear_and_free (mpz_t *M, int d) { mpz_vec_clear_and_free (M, d*d); }
static inline int mpz_matrix_equal (mpz_t *A, mpz_t *B, int d) { return mpz_vec_equal (A, B, d*d); }
static inline void mpz_matrix_set (mpz_t *A, mpz_t *B, int d) { mpz_vec_set (A, B, d*d); }

// height (upper bound on bits)
static inline long mpz_height (mpz_t x) { return GMP_NUMB_BITS*mpz_size(x); }
static inline long mpz_vec_height (mpz_t *A, long n) { return GMP_NUMB_BITS*mpz_vec_max_size(A,n); }
static inline long mpz_matrix_height (mpz_t *A, int d) { return mpz_vec_height (A, d*d); }

// naive r x d times d x d matrix multiply (exact, no modular reduction)
// ALIASING NOT ALLOWED
static inline void mpz_rmatrix_mult_naive (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    for ( int i = 0 ; i < r ; i++ ) for ( int j = 0 ; j < d ; j++ ) {
        mpz_mul (w,A[i*d],B[j]);
        for (  int k = 1 ; k < d ; k++ ) mpz_addmul (w,A[i*d+k],B[k*d+j]);
        mpz_set (C[i*d+j],w);
    }
}
static inline void mpz_matrix_mult_naive (mpz_t *C, mpz_t *A, mpz_t *B, int d, mpz_t w) { mpz_rmatrix_mult_naive (C, A, d, B, d, w); }

// FFT-based r x d times d x d matrix multiply (exact)
// NO ALIASING
void mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w);
static inline void mpz_matrix_mult_fft (mpz_t *C, mpz_t *A, mpz_t *B, int d, mpz_t w) { mpz_rmatrix_mult_fft (C, A, d, B, d, w); }

// GPU FFT-based r x d times d x d matrix multiply (exact)
// NO ALIASING
void cu_mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w);
static inline void cu_mpz_matrix_mult_fft (mpz_t *C, mpz_t *A, mpz_t *B, int d, mpz_t w) { cu_mpz_rmatrix_mult_fft (C, A, d, B, d, w); }

/* Returns 1 if GPU-FFT can handle this input (lgN in [GPU_MIN, GPU_MAX]),
 * 0 otherwise.  Requires hw_mpz_setup() to have been called first. */
int mpz_rmatrix_gpu_fft_suitable(mpz_t *A, int r, mpz_t *B, int d);

// naive r x d times d x d matrix multiply mod m
// ALIASING NOT ALLOWED
static inline void mpz_rmatrix_mult_mod_naive (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t m, mpz_t w)
{
    for ( int i = 0 ; i < r ; i++ ) for ( int j = 0 ; j < d ; j++ ) {
        mpz_mul (w,A[i*d],B[j]);
        for ( int k = 1 ; k < d ; k++ ) mpz_addmul (w,A[i*d+k],B[k*d+j]);
        mpz_fdiv_r (C[i*d+j],w,m);
    }
}
static inline void mpz_matrix_mult_mod_naive (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t m, mpz_t w) { mpz_rmatrix_mult_mod_naive (C, A, r, B, d, m, w); }

#ifdef __cplusplus
}
#endif

#endif
