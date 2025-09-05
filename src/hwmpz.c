#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <gmp.h>
#include "mpzfft.h"
#include "zzmem.h"
#include "hwmem.h"
#include "hwmpz.h"

#define HW_ZZ_PRIMES            2

int hw_disable_fft;
int mpzfft_threads = 1; // setting this to a value other than 1 is typically not all that helpful (better to parallelize at a higher level)

static zz_moduli_t zz_moduli;
extern cu_zz_moduli_t* cu_zz_moduli;
extern cu_zz_moduli_t* create_cu_zz_moduli_t();

static int mpzfft_initialized;

static inline void hw_mpzfft_setup ()
{
    if ( hw_disable_fft || mpzfft_initialized ) return;
    zz_moduli_init (&zz_moduli, ZZ_MAX_PRIMES);
    cu_zz_moduli = create_cu_zz_moduli_t();
    cu_zz_moduli_init (cu_zz_moduli, ZZ_MAX_PRIMES);
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
    mpzfft_initialized = 0;
}

void hw_mpz_setup (void) { hw_mpzfft_setup ();}
void hw_mpz_clear (void) { hw_mpzfft_clear (); }

mpz_t *mpz_vec_mod_fft (mpz_t *A, mpz_t *B, long n, mpz_t m)
{
    mpzfft_mod_t mod;
    long b, c;

    assert ( mpzfft_initialized );
    b = mpz_sizeinbase (m, 2);
    for ( long i = 0 ; i < n ; i++ ) { c = mpz_sizeinbase (B[i], 2);  if ( c > b ) b = c; }
    mpzfft_mod_init (&mod, b, m, HW_ZZ_PRIMES, &zz_moduli, mpzfft_threads);
    for ( long i = 0 ; i < n ; i++ ) { mpzfft_mod_mod (&mod, A[i], B[i], 1); }
    mpzfft_mod_clear (&mod);
    return A;
}

mpz_t *mpz_vec_mod_init_fft (mpz_t *A, mpz_t *B, long n, mpz_t m, mpz_t w)
{
    mpzfft_mod_t mod;
    long b, c;

    assert ( mpzfft_initialized );
    b = mpz_sizeinbase (m, 2);
    for ( long i = 0 ; i < n ; i++ ) { c = mpz_sizeinbase (B[i], 2);  if ( c > b ) b = c; }
    mpzfft_mod_init (&mod, b, m, HW_ZZ_PRIMES, &zz_moduli, mpzfft_threads);
    for ( long i = 0 ; i < n ; i++ ) { mpzfft_mod_mod (&mod, w, B[i], 1);  mpz_init_set (A[i], w); }
    mpzfft_mod_clear (&mod);
    return A;
}

static inline size_t mpz_rmatrix_product_max_bits (mpz_t *A, int r, mpz_t *B, int d)
{
    register int i;
    register size_t m, n;
    
    m = 0;
    for ( i = 0 ; i < r*d ; i++ ) { n = mpz_size(A[i]); if ( n > m ) m = n; }
    for ( i = 0 ; i < d*d ; i++ ) { n = mpz_size(B[i]); if ( n > m ) m = n; }
    return GMP_NUMB_BITS*(2*m+1);   // note that the +1 limb more than covers the extra log(d) bits due to additions
}

mpz_t *mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT;
    assert ( mpzfft_initialized );
    
    mpzfft_params_init (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);

    // transform input matrices
    AT = hw_malloc (r*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < r*d; i++) { mpzfft_init(AT[i], &params);  mpzfft_fft (AT[i], A[i], mpzfft_threads); }
    BT = hw_malloc (d*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < d*d; i++) { mpzfft_init(BT[i], &params);  mpzfft_fft (BT[i], B[i], mpzfft_threads); }

    // multiply matrices of Fourier coefficients
    mpzfft_matrix_mul(AT, AT, BT, r, d, d, mpzfft_threads);

    // inverse transform results and cleanup
    for ( int i = 0 ; i < d*d ; i++) { mpzfft_clear (BT[i]); }
    for ( int i = 0 ; i < r*d ; i++) { mpzfft_ifft (C[i], AT[i], mpzfft_threads); mpzfft_clear (AT[i]); }

    hw_free (AT, r*d*sizeof(mpzfft_t));
    hw_free (BT, d*d*sizeof(mpzfft_t));

    mpzfft_params_clear (&params);
    return C;
}

mpz_t* mpz_rmatrix_mult_fft2(mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT;
    assert ( mpzfft_initialized );
    
    mpzfft_params_init2 (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);

    // In this implementation, we effectively have a triple pointer (but everything is actually contiguous in 
    // implementation). First pointer (Adata) gives the location of the i-th fft matrix for i in 1 ... num_primes.
    // Second pointer (Adata[i][j]) gives the index of the fft representation of A[i][j] (there isn't actually a 
    // j in the implementation, the fft representations are just interpreted as contiguous blocks of length N). 
    // Adata[i][j][k] gives the inner fft index. So for the FFT, we want to do:
    // for (i in 1...num_primes)
    //   for (j = 0; j < r * d; j += N)
    //      fft(Adata[i][j ... j + N - 1])
    // 
    // Everything is continuous to make everything easily accessible to a GPU kernel through index arithmetic
    size_t Asz_0 = r * d * params.N;
    size_t Bsz_0 = d * d * params.N;

    uint64_t* Aptr = hw_malloc(Asz_0 * params.num_primes * sizeof(uint64_t));
    uint64_t* Bptr = hw_malloc(Bsz_0 * params.num_primes * sizeof(uint64_t));

    uint64_t* Adata[HW_ZZ_PRIMES];
    for (int i = 0; i < HW_ZZ_PRIMES; ++i) { Adata[i] = &Aptr[i * Asz_0]; }
    uint64_t* Bdata[HW_ZZ_PRIMES];
    for (int i = 0; i < HW_ZZ_PRIMES; ++i) { Bdata[i] = &Bptr[i * Bsz_0]; }

    for ( int i = 0 ; i < r*d; i++) { mpzfft_fft2(Adata, &params, A[i], mpzfft_threads, Asz_0); }
    for ( int i = 0 ; i < d*d; i++) { mpzfft_fft2(Bdata, &params, B[i], mpzfft_threads, Bsz_0); }
    
    assert (1 == 2);

    // // multiply matrices of Fourier coefficients
    // mpzfft_matrix_mul(AT, AT, BT, r, d, d, mpzfft_threads);

    // // inverse transform results and cleanup
    // for ( int i = 0 ; i < d*d ; i++) { mpzfft_clear (BT[i]); }
    // for ( int i = 0 ; i < r*d ; i++) { mpzfft_ifft (C[i], AT[i], mpzfft_threads); mpzfft_clear (AT[i]); }

    // hw_free (AT, r*d*sizeof(mpzfft_t));
    // hw_free (BT, d*d*sizeof(mpzfft_t));
    hw_free(Aptr, Asz_0 * params.num_primes * sizeof(uint64_t));
    hw_free(Bptr, Asz_0 * params.num_primes * sizeof(uint64_t));

    mpzfft_params_clear (&params);
    return C;
}

mpz_t* mpz_rmatrix_mult_fft3(mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT;
    assert ( mpzfft_initialized );
    
    mpzfft_params_init3 (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);

    AT = hw_malloc (r*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < r*d; i++) { mpzfft_init(AT[i], &params);  mpzfft_fft3 (AT[i], A[i], mpzfft_threads); }
    BT = hw_malloc (d*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < d*d; i++) { mpzfft_init(BT[i], &params);  mpzfft_fft3 (BT[i], B[i], mpzfft_threads); }

    // multiply matrices of Fourier coefficients
    mpzfft_matrix_mul(AT, AT, BT, r, d, d, mpzfft_threads);

    // inverse transform results and cleanup
    for ( int i = 0 ; i < d*d ; i++) { mpzfft_clear (BT[i]); }
    for ( int i = 0 ; i < r*d ; i++) { mpzfft_ifft (C[i], AT[i], mpzfft_threads); mpzfft_clear (AT[i]); }

    hw_free (AT, r*d*sizeof(mpzfft_t));
    hw_free (BT, d*d*sizeof(mpzfft_t));

    mpzfft_params_clear (&params);
    return C;
}