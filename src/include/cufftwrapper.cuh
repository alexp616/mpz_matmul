#ifndef CUFFTWRAPPER_CUH
#define CUFFTWRAPPER_CUH

#include <vector>
#include "mpzfft_moduli.h"
#include "cufft62_thresholds.h"

const int NUM_N = GPU_MAX_THRESHOLD - GPU_MIN_THRESHOLD + 1;

extern "C" {

// Struct containing precomputed data for gpu ffts
// for 1 prime
typedef struct {
    uint64_t p;
    Modulus<uint64_t> modulus;
    gpuntt::nttct_configuration<uint64_t> cfg[NUM_N];
    gpuntt::nttct_configuration<uint64_t> inverse_cfg[NUM_N];
} cu_fft62_mod_t;

// Struct containing tables of precomputed data for
// gpu ffts for all primes
typedef struct cu_zz_moduli_t {
    unsigned num_primes;
    uint64_t p[ZZ_MAX_PRIMES];

    cu_fft62_mod_t* fft62_mod[ZZ_MAX_PRIMES];
} cu_zz_moduli_t;

// Allocates gpu memory for ffts
void gpu_alloc_mem(size_t n);

// Frees gpu memory for ffts
void gpu_free_mem();

// C-friendly array access
cu_fft62_mod_t* get_mod_num(cu_zz_moduli_t* mod, int i);

// initializes gpu mod for use with p
void cu_fft62_mod_init(cu_fft62_mod_t* mod, uint64_t p);

// destroys gpu mod
void cu_fft62_mod_clear(cu_fft62_mod_t* mod);

/*
    yp: output
    xp: input
    size: number of relevant numbers in xp
    lgN: log2 fft length
    mod: mod
*/
void cu_fft62_fft(uint64_t* yp, uint64_t* xp, size_t size, unsigned lgN, cu_fft62_mod_t* mod);

/*
    yp: output
    xp: input
    lgN: log2 fft length
    mod: mod
*/
void cu_fft62_ifft(uint64_t* yp, uint64_t* xp, unsigned lgN, cu_fft62_mod_t* mod);

uint64_t* cu_fft62_fft_batch(uint64_t* data, int num_primes, unsigned lgN, cu_zz_moduli_t* mod, int datasz);

void cu_fft62_ifft_batch(uint64_t* host_ptr, uint64_t* d_data, int num_primes, unsigned lgN, cu_zz_moduli_t* mod, int datasz);

// C-friendly struct constructor
cu_zz_moduli_t* create_cu_zz_moduli_t();

// initializes list of gpu mods
void cu_zz_moduli_init(cu_zz_moduli_t* moduli, int numPrimes);

// destroys list of gpu mods
void cu_zz_moduli_clear(cu_zz_moduli_t* moduli);

// GPU matrix multiply of Fourier coefficients
void cu_mpzfft_matrix_mul(uint64_t* C, uint64_t* A, uint64_t* B,
                          int d1, int d2, int d3,
                          int num_primes, int n, cu_zz_moduli_t* mod);

// Pinned (page-locked) host memory helpers
void* cu_malloc_pinned(size_t n);
void  cu_free_pinned(void* p);

// Generic device memory helpers (thin wrappers around cudaMalloc/Free/Memcpy)
void* cu_device_malloc(size_t n);
void  cu_device_free(void* p);
void  cu_memcpy_d2h(void* dst, const void* src, size_t n);

// Split transfer+kernel functions used for per-step benchmarking
uint64_t* cu_h2d(uint64_t* host_data, int num_primes, int datasz);
void      cu_ntt_batch_only(uint64_t* d_data, int num_primes, unsigned lgN, int datasz);
void      cu_intt_batch_only(uint64_t* d_data, int num_primes, unsigned lgN, int datasz);
void      cu_d2h_batch_only(uint64_t* host_dst, uint64_t* d_data, int num_primes, int datasz);

}

#endif
