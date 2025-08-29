#include <vector>
#include "mpzfft_moduli.h"
#include "fft62/cufft62_thresholds.h"

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
typedef struct {
    unsigned num_primes;
    uint64_t p[ZZ_MAX_PRIMES];

    cu_fft62_mod_t* fft62_mod[ZZ_MAX_PRIMES];
} cu_zz_moduli_t;

// Allocates gpu memory for ffts
void gpu_alloc_mem(size_t n);

// Frees gpu memory for ffts
void gpu_free_mem();

// C-friendly array access
cu_fft62_mod_t* get_mod_num(cu_zz_moduli_t* mod, int i) {
    return mod->fft62_mod[i];
}

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

// C-friendly struct constructor
cu_zz_moduli_t* create_cu_zz_moduli_t();

// initializes list of gpu mods
void cu_zz_moduli_init(cu_zz_moduli_t* moduli, int numPrimes);

// destroys list of gpu mods
void cu_zz_moduli_clear(cu_zz_moduli_t* moduli);
}