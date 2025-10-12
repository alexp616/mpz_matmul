#include <GPUNTT-1.0/ntt_ct.cuh>
#include <GPUNTT-1.0/nttparameters.cuh>
#include <GPUNTT-1.0/modular_arith.cuh>
#include <cufftwrapper.cuh>
#include <assert.h>
#include "mod62.h"

#include <iostream>

#define THREADSPERBLOCK 512

void cuda_check() {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cudaGetErrorName(err)
        );
    }
}

static int cu_mpzfft_initialized;
static int cu_mem_alloced;
static uint64_t* d_arr;
static size_t d_arr_len;

extern "C" {
cu_zz_moduli_t* cu_zz_moduli;

// Only here for testing, never actually used in library
cu_fft62_mod_t cu_mod;

void gpu_alloc_mem(size_t n) {
    assert(cu_mem_alloced == 0);
    cu_mem_alloced = 1;
    cudaMalloc(&d_arr, n * sizeof(uint64_t));
    d_arr_len = n;
}

void gpu_free_mem() {
    assert(cu_mem_alloced == 1);
    cu_mem_alloced = 0;
    cudaFree(d_arr);
    d_arr_len = 0;
}

void cu_fft62_mod_init(cu_fft62_mod_t* mod, uint64_t p) {
    if (mod == nullptr) { exit(1); }

    mod->p = p;
    mod->modulus = Modulus<uint64_t>(p);

    uint64_t pinv = mod62_pinv(p);
    uint64_t npru = fft62_primitive_root_2(p, pinv, GPU_MAX_THRESHOLD);
    
    gpuntt::NTTFactors<uint64_t> nttfactors(mod->modulus, npru, 0);
    gpuntt::NTTParametersCT<uint64_t> params(GPU_MAX_THRESHOLD, nttfactors);

    // Make extra copy to avoid creating tables multiple times
    uint64_t* forward_table_copy;
    uint64_t* inverse_table_copy;
    cudaMalloc(&forward_table_copy, THREADSPERBLOCK * sizeof(uint64_t));
    cudaMalloc(&inverse_table_copy, THREADSPERBLOCK * sizeof(uint64_t));
    cudaMemcpy(forward_table_copy, params.forward_root_of_unity_table, THREADSPERBLOCK * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(inverse_table_copy, params.inverse_root_of_unity_table, THREADSPERBLOCK * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    uint64_t currnpru = npru;
    uint64_t currinpru = params.inverse_root_of_unity;
    for (int LOGN = GPU_MAX_THRESHOLD; LOGN >= GPU_MIN_THRESHOLD; --LOGN) {
        int curr_idx = LOGN - GPU_MIN_THRESHOLD;

        mod->cfg[curr_idx] = {
            .n_power = LOGN,
            .ntt_type = gpuntt::FORWARD,
            .shared_memory = 3 * THREADSPERBLOCK * sizeof(uint64_t),
            .root = currnpru,
            .root_table = forward_table_copy,
            .mod = mod->modulus,
            .stream = 0
        };

        mod->inverse_cfg[curr_idx] = {
            .n_power = LOGN,
            .ntt_type = gpuntt::INVERSE,
            .shared_memory = 3 * THREADSPERBLOCK * sizeof(uint64_t),
            .root = currinpru,
            .root_table = inverse_table_copy,
            .mod = mod->modulus,
            .scale_output = true,
            .stream = 0
        };

        currnpru = mod62_mul(currnpru, currnpru, p);
        currinpru = mod62_mul(currinpru, currinpru, p);
    }

    cuda_check();

    return;
}


void cu_fft62_mod_clear(cu_fft62_mod_t* mod) {
    cudaFree(mod->cfg[0].root_table);
    cudaFree(mod->inverse_cfg[0].root_table);
    cuda_check();

    return;
}

void cu_fft62_fft(uint64_t* yp, uint64_t* xp, size_t size, unsigned lgN, cu_fft62_mod_t* mod) {
    assert(lgN <= GPU_MAX_THRESHOLD && lgN >= GPU_MIN_THRESHOLD);

    int n = 1 << lgN;
    uint64_t* d_arr1;
    if (cu_mem_alloced) {
        assert(n == d_arr_len);
        d_arr1 = d_arr;
    } else {
        cudaMalloc(&d_arr1, n * sizeof(uint64_t));
    }

    // Input can be in range [0, 2p), so need to do this so doesn't 
    // overflow Barrett reduction
    for (int i = 0; i < n; ++i) { xp[i] = xp[i] % mod->p; }

    // xp has junk after first size elements, so need to set everything
    // to zero first
    cudaMemset(d_arr1, 0, n * sizeof(uint64_t));
    cudaMemcpy(d_arr1, xp, size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int modIdx = lgN - GPU_MIN_THRESHOLD;

    gpuntt::GPU_CT_NTT_Inplace(
        d_arr1,
        mod->cfg[modIdx]
    );

    cudaMemcpy(yp, d_arr1, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (!cu_mem_alloced) {
        cudaFree(d_arr1);
    }
    return;
}

void cu_fft62_ifft(uint64_t* yp, uint64_t* xp, unsigned lgN, cu_fft62_mod_t* mod) {
    assert(lgN <= GPU_MAX_THRESHOLD && lgN >= GPU_MIN_THRESHOLD);

    int n = 1 << lgN;

    uint64_t* d_arr1;
    if (cu_mem_alloced) {
        assert(n == d_arr_len);
        d_arr1 = d_arr;
    } else {
        cudaMalloc(&d_arr1, n * sizeof(uint64_t));
    }
    // Input can be in range [0, 2p), so need to do this so doesn't 
    // overflow Barrett reduction
    for (int i = 0; i < n; ++i) { xp[i] = xp[i] % mod->p; }

    cudaMemcpy(d_arr1, xp, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int modIdx = lgN - GPU_MIN_THRESHOLD;
    
    gpuntt::GPU_CT_NTT_Inplace(
        d_arr1,
        mod->inverse_cfg[modIdx]
    );

    cudaMemcpy(yp, d_arr1, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (!cu_mem_alloced) {
        cudaFree(d_arr1);
    }
    cuda_check();

    return;
}

uint64_t* cu_fft62_fft_batch(uint64_t* data, int num_primes, unsigned lgN, cu_zz_moduli_t* mod, int datasz) {
    assert(cu_mpzfft_initialized);
    assert(GPU_MIN_THRESHOLD <= lgN && lgN <= GPU_MAX_THRESHOLD);
    uint64_t* d_data;

    cudaMalloc(&d_data, datasz * num_primes * sizeof(uint64_t));
    cudaMemcpy(d_data, data, datasz * num_primes * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t* ptr = d_data;
    int modIdx = lgN - GPU_MIN_THRESHOLD;
    unsigned N = 1 << lgN;
    int batch_size = datasz / N;

    for (int i = 0; i < num_primes; ++i) {
        cu_fft62_mod_t* fft_data = mod->fft62_mod[i];
        gpuntt::nttct_configuration cfg = fft_data->cfg[modIdx];
        gpuntt::GPU_CT_NTT_Inplace_Batched(ptr, cfg, batch_size);
        ptr += datasz;
    }
    cuda_check();
    return d_data;
}

void cu_fft62_ifft_batch(uint64_t* host_ptr, uint64_t* d_data, int num_primes, unsigned lgN, cu_zz_moduli_t* mod, int datasz) {
    assert(cu_mpzfft_initialized);

    uint64_t* ptr = d_data;
    int modIdx = lgN - GPU_MIN_THRESHOLD;
    unsigned N = 1 << lgN;
    int batch_size = datasz / N;

    for (int i = 0; i < num_primes; ++i) {
        cu_fft62_mod_t* fft_data = mod->fft62_mod[i];
        gpuntt::nttct_configuration cfg = fft_data->inverse_cfg[modIdx];
        // std::cout << "cfg.n_power: " << cfg.n_power << std::endl;
        gpuntt::GPU_CT_NTT_Inplace_Batched(ptr, cfg, batch_size);
        ptr += batch_size * N;
    }

    cudaMemcpy(host_ptr, d_data, datasz * num_primes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cuda_check();
    return;
}

cu_zz_moduli_t* create_cu_zz_moduli_t() {
    cu_zz_moduli_t* result = (cu_zz_moduli_t*)malloc(sizeof(cu_zz_moduli_t));
    for (int i = 0; i < ZZ_MAX_PRIMES; ++i) {
        result->fft62_mod[i] = (cu_fft62_mod_t*)malloc(sizeof(cu_fft62_mod_t));
    }
    return result;
}

void cu_zz_moduli_init(cu_zz_moduli_t* moduli, int numPrimes) {
    moduli->num_primes = numPrimes;
    if (cu_mpzfft_initialized) { return; }
    
    for (int i = 0; i < numPrimes; ++i) {
        uint64_t p = global_p[i];
        moduli->p[i] = p;
        cu_fft62_mod_init(moduli->fft62_mod[i], p);
    }

    cuda_check();
    cu_mpzfft_initialized = 1;
    
    return;
}

void cu_zz_moduli_clear(cu_zz_moduli_t* moduli) {
    for (int i = 0; i < moduli->num_primes; ++i) {
        cu_fft62_mod_clear(moduli->fft62_mod[i]);
    }
    
    free(moduli);
    cu_mpzfft_initialized = 0;

    return;
}

__global__ void matmul_kernel(uint64_t* C, const uint64_t* B, uint64_t* aux, int d1, int d2, int d3, int n, Modulus64 mod) {
    int Aidx = threadIdx.x + blockIdx.x * blockDim.x;
    int Bidx = blockIdx.y * n + threadIdx.x + blockIdx.x * blockDim.x;
    int Cidx = blockIdx.y * n + threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t res = 0;
    uint64_t temp;
    for (int i = 0; i < d2; ++i) {
        temp = OPERATOR_GPU_64::mult(aux[Aidx], B[Bidx], mod);
        res = OPERATOR_GPU_64::add(res, temp, mod);
        Aidx += n;
        Bidx += d3 * n;
    }

    C[Cidx] = res;

    return;
}

void cu_mpzfft_matrix_mul(uint64_t* d_C, uint64_t* d_A, uint64_t* d_B, int d1, int d2, int d3, int num_primes, int n, cu_zz_moduli_t* mod) {
    uint64_t* aux_row;
    cudaMalloc(&aux_row, d2 * n * sizeof(uint64_t));
    assert (d_C == d_A);

    // uint64_t* d_A; uint64_t* d_B; uint64_t* d_C;
    // cudaMalloc(&d_A, d1 * d2 * n * num_primes * sizeof(uint64_t));
    // cudaMalloc(&d_B, d2 * d3 * n * num_primes * sizeof(uint64_t));
    // cudaMalloc(&d_C, d1 * d3 * n * num_primes * sizeof(uint64_t));
    
    // cudaMemcpy(d_A, A, d1 * d2 * n * num_primes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B, d2 * d3 * n * num_primes * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int totalThreads = n * d3;
    int threads = min(totalThreads, 512);
    // printf("totalThreads: %d, threads: %d\n", totalThreads, threads);

    int griddim_x = n / threads;
    int griddim_y = d3;

    uint64_t* a_ptr; uint64_t* b_ptr; uint64_t* c_ptr;
    for (int i = 0; i < num_primes; ++i) {
        Modulus64 modulus = mod->fft62_mod[i]->modulus;
        a_ptr = d_A + i * d1 * d2 * n;
        b_ptr = d_B + i * d2 * d3 * n;
        c_ptr = d_C + i * d1 * d3 * n;
        for (int row = 0; row < d1; ++row) {
            cudaMemcpy(aux_row, a_ptr, d2 * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            matmul_kernel<<<dim3(griddim_x, griddim_y), threads>>>(c_ptr, b_ptr, aux_row, d1, d2, d3, n, modulus);
            a_ptr += n * d2;
            c_ptr += n * d3;
        }
    }

    // cudaMemcpy(C, d_C, d1 * d3 * n * num_primes * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(aux_row);
    cuda_check();
    return;
}

// void cu_memcpy(void* host_ptr, void* dev_ptr, size_t bytes) {
//     cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
// }
}
