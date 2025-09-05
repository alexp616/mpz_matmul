#include <gmp.h>
#include <hwmpz.h>
#include <assert.h>

void matmul_test1() {
    hw_mpz_setup();

    int r = 3;
    int d = 3;

    mpz_t* A = malloc(r * d * sizeof(mpz_t));
    mpz_t* B = malloc(d * d * sizeof(mpz_t));
    mpz_t* C1 = malloc(r * d * sizeof(mpz_t));
    mpz_t* C2 = malloc(r * d * sizeof(mpz_t));

    gmp_randstate_t state;
    gmp_randinit_default(state);
    for (int t = 0; t < 10; ++t) {
    // between 50k bits and 100k bits
    int bits = (rand() % 50000) + 50000;

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(A[i*d + j]); mpz_init(C1[i*d + j]); mpz_init(C2[i*d + j]);
        mpz_urandomb(A[i*d + j], state, bits);
    }}
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(B[i*d + j]);
        mpz_urandomb(B[i*d + j], state, bits);
    }}

    mpz_t w; mpz_init(w);
    mpz_rmatrix_mult_naive(C1, A, r, B, d, w);

    mpz_rmatrix_mult_fft(C2, A, r, B, d, w);
    
    mpz_clear(w);

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        assert(mpz_cmp(C1[i*d + j], C2[i*d + j]) == 0);
    }}

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(A[i*d + j]); mpz_clear(C1[i*d + j]); mpz_clear(C2[i*d + j]);
    }}
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(B[i*d + j]);
    }}
    }
    free(A); free(B); free(C1); free(C2);
    gmp_randclear(state);
    hw_mpz_clear();
}

void matmul_test2() {
    hw_mpz_setup();

    int r = 3;
    int d = 3;

    mpz_t* A = malloc(r * d * sizeof(mpz_t));
    mpz_t* B = malloc(d * d * sizeof(mpz_t));
    mpz_t* C1 = malloc(r * d * sizeof(mpz_t));
    mpz_t* C2 = malloc(r * d * sizeof(mpz_t));

    gmp_randstate_t state;
    gmp_randinit_default(state);
    for (int t = 0; t < 10; ++t) {
    // between 50k bits and 100k bits
    int bits = (rand() % 50000) + 50000;

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(A[i*d + j]); mpz_init(C1[i*d + j]); mpz_init(C2[i*d + j]);
        mpz_urandomb(A[i*d + j], state, bits);
    }}
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(B[i*d + j]);
        mpz_urandomb(B[i*d + j], state, bits);
    }}

    mpz_t w; mpz_init(w);
    mpz_rmatrix_mult_fft3(C1, A, r, B, d, w);
    mpz_rmatrix_mult_fft(C2, A, r, B, d, w);
    
    mpz_clear(w);

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        assert(mpz_cmp(C1[i*d + j], C2[i*d + j]) == 0);
    }}

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(A[i*d + j]); mpz_clear(C1[i*d + j]); mpz_clear(C2[i*d + j]);
    }}
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(B[i*d + j]);
    }}
    printf("\n");
    }
    free(A); free(B); free(C1); free(C2);
    gmp_randclear(state);
    hw_mpz_clear();
}

void matmul_test3() {
    hw_mpz_setup();

    int r = 2;
    int d = 2;

    mpz_t* A = malloc(r * d * sizeof(mpz_t));
    mpz_t* B = malloc(d * d * sizeof(mpz_t));
    mpz_t* C1 = malloc(r * d * sizeof(mpz_t));
    mpz_t* C2 = malloc(r * d * sizeof(mpz_t));

    gmp_randstate_t state;
    gmp_randinit_default(state);
    // for (int t = 0; t < 10; ++t) {
    // between 50k bits and 100k bits
    // int bits = (rand() % 50000) + 50000;
    int bits = 60000;

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(A[i*d + j]); mpz_init(C1[i*d + j]); mpz_init(C2[i*d + j]);
        mpz_urandomb(A[i*d + j], state, bits);
    }}
    // mpz_set_str(A[0], "9301165293246235069759966068146313776551258669855356477271940698500929939755418247622530571466332330697816620308003246225290293476785304004840090056840661553451916748315356563734257724978000166406621823207925733850455027807451108123161768212073821382033500073069184011344280494573919716117539236653172", 10);
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_init(B[i*d + j]);
        mpz_urandomb(B[i*d + j], state, bits);
    }}
    // mpz_set_str(B[0], "9301165293246235069759966068146313776551258669855356477271940698500929939755418247622530571466332330697816620308003246225290293476785304004840090056840661553451916748315356563734257724978000166406621823207925733850455027807451108123161768212073821382033500073069184011344280494573919716117539236653171", 10);

    mpz_t w; mpz_init(w);
    printf("--------------------------------------------------- original impl: \n");
    mpz_rmatrix_mult_fft3(C2, A, r, B, d, w);
    printf("--------------------------------------------------- experimental impl: \n");
    mpz_rmatrix_mult_fft2(C1, A, r, B, d, w);
    
    mpz_clear(w);

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        assert(mpz_cmp(C1[i*d + j], C2[i*d + j]) == 0);
    }}

    for (int i = 0; i < r; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(A[i*d + j]); mpz_clear(C1[i*d + j]); mpz_clear(C2[i*d + j]);
    }}
    for (int i = 0; i < d; ++i) { for (int j = 0; j < d; ++j) {
        mpz_clear(B[i*d + j]);
    }}
    printf("\n");
    // }
    free(A); free(B); free(C1); free(C2);
    gmp_randclear(state);
    hw_mpz_clear();
}

int main() {
    // matmul_test1();
    // matmul_test2();
    matmul_test3();

    printf("All matmul tests passed!\n");
    return 0;
}