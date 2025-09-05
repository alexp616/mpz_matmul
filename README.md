# rforest

This repository is a fork of David Harvey and Andrew 
Sutherland's [rforest](https://math.mit.edu/~drew/rforest.html). 
We plan to accelerate the algorithm by implementing parts of it 
(and eventually maybe the entire thing) in CUDA. Primarily, we 
aim to optimize the main matrix multiplication function, 
`mpz_rmatrix_mult_fft()`, implemented in `src/hwmpz.c`