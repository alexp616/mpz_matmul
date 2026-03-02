cmake_minimum_required(VERSION 3.18)
project(matmul C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ── GPU-NTT dependency ─────────────────────────────────────────────────────────
include(FetchContent)
set(GPUNTT_BUILD_EXAMPLES OFF CACHE BOOL "Disable GPU-NTT examples" FORCE)
set(GPUNTT_BUILD_BENCHMARKS OFF CACHE BOOL "Disable GPU-NTT benchmarks" FORCE)
FetchContent_Declare(
    gpuntt
    GIT_REPOSITORY https://github.com/alexp616/GPU-NTT.git
    GIT_TAG c911051c184e78f1d680706a22cd4b271b240e6b
)
FetchContent_MakeAvailable(gpuntt)

# Compiler flags mirroring the original makefile
add_compile_options(
    -O3
    -fPIC
    -fomit-frame-pointer
    -funroll-loops
    -m64
)

# Find GMP
find_library(GMP_LIBRARY NAMES gmp REQUIRED
    HINTS /usr/local/lib /usr/lib)
find_path(GMP_INCLUDE_DIR NAMES gmp.h
    HINTS /usr/local/include /usr/include)

if(NOT GMP_LIBRARY OR NOT GMP_INCLUDE_DIR)
    message(FATAL_ERROR "GMP not found. Install libgmp-dev or set GMP_LIBRARY/GMP_INCLUDE_DIR manually.")
endif()

# ── Static library ─────────────────────────────────────────────────────────────

set(LIB_SOURCES
    src/lib/zzmisc.c
    src/lib/zzmem.c
    src/lib/moduli.c
    src/lib/split.c
    src/lib/reduce.c
    src/lib/split_reduce.c
    src/lib/crt.c
    src/lib/recompose.c
    src/lib/crt_recompose.c
    src/lib/mpnfft.c
    src/lib/fermat.c
    src/lib/mpnfft_mod.c
    src/lib/mpzfft.c
    src/lib/fft62.c
    src/lib/mod62.c
    src/lib/hwmpz.c
    src/lib/hwmem.c
)

set(CUDA_SOURCES
    src/lib/cufftwrapper.cu
    src/lib/cucrt.cu
    src/lib/curecompose.cu
)

add_library(matmul STATIC ${LIB_SOURCES} ${CUDA_SOURCES})

target_include_directories(matmul PUBLIC
    src/include
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(matmul
    PRIVATE
        ntt
        ${GMP_LIBRARY}
)

set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ── Optional per-step profiling (enabled by -DBENCH_PROFILE=ON) ───────────────
# When ON, hwmpz.c emits per-step wall-clock timings inside mpz_rmatrix_mult_fft
# and cu_mpz_rmatrix_mult_fft.  Used by 'make bench'.
option(BENCH_PROFILE "Enable per-step profiling output in FFT matmul functions" OFF)
if(BENCH_PROFILE)
    target_compile_definitions(matmul     PRIVATE BENCH_PROFILE)
endif()

# ── CPU test executable ────────────────────────────────────────────────────────

add_executable(test_matmul tests/test_matmul.c)
set_target_properties(test_matmul PROPERTIES LINKER_LANGUAGE CUDA)

target_link_libraries(test_matmul PRIVATE
    matmul
    ${GMP_LIBRARY}
    m
)

# ── GPU test executable ────────────────────────────────────────────────────────

add_executable(test_gpu_matmul tests/test_gpu_matmul.c)
set_target_properties(test_gpu_matmul PROPERTIES LINKER_LANGUAGE CUDA)

target_link_libraries(test_gpu_matmul PRIVATE
    matmul
    ${GMP_LIBRARY}
    m
)

# ── Benchmark executable ──────────────────────────────────────────────────────

add_executable(bench_matmul benchmark/bench_matmul.c)
set_target_properties(bench_matmul PROPERTIES LINKER_LANGUAGE CUDA)
# bench_matmul.c itself does not use BENCH_PROFILE but the library does;
# the flag is set via -DBENCH_PROFILE=ON at cmake configure time (see Makefile).

target_link_libraries(bench_matmul PRIVATE
    matmul
    ${GMP_LIBRARY}
    m
)

# ── Direct-FFT benchmark (temporary) ────────────────────────────────────────

add_executable(bench_fft_direct benchmark/bench_fft_direct.c)
set_target_properties(bench_fft_direct PROPERTIES LINKER_LANGUAGE CUDA)

target_link_libraries(bench_fft_direct PRIVATE
    matmul
    ${GMP_LIBRARY}
    m
)

enable_testing()

add_test(NAME matmul_50000_2_2     COMMAND test_matmul     50000 2 2)
add_test(NAME gpu_matmul_50000_2_2 COMMAND test_gpu_matmul 50000 2 2)
