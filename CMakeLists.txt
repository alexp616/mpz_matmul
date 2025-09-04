cmake_minimum_required(VERSION 3.26)
project(rforest VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
SET(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -fomit-frame-pointer -funroll-loops -m64 -pedantic -std=gnu11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fPIC -fomit-frame-pointer -funroll-loops -m64 -pedantic -std=gnu11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89 -ccbin=${CMAKE_C_COMPILER}")



include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/fft62
    /usr/local/include
)

link_directories(/usr/local/lib)

set(HWLPOLY_SRCS
    src/hwmem.c
    src/hwmpz.c
    src/hwmpz_tune.c
    src/rtree.c
    src/rforest.c
)

set(MPZFFT_SRCS
    src/zzmisc.c
    src/moduli.c
    src/split.c
    src/reduce.c
    src/split_reduce.c
    src/recompose.c
    src/crt.c
    src/crt_recompose.c
    src/mpnfft.c
    src/fermat.c
    src/mpnfft_mod.c
    src/mpzfft.c
    src/zzmem.c
    src/fft62/mod62.c
    src/fft62/fft62.c
)

set(CUDA_SRCS
    src/fft62/cufftwrapper.cu
)

# Static library target
add_library(rforest STATIC
    ${HWLPOLY_SRCS}
    ${MPZFFT_SRCS}
    ${CUDA_SRCS}
)

target_link_libraries(rforest
    /usr/local/lib/libntt-1.0.a
    gmp
    m
)

set_target_properties(rforest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Force CUDA language linkage
add_executable(test_rforest test_rforest.c)
set_target_properties(test_rforest PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_rforest rforest)

add_executable(test_fft tests/test_fft.c)
set_target_properties(test_fft PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_fft rforest)

add_executable(test_matmul tests/test_matmul.c)
set_target_properties(test_matmul PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_matmul rforest)

# Install rules
install(TARGETS rforest ARCHIVE DESTINATION lib)
install(FILES include/rforest.h DESTINATION include)