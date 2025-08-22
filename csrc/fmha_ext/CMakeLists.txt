cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deepwell_fmha LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enforce SM100a for tcgen05
# Set CUDA architectures to 100 (CMake requires non-empty); we still pass sm_100a to NVCC directly
set(CMAKE_CUDA_ARCHITECTURES 100)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_100a --generate-code=arch=compute_100,code=sm_100a>)

# CUTLASS from third_party
set(CUTLASS_DIR ${CMAKE_SOURCE_DIR}/../../third_party/cutlass)
include_directories(
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/examples/77_blackwell_fmha
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/collective
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/device
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/kernel
)

add_library(deepwell_fmha SHARED fmha_ext.cu)
target_compile_definitions(deepwell_fmha PRIVATE
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
  CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1
  CUTLASS_ENABLE_SM100_TCGEN05=1
  CUTE_ARCH_TMA_SM100_ENABLED=1
  CUTE_ARCH_TMA_SM90_ENABLED=1
  CUTLASS_ENABLE_SYNCLOG=0
)
target_compile_options(deepwell_fmha PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -rdc=true
>)
find_library(CUDA_CUDART cudart PATHS /usr/local/cuda/lib64)
find_library(CUDA_CUBLAS cublas PATHS /usr/local/cuda/lib64)
find_library(CUDA_CUBLASLT cublasLt PATHS /usr/local/cuda/lib64)
target_link_libraries(deepwell_fmha PRIVATE ${CUDA_CUDART} ${CUDA_CUBLAS} ${CUDA_CUBLASLT})


