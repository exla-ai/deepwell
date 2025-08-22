cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dw_fmha_bridge_min LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUTLASS include dirs
set(CUTLASS_DIR ${CMAKE_SOURCE_DIR}/../../third_party/cutlass)
include_directories(
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/examples/77_blackwell_fmha
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/collective
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/kernel
  ${CUTLASS_DIR}/examples/77_blackwell_fmha/device
)

# Clear any inherited CUDA architectures
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

add_library(dw_fmha_bridge_min SHARED fmha_bridge_min.cu)

# Use the exact same flags as the successful CUTLASS example build
# Force both PTX and SASS to use sm_100a to support tcgen05
set_property(TARGET dw_fmha_bridge_min PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(dw_fmha_bridge_min PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:
    -arch=sm_100a
    --use_fast_math
    -ftemplate-backtrace-limit=0
    --expt-relaxed-constexpr 
    --expt-extended-lambda 
    -rdc=true
    -Xptxas=-v
  >
)

# Define macros needed for CUTLASS SM100
target_compile_definitions(dw_fmha_bridge_min PRIVATE 
  CUTE_ARCH_TMA_SM100_ENABLED=1 
  CUTE_ARCH_TMA_SM90_ENABLED=1 
  CUTE_ARCH_TCGEN05_TMEM_ENABLED=1 
  CUTLASS_ENABLE_SM100_TCGEN05=1 
  CUTLASS_DEBUG_TRACE_LEVEL=0
)

set_target_properties(dw_fmha_bridge_min PROPERTIES 
  CUDA_SEPARABLE_COMPILATION ON 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON 
  OUTPUT_NAME "dw_fmha_bridge_min"
)

find_package(CUDAToolkit REQUIRED)
if (TARGET CUDA::cudadevrt)
  target_link_libraries(dw_fmha_bridge_min PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cudadevrt)
else()
  find_library(CUDA_CUDADEVRT_LIB cudadevrt HINTS /usr/local/cuda/lib64)
  if (NOT CUDA_CUDADEVRT_LIB)
    message(FATAL_ERROR "Could not find cudadevrt")
  endif()
  target_link_libraries(dw_fmha_bridge_min PRIVATE CUDA::cudart CUDA::cuda_driver ${CUDA_CUDADEVRT_LIB})
endif()


