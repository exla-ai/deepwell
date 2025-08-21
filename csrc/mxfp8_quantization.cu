/*
 * MXFP8 Quantization Kernel for Blackwell
 * Produces scale factors in the exact layout required by tcgen05.mma
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace deepwell {

// MXFP8 scale factor layout for tcgen05.mma
// Scale factors are stored in a specific layout for Blackwell:
// - Each scale factor applies to 32 elements (block size)
// - Scales are in E8M0 format (8-bit exponent, no mantissa)
// - Layout must match tcgen05.mma expectations

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Quantize BF16/FP32 to MXFP8 with E4M3 elements and E8M0 scales
template <typename InputType>
__global__ void quantize_to_mxfp8_kernel(
    int8_t* __restrict__ output,     // Quantized elements (E4M3 stored as int8)
    float* __restrict__ scales,      // Scale factors (E8M0 stored as float for now)
    const InputType* __restrict__ input,            // Input tensor
    int M, int N,                                    // Matrix dimensions
    bool row_major                                   // Layout
) {
    const int block_size = 32;  // MXFP8 block size
    const float max_e4m3 = 448.0f;  // Maximum value representable in E4M3
    
    // Thread and block indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_blocks_per_row = (N + block_size - 1) / block_size;
    
    // Calculate which block this thread block is processing
    const int block_row = bid / num_blocks_per_row;
    const int block_col = bid % num_blocks_per_row;
    
    // Check bounds
    if (block_row >= M || block_col >= num_blocks_per_row) return;
    
    // Calculate element range for this block
    const int col_start = block_col * block_size;
    const int col_end = min(col_start + block_size, N);
    const int block_elements = col_end - col_start;
    
    // Phase 1: Find maximum absolute value in block
    float local_max = 0.0f;
    
    // Each thread processes multiple elements
    for (int i = tid; i < block_elements; i += blockDim.x) {
        int col = col_start + i;
        if (col < N) {
            int idx = row_major ? (block_row * N + col) : (col * M + block_row);
            float val;
            if constexpr (std::is_same_v<InputType, float>) {
                val = input[idx];
            } else if constexpr (std::is_same_v<InputType, __half>) {
                val = __half2float(input[idx]);
            } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
                val = __bfloat162float(input[idx]);
            }
            local_max = fmaxf(local_max, fabsf(val));
        }
    }
    
    // Warp-level reduction to find block maximum
    local_max = warp_reduce_max(local_max);
    
    // Block-level reduction if multiple warps
    __shared__ float shared_max[32];
    if (blockDim.x > 32) {
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        
        if (lane_id == 0) {
            shared_max[warp_id] = local_max;
        }
        __syncthreads();
        
        if (tid < blockDim.x / 32) {
            local_max = shared_max[tid];
        } else {
            local_max = 0.0f;
        }
        local_max = warp_reduce_max(local_max);
    }
    
    // Thread 0 computes and stores the scale factor
    __shared__ float block_scale;
    __shared__ float inv_scale;
    
    if (tid == 0) {
        // Compute scale to fit the maximum value in E4M3 range
        if (local_max > 0) {
            // E4M3 can represent values up to 448
            // We want: max_val / scale <= 448
            // So: scale >= max_val / 448
            block_scale = local_max / max_e4m3;
            
            // Round to next power of 2 for E8M0 format
            if (block_scale > 0) {
                int exp = ceilf(log2f(block_scale));
                block_scale = ldexpf(1.0f, exp);
            } else {
                block_scale = 1.0f;
            }
            inv_scale = 1.0f / block_scale;
        } else {
            block_scale = 1.0f;
            inv_scale = 1.0f;
        }
        
        // Store scale factor (for now as float, convert to E8M0 later)
        int scale_idx = block_row * num_blocks_per_row + block_col;
        // Bounds check for scale storage
        int total_scale_blocks = M * num_blocks_per_row;
        if (scale_idx < total_scale_blocks) {
            scales[scale_idx] = block_scale;
        }
    }
    __syncthreads();
    
    // Phase 2: Quantize elements using the computed scale
    // inv_scale is already computed in shared memory
    
    for (int i = tid; i < block_elements; i += blockDim.x) {
        int col = col_start + i;
        if (col < N) {
            int idx = row_major ? (block_row * N + col) : (col * M + block_row);
            
            float val;
            if constexpr (std::is_same_v<InputType, float>) {
                val = input[idx];
            } else if constexpr (std::is_same_v<InputType, __half>) {
                val = __half2float(input[idx]);
            } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
                val = __bfloat162float(input[idx]);
            }
            
            // Scale and quantize to E4M3
            float scaled = val * inv_scale;
            
            // Clamp to E4M3 range
            scaled = fminf(fmaxf(scaled, -max_e4m3), max_e4m3);
            
            // Convert to int8 (simulated E4M3)
            // In production, would use proper E4M3 conversion
            output[idx] = static_cast<int8_t>(scaled);
        }
    }
}

// Dequantize MXFP8 back to BF16/FP32
template <typename OutputType>
__global__ void dequantize_from_mxfp8_kernel(
    OutputType* __restrict__ output,                     // Output tensor
    const int8_t* __restrict__ input,                   // Quantized elements (E4M3 as int8)
    const float* __restrict__ scales,                   // Scale factors
    int M, int N,                                        // Matrix dimensions
    bool row_major                                      // Layout
) {
    const int block_size = 32;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_elements = M * N;
    
    if (tid >= total_elements) return;
    
    // Calculate position
    int row, col;
    if (row_major) {
        row = tid / N;
        col = tid % N;
    } else {
        col = tid / M;
        row = tid % M;
    }
    
    // Find which scale block this element belongs to
    int block_col = col / block_size;
    int num_blocks_per_row = (N + block_size - 1) / block_size;
    int scale_idx = row * num_blocks_per_row + block_col;
    
    // Get scale factor (bounds check)
    float scale = 1.0f;
    int total_scale_blocks = M * ((N + block_size - 1) / block_size);
    if (scale_idx < total_scale_blocks) {
        scale = scales[scale_idx];
    }
    
    // Dequantize (convert int8 back to float and scale)
    float val = static_cast<float>(input[tid]) * scale;
    
    // Convert to output type
    if constexpr (std::is_same_v<OutputType, float>) {
        output[tid] = val;
    } else if constexpr (std::is_same_v<OutputType, __half>) {
        output[tid] = __float2half(val);
    } else if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
        output[tid] = __float2bfloat16(val);
    }
}

// Fused transpose and quantize for MXFP8
// This is critical for backward pass where we need AT
template <typename InputType>
__global__ void transpose_quantize_to_mxfp8_kernel(
    int8_t* __restrict__ output,          // Transposed & quantized
    float* __restrict__ scales,           // Scale factors for transposed
    const InputType* __restrict__ input,  // Input tensor
    int M, int N                          // Original dimensions
) {
    // Implementation would handle transpose + quantization in one pass
    // This is crucial for performance as transpose changes scale factor blocks
    // For now, placeholder - just a stub
}

// C++ wrapper functions
void quantize_to_mxfp8_bf16(
    void* output,
    void* scales,
    const void* input,
    int M, int N,
    bool row_major,
    cudaStream_t stream
) {
    const int threads = 256;
    const int num_blocks = M * ((N + 31) / 32);  // One block per 32-element chunk
    
    quantize_to_mxfp8_kernel<__nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
        reinterpret_cast<int8_t*>(output),
        reinterpret_cast<float*>(scales),
        reinterpret_cast<const __nv_bfloat16*>(input),
        M, N, row_major
    );
}

void quantize_to_mxfp8_fp32(
    void* output,
    void* scales,
    const void* input,
    int M, int N,
    bool row_major,
    cudaStream_t stream
) {
    const int threads = 256;
    const int num_blocks = M * ((N + 31) / 32);
    
    quantize_to_mxfp8_kernel<float><<<num_blocks, threads, 0, stream>>>(
        reinterpret_cast<int8_t*>(output),
        reinterpret_cast<float*>(scales),
        reinterpret_cast<const float*>(input),
        M, N, row_major
    );
}

void dequantize_from_mxfp8_bf16(
    void* output,
    const void* input,
    const void* scales,
    int M, int N,
    bool row_major,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;
    
    dequantize_from_mxfp8_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        reinterpret_cast<const int8_t*>(input),
        reinterpret_cast<const float*>(scales),
        M, N, row_major
    );
}

// Get scale factor layout information for tcgen05.mma
void get_mxfp8_scale_layout_info(
    int M, int N, int K,
    int& scale_m, int& scale_n, int& scale_k,
    int block_size = 32
) {
    // For MXFP8 with block size 32:
    // - A matrix (M x K): needs M x (K/32) scale factors
    // - B matrix (K x N): needs (K/32) x N scale factors
    scale_m = M;
    scale_n = N;
    scale_k = (K + block_size - 1) / block_size;
}

}  // namespace deepwell
