#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/cutlass.h"

#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "device/fmha.hpp"

using Element = cutlass::bfloat16_t;
using Accum = float;

using ProblemShape = cute::tuple<int, int, int, cute::tuple<int, int>>; // Q K D (H B)
using StrideQ = cute::tuple<int, cute::_1, cute::tuple<int, int>>;      // Q D (H B)
using StrideK = StrideQ;
using StrideV = StrideQ;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<cute::_1, cute::tuple<int, int>>;         // Q (H B)

using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
using ActiveMask = cutlass::fmha::collective::NoMask;
using TileScheduler = cutlass::fmha::kernel::IndividualTileScheduler;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
  Element, Accum, Accum, TileShape, StrideQ, StrideK, StrideV, ActiveMask>;

using Operation = cutlass::fmha::device::FMHA<
  cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape,
    Mainloop,
    cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      Element, Accum, typename Mainloop::TileShapePV, StrideO, StrideLSE>,
    TileScheduler
  >>;

extern "C" int dw_fmha_bf16_forward(
  void* q_ptr,
  void* k_ptr,
  void* v_ptr,
  void* o_ptr,
  int B, int H, int Q, int K, int D,
  int causal
) {
  int stride_hb_q = static_cast<int>(D * Q * H);
  int stride_hb_k = static_cast<int>(D * K * H);
  StrideQ stride_Q = cute::make_stride(D, cute::_1{}, cute::make_stride(static_cast<int>(D*Q), stride_hb_q));
  StrideK stride_K = cute::make_stride(D, cute::_1{}, cute::make_stride(static_cast<int>(D*K), stride_hb_k));
  StrideV stride_V = stride_K;
  StrideO stride_O = stride_Q;
  StrideLSE stride_LSE = cute::make_stride(cute::_1{}, cute::make_stride(Q, Q*H));

  ProblemShape problem_shape{Q, K, D, cute::make_tuple(H, B)};

  cutlass::KernelHardwareInfo hw_info;
  int dev = 0;
  cudaError_t st = cudaGetDevice(&dev);
  if (st != cudaSuccess) return -1;
  hw_info.device_id = dev;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  size_t lse_elems = static_cast<size_t>(B) * static_cast<size_t>(H) * static_cast<size_t>(Q);
  Accum* lse_ptr = nullptr;
  if (cudaMalloc(&lse_ptr, lse_elems * sizeof(Accum)) != cudaSuccess) return -6;

  typename Operation::Arguments arguments{
    problem_shape,
    { reinterpret_cast<Element*>(q_ptr), stride_Q,
      reinterpret_cast<Element*>(k_ptr), stride_K,
      reinterpret_cast<Element*>(v_ptr), stride_V },
    { reinterpret_cast<Element*>(o_ptr), stride_O,
      reinterpret_cast<Accum*>(lse_ptr), stride_LSE },
    hw_info
  };

  Operation op;
  auto status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) { cudaFree(lse_ptr); return -2; }

  size_t workspace_size = Operation::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) { cudaFree(lse_ptr); return -3; }
  }

  status = op.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) { if (workspace) cudaFree(workspace); cudaFree(lse_ptr); return -4; }
  status = op.run();
  if (workspace) cudaFree(workspace);
  cudaFree(lse_ptr);
  if (status != cutlass::Status::kSuccess) return -5;
  return 0;
}


