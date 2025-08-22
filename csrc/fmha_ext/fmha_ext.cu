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
  int dev = 0; if (cudaGetDevice(&dev) != cudaSuccess) return -1;

  ProblemShape problem_shape{Q, K, D, cute::make_tuple(H, B)};
  StrideQ stride_Q = cute::make_stride(D, cute::_1{}, cute::make_stride(D*Q, D*Q*H));
  StrideK stride_K = cute::make_stride(D, cute::_1{}, cute::make_stride(D*K, D*K*H));
  StrideV stride_V = stride_K;
  StrideO stride_O = stride_Q;
  StrideLSE stride_LSE = cute::make_stride(cute::_1{}, cute::make_stride(Q, Q*H));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = dev;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using DevPtr = void*;
  typename Operation::Arguments args{
    problem_shape,
    { reinterpret_cast<Element*>(q_ptr), stride_Q,
      reinterpret_cast<Element*>(k_ptr), stride_K,
      reinterpret_cast<Element*>(v_ptr), stride_V },
    { reinterpret_cast<Element*>(o_ptr), stride_O,
      nullptr, stride_LSE },
    hw_info
  };

  Operation op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return -2;

  size_t workspace_size = Operation::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) return -3;
  }
  status = op.initialize(args, workspace);
  if (status != cutlass::Status::kSuccess) { if (workspace) cudaFree(workspace); return -4; }
  status = op.run();
  if (workspace) cudaFree(workspace);
  if (status != cutlass::Status::kSuccess) return -5;
  return 0;
}


