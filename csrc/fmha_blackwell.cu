#define CUTE_ARCH_TMA_SM90_ENABLED 1
#define CUTE_ARCH_TMA_SM100_ENABLED 1
#define CUTE_ARCH_TCGEN05_TMEM_ENABLED 1
// Hard-disable CUTLASS synclog paths for standalone extension builds
#ifdef CUTLASS_ENABLE_SYNCLOG
#undef CUTLASS_ENABLE_SYNCLOG
#endif
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdexcept>
#include <vector>

// CUTLASS FMHA includes (from cloned CUTLASS in third_party)
#include "cutlass/cutlass.h"

#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "device/fmha.hpp"

using Element = cutlass::bfloat16_t;
using Accum = float;

// Problem shape aliases following CUTLASS examples
// Q K D (H B)
using ProblemShape = cute::tuple<int, int, int, cute::tuple<int, int>>;
using StrideQ = cute::tuple<int, cute::_1, cute::tuple<int, int>>;  // Q D (H B)
using StrideK = StrideQ;
using StrideV = StrideQ;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<cute::_1, cute::tuple<int, int>>;     // Q (H B)

// Mainloop and Operation definitions for SM100
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>; // CTA tile MxNxK (M:Q, N:D, K:K)
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

static inline void check_shape(const at::Tensor& t, int64_t last_dim) {
  TORCH_CHECK(t.dim() == 4, "Expected [B, H, S, D] tensor");
  TORCH_CHECK(t.size(3) == last_dim, "HeadDim mismatch");
}

// Minimal forward FMHA wrapper for tensors shaped [B, H, S, D]
at::Tensor fmha_forward_bf16(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                             bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be CUDA");
  TORCH_CHECK(q.dtype() == at::kBFloat16 && k.dtype() == at::kBFloat16 && v.dtype() == at::kBFloat16,
              "Tensors must be bfloat16");

  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();

  int B = q_c.size(0);
  int H = q_c.size(1);
  int SQ = q_c.size(2);
  int D = q_c.size(3);
  int SK = k_c.size(2);

  check_shape(q_c, D);
  check_shape(k_c, D);
  check_shape(v_c, D);

  // Strides: [S, D, (H, B)] with layouts matching examples
  StrideQ stride_Q = cute::make_stride(D, cute::_1{}, cute::make_stride(D*SQ, D*SQ*H));
  StrideK stride_K = cute::make_stride(D, cute::_1{}, cute::make_stride(D*SK, D*SK*H));
  StrideV stride_V = stride_K;
  StrideO stride_O = cute::make_stride(D, cute::_1{}, cute::make_stride(D*SQ, D*SQ*H));
  StrideLSE stride_LSE = cute::make_stride(cute::_1{}, cute::make_stride(SQ, SQ*H));

  // Allocate outputs
  auto opts = q.options();
  at::Tensor out = at::empty({B, H, SQ, D}, opts);
  at::Tensor lse = at::empty({B, H, SQ}, opts.dtype(at::kFloat));

  // Build CUTLASS arguments
  ProblemShape problem_shape{SQ, SK, D, cute::make_tuple(H, B)};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = at::cuda::current_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using DevPtr = void*;
  typename Operation::Arguments arguments{
    problem_shape,
    { reinterpret_cast<Element*>(q_c.data_ptr()), stride_Q,
      reinterpret_cast<Element*>(k_c.data_ptr()), stride_K,
      reinterpret_cast<Element*>(v_c.data_ptr()), stride_V },
    { reinterpret_cast<Element*>(out.data_ptr()), stride_O,
      reinterpret_cast<Accum*>(lse.data_ptr()), stride_LSE },
    hw_info
  };

  Operation op;
  auto status = op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "FMHA kernel not supported for given shapes");

  size_t workspace_size = Operation::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({static_cast<long>(workspace_size)}, q.options().dtype(at::kByte));

  status = op.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize FMHA op");

  status = op.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run FMHA op");

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Blackwell FMHA forward (CUTLASS)";
  m.def("fmha_forward_bf16", &fmha_forward_bf16, "FMHA forward bf16",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("causal") = false);
}


