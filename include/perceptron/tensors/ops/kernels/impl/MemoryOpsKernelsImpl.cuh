#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MEMORYOPSKERNELSIMPL_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MEMORYOPSKERNELSIMPL_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {
namespace details {

template<typename T, bool trans_src>
__global__
void
copy_kernel_imlp(const TensorReadOnly2D<T, trans_src> &src,
                 TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_nrows() && x_idx < src.get_ncols()) {
    dst(y_idx, x_idx) = src(y_idx, x_idx);
  }
}

template<typename T, bool trans_src, typename Predicate>
__global__
void
copy_if_kernel_imlp(const TensorReadOnly2D<T, trans_src> &src, const Predicate &predicate,
                 TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_nrows() && x_idx < src.get_ncols()) {
    dst(y_idx, x_idx) = src(y_idx, x_idx) * static_cast<T>(predicate(dst(y_idx, x_idx)));
  }
}

template<typename T>
__global__
void
set_kernel_impl(T value, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = value;
  }
}

template<typename T, typename Predicate>
__global__
void
set_if_kernel_impl(T value, const Predicate &predicate, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = value * static_cast<T>(predicate(dst(y_idx, x_idx)));
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MEMORYOPSKERNELSIMPL_CUH
