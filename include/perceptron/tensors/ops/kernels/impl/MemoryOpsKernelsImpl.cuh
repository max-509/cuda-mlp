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
                 TensorWriteable2D <T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_y_dim() && x_idx < src.get_x_dim()) {
    dst(y_idx, x_idx) = src(y_idx, x_idx);
  }
}

template<typename T>
__global__
void
set_kernel_impl(T value, TensorWriteable2D <T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_y_dim() && x_idx < dst.get_x_dim()) {
    dst(y_idx, x_idx) = value;
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MEMORYOPSKERNELSIMPL_CUH
