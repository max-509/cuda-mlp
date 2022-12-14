#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH

#include "perceptron/common/Common.h"
#include "perceptron/common/functions/MathFunctions.cuh"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {
namespace details {

template<typename T>
__global__
void
scal_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_y_dim() && x_idx < x.get_x_dim()) {
    x(y_idx, x_idx) *= alpha;
  }
}

template<typename T>
__global__
void
reverse_scal_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_y_dim() && x_idx < x.get_x_dim()) {
    x(y_idx, x_idx) = alpha / x(y_idx, x_idx);
  }
}

template<typename T>
__global__
void
add_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_y_dim() && x_idx < x.get_x_dim()) {
    x(y_idx, x_idx) += alpha;
  }
}

template<typename T>
__global__
void
add_negative_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_y_dim() && x_idx < x.get_x_dim()) {
    x(y_idx, x_idx) = alpha - x(y_idx, x_idx);
  }
}

template<typename T, bool trans>
__global__
void
exp_kernel_impl(TensorReadOnly2D<T, trans> &src,
                TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_y_dim() && x_idx < src.get_x_dim()) {
    dst(y_idx, x_idx) = common::functions::exp(src(y_idx, x_idx));
  }
}

template<typename T>
__global__
void
exp_kernel_impl(TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_y_dim() && x_idx < dst.get_x_dim()) {
    dst(y_idx, x_idx) = common::functions::exp(dst(y_idx, x_idx));
  }
}

template<typename T, bool trans>
__global__
void
negative_exp_kernel_impl(TensorReadOnly2D<T, trans> &src,
                         TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_y_dim() && x_idx < src.get_x_dim()) {
    dst(y_idx, x_idx) = common::functions::exp(-src(y_idx, x_idx));
  }
}

template<typename T>
__global__
void
negative_exp_kernel_impl(TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_y_dim() && x_idx < dst.get_x_dim()) {
    dst(y_idx, x_idx) = common::functions::exp(-dst(y_idx, x_idx));
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH


