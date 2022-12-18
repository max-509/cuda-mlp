#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH

#include "perceptron/common/Common.h"
#include "perceptron/common/functions/MathFunctions.cuh"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

#include <cuda_runtime.h>

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

  if (y_idx < x.get_nrows() && x_idx < x.get_ncols()) {
    x(y_idx, x_idx) *= alpha;
  }
}

template<typename T>
__global__
void
reverse_scal_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_nrows() && x_idx < x.get_ncols()) {
    x(y_idx, x_idx) = alpha / x(y_idx, x_idx);
  }
}

template<typename T>
__global__
void
add_kernel_impl(T alpha, TensorReadOnly1D<T> &x, T beta, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = beta * dst(y_idx, x_idx) + alpha * x(x_idx);
  }
}

template<typename T>
__global__
void
add_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_nrows() && x_idx < x.get_ncols()) {
    x(y_idx, x_idx) += alpha;
  }
}

template<typename T>
__global__
void
add_negative_kernel_impl(T alpha, TensorWriteable2D<T> &x) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < x.get_nrows() && x_idx < x.get_ncols()) {
    x(y_idx, x_idx) = alpha - x(y_idx, x_idx);
  }
}

template<typename T, bool trans_t1, bool trans_t2>
__global__
void
element_wise_mul_kernel_impl(TensorReadOnly2D<T, trans_t1> &t1,
                             TensorReadOnly2D<T, trans_t2> &t2,
                             TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = t1(y_idx, x_idx) * t2(y_idx, x_idx);
  }
}

template<typename T, bool trans_t1>
__global__
void
element_wise_mul_kernel_impl(TensorReadOnly2D<T, trans_t1> &t1,
                             TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) *= t1(y_idx, x_idx);
  }
}

template<typename T, bool trans_t1, bool trans_t2>
__global__
void
element_wise_div_kernel_impl(TensorReadOnly2D<T, trans_t1> &t1,
                             TensorReadOnly2D<T, trans_t2> &t2,
                             TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = t1(y_idx, x_idx) / t2(y_idx, x_idx);
  }
}

template<typename T, bool trans_t1>
__global__
void
element_wise_div_kernel_impl(TensorReadOnly2D<T, trans_t1> &t1,
                             TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) /= t1(y_idx, x_idx);
  }
}

template<typename T, bool trans>
__global__
void
exp_kernel_impl(TensorReadOnly2D<T, trans> &src,
                TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < src.get_nrows() && x_idx < src.get_ncols()) {
    dst(y_idx, x_idx) = common::functions::exp(src(y_idx, x_idx));
  }
}

template<typename T>
__global__
void
exp_kernel_impl(TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
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

  if (y_idx < src.get_nrows() && x_idx < src.get_ncols()) {
    dst(y_idx, x_idx) = common::functions::exp(-src(y_idx, x_idx));
  }
}

template<typename T>
__global__
void
negative_exp_kernel_impl(TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < dst.get_nrows() && x_idx < dst.get_ncols()) {
    dst(y_idx, x_idx) = common::functions::exp(-dst(y_idx, x_idx));
  }
}

template<typename T>
__global__
void
generate_uniform_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst,
                             T a, T b) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::uniform(&local_state, a, b);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

template<typename T>
__global__
void
generate_uniform_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::uniform<T>(&local_state);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

template<typename T>
__global__
void
generate_normal_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst,
                            T mean, T stddev) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::normal(&local_state, mean, stddev);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

template<typename T>
__global__
void
generate_normal_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::normal<T>(&local_state);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

template<typename T>
__global__
void
generate_log_normal_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst,
                                T mean, T stddev) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::log_normal(&local_state, mean, stddev);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

template<typename T>
__global__
void
generate_log_normal_kernel_impl(curandState_t *states, TensorWriteable2D<T> &dst) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto local_state = states[y_idx * dst.get_ncols() + x_idx];
  dst(y_idx, x_idx) = common::functions::log_normal<T>(&local_state);
  states[y_idx * dst.get_ncols() + x_idx] = local_state;
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH


