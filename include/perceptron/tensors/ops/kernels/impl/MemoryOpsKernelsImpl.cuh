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
copy_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                      T *__restrict__ dst, size_type dst_pitch,
                      size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, y_idx, x_idx, dst_pitch) =
        *tensors::get_elem<const T, trans_src>(src, y_idx, x_idx, src_pitch);
  }
}

template<typename T, bool trans_src, typename Predicate>
__global__
void
copy_or_zero_kernel_imlp(Predicate predicate,
                         const T *__restrict__ src, size_type src_pitch,
                         T *__restrict__ dst, size_type dst_pitch,
                         size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    auto src_elem = *tensors::get_elem<const T, trans_src>(src, y_idx, x_idx, src_pitch);
    auto pred_val = static_cast<T>(predicate(src_elem));
    *dst_elem = src_elem * pred_val;
  }
}

template<typename T, bool trans_src, typename Predicate, typename Transformer>
__global__
void
copy_or_transform_kernel_imlp(Predicate predicate, Transformer transformer,
                              const T *__restrict__ src, size_type src_pitch,
                              T *__restrict__ dst, size_type dst_pitch,
                              size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    auto src_elem = *tensors::get_elem<const T, trans_src>(src, y_idx, x_idx, src_pitch);
    auto pred_val = static_cast<T>(predicate(src_elem));
    *dst_elem = pred_val * src_elem + (static_cast<T>(1.0) - pred_val) * transformer(src_elem);
  }
}

template<typename T>
__global__
void
set_kernel_on_device(T value, T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, y_idx, x_idx, pitch) = value;
  }
}

template<typename T, typename Predicate>
__global__
void
set_or_zero_kernel_on_device(T value, Predicate predicate,
                             T *__restrict__ dst, size_type pitch,
                             size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem = value * static_cast<T>(predicate(*elem));
  }
}

template<typename T, bool trans, typename Predicate>
__global__
void
set_or_zero_kernel_on_device(T value, Predicate predicate,
                             const T *__restrict__ src, size_type src_pitch,
                             T *__restrict__ dst, size_type dst_pitch,
                             size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    auto src_elem = *tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch);
    *dst_elem = value * static_cast<T>(predicate(src_elem));
  }
}

template<typename T, bool trans, typename Predicate>
__global__
void
set1_or_set2_kernel_on_device(T value1, T value2, Predicate predicate,
                              const T *__restrict__ src, size_type src_pitch,
                              T *__restrict__ dst, size_type dst_pitch,
                              size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    auto src_elem = *tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch);
    auto pred_val = static_cast<T>(predicate(src_elem));
    *dst_elem = pred_val * value1 + (static_cast<T>(1.0) - pred_val) * value2;
  }
}

template<typename T, typename Predicate>
__global__
void
set1_or_set2_kernel_on_device(T value1, T value2, Predicate predicate,
                              T *__restrict__ dst, size_type pitch,
                              size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    auto pred_val = static_cast<T>(predicate(*elem));
    *elem = pred_val * value1 + (static_cast<T>(1.0) - pred_val) * value2;
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MEMORYOPSKERNELSIMPL_CUH
