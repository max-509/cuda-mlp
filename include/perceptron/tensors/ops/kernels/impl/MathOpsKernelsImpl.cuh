#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH

#include "perceptron/common/Common.h"
#include "perceptron/common/functions/MathFunctions.cuh"
#include "perceptron/common/utils/CudaUtils.h"
#include "perceptron/tensors/TensorGetter.h"

#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {
namespace details {

template<typename T, bool trans, typename TransformOp, typename ReduceOp, size_type block_size = utils::DEFAULT_BLOCK_SIZE_2D>
__global__
void
reduce_kernel_on_device(TransformOp transform_op, ReduceOp reduce_op,
                        const T *__restrict__ src, size_type src_pitch,
                        T *__restrict__ dst, size_type dst_pitch,
                        size_type nrows, size_type ncols) {
  using BlockReduce = cub::BlockReduce<T, block_size, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_size>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto val = static_cast<T>(0.0);
  if (x_idx < nrows && y_idx < ncols) {
    val = transform_op(*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch));
  }

  auto block_acc = BlockReduce(temp_storage).Reduce(val, reduce_op);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    *tensors::get_elem(dst, blockIdx.y, blockIdx.x, dst_pitch) = block_acc;
  }
}

template<typename T>
__global__
void
scal_kernel_on_device(T alpha, T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem = (*elem) * alpha;
  }
}

template<typename T, bool trans>
__global__
void
scal_kernel_on_device(T alpha,
                      const T *__restrict__ src, size_type src_pitch,
                      T *__restrict__ dst, size_type dst_pitch,
                      size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    auto src_elem = *tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch);
    *dst_elem = alpha * src_elem;
  }
}

template<typename T>
__global__
void
reverse_scal_kernel_on_device(T alpha, T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem = alpha / (*elem);
  }
}

template<typename T>
__global__
void
add_row_kernel_on_device(T alpha, const T *__restrict__ row, size_type row_pitch,
                         T beta, T *__restrict__ dst, size_type dst_pitch,
                         size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    *dst_elem = beta * (*dst_elem) + alpha * (*tensors::get_elem(row, x_idx, row_pitch));
  }
}

template<typename T>
__global__
void
add_col_kernel_on_device(T alpha, const T *__restrict__ col, size_type col_pitch,
                         T beta, T *__restrict__ dst, size_type dst_pitch,
                         size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    *dst_elem = beta * (*dst_elem) + alpha * (*tensors::get_elem(col, y_idx, col_pitch));
  }
}

template<typename T>
__global__
void
add_kernel_on_device(T alpha, T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem = (*elem) + alpha;
  }
}

template<typename T>
__global__
void
add_negative_kernel_on_device(T alpha, T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem = alpha - (*elem);
  }
}

template<typename T, bool trans_t1, bool trans_t2>
__global__
void
element_wise_mul_kernel_on_device(const T *__restrict__ t1, size_type t1_pitch,
                                  const T *__restrict__ t2, size_type t2_pitch,
                                  T *__restrict__ dst, size_type dst_pitch,
                                  size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, y_idx, x_idx, dst_pitch) =
        ((*tensors::get_elem<const T, trans_t1>(t1, y_idx, x_idx, t1_pitch)) *
            (*tensors::get_elem<const T, trans_t2>(t2, y_idx, x_idx, t2_pitch)));
  }
}

template<typename T, bool trans>
__global__
void
element_wise_mul_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                                  T *__restrict__ dst, size_type dst_pitch,
                                  size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    *dst_elem = (*dst_elem) * (*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch));
  }
}

template<typename T, bool trans_t1, bool trans_t2>
__global__
void
element_wise_div_kernel_on_device(const T *__restrict__ t1, size_type t1_pitch,
                                  const T *__restrict__ t2, size_type t2_pitch,
                                  T *__restrict__ dst, size_type dst_picth,
                                  size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, y_idx, x_idx, dst_picth) =
        ((*tensors::get_elem<const T, trans_t1>(t1, y_idx, x_idx, t1_pitch)) /
            (*tensors::get_elem<const T, trans_t2>(t2, y_idx, x_idx, t2_pitch)));
  }
}

template<typename T, bool trans>
__global__
void
element_wise_div_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                                  T *__restrict__ dst, size_type dst_pitch,
                                  size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto dst_elem = tensors::get_elem(dst, y_idx, x_idx, dst_pitch);
    *dst_elem = *dst_elem / tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch);
  }
}

template<typename T, bool trans>
__global__
void
exp_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                     T *__restrict__ dst, size_type dst_pitch,
                     size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, nrows, ncols, dst_pitch) =
        functions::exp(*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch));
  }
}

template<typename T>
__global__
void
exp_kernel_on_device(T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem_ptr = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem_ptr = functions::exp(*elem_ptr);
  }
}

template<typename T, bool trans>
__global__
void
cos_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                     T *__restrict__ dst, size_type dst_pitch,
                     size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, nrows, ncols, dst_pitch) =
        functions::cos(*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch));
  }
}

template<typename T>
__global__
void
cos_kernel_on_device(T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem_ptr = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem_ptr = functions::cos(*elem_ptr);
  }
}

template<typename T, bool trans>
__global__
void
sin_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                     T *__restrict__ dst, size_type dst_pitch,
                     size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, nrows, ncols, dst_pitch) =
        functions::sin(*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch));
  }
}

template<typename T>
__global__
void
sin_kernel_on_device(T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem_ptr = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem_ptr = functions::sin(*elem_ptr);
  }
}

template<typename T, bool trans>
__global__
void
negative_exp_kernel_on_device(const T *__restrict__ src, size_type src_pitch,
                              T *__restrict__ dst, size_type dst_pitch,
                              size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    *tensors::get_elem(dst, y_idx, x_idx, dst_pitch) =
        functions::exp(-(*tensors::get_elem<const T, trans>(src, y_idx, x_idx, src_pitch)));
  }
}

template<typename T>
__global__
void
negative_exp_kernel_on_device(T *__restrict__ dst, size_type pitch, size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto elem_ptr = tensors::get_elem(dst, y_idx, x_idx, pitch);
    *elem_ptr = functions::exp(-(*elem_ptr));
  }
}

template<typename T, typename Pdf>
__global__
void
generate_kernel_on_device(Pdf pdf, curandState_t *__restrict__ states,
                          T *__restrict__ dst, size_type pitch,
                          size_type nrows, size_type ncols) {
  const auto y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const auto x_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx < nrows && x_idx < ncols) {
    auto state_idx = y_idx * ncols + x_idx;
    auto local_state = states[state_idx];
    *tensors::get_elem(dst, y_idx, x_idx, pitch) = pdf(&local_state);
    states[state_idx] = local_state;
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_MATHOPSKERNELSIMPL_CUH


