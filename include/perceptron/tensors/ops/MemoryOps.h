#ifndef PERCEPTRON_TENSORS_OPS_MEMORYOPS_H
#define PERCEPTRON_TENSORS_OPS_MEMORYOPS_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/kernels/MemoryOpsKernels.cuh"
#include "perceptron/common/utils/CudaUtils.h"
#include "perceptron/common/utils/MemoryUtils.h"

#include <cuda_runtime.h>

#include <utility>

namespace perceptron {
namespace tensors {
namespace ops {

template<typename T, bool trans_src>
void
copy(TensorReadOnly2D<T, trans_src> src,
     TensorWriteable2D<T> dst,
     cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  if constexpr (trans_src) {
    dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D,
                 utils::DEFAULT_BLOCK_SIZE_2D);
    dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
                utils::block_size_by_threads(src.get_nrows(), threads.y));

    kernels::copy_kernel(blocks, threads, 0, stream, src, dst);
  } else {
    utils::cu_memcpy2D_async(dst.get(), dst.get_stride(),
                             src.get(), src.get_stride(),
                             src.get_ncols(), src.get_nrows(),
                             cudaMemcpyDefault,
                             stream);
  }
}

template<typename T, bool trans_src>
TensorOwner2D<T>
copy(TensorReadOnly2D<T, trans_src> src,
     cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice2D<T>(src.get_y_dim(), src.get_x_dim(), DEFAULT_2D_STRIDE, stream);
  copy(src, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T, bool trans_src, typename Predicate>
void
copy_or_zero(TensorReadOnly2D<T, trans_src> src,
             Predicate &&predicate,
             TensorWriteable2D<T> dst,
             cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::copy_or_zero_kernel(blocks, threads, 0, stream,
                               src, utils::device_forward<Predicate>(predicate), dst);
}

template<typename T, bool trans_src, typename Predicate>
TensorOwner2D<T>
copy_or_zero(TensorReadOnly2D<T, trans_src> src,
             Predicate &&predicate,
             cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(src.get_nrows(), src.get_ncols(), DEFAULT_2D_STRIDE, stream);
  copy_or_zero(src, utils::device_forward<Predicate>(predicate), dst_owner.tensor_view(), stream);
  return dst_owner;
}

template<typename T, bool trans_src, typename Predicate, typename Transformer>
void
copy_or_transform(TensorReadOnly2D<T, trans_src> src,
                  Predicate &&predicate,
                  Transformer &&transformer,
                  TensorWriteable2D<T> dst,
                  cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::copy_or_transform_kernel(blocks, threads, 0, stream,
                                    src,
                                    utils::device_forward<Predicate>(predicate),
                                    utils::device_forward<Transformer>(transformer),
                                    dst);
}

template<typename T, bool trans>
void
copy_rows_by_indices(TensorReadOnly2D<T, trans> src,
                     TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<T> dst,
                     cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::copy_rows_by_indices(blocks, threads, 0, stream,
                                src, indices, dst);
}

template<typename T>
void
set(T value,
    TensorWriteable2D<T> dst,
    cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set_kernel(blocks, threads, 0, stream,
                      value, dst);
}

template<typename T>
TensorOwner2D<T>
set(T value,
    size_type nrows, size_type ncols,
    size_type stride = DEFAULT_2D_STRIDE, cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(nrows, ncols, stride, stream);
  set(value, dst_owner.tensor_view(), stream);
  return dst_owner;
}

template<typename T>
void
set(int value,
    TensorWriteable2D<T> dst,
    cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  utils::cu_memset2D_async(dst.get(),
                           dst.get_stride(),
                           value,
                           dst.get_ncols(),
                           dst.get_nrows(),
                           stream);
}

template<typename T>
TensorOwner2D<T>
set(int value,
    size_type nrows, size_type ncols,
    size_type stride = DEFAULT_2D_STRIDE, cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(nrows, ncols, stride, stream);
  set(value, dst_owner.tensor_view(), stream);
  return dst_owner;
}

template<typename T, typename Predicate>
void
set_or_zero(T value,
            Predicate &&predicate,
            TensorWriteable2D<T> dst,
            cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set_or_zero_kernel(blocks, threads, 0, stream,
                              value, utils::device_forward<Predicate>(predicate), dst);
}

template<typename T, bool trans, typename Predicate>
void
set_or_zero(T value,
            TensorReadOnly2D<T, trans> src,
            Predicate &&predicate,
            TensorWriteable2D<T> dst,
            cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set_or_zero_kernel(blocks, threads, 0, stream,
                              value, src, utils::device_forward<Predicate>(predicate), dst);
}

template<typename T, bool trans, typename Predicate>
void
set1_or_set2(T value1, T value2,
             TensorReadOnly2D<T, trans> src,
             Predicate &&predicate,
             TensorWriteable2D<T> dst,
             cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D,
               utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set1_or_set2_kernel(blocks, threads, 0, stream,
                               value1, value2, src, utils::device_forward<Predicate>(predicate), dst);
}

template<typename T, typename Predicate>
void
set1_or_set2(T value1, T value2,
             Predicate &&predicate,
             TensorWriteable2D<T> dst,
             cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D,
               utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set1_or_set2_kernel(blocks, threads, 0, stream,
                               value1, value2, utils::device_forward<Predicate>(predicate), dst);
}

} // perceptron
} // tensors
} // ops

#endif //PERCEPTRON_TENSORS_OPS_MEMORYOPS_H
