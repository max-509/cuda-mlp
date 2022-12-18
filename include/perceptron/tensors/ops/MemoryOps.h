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
    dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
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

template<typename T, bool trans_src, typename Predicate>
void
copy_if(TensorReadOnly2D<T, trans_src> src,
        Predicate &&predicate,
        TensorWriteable2D<T> dst,
        cudaStream_t stream = nullptr) {
  is_valid_type<T>();
  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::copy_if_kernel(blocks, threads, 0, stream, src, utils::device_forward<Predicate>(predicate), dst);
}

template<typename T>
void
set(T value,
    TensorWriteable2D<T> dst,
    cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set_kernel(blocks, threads, 0, stream, value, dst);
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

template<typename T, typename Predicate>
void
set_if(T value,
       Predicate &&predicate,
       TensorWriteable2D<T> dst,
       cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::set_if_kernel(blocks, threads, 0, stream, value, utils::device_forward<Predicate>(predicate), dst);
}

} // perceptron
} // tensors
} // ops

#endif //PERCEPTRON_TENSORS_OPS_MEMORYOPS_H
