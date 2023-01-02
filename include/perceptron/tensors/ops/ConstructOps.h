#ifndef PERCEPTRON_TENSORS_OPS_CONSTRUCTOPS_H
#define PERCEPTRON_TENSORS_OPS_CONSTRUCTOPS_H

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/CudaUtils.h"
#include "perceptron/tensors/ops/kernels/ConstructOpsKernels.cuh"

namespace perceptron {
namespace tensors {
namespace ops {

template<typename T>
void
linspace(T start, T end, size_type count,
         TensorWriteable1D<T> dst,
         cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_1D);
  dim3 blocks(utils::block_size_by_threads(dst.get_size(), threads.x));

  kernels::linspace_kernel(blocks, threads, 0, stream, start, end, count, dst);
}

template<typename T>
TensorOwner1D<T>
linspace(T start, T end, size_type count,
         cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice1D<T>(count, DEFAULT_1D_STRIDE, stream);
  linspace(start, end, count, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T>
void
arange(T start, T end, T step,
       TensorWriteable1D<T> dst,
       cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_1D);
  dim3 blocks(utils::block_size_by_threads(dst.get_size(), threads.x));

  kernels::arange_kernel(blocks, threads, 0, stream, start, end, step, dst);
}

template<typename T>
TensorOwner1D<T>
arange(T start, T end, T step = static_cast<T>(1.0), bool right_include = false,
       cudaStream_t stream = nullptr) {
  auto count = static_cast<size_type>((end - start) / step) + static_cast<size_type>(right_include);
  auto output_owner = constructTensorOwnerDevice1D<T>(count, DEFAULT_1D_STRIDE, stream);
  arange(start, end, step, output_owner.tensor_view(), stream);
  return output_owner;
}

} // perceptron
} // tensors
} // ops

#endif //PERCEPTRON_TENSORS_OPS_CONSTRUCTOPS_H
