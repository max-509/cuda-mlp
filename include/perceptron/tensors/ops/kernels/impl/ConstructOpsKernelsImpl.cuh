#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_CONSTRUCTOPSKERNELSIMPL_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_CONSTRUCTOPSKERNELSIMPL_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/TensorGetter.h"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {
namespace details {

template<typename T>
__global__
void
arange_kernel_on_device(T begin, T h,
                        T *__restrict__ dst, size_type size, size_type pitch) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    *tensors::get_elem(dst, idx, pitch) = begin + h * idx;
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
} // details

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_IMPL_CONSTRUCTOPSKERNELSIMPL_CUH
