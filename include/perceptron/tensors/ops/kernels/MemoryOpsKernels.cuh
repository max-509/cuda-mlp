#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst);

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst);

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst);

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst);

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           float value, TensorWriteable2D<float> dst);

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           double value, TensorWriteable2D<double> dst);

} // perceptron
} // tensors
} // ops
} // kernels

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
