#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_CONSTRUCTOPSKERNELS_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_CONSTRUCTOPSKERNELS_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
linspace_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                float start, float end, size_type count, TensorWriteable1D<float> output);

void
linspace_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                double start, double end, size_type count, TensorWriteable1D<double> output);

void
arange_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                float start, float end, float step, TensorWriteable1D<float> output);

void
arange_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                double start, double end, double step, TensorWriteable1D<double> output);

} // perceptron
} // tenrors
} // ops
} // kernels

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_CONSTRUCTOPSKERNELS_CUH
