#include "perceptron/tensors/ops/kernels/ConstructOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/ConstructOpsKernelsImpl.cuh"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

template<typename T>
void
linspace_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     T begin, T end, size_type count, TensorWriteable1D<T> x) {
  auto h = (end - begin) / static_cast<T>(count - 1);
  details::arange_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(begin, h, x.get(), x.get_size(), x.get_stride());
}

void
linspace_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                float begin, float end, size_type count, TensorWriteable1D<float> x) {
  linspace_kernel_impl(blocks, threads, shared_mem, stream, begin, end, count, x);
}

void
linspace_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                double begin, double end, size_type count, TensorWriteable1D<double> x) {
  linspace_kernel_impl(blocks, threads, shared_mem, stream, begin, end, count, x);
}

template<typename T>
void
arange_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                   T start, T end, T step, TensorWriteable1D<T> output) {
  details::arange_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(start, step, output.get(), output.get_size(), output.get_stride());
}

void
arange_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
              float start, float end, float step, TensorWriteable1D<float> output) {
  arange_kernel_impl(blocks, threads, shared_mem, stream, start, end, step, output);
}

void
arange_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
              double start, double end, double step, TensorWriteable1D<double> output) {
  arange_kernel_impl(blocks, threads, shared_mem, stream, start, end, step, output);
}

} // perceptron
} // tensors
} // ops
} // kernels
