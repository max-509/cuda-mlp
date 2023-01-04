#include "perceptron/tensors/ops/kernels/MemoryOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

template<typename T, bool trans>
void
copy_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                 TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::copy_kernel_on_device<T,
  trans
      ><<<blocks, threads, shared_mem, stream>>>(src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  copy_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  copy_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  copy_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  copy_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

template<typename T, bool trans>
void
copy_rows_by_indices_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                          TensorReadOnly2D<T, trans> src, TensorReadOnly1D<size_type> indices,
                          TensorWriteable2D<T> dst) {
  details::copy_rows_by_indices_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      src.get(), src.get_stride(), indices.get(), indices.get_stride(), dst.get(), dst.get_stride(),
          dst.get_nrows(), dst.get_ncols());
}

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<float, false> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<float> dst) {
  copy_rows_by_indices_impl(blocks, threads, shared_mem, stream, src, indices, dst);
}

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<float, true> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<float> dst) {
  copy_rows_by_indices_impl(blocks, threads, shared_mem, stream, src, indices, dst);
}

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<double, false> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<double> dst) {
  copy_rows_by_indices_impl(blocks, threads, shared_mem, stream, src, indices, dst);
}

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<double, true> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<double> dst) {
  copy_rows_by_indices_impl(blocks, threads, shared_mem, stream, src, indices, dst);
}

template<typename T>
void
set_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                T value, TensorWriteable2D<T> dst) {
  details::set_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(value, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
set_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           float value, TensorWriteable2D<float> dst) {
  set_kernel_impl(blocks, threads, shared_mem, stream, value, dst);
}

void
set_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           double value, TensorWriteable2D<double> dst) {
  set_kernel_impl(blocks, threads, shared_mem, stream, value, dst);
}

} // perceptron
} // tensors
} // ops
} // kernels
