#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/CudaUtils.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh"

#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst);

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst);

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst);

void
copy_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst);

template<typename T, bool trans_src, typename Predicate>
void
copy_or_zero_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<T, trans_src> src, Predicate &&predicate, TensorWriteable2D<T> dst) {
  details::copy_or_zero_kernel_on_device<T, trans_src, Predicate><<<blocks, threads, shared_mem, stream>>>(
      predicate,
          src.get(), src.get_stride(),
          dst.get(), dst.get_stride(),
          src.get_nrows(), src.get_ncols());
}

template<typename T, bool trans_src, typename Predicate, typename Transformer>
void
copy_or_transform_kernel(dim3 blocks,
                         dim3 threads,
                         size_type shared_mem,
                         cudaStream_t stream,
                         TensorReadOnly2D<T, trans_src> src,
                         Predicate &&predicate,
                         Transformer &&transformer,
                         TensorWriteable2D<T> dst) {
  details::copy_or_transform_kernel_on_device<T, trans_src><<<blocks, threads, shared_mem, stream>>>(
      predicate, transformer, src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<float, false> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<float> dst);

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<float, true> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<float> dst);

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<double, false> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<double> dst);

void
copy_rows_by_indices(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     TensorReadOnly2D<double, true> src, TensorReadOnly1D<size_type> indices,
                     TensorWriteable2D<double> dst);

void
set_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           float value, TensorWriteable2D<float> dst);

void
set_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           double value, TensorWriteable2D<double> dst);

template<typename T, typename Predicate>
void
set_or_zero_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                   T value, Predicate &&predicate, TensorWriteable2D<T> dst) {
  details::set_or_zero_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      value, predicate, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

template<typename T, bool trans, typename Predicate>
void
set_or_zero_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                   T value, TensorReadOnly2D<T, trans> src, Predicate &&predicate, TensorWriteable2D<T> dst) {
  details::set_or_zero_kernel_on_device<T, trans, Predicate><<<blocks, threads, shared_mem, stream>>>(
      value, predicate,
          src.get(), src.get_stride(),
          dst.get(), dst.get_stride(),
          src.get_nrows(), src.get_ncols());
}

template<typename T, bool trans, typename Predicate>
void
set1_or_set2_kernel(dim3 blocks,
                    dim3 threads,
                    size_type shared_mem,
                    cudaStream_t stream,
                    T value1,
                    T value2,
                    TensorReadOnly2D<T, trans> src,
                    Predicate &&predicate,
                    TensorWriteable2D<T> dst) {
  details::set1_or_set2_kernel_on_device<T, trans, Predicate><<<blocks, threads, shared_mem, stream>>>(
      value1, value2, utils::device_forward<Predicate>(predicate),
          src.get(), src.get_stride(),
          dst.get(), dst.get_stride(),
          src.get_nrows(), src.get_ncols());
}

template<typename T, typename Predicate>
void
set1_or_set2_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    T value1, T value2, Predicate &&predicate, TensorWriteable2D<T> dst) {
  details::set1_or_set2_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      value1, value2, utils::device_forward<Predicate>(predicate),
          dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

} // perceptron
} // tensors
} // ops
} // kernels

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
