#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh"

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

template<typename T, bool trans_src, typename Predicate>
void
copy_if_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
               TensorReadOnly2D<T, trans_src> src, Predicate &&predicate, TensorWriteable2D<T> dst) {
  auto &&src_pinned_owner = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::copy_if_kernel_imlp<<<threads, blocks, shared_mem, stream>>>(*src_pinned_owner, device_forward<Predicate>(predicate), *dst_pinned_owner);
}

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           float value, TensorWriteable2D<float> dst);

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           double value, TensorWriteable2D<double> dst);

template<typename T, typename Predicate>
void
set_if_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
              T value, Predicate &&predicate, TensorWriteable2D<T> dst) {
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::set_if_kernel_impl<<<threads, blocks, shared_mem, stream>>>(value, device_forward<Predicate>(predicate), *dst_pinned_owner);
}

} // perceptron
} // tensors
} // ops
} // kernels

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_MEMORYOPSDEVICE_CUH
