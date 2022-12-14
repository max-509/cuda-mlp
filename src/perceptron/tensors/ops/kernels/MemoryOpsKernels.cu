#include "perceptron/tensors/ops/kernels/MemoryOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned_owner = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::copy_kernel_imlp<<<threads, blocks, shared_mem, stream>>>(*src_pinned_owner, *dst_pinned_owner);
}

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned_owner = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::copy_kernel_imlp<<<threads, blocks, shared_mem, stream>>>(*src_pinned_owner, *dst_pinned_owner);
}

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned_owner = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::copy_kernel_imlp<<<threads, blocks, shared_mem, stream>>>(*src_pinned_owner, *dst_pinned_owner);
}

void
copy_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned_owner = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::copy_kernel_imlp<<<threads, blocks, shared_mem, stream>>>(*src_pinned_owner, *dst_pinned_owner);
}

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           float value, TensorWriteable2D<float> dst) {
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::set_kernel_impl<<<threads, blocks, shared_mem, stream>>>(value, *dst_pinned_owner);
}

void
set_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           double value, TensorWriteable2D<double> dst) {
  auto &&dst_pinned_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::set_kernel_impl<<<threads, blocks, shared_mem, stream>>>(value, *dst_pinned_owner);
}

} // perceptron
} // tensors
} // ops
} // kernels
