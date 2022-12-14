#include "perceptron/tensors/ops/kernels/MathOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/MathOpsKernelsImpl.cuh"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::scal_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::scal_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
reverse_scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::reverse_scal_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
reverse_scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::reverse_scal_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_negative_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_negative_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_negative_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_negative_kernel_impl<<<threads, blocks, shared_mem, stream>>>(alpha, *x_pinned);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*dst_pinned);
}

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<float> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*dst_pinned);
}

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<double> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<threads, blocks, shared_mem, stream>>>(*dst_pinned);
}

} // perceptron
} // tensors
} // ops
} // kernels
