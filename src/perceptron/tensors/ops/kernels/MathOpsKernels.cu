#include "perceptron/tensors/ops/kernels/MathOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/MathOpsKernelsImpl.cuh"

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::scal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::scal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
reverse_scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::reverse_scal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
reverse_scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::reverse_scal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           float alpha, TensorReadOnly1D<float> x, float beta, TensorWriteable2D<float> dst) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::add_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned, beta, *dst_pinned);
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           double alpha, TensorReadOnly1D<double> x, double beta, TensorWriteable2D<double> dst) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::add_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned, beta, *dst_pinned);
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_negative_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_negative_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
add_negative_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  auto &&x_pinned = utils::cu_make_pinned_memory_unique(&x);
  details::add_negative_kernel_impl<<<blocks, threads, shared_mem, stream>>>(alpha, *x_pinned);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&t2_owner = utils::cu_make_pinned_memory_unique(&t2);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *t2_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorWriteable2D<float> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorWriteable2D<double> dst) {
  auto &&t1_owner = utils::cu_make_pinned_memory_unique(&t1);
  auto &&dst_owner = utils::cu_make_pinned_memory_unique(&dst);
  details::element_wise_mul_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*t1_owner, *dst_owner);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*dst_pinned);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  auto &&src_pinned = utils::cu_make_pinned_memory_unique(&src);
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*src_pinned, *dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<float> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*dst_pinned);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<double> dst) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  details::negative_exp_kernel_impl<<<blocks, threads, shared_mem, stream>>>(*dst_pinned);
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_uniform_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.a - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.b - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_uniform_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_uniform_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned,
        static_cast<float>(tag.a), static_cast<float>(tag.b));
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_log_normal_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_log_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_log_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned,
                                                                                      static_cast<float>(tag.mean), static_cast<float>(tag.stddev));
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_normal_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned,
                                                                                  static_cast<float>(tag.mean), static_cast<float>(tag.stddev));
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_uniform_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.a - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.b - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_uniform_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_uniform_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned, tag.a, tag.b);
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_log_normal_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_log_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_log_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned, tag.mean, tag.stddev);
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_normal_tag tag) {
  auto &&dst_pinned = utils::cu_make_pinned_memory_unique(&dst);
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    details::generate_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned);
  } else {
    details::generate_normal_kernel_impl<<<blocks, threads, shared_mem, stream>>>(states, *dst_pinned, tag.mean, tag.stddev);
  }
}

} // perceptron
} // tensors
} // ops
} // kernels
