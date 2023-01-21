#include "perceptron/tensors/ops/kernels/MathOpsKernels.cuh"

#include "perceptron/tensors/ops/kernels/impl/MathOpsKernelsImpl.cuh"
#include "perceptron/common/functions/MathFunctions.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

template<typename T, bool trans, typename TransformOp, typename ReduceOp, size_type block_size = utils::DEFAULT_BLOCK_SIZE_2D>
void
reduce_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                   TransformOp transform_op, ReduceOp reduce_op,
                   TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::reduce_kernel_on_device<T,
                                   trans,
                                   TransformOp,
  ReduceOp, block_size ><<<blocks, threads, shared_mem, stream>>>(
      transform_op, reduce_op,
          src.get(), src.get_stride(),
          dst.get(), dst.get_stride(),
          src.get_nrows(), src.get_ncols());
}

template<typename T, bool trans>
T
nrm2_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                 TensorReadOnly2D<T, trans> src) {
  auto acc_blocks_reduce_owner = constructTensorOwnerDevice2D<T>(blocks.y, blocks.x, blocks.x, stream);
  auto acc_blocks_reduce_view = acc_blocks_reduce_owner.tensor_view();
  reduce_kernel_impl(blocks,
                     threads,
                     shared_mem,
                     stream,
                     []DEVICE_CALLABLE(const T &x) { return x * x; },
                     cub::Sum{},
                     src, acc_blocks_reduce_view);
  auto acc_global_reduce_ptr = utils::CudaDeviceOwner<T>(nullptr);
  std::size_t acc_global_reduce_size = 0;
  auto res_host = utils::cu_make_host_memory_unique<T>(stream);
  CUDA_CHECK(cub::DeviceReduce::Sum(static_cast<void *>(acc_global_reduce_ptr.get()),
                                    acc_global_reduce_size,
                                    acc_blocks_reduce_view.get(), res_host.get(),
                                    acc_blocks_reduce_view.get_y_dim() * acc_blocks_reduce_view.get_stride(),
                                    stream));
  acc_global_reduce_ptr = utils::cu_make_memory_unique<T>((acc_global_reduce_size / sizeof(T)) + 1, stream);

  CUDA_CHECK(cub::DeviceReduce::Sum(static_cast<void *>(acc_global_reduce_ptr.get()),
                                    acc_global_reduce_size,
                                    acc_blocks_reduce_view.get(), res_host.get(),
                                    acc_blocks_reduce_view.get_y_dim() * acc_blocks_reduce_view.get_stride(),
                                    stream));
  auto res = static_cast<T>(0.0);
  utils::cu_memcpy_async(&res, res_host.get(), 1, cudaMemcpyDefault, stream);
  return std::sqrt(res);
}

float
nrm2_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, false> src) {
  return nrm2_kernel_impl(blocks, threads, shared_mem, stream, src);
}

float
nrm2_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<float, true> src) {
  return nrm2_kernel_impl(blocks, threads, shared_mem, stream, src);
}

double
nrm2_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, false> src) {
  return nrm2_kernel_impl(blocks, threads, shared_mem, stream, src);
}

double
nrm2_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            TensorReadOnly2D<double, true> src) {
  return nrm2_kernel_impl(blocks, threads, shared_mem, stream, src);
}

template<typename T>
void
scal_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                 T alpha, TensorWriteable2D<T> x) {
  details::scal_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, x.get(), x.get_stride(), x.get_nrows(), x.get_ncols());
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorWriteable2D<float> x) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorWriteable2D<double> x) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

template<typename T, bool trans>
void
scal_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                 T alpha, TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::scal_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      alpha, src.get(), src.get_stride(),dst.get(),dst.get_stride(), src.get_nrows(), src.get_ncols());
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, src, dst);
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, src, dst);
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, src, dst);
}

void
scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, src, dst);
}

template<typename T>
void
reverse_scal_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                         T alpha, TensorWriteable2D<T> x) {
  details::reverse_scal_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, x.get(), x.get_stride(), x.get_nrows(), x.get_ncols());
}

void
reverse_scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  reverse_scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

void
reverse_scal_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  reverse_scal_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

template<typename T>
void
add_row_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    T alpha, TensorReadOnly1D<T> row, T beta, TensorWriteable2D<T> dst) {
  details::add_row_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, row.get(), row.get_stride(), beta, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
add_row_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
               float alpha, TensorReadOnly1D<float> row, float beta, TensorWriteable2D<float> dst) {
  add_row_kernel_impl(blocks, threads, shared_mem, stream, alpha, row, beta, dst);
}

void
add_row_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
               double alpha, TensorReadOnly1D<double> row, double beta, TensorWriteable2D<double> dst) {
  add_row_kernel_impl(blocks, threads, shared_mem, stream, alpha, row, beta, dst);
}

template<typename T>
void
add_col_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    T alpha, TensorReadOnly1D<T> col, T beta, TensorWriteable2D<T> dst) {
  details::add_col_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, col.get(), col.get_stride(), beta, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
add_col_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
               float alpha, TensorReadOnly1D<float> col, float beta, TensorWriteable2D<float> dst) {
  add_col_kernel_impl(blocks, threads, shared_mem, stream, alpha, col, beta, dst);
}

void
add_col_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
               double alpha, TensorReadOnly1D<double> col, double beta, TensorWriteable2D<double> dst) {
  add_col_kernel_impl(blocks, threads, shared_mem, stream, alpha, col, beta, dst);
}

template<typename T>
void
add_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                T alpha, TensorWriteable2D<T> x) {
  details::add_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, x.get(), x.get_stride(), x.get_nrows(), x.get_ncols());
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           float alpha, TensorWriteable2D<float> x) {
  add_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

void
add_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           double alpha, TensorWriteable2D<double> x) {
  add_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

template<typename T>
void
add_negative_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                         T alpha, TensorWriteable2D<T> x) {
  details::add_negative_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(alpha, x.get(), x.get_stride(), x.get_nrows(), x.get_ncols());
}

void
add_negative_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x) {
  add_negative_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

void
add_negative_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x) {
  add_negative_kernel_impl(blocks, threads, shared_mem, stream, alpha, x);
}

template<typename T, bool trans_t1, bool trans_t2>
void
element_wise_mul_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                             TensorReadOnly2D<T, trans_t1> t1,
                             TensorReadOnly2D<T, trans_t2> t2,
                             TensorWriteable2D<T> dst) {
  details::element_wise_mul_kernel_on_device<T, trans_t1, trans_t2><<<blocks, threads, shared_mem, stream>>>(
      t1.get(), t1.get_stride(), t2.get(), t2.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, t2, dst);
}

template<typename T, bool trans>
void
element_wise_mul_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                             TensorReadOnly2D<T, trans> t1, TensorWriteable2D<T> dst) {
  details::element_wise_mul_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      t1.get(), t1.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorWriteable2D<float> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, dst);
}

void
element_wise_mul_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorWriteable2D<double> dst) {
  element_wise_mul_kernel_impl(blocks, threads, shared_mem, stream, t1, dst);
}

template<typename T, bool trans>
void
exp_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::exp_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

template<typename T>
void
exp_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorWriteable2D<T> dst) {
  details::exp_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

void
exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst) {
  exp_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

template<typename T, bool trans>
void
cos_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::cos_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

template<typename T>
void
cos_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorWriteable2D<T> dst) {
  details::cos_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

void
cos_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst) {
  cos_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

template<typename T, bool trans>
void
sin_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::sin_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

template<typename T>
void
sin_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                TensorWriteable2D<T> dst) {
  details::sin_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

void
sin_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst) {
  sin_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

template<typename T, bool trans>
void
negative_exp_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                         TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst) {
  details::negative_exp_kernel_on_device<T, trans><<<blocks, threads, shared_mem, stream>>>(
      src.get(), src.get_stride(), dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, src, dst);
}

template<typename T>
inline void
negative_exp_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                         TensorWriteable2D<T> dst) {
  details::negative_exp_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(
      dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<float> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

void
negative_exp_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<double> dst) {
  negative_exp_kernel_impl(blocks, threads, shared_mem, stream, dst);
}

template<typename T>
inline void
generate_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     curandState_t *states, TensorWriteable2D<T> dst, utils::curand_uniform_tag tag) {
  if (std::abs(tag.a - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.b - 1.0) < std::numeric_limits<double>::min()) {
    auto pdf = []__device__(curandState_t *state) { return functions::uniform<T>(state); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  } else {
    auto a = tag.a;
    auto b = tag.b;
    auto pdf = [a, b]__device__(curandState_t *state) { return functions::uniform(state, a, b); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_uniform_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_uniform_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

template<typename T>
inline void
generate_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     curandState_t *states, TensorWriteable2D<T> dst, utils::curand_log_normal_tag tag) {
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    auto pdf = []__device__(curandState_t *state) { return functions::log_normal<T>(state); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  } else {
    auto mean = tag.mean;
    auto stddev = tag.stddev;
    auto pdf = [mean, stddev]__device__(curandState_t *state) { return functions::log_normal(state, mean, stddev); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_log_normal_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_log_normal_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

template<typename T>
inline void
generate_kernel_impl(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                     curandState_t *states, TensorWriteable2D<T> dst, utils::curand_normal_tag tag) {
  if (std::abs(tag.mean - 0.0) < std::numeric_limits<double>::min() &&
      std::abs(tag.stddev - 1.0) < std::numeric_limits<double>::min()) {
    auto pdf = []__device__(curandState_t *state) { return functions::normal<T>(state); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  } else {
    auto mean = tag.mean;
    auto stddev = tag.stddev;
    auto pdf = [mean, stddev]__device__(curandState_t *state) { return functions::normal(state, mean, stddev); };
    details::generate_kernel_on_device<<<blocks, threads, shared_mem, stream>>>(pdf, states, dst.get(), dst.get_stride(), dst.get_nrows(), dst.get_ncols());
  }
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<float> dst, utils::curand_normal_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

void
generate_kernel(dim3 blocks, dim3 threads, size_type shared_mem, cudaStream_t stream,
                curandState_t *states, TensorWriteable2D<double> dst, utils::curand_normal_tag tag) {
  generate_kernel_impl(blocks, threads, shared_mem, stream, states, dst, tag);
}

} // perceptron
} // tensors
} // ops
} // kernels
