#ifndef PERCEPTRON_TENSORS_OPS_MATHOPS_H
#define PERCEPTRON_TENSORS_OPS_MATHOPS_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/common/utils/CuBLASUtils.h"
#include "perceptron/common/utils/CudaUtils.h"
#include "perceptron/common/utils/CurandUtils.h"
#include "perceptron/tensors/ops/kernels/MathOpsKernels.cuh"

#include <cublas_v2.h>

#include <sstream>
#include <cassert>
#include <type_traits>

namespace perceptron {
namespace tensors {
namespace ops {

template<typename T, bool transa, bool transb>
void
gemm(T alpha,
     TensorReadOnly2D<T, transa> A,
     TensorReadOnly2D<T, transb> B,
     T beta,
     TensorWriteable2D<T> C,
     cudaStream_t stream = nullptr) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  utils::CuBLASHandle::set_stream(stream);
  auto cu_transa = utils::trans2operation<transa>();
  auto cu_transb = utils::trans2operation<transb>();

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(B.get_ncols());
  auto k = static_cast<int>(A.get_ncols());

  if (k != B.get_nrows()) {
    std::stringstream strstream;
    strstream << "Bad sizes for A@B matrices multiplication: " << A.shape_repr() << " and " << B.shape_repr();
    throw std::invalid_argument{strstream.str()};
  }
  if (m != C.get_nrows() && n != C.get_ncols()) {
    std::stringstream strstream;
    strstream << "Bad C matrix size " << C.shape_repr() << " for A@B result (" << m << ", " << k << ")";
    throw std::invalid_argument{strstream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    // Because CuBLAS is column-majored, we solve (AB)^T = B^T A^T = C^T
    CUBLAS_CHECK(cublasSgemm(handle,
                             cu_transb, cu_transa,
                             n, m, k,
                             &alpha,
                             B.get(), B.get_stride(),
                             A.get(), A.get_stride(),
                             &beta,
                             C.get(), C.get_stride()));
  } else {
    // Because CuBLAS is column-majored, we solve (AB)^T = B^T A^T = C^T
    CUBLAS_CHECK(cublasDgemm(handle,
                             cu_transb, cu_transa,
                             n, m, k,
                             &alpha,
                             B.get(), B.get_stride(),
                             A.get(), A.get_stride(),
                             &beta,
                             C.get(), C.get_stride()));
  }
}

template<typename T, bool transa, bool transb>
void
geam(TensorReadOnly2D<T, transa> A,
     T alpha,
     TensorReadOnly2D<T, transb> B,
     T beta,
     TensorWriteable2D<T> C,
     cudaStream_t stream = nullptr) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  utils::CuBLASHandle::set_stream(stream);
  auto cu_transa = utils::trans2operation<transa>();
  auto cu_transb = utils::trans2operation<transb>();

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(A.get_ncols());
  if (m != B.get_nrows() || n != B.get_ncols()) {
    std::stringstream strstream;
    strstream << "Bad sizes for A+B summation: " << A.shape_repr() << " and " << B.shape_repr();
    throw std::invalid_argument{strstream.str()};
  }
  if (m != C.get_nrows() || n != C.get_ncols()) {
    std::stringstream strstream;
    strstream << "Bad C matrix size " << C.shape_repr() << " for A+B result: (" << n << ", " << m << ")";
    throw std::invalid_argument{strstream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    // Because of column-major format, we compute A^T + B^T = C^T
    CUBLAS_CHECK(cublasSgeam(handle,
                             cu_transa, cu_transb,
                             n, m,
                             &alpha, A.get(), A.get_stride(),
                             &beta, B.get(), B.get_stride(),
                             C.get(), C.get_stride()));
  } else {
    // Because of column-major format, we compute A^T + B^T = C^T
    CUBLAS_CHECK(cublasDgeam(handle,
                             cu_transa, cu_transb,
                             n, m,
                             &alpha, A.get(), A.get_stride(),
                             &beta, B.get(), B.get_stride(),
                             C.get(), C.get_stride()));
  }
}

template<typename T, bool transa>
void
geam(TensorReadOnly2D<T, transa> A,
     T alpha,
     TensorWriteable2D<T> C,
     T beta,
     cudaStream_t stream = nullptr) {
  geam(A, alpha, C.to_read_only(), beta, C, stream);
}

template<typename T, bool transa>
void
gemv(T alpha,
     TensorReadOnly2D<T, transa> A,
     TensorReadOnly1D<T> x,
     T beta,
     TensorWriteable1D<T> y,
     cudaStream_t stream = nullptr) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  utils::CuBLASHandle::set_stream(stream);
  auto cu_transa = utils::trans2operation<transa>();

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(A.get_ncols());
  if (n != x.get_size()) {
    std::stringstream strstream;
    strstream << "Bad size for A@x multiplication operation: " << A.shape_repr() << " and " << x.get_size();
    throw std::invalid_argument{strstream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    // We transpose of matrix A, because of column-wise
    CUBLAS_CHECK(cublasSgemv(handle, utils::inverse_trans(cu_transa),
                             m, n,
                             &alpha,
                             A.get(), A.get_stride(),
                             x.get(), x.get_stride(),
                             &beta,
                             y.get(), y.get_stride()));
  } else {
    // Because of column-major format, we compute A^T + C^T = C^T
    CUBLAS_CHECK(cublasDgemv(handle, utils::inverse_trans(cu_transa),
                             m, n,
                             &alpha,
                             A.get(), A.get_stride(),
                             x.get(), x.get_stride(),
                             &beta,
                             y.get(), y.get_stride()));
  }
}

template<typename T>
T
nrm2(TensorReadOnly1D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  auto handle = utils::CuBLASHandle::getInstance();
  utils::CuBLASHandle::set_stream(stream);
  auto n = static_cast<int>(x.get_size());
  auto incx = static_cast<int>(x.get_stride());

  T result{};
  if constexpr (std::is_same_v<float, T>) {
    CUBLAS_CHECK(cublasSnrm2(handle, n, x.get(), incx, &result));
  } else {
    CUBLAS_CHECK(cublasDnrm2(handle, n, x.get(), incx, &result));
  }

  return result;
}

template<typename T, bool trans>
T
nrm2(TensorReadOnly2D<T, trans> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  return kernels::nrm2_kernel(blocks, threads, 0, stream, x);
}

template<typename T>
void
scal(T alpha, TensorWriteable1D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  auto handle = utils::CuBLASHandle::getInstance();
  utils::CuBLASHandle::set_stream(stream);
  auto n = static_cast<int>(x.get_size());
  auto incx = static_cast<int>(x.get_stride());

  if constexpr (std::is_same_v<float, T>) {
    CUBLAS_CHECK(cublasSscal(handle, n, &alpha, x.get(), incx));
  } else {
    CUBLAS_CHECK(cublasDscal(handle, n, &alpha, x.get(), incx));
  }
}

template<typename T>
void
scal(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::scal_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T, bool trans>
void
scal(T alpha, TensorReadOnly2D<T, trans> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::scal_kernel(blocks, threads, 0, stream, alpha, src, dst);
}

template<typename T, bool trans>
TensorOwner2D<T>
scal(T alpha, TensorReadOnly2D<T, trans> src, cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(src.get_nrows(), src.get_ncols(), DEFAULT_2D_STRIDE, stream);
  scal(alpha, src, dst_owner.tensor_view(), stream);
  return dst_owner;
}

template<typename T>
void
reverse_scal(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::reverse_scal_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T>
void
add_row(T alpha,
        TensorReadOnly1D<T> row,
        T beta,
        TensorWriteable2D<T> dst,
        cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::add_row_kernel(blocks, threads, 0, stream, alpha, row, beta, dst);
}

template<typename T>
void
add_col(T alpha,
        TensorReadOnly1D<T> col,
        T beta,
        TensorWriteable2D<T> dst,
        cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::add_col_kernel(blocks, threads, 0, stream, alpha, col, beta, dst);
}

template<typename T>
void
add(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::add_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T>
void
add_negative(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::add_negative_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T, bool trans_t1, bool trans_t2>
void
element_wise_mul(TensorReadOnly2D<T, trans_t1> t1,
                 TensorReadOnly2D<T, trans_t2> t2,
                 TensorWriteable2D<T> dst,
                 cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::element_wise_mul_kernel(blocks, threads, 0, stream, t1, t2, dst);
}

template<typename T, bool trans_t1, bool trans_t2>
TensorOwner2D<T>
element_wise_mul(TensorReadOnly2D<T, trans_t1> t1,
                 TensorReadOnly2D<T, trans_t2> t2,
                 cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice2D<T>(t1.get_y_dim(), t1.get_x_dim(), DEFAULT_2D_STRIDE, stream);
  element_wise_mul(t1, t2, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T, bool trans_t1>
void
element_wise_mul(TensorReadOnly2D<T, trans_t1> t1,
                 TensorWriteable2D<T> dst,
                 cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::element_wise_mul_kernel(blocks, threads, 0, stream, t1, dst);
}

template<typename T, bool trans_src>
void
exp(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::exp_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T, bool trans_src>
TensorOwner2D<T>
exp(TensorReadOnly2D<T, trans_src> src, cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice2D<T>(src.get_y_dim(), src.get_x_dim(), DEFAULT_2D_STRIDE, stream);
  exp(src, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T>
void
exp(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::exp_kernel(blocks, threads, 0, stream, dst);
}

template<typename T, bool trans_src>
void
cos(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::cos_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T, bool trans_src>
TensorOwner2D<T>
cos(TensorReadOnly2D<T, trans_src> src, cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice2D<T>(src.get_y_dim(), src.get_x_dim(), DEFAULT_2D_STRIDE, stream);
  cos(src, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T>
void
cos(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::cos_kernel(blocks, threads, 0, stream, dst);
}

template<typename T, bool trans_src>
void
sin(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::sin_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T, bool trans_src>
TensorOwner2D<T>
sin(TensorReadOnly2D<T, trans_src> src, cudaStream_t stream = nullptr) {
  auto output_owner = constructTensorOwnerDevice2D<T>(src.get_y_dim(), src.get_x_dim(), DEFAULT_2D_STRIDE, stream);
  sin(src, output_owner.tensor_view(), stream);
  return output_owner;
}

template<typename T>
void
sin(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::sin_kernel(blocks, threads, 0, stream, dst);
}

template<typename T, bool trans_src>
void
negative_exp(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::negative_exp_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T, bool trans_src>
TensorOwner2D<T>
negative_exp(TensorReadOnly2D<T, trans_src> src, cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(src.get_nrows(), src.get_ncols(), DEFAULT_2D_STRIDE, stream);
  negative_exp(src, dst_owner.tensor_view(), stream);
  return dst_owner;
}

template<typename T>
void
negative_exp(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::negative_exp_kernel(blocks, threads, 0, stream, dst);
}

template<typename T, typename curand_distr_tag>
void
generate(curand_distr_tag tag,
         TensorWriteable2D<T> dst,
         size_type seed = 42,
         cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  auto curand_states = utils::curand_create_states(seed, dst.get_nrows() * dst.get_ncols(), stream);

  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D, utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::generate_kernel(blocks, threads, 0, stream, curand_states.get(), dst, tag);
}

template<typename T, typename curand_distr_tag>
TensorOwner2D<T>
generate(curand_distr_tag tag,
         size_type nrows, size_type ncols,
         size_type seed = 42,
         cudaStream_t stream = nullptr) {
  auto dst_owner = constructTensorOwnerDevice2D<T>(nrows, ncols, DEFAULT_2D_STRIDE, stream);
  generate(tag, dst_owner.tensor_view(), seed, stream);
  return dst_owner;
}

} // perceptron
} // tensors
} // ops

#endif //PERCEPTRON_TENSORS_OPS_MATHOPS_H
