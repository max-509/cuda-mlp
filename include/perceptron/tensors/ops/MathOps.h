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
     TensorWriteable2D<T> C) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  auto cu_transa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto cu_transb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto m  = static_cast<int>(A.get_nrows());
  auto n  = static_cast<int>(A.get_ncols());
  auto k = static_cast<int>(B.get_ncols());

  if (n != B.get_nrows()) {
    std::stringstream stream;
    stream << "Bad sizes for A@B matrices multiplication: " << A.shape_repr() << " and " << B.shape_repr();
    throw std::invalid_argument{stream.str()};
  }
  if (m != C.get_nrows() && k != C.get_ncols()) {
    std::stringstream stream;
    stream << "Bad C matrix size " << C.shape_repr() << " for A@B result (" << m << ", " << k << ")";
    throw std::invalid_argument{stream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    CUBLAS_CHECK(cublasSgemm(handle,
                             cu_transa, cu_transb,
                             m, n, k,
                             &alpha,
                             A.get(), A.get_stride(),
                             B.get(), B.get_stride(),
                             &beta,
                             C.get(), C.get_stride()));
  } else {
    CUBLAS_CHECK(cublasDgemm(handle,
                             cu_transa, cu_transb,
                             m, n, k,
                             &alpha,
                             A.get(), A.get_stride(),
                             B.get(), B.get_stride(),
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
     TensorWriteable2D<T> C) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  auto cu_transa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto cu_transb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(A.get_ncols());
  if (m != B.get_nrows() || n != B.get_ncols()) {
    std::stringstream stream;
    stream << "Bad sizes for A+B summation: " << A.shape_repr() << " and " << B.shape_repr();
    throw std::invalid_argument{stream.str()};
  }
  if (m != C.get_nrows() || n != C.get_ncols()) {
    std::stringstream stream;
    stream << "Bad C matrix size " << C.shape_repr() <<  " for A+B result: (" << n << ", " << m << ")";
    throw std::invalid_argument{stream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    CUBLAS_CHECK(cublasSgeam(handle,
                             cu_transa, cu_transb,
                             m, n,
                             &alpha, A.get(), A.get_stride(),
                             &beta, B.get(), B.get_stride(),
                             C.get(), C.get_stride()));
  } else {
    CUBLAS_CHECK(cublasDgeam(handle,
                             cu_transa, cu_transb,
                             m, n,
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
     T beta) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  auto cu_transa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(A.get_ncols());
  if (m != C.get_nrows() || n != C.get_ncols()) {
    std::stringstream stream;
    stream << "Bad C matrix size " << C.shape_repr() << " for summation with matrix A with size " << A.shape_repr();
    throw std::invalid_argument{stream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    CUBLAS_CHECK(cublasSgeam(handle,
                             cu_transa, CUBLAS_OP_N,
                             m, n,
                             &alpha, A.get(), A.get_stride(),
                             &beta, C.get(), C.get_stride(),
                             C.get(), C.get_stride()));
  } else {
    CUBLAS_CHECK(cublasDgeam(handle,
                             cu_transa, CUBLAS_OP_N,
                             m, n,
                             &alpha, A.get(), A.get_stride(),
                             &beta, C.get(), C.get_stride(),
                             C.get(), C.get_stride()));
  }
}

template<typename T, bool transa>
void
gemv(T alpha,
     TensorReadOnly2D<T, transa> A,
     TensorReadOnly1D<T> x,
     T beta,
     TensorWriteable1D<T> y) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
  auto cu_transa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto m = static_cast<int>(A.get_nrows());
  auto n = static_cast<int>(A.get_ncols());
  if (n != x.get_size()) {
    std::stringstream stream;
    stream << "Bad size for A@x multiplication operation: " << A.shape_repr() << " and " << x.get_size();
    throw std::invalid_argument{stream.str()};
  }

  if constexpr (std::is_same_v<T, float>) {
    CUBLAS_CHECK(cublasSgemv(handle, cu_transa,
                             m, n,
                             &alpha,
                             A.get(), A.get_stride(),
                             x.get(), x.get_stride(),
                             &beta,
                             y.get(), y.get_stride()));
  } else {
    CUBLAS_CHECK(cublasDgemv(handle, cu_transa,
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
nrm2(TensorReadOnly1D<T> x) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
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

template<typename T>
void
scal(T alpha, TensorWriteable1D<T> x) {
  is_valid_type<T>();
  auto handle = utils::CuBLASHandle::getInstance();
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

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::scal_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T>
void
reverse_scal(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::reverse_scal_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T>
void
add(T alpha,
    TensorReadOnly1D<T> x,
    T beta,
    TensorWriteable2D<T> dst,
    cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::add_kernel(blocks, threads, 0, stream, alpha, x, beta, dst);
}

template<typename T>
void
add(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(x.get_ncols(), threads.x),
              utils::block_size_by_threads(x.get_nrows(), threads.y));

  kernels::add_kernel(blocks, threads, 0, stream, alpha, x);
}

template<typename T>
void
add_negative(T alpha, TensorWriteable2D<T> x, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
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

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::element_wise_mul_kernel(blocks, threads, 0, stream, t1, t2, dst);
}

template<typename T, bool trans_t1>
void
element_wise_mul(TensorReadOnly2D<T, trans_t1> t1,
                 TensorWriteable2D<T> dst,
                 cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::element_wise_mul_kernel(blocks, threads, 0, stream, t1, dst);
}

template<typename T, bool trans_src>
void
exp(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::exp_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T>
void
exp(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::exp_kernel(blocks, threads, 0, stream, dst);
}

template<typename T, bool trans_src>
void
negative_exp(TensorReadOnly2D<T, trans_src> src, TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(src.get_ncols(), threads.x),
              utils::block_size_by_threads(src.get_nrows(), threads.y));

  kernels::negative_exp_kernel(blocks, threads, 0, stream, src, dst);
}

template<typename T>
void
negative_exp(TensorWriteable2D<T> dst, cudaStream_t stream = nullptr) {
  is_valid_type<T>();

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
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

  dim3 threads(utils::DEFAULT_BLOCK_SIZE, utils::DEFAULT_BLOCK_SIZE);
  dim3 blocks(utils::block_size_by_threads(dst.get_ncols(), threads.x),
              utils::block_size_by_threads(dst.get_nrows(), threads.y));

  kernels::generate_kernel(blocks, threads, 0, stream, curand_states.get(), dst, tag);
}

} // perceptron
} // tensors
} // ops

#endif //PERCEPTRON_TENSORS_OPS_MATHOPS_H
