#include "perceptron/common/utils/CuBLASUtils.h"

namespace perceptron {
namespace utils {

cublasOperation_t
trans2operation(bool trans) {
  return trans ? CUBLAS_OP_T : CUBLAS_OP_N;
}

cublasOperation_t
inverse_trans(cublasOperation_t trans) {
  return trans == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T;
}

std::unique_ptr<cublasHandle_t, CuBLASHandle::cublas_handle_deleter> CuBLASHandle::m_handle{};

cublasHandle_t
CuBLASHandle::getInstance() {
  if (nullptr == m_handle) {
    m_handle = cublas_init_handle();
  }

  return *m_handle;
}

std::unique_ptr<cublasHandle_t, CuBLASHandle::cublas_handle_deleter>
CuBLASHandle::cublas_init_handle() {
  std::unique_ptr<cublasHandle_t> handle_ptr{new cublasHandle_t};
  CUBLAS_CHECK(cublasCreate(handle_ptr.get()));

  return std::unique_ptr<cublasHandle_t, cublas_handle_deleter>{handle_ptr.release(), cublas_handle_deleter{}};
}

void CuBLASHandle::set_stream(cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(getInstance(), stream));
}

} // perceptron
} // utils
