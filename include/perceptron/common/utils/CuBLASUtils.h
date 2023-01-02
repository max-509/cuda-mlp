#ifndef PERCEPTRON_COMMON_UTILS_CUBLASUTILS_H
#define PERCEPTRON_COMMON_UTILS_CUBLASUTILS_H

#include <cublas_v2.h>

#include <type_traits>
#include <memory>
#include <stdexcept>
#include <cstdio>

#define CUBLAS_CHECK(err)                                                           \
  do                                                                                \
  {                                                                                 \
    cublasStatus_t err_ = (err);                                                    \
    if (err_ != CUBLAS_STATUS_SUCCESS)                                              \
    {                                                                               \
      std::fprintf(stderr, "cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                                     \
    }                                                                               \
  } while (0)

namespace perceptron {
namespace utils {

template<bool trans>
cublasOperation_t
trans2operation() {
  if constexpr(trans) {
    return CUBLAS_OP_T;
  } else {
    return CUBLAS_OP_N;
  }
}

cublasOperation_t
trans2operation(bool trans);

cublasOperation_t
inverse_trans(cublasOperation_t trans);

class CuBLASHandle {
public:
  static cublasHandle_t getInstance();

  CuBLASHandle() = delete;
  CuBLASHandle(const CuBLASHandle &) = delete;
  CuBLASHandle(CuBLASHandle &&) = delete;
  CuBLASHandle &operator=(const CuBLASHandle &) = delete;
  CuBLASHandle &operator=(CuBLASHandle &&) = delete;

  static void set_stream(cudaStream_t stream);

private:
  struct cublas_handle_deleter {
    void operator()(cublasHandle_t *handle_ptr) {
      CUBLAS_CHECK(cublasDestroy(*handle_ptr));
      delete handle_ptr;
    }
  };

  static std::unique_ptr<cublasHandle_t, cublas_handle_deleter>
  cublas_init_handle();

  static std::unique_ptr<cublasHandle_t, cublas_handle_deleter> m_handle;
};

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_CUBLASUTILS_H
