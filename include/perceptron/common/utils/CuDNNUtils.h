#ifndef PERCEPTRON_COMMON_UTILS_CUDNNUTILS_H
#define PERCEPTRON_COMMON_UTILS_CUDNNUTILS_H

#include <stdexcept>
#include <cstdio>
#include <memory>
#include <cstdint>

#include <cudnn.h>

#define CUDNN_CHECK(err)                                                            \
  do                                                                                \
  {                                                                                 \
    cudnnStatus_t err_ = (err);                                                     \
    if (err_ != CUDNN_STATUS_SUCCESS)                                               \
    {                                                                               \
      std::fprintf(stderr, "cudnn error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
      throw std::runtime_error("cudnn error");                                      \
    }                                                                               \
  } while (0)

namespace perceptron {
namespace utils {

namespace details {
template<typename T>
struct cudnn_data_type;

template<>
struct cudnn_data_type<float> {
  static constexpr auto value = CUDNN_DATA_FLOAT;
};

template<>
struct cudnn_data_type<double> {
  static constexpr auto value = CUDNN_DATA_DOUBLE;
};

template<>
struct cudnn_data_type<std::int32_t> {
  static constexpr auto value = CUDNN_DATA_INT32;
};

template<>
struct cudnn_data_type<std::int64_t> {
  static constexpr auto value = CUDNN_DATA_INT64;
};

template<>
struct cudnn_data_type<std::int8_t> {
  static constexpr auto value = CUDNN_DATA_INT8;
};

template<>
struct cudnn_data_type<bool> {
  static constexpr auto value = CUDNN_DATA_BOOLEAN;
};
}

class CuDNNHandle {
public:
  static cudnnHandle_t
  getInstance();

  template<typename T>
  static cudnnDataType_t
  get_data_type();

  CuDNNHandle() = delete;
  CuDNNHandle(const CuDNNHandle &) = delete;
  CuDNNHandle(CuDNNHandle &&) = delete;
  CuDNNHandle &operator=(const CuDNNHandle &) = delete;
  CuDNNHandle &operator=(CuDNNHandle &&) = delete;

private:
  struct cudnn_handle_deleter {
    void operator()(cudnnHandle_t *handle_ptr) {
      CUDNN_CHECK(cudnnDestroy(*handle_ptr));
      delete handle_ptr;
    }
  };

  static std::unique_ptr<cudnnHandle_t, cudnn_handle_deleter>
  cudnn_init_handle();

  static std::unique_ptr<cudnnHandle_t, cudnn_handle_deleter> m_handle;
};

template<typename T>
cudnnDataType_t
CuDNNHandle::get_data_type() {
  return details
  ::cudnn_data_type<T>::value;
}

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_CUDNNUTILS_H
