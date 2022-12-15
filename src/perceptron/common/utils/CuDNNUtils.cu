#include "perceptron/common/utils/CuDNNUtils.h"

namespace perceptron {
namespace utils {

std::unique_ptr<cudnnHandle_t, CuDNNHandle::cudnn_handle_deleter> CuDNNHandle::m_handle{};

cudnnHandle_t
CuDNNHandle::getInstance() {
  if (nullptr == m_handle) {
    m_handle = cudnn_init_handle();
  }

  return *m_handle;
}

std::unique_ptr<cudnnHandle_t, CuDNNHandle::cudnn_handle_deleter>
CuDNNHandle::cudnn_init_handle() {
  std::unique_ptr<cudnnHandle_t> handle_ptr{new cudnnHandle_t};
  CUDNN_CHECK(cudnnCreate(handle_ptr.get()));

  return std::unique_ptr<cudnnHandle_t, cudnn_handle_deleter>{handle_ptr.release(), cudnn_handle_deleter{}};
}

} // perceptron
} // utils
