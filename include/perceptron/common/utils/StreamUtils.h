#ifndef PERCEPTRON_COMMON_UTILS_STREAMUTILS_H
#define PERCEPTRON_COMMON_UTILS_STREAMUTILS_H

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/CudaUtils.h"

#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace perceptron {
namespace utils {

struct cu_stream_deleter {
  void operator()(cudaStream_t *stream_ptr) const {
    CUDA_CHECK(cudaStreamDestroy(*stream_ptr));
    delete stream_ptr;
  }
};

struct cu_streams_deleter {
  explicit cu_streams_deleter(std::size_t n_streams) : m_n_streams{n_streams} {}
  void operator()(cudaStream_t *streams_ptr) const {
    for (std::size_t i = m_n_streams; i > 0; --i) {
      CUDA_CHECK(cudaStreamDestroy(streams_ptr[i - 1]));
    }

    delete[] streams_ptr;
  }

private:
  std::size_t m_n_streams;
};

using CudaStreamOwner = std::unique_ptr<cudaStream_t, cu_stream_deleter>;
using CudaStreamsOwner = std::vector<CudaStreamOwner>;

CudaStreamOwner
cu_create_stream();

CudaStreamsOwner
cu_create_streams(size_type size);

void
cu_wait_streams(const CudaStreamsOwner &streams);

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_STREAMUTILS_H
